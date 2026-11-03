"""Soundings and color tables on this computer.

The viewer looks for them in folders: its own data folders, the usual places
on any PC (Documents, Downloads, Desktop, SHARPpy's folder, GR2Analyst's and
GRLevelX's color tables under Program Files on every drive), folders the user
adds, and the folders a scan of every fixed drive turned up.  A file counts by
its content, not its name: a SHARPpy (%RAW%), University of Wyoming or WRF/CM1
input_sounding sounding, or a GR2Analyst/GRLevelX color table (.pal with
Color: lines).

The folders the user adds and the scan found are kept in config/library.json;
what was found in them in output/cache/library_index.json (re-read only when a
file changes).  Imported files are copied to data/imported/.
"""
from __future__ import annotations

import json
import logging
import os
import re
import string
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import PROJECT

log = logging.getLogger(__name__)

CONFIG = PROJECT / "config" / "library.json"
INDEX = PROJECT / "output" / "cache" / "library_index.json"
IMPORTED = PROJECT / "data" / "imported"

SOUNDING_EXT = {"", ".txt", ".snd", ".dat", ".csv", ".sounding", ".raw", ".sharppy", ".json"}
PALETTE_EXT = {".pal"}
MAX_BYTES = 3_000_000           # bigger files are not soundings or color tables
HEAD = 4096
MAX_DEPTH = 6                   # below a library folder
MAX_ENTRIES = 60_000            # directory entries looked at per library folder
# never worth walking: system, package and cache folders
SKIP_DIRS = {
    "windows", "$recycle.bin", "system volume information", "recovery", "perflogs", "programdata",
    "appdata", "windowsapps", "winsxs", "msocache", "$windows.~bt", "$windows.~ws", "$sysreset",
    "node_modules", ".git", ".svn", ".hg", "__pycache__", "site-packages", "dist-packages", ".venv", "venv",
    ".conda", "conda", "anaconda3", "miniconda3", "miniforge3", ".cache", ".npm", ".nuget", ".gradle", ".cargo",
    ".rustup", ".vscode", ".idea", "msys64", "cygwin64", "steamapps", "windows kits", "microsoft visual studio",
}
GENERIC_NAMES = {"input_sounding", "input_sharppy", "sounding", "snd", "raw", "data"}
# OneDrive / cloud placeholders: reading one downloads it, so they are left alone
_CLOUD = 0x1000 | 0x40000 | 0x400000     # OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS


def _in_cloud(st) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & _CLOUD)


# ---------------------------------------------------------------- what a file is
_NUM = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")


def _numbers(tokens) -> bool:
    return bool(tokens) and all(_NUM.match(t) for t in tokens)


def classify(path: Path, head: str):
    """("sounding", format) | ("palette", product) | None, from a file's first few KB."""
    suf = path.suffix.lower()
    if suf in PALETTE_EXT:
        if re.search(r"^\s*(solid)?color4?\s*:", head, re.I | re.M):
            m = re.search(r"^\s*product\s*:\s*(\S+)", head, re.I | re.M)
            return "palette", (m.group(1).upper() if m else "")
        return None
    if suf not in SOUNDING_EXT:
        return None
    if head.lstrip().startswith("{"):
        # JSON soundings are small: the columns must all be in the first few KB's keys
        keys = {k.lower() for k in re.findall(r'"([A-Za-z_]+)"\s*:\s*\[', head)}
        from .sounding import JSON_COLUMNS
        if all(any(a in keys for a in names) for names in JSON_COLUMNS.values()):
            return "sounding", "JSON"
        return None
    if "%RAW%" in head:
        return "sounding", "SHARPpy"
    if "PRES" in head and "HGHT" in head and "DWPT" in head:
        return "sounding", "Wyoming"
    lines = [ln.split() for ln in head.splitlines() if ln.strip()]
    if (len(lines) >= 4 and len(lines[0]) == 3 and _numbers(lines[0])
            and all(len(ln) == 5 and _numbers(ln) for ln in lines[1:4])):
        return "sounding", "input_sounding"
    # unmarked SHARPpy rows: pressure, height, T, Td, wind direction, speed, pressure falling
    rows = [ln.replace(",", " ").split() for ln in head.splitlines() if ln.count(",") >= 5]
    rows = [r for r in rows if len(r) >= 6 and _numbers(r[:6])]
    if len(rows) >= 5:
        p = [float(r[0]) for r in rows]
        if 1100.0 >= p[0] > p[-1] > 0.0 and all(a >= b for a, b in zip(p, p[1:])):
            return "sounding", "SHARPpy table"
    return None


def _read_head(path: Path) -> str:
    with open(path, "rb") as f:
        return f.read(HEAD).decode("utf-8", errors="replace")


# ---------------------------------------------------------------- where to look
def _fixed_drives() -> list[Path]:
    if os.name != "nt":
        return [Path("/")]
    import ctypes
    out = []
    mask = ctypes.windll.kernel32.GetLogicalDrives()
    for i, letter in enumerate(string.ascii_uppercase):
        if mask & (1 << i) and ctypes.windll.kernel32.GetDriveTypeW(f"{letter}:\\") == 3:   # DRIVE_FIXED
            out.append(Path(f"{letter}:\\"))
    return out


def _known_folder(guid: str) -> Path | None:
    """A Windows known folder (follows OneDrive / moved folders), independent of
    the HOME the package redirects for gpuwm."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD), ("d", ctypes.c_ubyte * 8)]

    h = guid.strip("{}").split("-")
    g = GUID(int(h[0], 16), int(h[1], 16), int(h[2], 16),
             (ctypes.c_ubyte * 8)(*bytes.fromhex(h[3] + h[4])))
    p = ctypes.c_wchar_p()
    try:
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(p)) != 0:
            return None
        return Path(p.value)
    except OSError:
        return None
    finally:
        if p:
            ctypes.windll.ole32.CoTaskMemFree(p)


def user_home() -> Path:
    from . import USER_HOME
    return USER_HOME


def usual_places() -> list[dict]:
    """Where soundings and color tables usually are on a PC."""
    out = []
    folders = {"Documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
               "Downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
               "Desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}"}
    for label, guid in folders.items():
        p = _known_folder(guid) or (user_home() / label)
        out.append({"path": str(p), "label": label, "recursive": True})
    out.append({"path": str(user_home() / ".sharppy"), "label": "SHARPpy", "recursive": True})
    for drive in _fixed_drives():
        for pf in ("Program Files", "Program Files (x86)"):
            base = drive / pf
            try:
                apps = [d for d in base.iterdir() if d.is_dir() and d.name.lower().startswith(("gr2analyst", "grlevel", "supercell"))]
            except OSError:
                continue
            for app in apps:
                out.append({"path": str(app), "label": app.name, "recursive": True})
    return out


def app_folders() -> list[dict]:
    return [{"path": str(PROJECT / "data" / "soundings"), "label": "Built-in library", "recursive": True},
            {"path": str(PROJECT / "data" / "palettes"), "label": "Built-in color tables", "recursive": True},
            {"path": str(IMPORTED), "label": "Imported", "recursive": True}]


# ---------------------------------------------------------------- the library
class Library:
    def __init__(self):
        self.lock = threading.RLock()
        self.entries: dict[str, dict] = {}          # path -> entry
        self.status = {"state": "idle", "message": "", "dirs": 0, "found": 0, "current": ""}
        self._stop = threading.Event()
        self._thread = None
        self._pending = False
        self._cache = self._load_json(INDEX, {})
        self.config = self._load_json(CONFIG, {"folders": [], "found": []})

    # ------------------------------------------------------------ files
    @staticmethod
    def _load_json(path: Path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _save_config(self):
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(self.config, indent=1), encoding="utf-8")

    def _save_cache(self):
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        INDEX.write_text(json.dumps(self._cache), encoding="utf-8")

    # ------------------------------------------------------------ folders
    def roots(self) -> list[dict]:
        out, seen = [], set()
        groups = (("app", app_folders()), ("yours", self.config.get("folders", [])),
                  ("usual", usual_places()),
                  ("found", [{"path": p, "label": Path(p).name or p, "recursive": False}
                             for p in self.config.get("found", [])]))
        for origin, items in groups:
            for r in items:
                key = os.path.normcase(os.path.abspath(r["path"]))
                if key in seen:
                    continue
                seen.add(key)
                out.append({**r, "origin": origin, "exists": Path(r["path"]).is_dir()})
        return out

    def add_folder(self, path: str, recursive: bool = True) -> dict:
        p = Path(path).expanduser()
        if not p.is_dir():
            raise ValueError(f"not a folder: {path}")
        p = p.resolve()
        with self.lock:
            folders = self.config.setdefault("folders", [])
            if not any(os.path.normcase(f["path"]) == os.path.normcase(str(p)) for f in folders):
                folders.append({"path": str(p), "label": p.name or str(p), "recursive": bool(recursive)})
                self._save_config()
        self.refresh()
        return {"path": str(p)}

    def remove_folder(self, path: str) -> None:
        key = os.path.normcase(os.path.abspath(path))
        with self.lock:
            for name in ("folders", "found"):
                items = self.config.get(name, [])
                keep = [f for f in items
                        if os.path.normcase(os.path.abspath(f["path"] if isinstance(f, dict) else f)) != key]
                self.config[name] = keep
            self._save_config()
        self.refresh()

    def allowed(self, path: Path) -> bool:
        """A file inside one of the library's folders (or found in them)."""
        try:
            rp = path.resolve()
        except OSError:
            return False
        if str(rp) in self.entries:
            return True
        for r in self.roots():
            try:
                root = Path(r["path"]).resolve()
            except OSError:
                continue
            if rp.parent == root or (r["recursive"] and root in rp.parents):
                return True
        return False

    # ------------------------------------------------------------ lists
    def soundings(self) -> list[dict]:
        with self.lock:
            items = [e for e in self.entries.values() if e["kind"] == "sounding"]
        items.sort(key=lambda e: (e["order"], e["group"].lower(), e["name"].lower()))
        return [{"group": e["group"], "name": e["name"], "path": e["path"], "format": e["fmt"]} for e in items]

    def palettes(self) -> list[dict]:
        with self.lock:
            items = [e for e in self.entries.values() if e["kind"] == "palette"]
        items.sort(key=lambda e: (e["order"], e["name"].lower()))
        return [{"name": e["name"], "path": e["path"], "product": e["fmt"], "group": e["group"]} for e in items]

    def summary(self) -> dict:
        with self.lock:
            counts = {}
            for e in self.entries.values():
                c = counts.setdefault(e["root"], {"soundings": 0, "palettes": 0})
                c["soundings" if e["kind"] == "sounding" else "palettes"] += 1
            roots = [{**r, **counts.get(os.path.normcase(os.path.abspath(r["path"])),
                                        {"soundings": 0, "palettes": 0})} for r in self.roots()]
            return {"status": dict(self.status), "roots": roots,
                    "soundings": sum(1 for e in self.entries.values() if e["kind"] == "sounding"),
                    "palettes": sum(1 for e in self.entries.values() if e["kind"] == "palette")}

    # ------------------------------------------------------------ indexing
    def _entry(self, f: Path, st, root: dict, order: int, rel: Path):
        key = str(f)
        c = self._cache.get(key)
        if c and c[0] == st.st_mtime and c[1] == st.st_size:
            kind, fmt = c[2], c[3]
        else:
            try:
                got = classify(f, _read_head(f))
            except OSError:
                got = None
            kind, fmt = got if got else ("", "")
            self._cache[key] = [st.st_mtime, st.st_size, kind, fmt]
        if not kind:
            return None
        name = f.stem
        if name.lower() in GENERIC_NAMES:
            name = f"{f.name} ({f.parent.name})"
        group = root["label"] + ("" if str(rel.parent) in (".", "") else " › " + " › ".join(rel.parent.parts))
        return {"kind": kind, "fmt": fmt, "name": name, "path": key, "group": group, "order": order,
                "root": os.path.normcase(os.path.abspath(root["path"]))}

    def _walk(self, root: dict, order: int, out: dict) -> None:
        base = Path(root["path"])
        if not base.is_dir():
            return
        stack, seen = [(base, 0)], 0
        while stack and not self._stop.is_set():
            d, depth = stack.pop()
            try:
                it = list(os.scandir(d))
            except OSError:
                continue
            for e in it:
                seen += 1
                if seen > MAX_ENTRIES:
                    return
                try:
                    if e.is_dir(follow_symlinks=False):
                        if (root["recursive"] and depth < MAX_DEPTH and e.name.lower() not in SKIP_DIRS
                                and not e.name.startswith("$") and not _is_junction(e)
                                and not _in_cloud(e.stat(follow_symlinks=False))):
                            stack.append((Path(e.path), depth + 1))
                        continue
                    suf = os.path.splitext(e.name)[1].lower()
                    if suf not in SOUNDING_EXT and suf not in PALETTE_EXT:
                        continue
                    st = e.stat()
                    if st.st_size > MAX_BYTES or st.st_size == 0 or _in_cloud(st):
                        continue
                    f = Path(e.path)
                    ent = self._entry(f, st, root, order, f.relative_to(base))
                    if ent:
                        out[ent["path"]] = ent
                except OSError:
                    continue

    def refresh(self, wait: bool = False) -> None:
        """Look through every library folder again (in the background)."""
        with self.lock:
            if self._thread is not None and self._thread.is_alive():
                self._pending = True
                return
            self._pending = False
            self._stop.clear()
            self._thread = threading.Thread(target=self._refresh_run, name="library", daemon=True)
            self._thread.start()
        if wait:
            self._thread.join()

    def _refresh_run(self):
        while True:
            self.status.update(state="indexing", message="Looking through the library folders…")
            out = {}
            first = not self.entries
            for order, root in enumerate(self.roots()):
                self.status["current"] = root["path"]
                self._walk(root, order, out)
                if first:
                    # the first look-through on a new PC reads every file once (minutes in a big
                    # Desktop): show each folder as it is done, the viewer's own soundings first
                    with self.lock:
                        self.entries = dict(out)
            with self.lock:
                self.entries = out
                self._save_cache()
                again = getattr(self, "_pending", False)
                self._pending = False
                self.status.update(state="idle", current="", message=(
                    f"{sum(e['kind'] == 'sounding' for e in out.values())} soundings, "
                    f"{sum(e['kind'] == 'palette' for e in out.values())} color tables"))
            if not again:
                return

    # ------------------------------------------------------------ the whole computer
    def scan_computer(self) -> None:
        """Walk every fixed drive (skipping system and package folders) for
        folders with soundings or color tables, and add them to the library."""
        with self.lock:
            if self._thread is not None and self._thread.is_alive():
                if self.status["state"] == "scanning":
                    return
                self._stop.set()
                self._thread.join(timeout=10)
            self._stop.clear()
            self._thread = threading.Thread(target=self._scan_run, name="library-scan", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _scan_run(self):
        t0 = time.time()
        self.status.update(state="scanning", dirs=0, found=0, current="", message="Scanning this computer…")
        skip_paths = {os.path.normcase(str(PROJECT / x)) for x in ("runtime", "output")}
        found_dirs: set[str] = set()
        hits = 0
        for drive in _fixed_drives():
            stack = [drive]
            while stack and not self._stop.is_set():
                d = stack.pop()
                self.status["dirs"] += 1
                if self.status["dirs"] % 200 == 0:
                    self.status["current"] = str(d)
                try:
                    it = list(os.scandir(d))
                except OSError:
                    continue
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            n = e.name.lower()
                            if (n in SKIP_DIRS or n.startswith("$") or _is_junction(e)
                                    or os.path.normcase(e.path) in skip_paths
                                    or _in_cloud(e.stat(follow_symlinks=False))):
                                continue
                            stack.append(Path(e.path))
                            continue
                        suf = os.path.splitext(e.name)[1].lower()
                        if suf not in SOUNDING_EXT and suf not in PALETTE_EXT:
                            continue
                        st = e.stat()
                        if st.st_size > MAX_BYTES or st.st_size == 0 or _in_cloud(st):
                            continue
                        key = e.path
                        c = self._cache.get(key)
                        if c and c[0] == st.st_mtime and c[1] == st.st_size:
                            kind = c[2]
                        else:
                            got = classify(Path(key), _read_head(Path(key)))
                            kind = got[0] if got else ""
                            self._cache[key] = [st.st_mtime, st.st_size, kind, got[1] if got else ""]
                        if kind:
                            hits += 1
                            found_dirs.add(str(Path(key).parent))
                            self.status["found"] = hits
                    except OSError:
                        continue
        stopped = self._stop.is_set()
        with self.lock:
            covered = [Path(r["path"]) for r in self.roots() if r["origin"] != "found"]
            new = sorted(p for p in found_dirs if not _inside(Path(p), covered))
            self.config["found"] = sorted(set(self.config.get("found", [])) | set(new))
            self._save_config()
            self._save_cache()
        self.status.update(state="idle", current="", message=(
            f"{'Stopped' if stopped else 'Scanned'} in {time.time() - t0:.0f} s: {hits} files in "
            f"{len(found_dirs)} folders ({len(new)} new)"))
        self._stop.clear()
        self.refresh()

    # ------------------------------------------------------------ imports
    def import_file(self, kind: str, name: str, text: str) -> dict:
        """Copy an opened file into data/imported/ so runs and later sessions find it."""
        if kind not in ("sounding", "palette"):
            raise ValueError("kind must be sounding or palette")
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(name).name).strip(" .") or "imported"
        if kind == "palette" and not safe.lower().endswith(".pal"):
            safe += ".pal"
        if kind == "sounding" and Path(safe).suffix.lower() not in SOUNDING_EXT:
            safe += ".txt"
        got = classify(Path(safe), text[:HEAD])
        if not got or got[0] != kind:
            if kind == "palette":
                raise ValueError(f"{name} is not a GR2Analyst / GRLevelX color table")
            from . import sounding as snd
            try:
                s = snd.parse_any(text)
                ok = len(s.z) >= 5
            except Exception:
                ok = False
            if not ok:
                raise ValueError(f"{name} is not a sounding the viewer can read "
                                 "(SHARPpy, University of Wyoming text, WRF/CM1 input_sounding or JSON)")
            got = ("sounding", "text")
        d = IMPORTED / ("soundings" if kind == "sounding" else "palettes")
        d.mkdir(parents=True, exist_ok=True)
        f = d / safe
        f.write_text(text, encoding="utf-8")
        st = f.stat()
        self._cache[str(f)] = [st.st_mtime, st.st_size, got[0], got[1]]
        root = next(r for r in self.roots() if r["path"] == str(IMPORTED))
        ent = self._entry(f, st, root, 2, f.relative_to(IMPORTED))
        with self.lock:
            self.entries[ent["path"]] = ent
        return {"path": ent["path"], "name": ent["name"], "kind": kind, "format": ent["fmt"]}


def _is_junction(e) -> bool:
    try:
        return e.is_junction()
    except AttributeError:          # Python < 3.12
        return False


def _inside(p: Path, roots: list[Path]) -> bool:
    try:
        rp = p.resolve()
    except OSError:
        return False
    for r in roots:
        try:
            rr = r.resolve()
        except OSError:
            continue
        if rp == rr or rr in rp.parents:
            return True
    return False


def pick_folder(title: str = "Choose a folder with soundings or color tables") -> str | None:
    """The system's folder dialog (the viewer runs on this computer).  None if cancelled."""
    code = ("import tkinter as tk\nfrom tkinter import filedialog\nr = tk.Tk(); r.withdraw()\n"
            "r.attributes('-topmost', True)\n"
            f"print(filedialog.askdirectory(title={title!r}, mustexist=True) or '')\n")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    p = out.stdout.strip()
    return p or None


LIBRARY = Library()
