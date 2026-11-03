"""Saved setups: everything the Setup panel and the map hold for a run (the sounding and its
changes, grid, domain, physics, storms, radars, fronts, initiation areas, nudgers, OutbreakLab
mode) kept under a name as setups\\<name>.json.

A setup is self-contained so it can be sent to someone else: a sounding from a file travels as
the file's text.  A setup someone sent (or a run's settings.json) is imported the same way; its
sounding, if it is only a file on the computer it came from, is replaced by the viewer's
default and the import says so.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path

from . import PROJECT, __version__

SETUPS = PROJECT / "setups"
FORMAT = "arht-model-viewer-setup"
OLD_FORMATS = (FORMAT, "atmosphere-sandbox-setup")   # setups saved before the rename
MAX_BYTES = 8 * 1024 * 1024
# what a setup may hold (a run's settings): anything else in an imported file is left out
KEYS = ("sounding", "grid", "physics", "map", "start_time", "speed_limit", "frame_interval", "radars",
        "radar_refine", "grid_motion", "grid_follow_min", "refine", "large_scale", "domain", "storms", "nudgers", "zones", "boundaries", "low", "auto_storms",
        "persistence", "engine", "seed", "fire_area", "wind_brushes", "synoptic")
SOUNDING_MODS = ("mix", "moist", "mid_moist", "cap", "wind_scale", "ice_cap", "llj", "aloft")


def safe_name(name) -> str:
    """A setup's name as a file name (Windows-safe, at most 80 characters)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", str(name or ""))
    s = re.sub(r"\s+", " ", s).strip(" .")[:80].strip(" .")
    return s or "setup"


def _path(name) -> Path:
    return SETUPS / f"{safe_name(name)}.json"


def _readable_sounding(p: Path) -> bool:
    """A sounding file the viewer may read: one of the library's (or the viewer's own data)."""
    from .library import LIBRARY
    try:
        rp = p.resolve()
    except OSError:
        return False
    return p.is_file() and (LIBRARY.allowed(p) or (PROJECT / "data").resolve() in rp.parents)


def portable(settings: dict) -> tuple[dict, list[str]]:
    """A complete, self-contained copy of run settings (missing keys from the defaults, the
    sounding's file as its text) and what had to change on the way."""
    from .sim import DEFAULT_SETTINGS, load_sounding
    from .sounding import parse_file
    s = json.loads(json.dumps(DEFAULT_SETTINGS))
    s.update({k: v for k, v in json.loads(json.dumps(settings)).items() if k in KEYS})
    s.setdefault("persistence", 0.5)
    warnings = []
    snd = dict(s.get("sounding") or {})
    if snd.get("file"):
        p = Path(str(snd["file"]))
        if not p.is_absolute():
            p = PROJECT / p
        if _readable_sounding(p):
            title = snd.get("title") or parse_file(p).title
            snd = {k: v for k, v in snd.items() if k != "file"}
            snd.update(text=p.read_text(errors="replace"), title=title, source=p.name)
        else:
            warnings.append(f"Its sounding was a file on the computer it came from ({p.name}), not part of the "
                            "setup: the viewer's default sounding stands in. Pick another under Atmosphere.")
            snd = None
    if snd is not None:
        try:
            load_sounding(snd)
        except Exception as exc:
            warnings.append(f"Its sounding could not be read ({exc}): the viewer's default sounding stands in.")
            snd = None
    if snd is None:
        d = dict(DEFAULT_SETTINGS["sounding"])
        p = PROJECT / d.pop("file")
        snd = dict(d, text=p.read_text(errors="replace"), title=parse_file(p).title, source=p.name)
    s["sounding"] = snd
    return s, warnings


def summary(settings: dict) -> dict:
    """What a setup is, in a few words for the list."""
    snd = settings.get("sounding") or {}
    title = snd.get("title") or (Path(snd["file"]).stem if snd.get("file") else
                                 "Weisman-Klemp analytic" if snd.get("analytic") else "sounding")
    dom = settings.get("domain") or {}
    return {
        "sounding": str(title), "changes": [k for k in SOUNDING_MODS if snd.get(k) and k != "ice_cap"],
        "grid": settings.get("grid"), "domain": {"mode": dom.get("mode", "unlimited"), "size_km": dom.get("size_km"),
                                                 "max_km": dom.get("max_km") or 0},
        "storms": len(settings.get("storms") or []), "radars": [str(r.get("icao", "")) for r in settings.get("radars") or []],
        "boundaries": len(settings.get("boundaries") or []), "zones": len(settings.get("zones") or []),
        "nudgers": len(settings.get("nudgers") or []), "auto": bool((settings.get("auto_storms") or {}).get("on")),
        "synoptic": bool((settings.get("synoptic") or {}).get("on")),
        "map": settings.get("map"), "start_time": settings.get("start_time"),
    }


def _read(f: Path) -> dict | None:
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(d, dict) and d.get("format") in OLD_FORMATS and isinstance(d.get("settings"), dict):
        return d
    return None


def entries() -> list[dict]:
    """The saved setups, newest first."""
    out = []
    if not SETUPS.is_dir():
        return out
    for f in sorted(SETUPS.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        d = _read(f)
        if d is None:
            continue
        out.append({"name": f.stem, "saved": d.get("saved") or "", "note": d.get("note") or "",
                    "summary": summary(d["settings"]), "bytes": f.stat().st_size})
    return out


def save(name: str, settings: dict, note: str = "") -> dict:
    if not isinstance(settings, dict):
        raise ValueError("no settings to save")
    s, warnings = portable(settings)
    SETUPS.mkdir(parents=True, exist_ok=True)
    f = _path(name)
    doc = {"format": FORMAT, "version": 1, "name": f.stem, "note": str(note or ""),
           "saved": dt.datetime.now().isoformat(timespec="seconds"), "viewer": __version__, "settings": s}
    part = f.with_name(f.name + ".part")
    part.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(part, f)
    return {"name": f.stem, "warnings": warnings, "summary": summary(s)}


def load(name: str) -> dict:
    f = _path(name)
    d = _read(f) if f.is_file() else None
    if d is None:
        raise LookupError(name)
    return {"name": f.stem, "note": d.get("note") or "", "saved": d.get("saved") or "", "settings": d["settings"]}


def delete(name: str) -> None:
    f = _path(name)
    if not f.is_file():
        raise LookupError(name)
    f.unlink()


def file_path(name: str) -> Path:
    f = _path(name)
    if not f.is_file():
        raise LookupError(name)
    return f


def free_name(name: str) -> str:
    """``name``, or ``name (2)``, ``name (3)`` ... if a setup has it already."""
    base = safe_name(name)
    n, i = base, 2
    while _path(n).exists():
        n, i = f"{base} ({i})", i + 1
    return n


def import_text(name: str, text: str) -> dict:
    """A setup someone sent (or a run's settings.json), saved under a free name."""
    if len(text) > MAX_BYTES:
        raise ValueError("that file is too big for a setup")
    try:
        d = json.loads(text)
    except ValueError:
        raise ValueError("not a setup: it is not JSON") from None
    if isinstance(d, dict) and d.get("format") in OLD_FORMATS and isinstance(d.get("settings"), dict):
        settings, note, name = d["settings"], d.get("note") or "", name or d.get("name") or ""
    elif isinstance(d, dict) and ("sounding" in d or "grid" in d):
        settings, note = d, ""                     # a run's settings.json
    else:
        raise ValueError("not an Atmos-WRF setup (or a run's settings.json)")
    if not name or name.lower() in ("settings", "setup", "imported setup"):
        s = summary(settings)                      # a run's settings.json: named after what it is
        name = f"{s['sounding']} {s['grid'] or ''}".strip()
    return save(free_name(name), settings, note)


def reveal(name: str | None = None) -> None:
    """The setups folder in Explorer, with the setup's file selected (to attach it to a message)."""
    SETUPS.mkdir(parents=True, exist_ok=True)
    f = _path(name) if name else None
    if os.name == "nt":
        if f is not None and f.is_file():
            subprocess.Popen(f'explorer /select,"{f}"')
        else:
            os.startfile(SETUPS)                     # noqa: S606 - our own folder
    else:
        subprocess.Popen(["xdg-open", str(SETUPS)])
