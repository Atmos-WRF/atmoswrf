"""Atmos-WRF web server: the browser UI, the run API and
GR2Analyst polling (the Iowa-State-style layout: grlevel2.cfg, <SITE>/dir.list,
<SITE>/<file>; polling.py).

    runtime\\venv\\Scripts\\python.exe -m arhtmodelviewer serve [port] [--lan]

--lan also shares the polling feed (and nothing else) with other computers.
"""
from __future__ import annotations

import email.utils
import gzip
import ipaddress
import json
import logging
import math
import mimetypes
import os
import re
import socket
import threading
import time
import urllib.parse
import struct
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import (PROJECT, boundaries, geo, palettes, placefile, polling, products, radar as rd, setups,
               sounding as snd, synoptic)
from .environment import GRID_PRESETS, MICROPHYSICS, MICROPHYSICS_MENU, physics_suites
from .library import LIBRARY, pick_folder

log = logging.getLogger(__name__)

WEB = PROJECT / "web"
_STARTED = time.time()          # Last-Modified of grlevel2.cfg and of a radar's empty dir.list
_POLLERS: set = set()           # (client, site) pairs already logged
_HEARD: dict = {}               # (client, request line + status) -> when last logged
_SWATHS: dict = {}              # (run, swath, version) -> (PNG, box) of the placefiles' images, the newest few
_SWATHS_LOCK = threading.Lock()
_NET = {"port": 8830, "lan": []}   # this server's port and, with --lan, its addresses on the network


def _lan_addresses() -> list[str]:
    """This PC's IPv4 addresses that other computers can reach (LAN, VPN), the default route's first."""
    out = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))           # picks the outgoing interface; sends nothing
            out.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        out += [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    keep = []
    for a in out:
        ip = ipaddress.ip_address(a)
        if not (ip.is_loopback or ip.is_link_local or ip.is_unspecified) and a not in keep:
            keep.append(a)
    return keep


def _poll_links() -> dict:
    """The polling URLs to give GR2Analyst: on this PC, and from other computers (--lan)."""
    port = _NET["port"]
    return {"local": f"http://127.0.0.1:{port}/poll/",
            "lan": [f"http://{a}:{port}/poll/" for a in _NET["lan"]]}


class App:
    """Holds the current Simulation (one at a time)."""

    def __init__(self):
        self.sim = None
        self.changed = time.time()      # when the run changed (a polling viewer must list it again)
        self.lock = threading.RLock()
        self._sites = None

    def sites(self):
        if self._sites is None:
            self._sites = geo.radar_sites()
        return self._sites

    def new_run(self, settings: dict):
        from .sim import Simulation, free_gpu
        with self.lock:
            # read the settings (sounding) first: settings that fail leave the current run going
            new = Simulation(settings)
            old = self.sim
            if old is not None:
                old.stop()
            self.sim = new
            self.changed = time.time()
            # the new run sizes its memory budget from the free GPU memory when it builds:
            # give the old run's arrays back first, or a run started right after a big one
            # gets a fraction of the card
            if old is not None and old.release_gpu():
                del old
                free_gpu()
            return self.sim


APP = App()

# the synoptic preview: the simulated weather of the page's setup, before a run (one at a time)
_SYN = {"key": None, "world": None, "progress": None, "lock": threading.Lock()}


def _synoptic_world(body: dict):
    """(world, model time, extra payload) for a synoptic request: the running run's weather at its current
    time when ``run`` is set and the run is in synoptic mode, else the preview of ``settings`` at ``hour``
    (built the first time: the weather bank takes ~3 min once, a new day ~20 s)."""
    import datetime as _dt
    import random
    sim = APP.sim
    if body.get("run") and sim is not None and getattr(sim, "synoptic", None) is not None and sim.engine is not None:
        return sim.synoptic, float(sim.engine.t), {"source": "run", "seed": sim.synoptic.seed}
    s = body.get("settings") or {}
    m = s.get("map") or {}
    proj = geo.Lambert(float(m.get("lat0", 35.2)), float(m.get("lon0", -97.6)))
    start = _dt.datetime.fromisoformat(str(s.get("start_time") or "2024-05-06T15:00:00Z").replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=_dt.timezone.utc)
    cfg = {**synoptic.DEFAULTS, **(s.get("synoptic") or {})}
    seed = int(cfg.get("seed") or random.SystemRandom().randrange(1, 2 ** 31))
    cfg["seed"] = seed
    key = synoptic.world_key(proj, start, cfg, seed)
    with _SYN["lock"]:
        if _SYN["key"] != key:
            _SYN["world"] = None
            _SYN["progress"] = (0.0, "starting")

            def progress(f, msg):
                _SYN["progress"] = (float(f), msg)

            try:
                w = synoptic.world_for(proj, start, cfg, seed, progress=progress)
            finally:
                _SYN["progress"] = None
            w.keep_hourly = True
            _SYN.update(key=key, world=w, ready=set())
            threading.Thread(target=_synoptic_prefetch, args=(key, w), name="synoptic-prefetch", daemon=True).start()
        w = _SYN["world"]
    hour = min(max(round(float(body.get("hour", 0.0)) * 4.0) / 4.0, 0.0), float(getattr(w, "max_hour", 24)))
    return w, hour * 3600.0, {"source": "preview", "seed": seed, "hour": hour, "ready": sorted(_SYN.get("ready", ()))}


def _synoptic_prefetch(key, w) -> None:
    """Work out every forecast hour of a new preview in the background, in order, so the hour slider and the
    loop show them at once; stops when another preview takes over."""
    for h in range(0, int(getattr(w, "max_hour", 24)) + 1):
        if _SYN["key"] != key:
            return
        try:
            w.payload(h * 3600.0)
        except Exception:
            log.exception("synoptic prefetch")
            return
        if _SYN["key"] == key:
            _SYN["ready"].add(h)


def _synoptic_bank_warmup() -> None:
    """The weather bank (the QG model's climate, ~2.5 min once per PC) built while the viewer starts, not when the
    first preview asks for it."""
    try:
        synoptic.bank(lambda f, msg: _SYN.__setitem__("bank", (float(f), msg)))
    except Exception:
        log.exception("synoptic bank")
    _SYN["bank"] = None


def _sounding_payload(s: snd.Sounding) -> dict:
    out = {"sounding": s.to_json(), "diag": snd.diagnostics(s), "sharppy": s.to_sharppy()}
    try:
        out["parcel"] = snd.parcel_trace(s)          # the sounding editor's skew-T draws it
    except Exception:                                # a sounding too short to lift a parcel through
        log.debug("no parcel trace", exc_info=True)
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "Atmos-WRF/1.2"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        """The browser UI's own requests stay quiet; anything else (a radar viewer polling, a wrong
        address) is logged, each distinct request at most every 10 minutes."""
        path = urllib.parse.urlparse(getattr(self, "path", "") or "").path
        line = fmt % args
        if path.startswith(("/api/", "/static/")) or path in ("/", "/index.html", "/favicon.ico"):
            log.debug("%s - %s", self.address_string(), line)
            return
        key, now = (self.client_address[0], line), time.time()
        if now - _HEARD.get(key, 0.0) > 600.0:
            _HEARD[key] = now
            headers = getattr(self, "headers", None)
            log.info("request from %s: %s  (%s)", self.client_address[0], line,
                     headers.get("User-Agent", "no user agent") if headers else "no headers")

    # ------------------------------------------------------------ helpers
    def _send(self, code, body: bytes, ctype="application/json", encoding=None, cache=False, modified=None,
              headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if encoding:
            self.send_header("Content-Encoding", encoding)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if modified is not None:
            self.send_header("Last-Modified", email.utils.formatdate(modified, usegmt=True))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _not_modified(self, modified: float) -> bool:
        """A conditional GET/HEAD (If-Modified-Since) for something unchanged since: answer 304."""
        since = self.headers.get("If-Modified-Since")
        try:
            if since and int(modified) <= email.utils.parsedate_to_datetime(since).timestamp():
                self.send_response(304)
                self.send_header("Last-Modified", email.utils.formatdate(modified, usegmt=True))
                self.end_headers()
                return True
        except (TypeError, ValueError):
            pass
        return False

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=_jsonable).encode()
        if len(body) > 4096 and "gzip" in self.headers.get("Accept-Encoding", ""):
            self._send(code, gzip.compress(body, 5), encoding="gzip")
        else:
            self._send(code, body)

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode() or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    _FORWARDED = ("CF-Connecting-IP", "X-Forwarded-For", "Forwarded", "X-Real-IP")
    _LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")

    def _tunneled(self) -> bool:
        """A request that came in through a tunnel or proxy on this PC (chase mode's internet link, or anything
        else forwarding to us): from this PC's own address, but someone else's request -- it carries a forwarding
        header, or a Host that is not this PC's own name."""
        if any(self.headers.get(h) for h in self._FORWARDED):
            return True
        host = (self.headers.get("Host") or "127.0.0.1").rsplit(":", 1)[0].lower()
        if host.startswith("[") and "]" in (self.headers.get("Host") or ""):
            host = (self.headers.get("Host") or "").split("]")[0].lower() + "]"
        return not (host in self._LOCAL_HOSTS or host.endswith(".localhost"))

    def _is_local(self) -> bool:
        """This PC's own browser (or program): may use everything."""
        try:
            loop = ipaddress.ip_address(self.client_address[0]).is_loopback
        except ValueError:
            loop = False
        return loop and not self._tunneled()

    def _client_ip(self) -> str:
        """Who is asking: through a tunnel, the address it forwards for (for the join code's slow-down)."""
        for h in ("CF-Connecting-IP", "X-Real-IP"):
            if self.headers.get(h):
                return self.headers[h].strip()[:64]
        xf = self.headers.get("X-Forwarded-For")
        return xf.split(",")[0].strip()[:64] if xf else self.client_address[0]

    def _remote_refused(self, path: str) -> bool:
        """With --lan other computers get the polling feed only: the page and its API can change
        runs and files on this PC.  Through a tunnel (chase mode's internet link) only chase mode."""
        if self._is_local():
            return False
        from . import chase
        tunneled = self._tunneled()
        if not tunneled and path.startswith("/poll"):
            return False
        if not tunneled and (path == "/remote" or path.startswith("/api/r/")):
            return False                          # the phone remote: its API checks its own PIN tokens (remote.py)
        if chase.allowed(path, self.command, self.headers.get("X-Chase-Token")):
            return False                          # chase mode's friends, while hosting is on (chase.py)
        self._send(403, b"Only the GR2Analyst polling feed (/poll/) is shared with other computers.\n",
                   "text/plain")
        return True

    # ------------------------------------------------------------ GET
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        path = url.path
        if self._remote_refused(path):
            return None
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path.startswith("/poll"):
                # "/poll/KTLX/dir.list", or "/pollKTLX/dir.list" from a URL entered without its "/"
                return self._poll(path[len("/poll"):])
            if path == "/api/state":
                return self._state()
            if path == "/api/c/air":
                return self._chase_air(q)
            if path == "/api/defaults":
                from . import backend
                from .engine import CELL_BYTES
                from .sim import DEFAULT_SETTINGS
                defaults = DEFAULT_SETTINGS
                if backend.is_cpu():                # 15-40x slower than an NVIDIA card: start at 3 km, and on the
                    defaults = {**DEFAULT_SETTINGS, "grid": "3km",          # built-in core (suites untested there)
                                "physics": {**DEFAULT_SETTINGS["physics"], "suite": ""}}
                return self._json({"settings": defaults, "cell_bytes": CELL_BYTES,
                                   "backend": backend.describe(), "cpu": backend.is_cpu(),
                                   "grids": {k: {"label": g.label, "dx": g.dx, "nz": g.nz, "dt": g.dt}
                                             for k, g in GRID_PRESETS.items()},
                                   "vcps": rd.VCP_INFO, "products": {k: v[3] for k, v in products.CODECS.items()},
                                   "boundary_kinds": boundaries.KINDS,
                                   "microphysics": [{"id": k, "label": MICROPHYSICS[k]} for k in MICROPHYSICS_MENU],
                                   "suites": _suites(), "poll": _poll_links()})
            if path.startswith("/tiles/"):
                # a map tile (tiles.py): fetched from its provider once, then from the cache
                from . import tiles
                parts = path.split("/")
                try:
                    src, z, x, y = parts[2], int(parts[3]), int(parts[4]), int(parts[5].split(".")[0])
                    data = tiles.tile(src, z, x, y)
                except (IndexError, ValueError) as e:
                    return self._error(404, str(e))
                except Exception as e:
                    return self._error(502, f"map tile unavailable: {e}")
                ctype = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
                return self._send(200, data, ctype, cache=True)
            if path == "/api/tiles":
                from . import tiles
                return self._json(tiles.info())
            if path == "/api/volume/ground":
                # the 3-D view's ground: the map tiles of its box, on the model's plane (tiles.ground)
                from . import tiles
                sim = APP.sim
                if sim is None:
                    return self._error(404, "no run")
                srcs = [s for s in [q.get("src", "sat")] + (["roads", "places"] if q.get("labels") == "1" else []) if s in tiles.SOURCES]
                try:
                    png = tiles.ground(sim.proj, [float(q[k]) for k in ("x0", "y0", "x1", "y1")], srcs,
                                       max(256, min(int(q.get("px", 1024)), 2048)))
                except Exception as e:
                    return self._error(404, str(e))
                return self._send(200, png, "image/png")
            if path in ("/api/volume/landcover", "/api/volume/roads"):
                # the 3-D view's landscape from real data (landscape.py): WorldCover's land cover of a box, one
                # byte a pixel (row 0 north), or OpenStreetMap's roads in it (fetched in the background)
                from . import landscape
                sim = APP.sim
                if sim is None:
                    return self._error(404, "no run")
                try:
                    box = [float(q[k]) for k in ("x0", "y0", "x1", "y1")]
                    if path.endswith("roads"):
                        return self._json(landscape.roads(sim.proj, box, q.get("major") == "1"))
                    W, H, a = landscape.landcover(sim.proj, box, max(64, min(int(q.get("px", 1024)), 2048)))
                except Exception as e:
                    return self._error(404, str(e))
                head = json.dumps({"W": W, "H": H, "classes": landscape.CLASSES}).encode()
                return self._send(200, zlib.compress(struct.pack("<I", len(head)) + head + a.tobytes(), 1),
                                  "application/octet-stream", encoding="deflate")
            if path == "/remote":
                return self._send(200, (WEB / "remote.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/chase":
                return self._send(200, (WEB / "chase.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/volume/buildings":
                # chase mode's buildings (chase.py): OpenStreetMap's footprints in the box, fetched in the background
                from . import chase
                sim = APP.sim
                if sim is None:
                    return self._error(404, "no run")
                try:
                    return self._json(chase.buildings(sim.proj, [float(q[k]) for k in ("x0", "y0", "x1", "y1")]))
                except (KeyError, ValueError) as e:
                    return self._error(400, str(e))
            if path == "/api/c/host":
                from . import chase
                if not self._is_local():
                    return self._error(403, "only the host's own PC")
                return self._json(dict(chase.host(), urls=self._chase_urls(), tunnel=chase.tunnel_state()))
            if path.startswith("/api/r/"):
                return self._remote_get(path[len("/api/r/"):], q)
            if path == "/api/remote/config":
                from . import remote
                return self._json(dict(remote.config(), urls=self._remote_urls()))
            if path == "/api/loop/sources":
                from . import loops
                d = self._run_dir(q.get("run"))
                return self._json({"run": d.name, "sources": loops.sources(d)})
            if path == "/api/loop/status":
                from .loops import LOOPS
                return self._json(LOOPS.status(q.get("job", "")))
            if path == "/api/loop/file":
                from .loops import OUT
                f = OUT / Path(q.get("name", "")).name
                if f.suffix.lower() != ".gif" or not f.is_file():
                    return self._error(404, "no such loop")
                data = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/gif")
                self.send_header("Content-Length", str(len(data)))
                if q.get("download"):
                    self.send_header("Content-Disposition", f'attachment; filename="{f.name}"')
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
                return None
            if path == "/api/survey":
                return self._survey_json(q.get("run"))
            if path == "/api/survey/download":
                return self._survey_download(q.get("run"), q.get("kind", "kml"))
            if path == "/api/soundings":
                return self._json(LIBRARY.soundings())
            if path == "/api/library":
                return self._json(LIBRARY.summary())
            if path == "/api/sounding":
                p = Path(q.get("path", ""))
                if not p.is_file() or not LIBRARY.allowed(p):
                    return self._error(404, "sounding not found")
                return self._json(_sounding_payload(snd.parse_file(p)))
            if path == "/api/map":
                proj = geo.Lambert(float(q.get("lat0", 35.0)), float(q.get("lon0", -97.0)))
                lines = geo.map_lines(proj)
                body = json.dumps(lines, separators=(",", ":")).encode()
                return self._send(200, gzip.compress(body, 5), encoding="gzip", cache=True)
            if path == "/api/sites":
                proj = geo.Lambert(float(q.get("lat0", 35.0)), float(q.get("lon0", -97.0)))
                out = []
                for s in APP.sites():
                    x, y = proj.fwd(s["lat"], s["lon"])
                    out.append(dict(s, x=float(x) / 1000.0, y=float(y) / 1000.0))
                return self._json(out)
            if path == "/api/cities":
                proj = geo.Lambert(float(q.get("lat0", 35.0)), float(q.get("lon0", -97.0)))
                minpop = float(q.get("minpop", 2500))
                cs = [c for c in geo.cities() if c["pop"] >= minpop]
                if cs:
                    x, y = proj.fwd(np.array([c["lat"] for c in cs]), np.array([c["lon"] for c in cs]))
                    keep = (np.abs(x) < 2.6e6) & (np.abs(y) < 2.6e6)
                    cs = [dict(name=c["name"], pop=c["pop"], x=round(float(xx) / 1000, 2), y=round(float(yy) / 1000, 2))
                          for c, xx, yy, k in zip(cs, x, y, keep) if k]
                return self._json(cs)
            if path == "/api/palettes":
                return self._json(LIBRARY.palettes())
            if path == "/api/palette":
                p = Path(q.get("path", ""))
                if p.suffix.lower() != ".pal" or not p.is_file() or not LIBRARY.allowed(p):
                    return self._error(404, "palette not found")
                return self._json(palettes.parse_pal(p.read_text(errors="replace")))
            if path == "/api/frames":
                sim = APP.sim
                if sim is None:
                    return self._json([])
                fr = sim.frames
                return self._json([{"i": i, "t": f.t, "ms": f.epoch_ms, "patches": f.patches,
                                    "radars": list(f.radars)} for i, f in enumerate(fr)])
            if path == "/api/frame":
                return self._frame(q)
            if path in ("/api/metars", "/api/metar"):
                # the simulated METARs (metars.py): the map's station plots at a model time, or one station's
                sim = APP.sim
                net = getattr(sim, "metars", None)
                t = float(q["t"]) if q.get("t") not in (None, "") else None
                if path == "/api/metars":          # finished observations only: no need to wait for the run's lock
                    return self._json(net.to_json(t) if net else {"t": None, "stations": []})
                d = net.detail(q.get("icao", "").upper(), t) if net else None
                return self._json(d) if d else self._error(404, "no observations from that station")
            if path == "/api/ppi":
                return self._ppi(q)
            if path == "/api/volume":
                return self._volume(q)
            if path in ("/api/pmw/frame", "/api/pmw/targets"):
                return self._pmw(path, q)
            if path == "/api/volume/times":
                # the 3-D replay's snapshots (volume.snapshot): model times (s), and the radar frames' clock
                from . import volume
                sim = APP.sim
                times = volume.history_times(sim.run_dir / "vol3d") if sim is not None else []
                t0 = sim.frames[0].epoch_ms - 1000.0 * sim.frames[0].t if sim is not None and sim.frames else None
                return self._json({"times": times, "ms0": t0, "every": volume.HISTORY_EVERY})
            if path == "/api/tornado/paths":
                return self._tornado_paths()
            if path == "/api/runs":
                runs = sorted((PROJECT / "output" / "runs").glob("*"), reverse=True)[:50]
                return self._json([{"id": r.name, "radars": sorted(d.name for d in r.iterdir() if d.is_dir())}
                                   for r in runs if r.is_dir()])
            if path == "/api/synoptic/status":
                sim = APP.sim
                run = getattr(sim, "syn_progress", None) if sim is not None else None
                return self._json({"preview": _SYN["progress"] or _SYN.get("bank"), "run": run,
                                   "ready": sorted(_SYN.get("ready", ())),
                                   "scenarios": {k: v["label"] for k, v in synoptic.SCENARIOS.items()}})
            if path == "/api/hrrr/latest":
                from . import hrrr
                c = hrrr.latest_cycle()
                return self._json({"cycle": c.strftime("%Y-%m-%dT%H:00:00Z"), "max_fh": hrrr.max_hours(c)})
            if path == "/api/hrrr/for":
                # the run for a valid time (past, now or up to ~2 days ahead)
                import datetime as _dt
                from . import hrrr
                v = _dt.datetime.fromisoformat(q.get("valid", "").replace("Z", "+00:00"))
                v = v if v.tzinfo else v.replace(tzinfo=_dt.timezone.utc)
                try:
                    c, fh = hrrr.cycle_for(v)
                except LookupError as exc:
                    return self._json({"error": str(exc)})
                return self._json({"cycle": c.strftime("%Y-%m-%dT%H:00:00Z"), "start_fh": fh, "max_fh": hrrr.max_hours(c)})
            if path == "/api/hrrr/check":
                # is this cycle in NOAA's archive, and how long is its forecast
                import datetime as _dt
                from . import hrrr
                c = _dt.datetime.fromisoformat(q.get("cycle", "").replace("Z", "+00:00"))
                c = (c if c.tzinfo else c.replace(tzinfo=_dt.timezone.utc)).replace(minute=0, second=0)
                try:
                    import eccodes  # noqa: F401
                    decoder = True
                except Exception:
                    decoder = False
                return self._json({"cycle": c.strftime("%Y-%m-%dT%H:00:00Z"), "available": hrrr.available(c, 0),
                                   "max_fh": hrrr.max_hours(c), "decoder": decoder})
            if path == "/api/setups":
                return self._json(setups.entries())
            if path == "/api/setups/get":
                return self._json(setups.load(q.get("name", "")))
            if path == "/api/setups/file":
                return self._setup_file(q.get("name", ""))
            return self._error(404, "not found")
        except LookupError:
            return self._error(404, "no such setup")
        except Exception as exc:
            log.exception("GET %s", self.path)
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def _static(self, rel):
        p = (WEB / rel).resolve()
        if WEB.resolve() not in p.parents and p != WEB.resolve() or not p.is_file():
            return self._error(404, "not found")
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if p.suffix == ".js":
            ctype = "text/javascript"
        self._send(200, p.read_bytes(), ctype)

    def _state(self):
        sim = APP.sim
        if sim is None:
            return self._json(dict({"status": "none"}, **_gpu_notice()))
        # never hang behind a run that is stuck inside the GPU: read without the lock
        got = sim.lock.acquire(timeout=2.0)
        try:
            info = sim.info()
        except RuntimeError:                      # changed while read without the lock
            info = {"run_id": sim.run_id, "status": sim.status, "error": sim.error}
        finally:
            if got:
                sim.lock.release()
        return self._json(info)

    # ------------------------------------------------------------ chase mode's multiplayer (chase.py)
    @staticmethod
    def _chase_urls() -> list:
        port = _NET["port"]
        return [f"http://{a}:{port}/chase" for a in _NET["lan"]]

    def _chase_post(self, what: str, body: dict):
        from . import chase
        local = self._is_local()
        tok = self.headers.get("X-Chase-Token")
        try:
            if what in ("host", "tunnel") and not local:
                return self._error(403, "only the host's own PC switches hosting")
            if what == "host":
                return self._json(dict(chase.host(body.get("on"), bool(body.get("new_code"))), urls=self._chase_urls()))
            if what == "tunnel":
                return self._json(chase.tunnel(bool(body.get("on")), _NET["port"]))
            if what == "join":
                return self._json(chase.join(body, self._client_ip(), local))
            if what == "sync":
                return self._json(chase.sync(tok, body))
            if what == "leave":
                return self._json(chase.leave(tok))
        except PermissionError as e:
            return self._error(403, str(e))
        return self._error(404, "unknown chase call")

    # ------------------------------------------------------------ phone remote (remote.py)
    @staticmethod
    def _remote_urls() -> list:
        port = _NET["port"]
        return [f"http://{a}:{port}/remote" for a in _NET["lan"]] or [f"http://127.0.0.1:{port}/remote"]

    def _remote_role(self):
        from . import remote
        return remote.role_of(self.headers.get("X-Remote-Token"))

    def _remote_get(self, what: str, q: dict):
        from . import remote
        role = self._remote_role()
        if role is None:
            return self._error(401, "sign in with a PIN")
        sim = APP.sim
        if what == "state":
            return self._json(dict(remote.summary(sim), role=role))
        if what == "image":
            try:
                d = self._run_dir(None)
                site = (q.get("site") or (sim.display_radar if sim is not None else "") or "").upper()
                if not site:
                    return self._error(404, "no radar")
                view = q.get("view", "storm")
                if view not in ("storm", "60", "150", "230"):
                    view = "storm"
                prod = q.get("product", "BR").upper()
                if prod not in ("BR", "BV", "CC", "ZDR"):
                    prod = "BR"
                px = max(400, min(int(q.get("px", 900)), 1400))
                return self._send(200, remote.picture(d, site, prod, view, (px, px)), "image/png")
            except Exception as e:
                return self._error(404, str(e))
        if what == "survey":
            try:
                s = self._survey_of(None)
            except Exception as e:
                return self._error(404, str(e))
            rows = [{k: v for k, v in r.items() if not k.startswith("_") and k != "path"} for r in s.rows]
            return self._json({"summary": s.summary(), "rows": rows})
        return self._error(404, "unknown")

    def _remote_post(self, what: str, body: dict):
        from . import remote
        if what == "login":
            try:
                return self._json(remote.login(str(body.get("pin", "")), self.client_address[0]))
            except PermissionError as e:
                return self._error(403, str(e))
        if what == "control":
            if self._remote_role() != "control":
                return self._error(403, "this needs the control PIN")
            sim = APP.sim
            if sim is None:
                return self._error(404, "no run")
            act = body.get("action")
            if act == "pause":
                sim.pause()
            elif act == "resume":
                sim.resume()
            elif act == "stop":
                threading.Thread(target=sim.stop, name="remote-stop", daemon=True).start()
            else:
                return self._error(400, "action is pause, resume or stop")
            log.info("phone remote (%s): %s", self.client_address[0], act)
            return self._json({"ok": True})
        return self._error(404, "unknown")

    @staticmethod
    def _run_dir(run_id):
        """A run's folder: run_id's, else the run on the map's, else the newest."""
        runs = PROJECT / "output" / "runs"
        if run_id:
            d = runs / Path(str(run_id)).name
            if not d.is_dir():
                raise ValueError(f"no run {run_id}")
            return d
        sim = APP.sim
        if sim is not None and getattr(sim, "run_dir", None) is not None and Path(sim.run_dir).is_dir():
            return Path(sim.run_dir)
        got = sorted(p for p in runs.glob("*") if p.is_dir())
        if not got:
            raise ValueError("no runs yet")
        return got[-1]

    # ------------------------------------------------------------ tornado list / damage survey
    def _survey_of(self, run_id):
        """survey.Survey of run ``run_id`` (its state.json), or of the run on the map (live) / the newest run."""
        from .survey import Survey
        sim = APP.sim
        if (not run_id or (sim is not None and run_id == sim.run_id)) and sim is not None and sim.engine is not None:
            got = sim.lock.acquire(timeout=5.0)
            try:
                state = sim.info()
            finally:
                if got:
                    sim.lock.release()
            return Survey(json.loads(json.dumps(state, default=_jsonable)))
        runs = PROJECT / "output" / "runs"
        d = runs / run_id if run_id else (sorted(runs.glob("*"))[-1] if any(runs.glob("*")) else None)
        if d is None or not (d / "state.json").is_file():
            raise ValueError("that run has no saved state (state.json)")
        return Survey(json.loads((d / "state.json").read_text()))

    def _survey_json(self, run_id):
        try:
            s = self._survey_of(run_id)
        except Exception as e:
            return self._error(404, str(e))
        return self._json({"run_id": s.run_id, "summary": s.summary(), "rows": s.table()})

    def _survey_download(self, run_id, kind: str):
        try:
            s = self._survey_of(run_id)
        except Exception as e:
            return self._error(404, str(e))
        text, ext, ctype = {"kml": (s.kml, "kml", "application/vnd.google-earth.kml+xml"),
                            "csv": (s.csv, "csv", "text/csv"), "md": (s.markdown, "md", "text/markdown"),
                            "placefile": (lambda: s.placefile(refresh=0), "txt", "text/plain")}.get(kind, (None,) * 3)
        if text is None:
            return self._error(400, "kind is kml, csv, md or placefile")
        data = text().encode("utf-8")
        name = f"{s.run_id} {'damage survey' if kind in ('kml', 'placefile') else 'tornadoes'}.{ext}"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _airmass(self, body):
        return self._json(_airmass_map(body))

    def _setup_file(self, name: str):
        """A saved setup as a download (to send to someone)."""
        f = setups.file_path(name)
        data = f.read_bytes()
        plain = re.sub(r'[^A-Za-z0-9 ._()-]', "_", f.name)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition",
                         f"attachment; filename=\"{plain}\"; filename*=UTF-8''{urllib.parse.quote(f.name)}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _setups(self, action: str, body: dict):
        """Saved setups (setups.py): save, import a file someone sent, delete, show in Explorer."""
        try:
            if action == "save":
                return self._json(setups.save(str(body.get("name", "")), body.get("settings"), str(body.get("note", ""))))
            if action == "import":
                return self._json(setups.import_text(str(body.get("name", "")), str(body.get("text", ""))))
            if action == "delete":
                setups.delete(str(body.get("name", "")))
                return self._json({"ok": True})
            if action == "reveal":
                setups.reveal(str(body.get("name") or "") or None)
                return self._json({"ok": True})
        except LookupError:
            return self._error(404, "no such setup")
        except ValueError as exc:
            return self._error(400, str(exc))
        return self._error(404, "not found")

    def _frame(self, q):
        sim = APP.sim
        if sim is None:
            return self._error(404, "no run")
        i = int(q.get("i", -1))
        fr = sim.frames
        if not fr:
            return self._error(404, "no frames yet")
        f = fr[i] if -len(fr) <= i < len(fr) else fr[-1]
        name = q.get("product", "cref")
        blob = f.products.get(name)
        if blob is None:
            return self._error(404, f"no product {name}")
        self._send(200, blob, "application/octet-stream", encoding="deflate")

    def _sst_preview(self, body):
        """The hurricane tab's SST map before a run: the procedural SST (hurricane.ProceduralSST) over the sea
        around the map centre, packed like a map product (one patch, ``step_km`` cells)."""
        from . import hurricane as hur
        proj = geo.Lambert(float(body.get("lat0", 35.0)), float(body.get("lon0", -97.0)))
        half = min(max(float(body.get("half_km", 1800.0)), 200.0), 3000.0) * 1000.0
        step = min(max(float(body.get("step_km", 10.0)), 4.0), 40.0) * 1000.0
        n = int(2 * half / step)
        c = (np.arange(n) + 0.5) * step - half
        X, Y = np.meshgrid(c, c)
        lat, lon = proj.inv(X, Y)
        sst = hur.ProceduralSST(int(body.get("seed", 1) or 1), float(body.get("offset", 0.0) or 0.0)).celsius(lat, lon)
        land = hur.land_mask()
        water = ~land(lat, lon) if land is not None else np.ones(X.shape, bool)
        scale, offset = products.CODECS["sst"][:2]
        codes = np.where(water, np.clip(np.rint(sst * scale + offset), 1, 255), 0).astype(np.uint8)
        meta = {"x0": -half, "y0": -half, "x1": half, "y1": half, "nx": n, "ny": n}
        blob = products.compress(products.pack_frame_product(0.0, "sst", [(meta, codes)]))
        self._send(200, blob, "application/octet-stream", encoding="deflate")

    def _volume(self, q):
        """The 3-D view's volume (volume.py) of the box x0, y0, x1, y1 (km) from the run's latest radar fields."""
        from . import volume
        sim = APP.sim
        if q.get("live") and sim is not None and sim.engine is not None:
            # the tornado view: the patches' state now, the finest grid wherever there is one, up to ztop
            try:
                box = [1000.0 * float(q[k]) for k in ("x0", "y0", "x1", "y1")]
                ztop = min(18000.0, max(1000.0, 1000.0 * float(q.get("ztop", 6))))
            except (KeyError, ValueError):
                return self._error(400, "x0, y0, x1, y1 (km) needed")
            try:
                with sim.lock:
                    blob = volume.extract_live(sim.engine.patches + sim.engine.nest_patches(), box,
                                               q.get("field", "cloud"), ztop)
            except ValueError as exc:
                return self._error(400, str(exc))
            if blob is None:
                return self._error(404, "no storm patch in that box")
            return self._send(200, zlib.compress(blob, 1), "application/octet-stream", encoding="deflate")
        if q.get("t") not in (None, ""):
            # the replay: the snapshot nearest model time t (s)
            if sim is None:
                return self._error(404, "no run")
            try:
                box = [1000.0 * float(q[k]) for k in ("x0", "y0", "x1", "y1")]
                blob, tt = volume.extract_past(sim.run_dir / "vol3d", float(q["t"]), box, q.get("field", "dbz"))
            except (KeyError, ValueError) as exc:
                return self._error(400, str(exc))
            if blob is None:
                return self._error(404, "no 3-D snapshot of that box then" if tt is not None
                                   else "no 3-D snapshots yet: they are saved with the radar frames")
            return self._send(200, zlib.compress(blob, 1), "application/octet-stream", encoding="deflate")
        vf = getattr(sim, "vol_fields", None) if sim is not None else None
        if not vf:
            return self._error(404, "no storm fields yet: start a run (the 3-D view shows the latest model step)")
        fields, t = vf
        try:
            box = [1000.0 * float(q[k]) for k in ("x0", "y0", "x1", "y1")]
            since = float(q["since"]) if q.get("since") not in (None, "") else None
        except (KeyError, ValueError):
            return self._error(400, "x0, y0, x1, y1 (km) needed")
        # chase mode asks about once a second with the model time of the volume it has (since=): while the model
        # has not stepped on (paused, or one long step) it gets 204 without waiting for the run's lock
        eng = sim.engine
        if since is not None and eng is not None and abs(float(eng.t) - since) < 1e-3:
            return self._send(204, b"", "application/octet-stream", headers={"X-Model-Time": f"{float(eng.t):.3f}"})
        try:
            # chase mode's far cloud (far=1): a wide box on a coarse grid, no nests, no wind
            far = q.get("far") == "1"
            side = max(32, min(320, int(q.get("side", 256)))) if far else volume.MAX_SIDE
            nz = max(8, min(volume.NZ, int(q.get("nz", 40)))) if far else volume.NZ
        except ValueError:
            return self._error(400, "side, nz: whole numbers")
        with sim.lock:
            patches = ({p.id: p for p in sim.engine.patches + sim.engine.nest_patches()}
                       if sim.engine is not None else {})
            t_now = float(sim.engine.t) if sim.engine is not None else float(t)
            if far:
                blob = volume.extract(fields, box, "cloud", patches, max_side=side, nz=nz, insets=False)
            elif q.get("wind") == "1":
                # chase mode's moving clouds: the model's own wind too, to carry the cloud along it between volumes
                drift = getattr(sim.env, "storm_motion", None) if sim.env is not None else None
                blob = volume.extract(fields, box, q.get("field", "dbz"), patches, wind=True, t=t_now, drift=drift)
            else:
                blob = volume.extract(fields, box, q.get("field", "dbz"), patches)
        if blob is None:
            return self._error(404, "no storm patch in that box")
        self._send(200, zlib.compress(blob, 1), "application/octet-stream", encoding="deflate",
                   headers={"X-Model-Time": f"{t_now:.3f}"})

    def _chase_air(self, q):
        """The model's wind, gust, temperature and dew point at chase mode's cars: p=x,y;x,y (world m); grid=1: and
        round the first, the model's 10 m wind on its finest grid (chase.air)."""
        from . import chase
        sim = APP.sim
        if sim is None or sim.engine is None:
            return self._error(404, "no run")
        try:
            pts = [tuple(float(v) for v in s.split(",")) for s in str(q.get("p", "")).split(";") if s.strip()]
            if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in pts):
                raise ValueError
        except ValueError:
            return self._error(400, "p=x,y;x,y (world m)")
        with sim.lock:
            out = chase.air(sim, pts, grid=q.get("grid") == "1")
        self._json({"t": float(sim.engine.t), "air": out})

    def _pmw(self, path, q):
        """The PMWeather addon's live feed (pmw.py): targets as JSON, or a frame around x, y (m; none: the tornado
        zoom, else a tornado, else a storm) with a storm box storm_km and a fine box fine_km a side.  since=t
        (model s): 204 until the model has moved on past t."""
        from . import pmw
        sim = APP.sim
        if sim is None or sim.engine is None:
            return self._error(404, "no run")
        if path == "/api/pmw/targets":
            with sim.lock:
                d = pmw.targets(sim)
            return self._json(d)
        try:
            since = float(q["since"]) if q.get("since") not in (None, "") else None
            cx = float(q["x"]) if q.get("x") not in (None, "") else None
            cy = float(q["y"]) if q.get("y") not in (None, "") else None
            storm_km = float(q.get("storm_km", 40))
            fine_km = float(q.get("fine_km", 3))
        except ValueError:
            return self._error(400, "x, y (m), storm_km, fine_km, since must be numbers")
        if since is not None and sim.engine.t <= since:
            return self._send(204, b"", "application/octet-stream")
        with sim.lock:
            blob = pmw.frame(sim, cx, cy, storm_km, fine_km, q.get("fine", "1") != "0", q.get("cloud", "1") != "0")
        if blob is None:
            return self._error(404, "no storm patch there")
        return self._send(200, zlib.compress(blob, 1), "application/octet-stream", encoding="deflate")

    def _tornado_paths(self):
        """The tornado zoom's track and its parcels' paths (tornado.py): x, y, z m triples per path."""
        sim = APP.sim
        tz = getattr(sim.engine, "tornado", None) if sim is not None and sim.engine is not None else None
        if tz is None:
            return self._error(404, "no tornado zoom: drag one on the map during a run")
        with sim.lock:
            d = tz.paths_json()
        return self._json(d)

    def _ppi(self, q):
        sim = APP.sim
        if sim is None:
            return self._error(404, "no run")
        fr = sim.frames
        if not fr:
            return self._error(404, "no frames yet")
        i = int(q.get("i", -1))
        f = fr[i] if -len(fr) <= i < len(fr) else fr[-1]
        d = f.radars.get(q.get("icao", "").upper())
        if not d:
            return self._error(404, "radar not in this frame")
        blob = d.get(q.get("moment", "REF").upper())
        if blob is None:
            return self._error(404, "moment not available yet")
        self._send(200, blob, "application/octet-stream", encoding="deflate")

    # ------------------------------------------------------------ polling
    def _poll(self, rest):
        """The layout Iowa State's Level II server uses and GR2Analyst polls: grlevel2.cfg (one
        "Site:" line per radar), <SITE>/dir.list ("size name" per volume, oldest first) and the
        volumes, re-timed to when they were made (polling.py).  GET and HEAD, with Last-Modified,
        If-Modified-Since and Range (GR2Analyst downloads volumes incrementally), like a plain
        web server."""
        sim = APP.sim
        parts = [p for p in rest.split("/") if p]
        if not parts or parts[0].lower() == "grlevel2.cfg":
            # every WSR-88D and TDWR, as Iowa State lists them: GR2Analyst keeps the list it read
            # when the source was added, so radars added to a run later are on it already
            sites = {s["icao"] for s in APP.sites()} | set(rd.DOW_IDS)      # the DOWs, wherever they park
            if sim is not None:
                sites |= {r.icao for r in list(sim.radars)}
            body = "".join(f"Site: {s}\n" for s in sorted(sites)).encode()
            return self._send(200, body, "text/plain", modified=_STARTED)
        if parts[0].lower() == "warnings.txt" and len(parts) == 1:
            # automatic warnings (warnings_auto.py) in the polling feed's clock
            if sim is None or sim.engine is None:
                body = "Title: Atmos-WRF warnings" + chr(10) + "RefreshSeconds: 15" + chr(10)
            else:
                body = sim.warnings.placefile(sim.proj, sim.start_ms / 1000.0, sim.speed)
            return self._send(200, body.encode(), "text/plain")
        if parts[0].lower() == "metars.txt" and len(parts) == 1:
            # the simulated METARs (metars.py): wind barbs, temperature and dewpoint, the METAR on hover
            from . import metars
            host = self.headers.get("Host") or f"127.0.0.1:{_NET['port']}"
            icon = f"http://{host}/poll/metar-barbs.png"
            net = getattr(sim, "metars", None)
            body = net.placefile(icon) if net is not None else metars.empty_placefile(icon)
            return self._send(200, body.encode(), "text/plain")
        if parts[0].lower() == "survey.txt" and len(parts) == 1:
            # the damage survey (survey.py) of the run on the map, or the newest one: EF-coloured tracks, labels
            try:
                body = self._survey_of(None).placefile(refresh=60)
            except Exception:
                body = "Title: Atmos-WRF damage survey\nRefreshSeconds: 60\n"
            return self._send(200, body.encode(), "text/plain")
        if parts[0].lower() == "metar-barbs.png" and len(parts) == 1:
            from . import metars
            return self._send(200, metars.barb_sheet(), "image/png", cache=True)
        m = re.fullmatch(r"(uhtrack|windtrack)(?:\.txt|-(\d{8}-\d{6})-(\d+)\.png)", parts[0].lower())
        if m and len(parts) == 1:
            return self._placefile(m[1]) if m[2] is None else self._swath_png(m[1], m[2], int(m[3]))
        site = parts[0].upper()
        run_dir = sim.run_dir if sim is not None else None
        if run_dir is None:
            runs = sorted((PROJECT / "output" / "runs").glob("*"))
            run_dir = runs[-1] if runs else None
        since = sim.dow_since.get(site, 0.0) if sim is not None else 0.0     # a DOW: its current deployment
        if len(parts) == 1 or parts[1].lower() == "dir.list":
            body, newest, n = polling.dir_list(run_dir, site, since)
            # never older than this server or the run, so a viewer holding a list from before
            # (another run) fetches it again
            modified = max(newest, _STARTED, APP.changed)
            if self._not_modified(modified):
                return None
            if (self.client_address[0], site) not in _POLLERS:
                _POLLERS.add((self.client_address[0], site))
                log.info("polling: %s asked for %s (%d volumes in %s; %s)", self.client_address[0], site, n,
                         run_dir.name if run_dir is not None else "no run", self.headers.get("User-Agent", "?"))
            return self._send(200, body, "text/plain", modified=modified)
        f, modified = polling.volume_file(run_dir, site, Path(parts[1]).name, since)
        if f is None:
            return self._error(404, "no such volume")
        if self._not_modified(modified):
            return None
        try:
            data = f.read_bytes()
        except OSError:                       # dropped off the list meanwhile
            return self._error(404, "no such volume")
        return self._send_range(data, modified)

    def _placefile(self, name: str):
        """/poll/uhtrack.txt, /poll/windtrack.txt: the run's run-max swath as a GR2Analyst /
        Supercell-Wx placefile (placefile.py), its image at a URL of its own for each update."""
        sim = APP.sim
        env = getattr(sim, "env", None)
        box = url = peak = None
        label = ""
        if sim is not None and env is not None:
            ver = sim.swath_version
            key = (sim.run_id, name, ver)
            with _SWATHS_LOCK:
                got = _SWATHS.get(key)
                if got is None:
                    got = placefile.image(sim.tracks, name, env.grid.dx)
                    if got is not None:
                        _SWATHS[key] = got
                        old = [k for k in _SWATHS if k[0] != sim.run_id]
                        old += sorted((k for k in _SWATHS if k[:2] == key[:2]), key=lambda k: k[2])[:-8]
                        for k in old:
                            del _SWATHS[k]
            if got is not None:
                box = got[1]
                host = self.headers.get("Host") or f"127.0.0.1:{_NET['port']}"
                url = f"http://{host}/poll/{name}-{sim.run_id}-{ver}.png"
            p = sim.uh_peaks[-1] if name == "uhtrack" and sim.uh_peaks else None
            if p is not None:
                peak = p
                from . import level2
                when = level2.utc(sim.start_ms + p["t"] * 1000.0).strftime("%H:%M:%S")     # before 1970 too
                label = (f"Run-max 2-5 km UH {p['value']:.0f} m2/s2 at +{int(p['t'] // 3600)}:"
                         f"{int(p['t'] // 60) % 60:02d} ({when}Z model time)")
        title = f"Atmos-WRF {'storm tracks (run-max UH)' if name == 'uhtrack' else 'wind swaths (run-max)'}" + (
            f" {sim.run_id}" if sim is not None else "")
        body = placefile.text(name, box, sim.proj if sim is not None else None, url, title, peak=peak, label=label)
        return self._send(200, body.encode(), "text/plain")

    def _swath_png(self, name: str, run: str, ver: int):
        with _SWATHS_LOCK:
            got = _SWATHS.get((run, name, ver))
        if got is None:
            return self._error(404, "no such swath image (a newer one is in the placefile)")
        return self._send(200, got[0], "image/png", cache=True)

    def _send_range(self, data: bytes, modified: float):
        """A file, or the part of it a "Range: bytes=a-b" header asks for (one range)."""
        size = len(data)
        m = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", self.headers.get("Range") or "")
        if_range = self.headers.get("If-Range")
        if m and (m[1] or m[2]) and (not if_range or if_range == email.utils.formatdate(modified, usegmt=True)):
            if m[1]:
                a, b = int(m[1]), min(int(m[2]) if m[2] else size - 1, size - 1)
            else:                                   # bytes=-n: the last n
                a, b = max(size - int(m[2]), 0), size - 1
            if a >= size or a > b:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            part = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        else:
            part = data
            self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(part)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", email.utils.formatdate(modified, usegmt=True))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(part)
        return None

    # ------------------------------------------------------------ POST
    def _library(self, action: str, body: dict):
        try:
            return self._json(_library_post(action, body))
        except LookupError:
            return self._error(404, "not found")
        except ValueError as exc:
            return self._error(400, str(exc))

    def do_HEAD(self):
        self.do_GET()               # _send leaves the body out

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        body = self._body()
        if self._remote_refused(path):
            return None
        try:
            if path == "/api/sounding/load":
                from .sim import load_sounding
                spec = body.get("spec", {})
                if spec.get("file"):
                    p = Path(spec["file"])
                    if not p.is_absolute():
                        p = PROJECT / p
                    if not p.is_file() or not LIBRARY.allowed(p):
                        return self._error(404, "sounding not found")
                return self._json(_sounding_payload(load_sounding(spec)))
            if path == "/api/sounding/parse":
                s = snd.parse_any(body.get("text", ""), body.get("title"))
                return self._json(_sounding_payload(s))
            if path == "/api/sounding/analytic":
                s = snd.weisman_klemp(theta_sfc=float(body.get("theta_sfc", 300)), qv0=float(body.get("qv0", 14)),
                                      shear_ms=float(body.get("shear", 35)),
                                      hodograph=body.get("hodograph", "quarter"))
                return self._json(_sounding_payload(s))
            if path == "/api/airmass":
                return self._airmass(body)
            if path == "/api/synoptic":
                w, t, extra = _synoptic_world(body)
                return self._json(dict(w.payload(t), **extra))
            if path == "/api/synoptic/sounding":
                w, t, extra = _synoptic_world(body)
                s = w.sounding(float(body["x"]) * 1000.0, float(body["y"]) * 1000.0, t)
                return self._json(dict(_sounding_payload(s.ice_capped()), **extra))
            if path == "/api/hurricane/sst":
                return self._sst_preview(body)
            if path == "/api/gr2/sites":
                from . import gr2sites
                return self._json(gr2sites.install() if body.get("install") else gr2sites.status())
            if path.startswith("/api/r/"):
                return self._remote_post(path[len("/api/r/"):], body)
            if path.startswith("/api/c/"):
                return self._chase_post(path[len("/api/c/"):], body)
            if path == "/api/remote/config":
                from . import remote
                return self._json(dict(remote.update(body), urls=self._remote_urls()))
            if path == "/api/loop/make":
                # a radar loop (loops.py) in the background: its job id, then /api/loop/status
                from .loops import LOOPS, PRODUCTS
                d = self._run_dir(body.get("run"))
                prod = str(body.get("product", "BR")).upper()
                if prod not in PRODUCTS:
                    return self._error(400, f"product is one of {', '.join(PRODUCTS)}")
                pal = body.get("palette") or None
                if pal and not (Path(pal).suffix.lower() == ".pal" and LIBRARY.allowed(Path(pal))):
                    return self._error(400, "that color table is not in the viewer's library")
                size = body.get("size") or [1280, 720]
                job = LOOPS.start(run_dir=d, site=str(body["site"]).upper(), product=prod, tilt=float(body.get("tilt", 0.5)),
                                  first=body.get("first"), last=body.get("last"), box=body["box"], palette_path=pal,
                                  size=(max(320, min(int(size[0]), 3840)), max(240, min(int(size[1]), 2160))),
                                  frame_ms=max(40, min(int(body.get("frame_ms", 250)), 5000)),
                                  tracks=bool(body.get("tracks", True)))
                return self._json({"job": job})
            if path == "/api/gr2/dow":
                # a DOW's new parking spot into GR2Analyst's site table (one Windows prompt); restart GR2Analyst
                from . import gr2sites
                icao = str(body["icao"]).upper()
                sim = APP.sim
                # the truck's ground: the run's (the deployment may still be queued), its mast
                elev = (float(sim.sounding.zsfc) + 4.0 if sim is not None and getattr(sim, "sounding", None) is not None
                        else float(body.get("elev", 300.0)))
                return self._json(gr2sites.set_dow(icao, float(body["lat"]), float(body["lon"]), elev))
            if path.startswith("/api/library/"):
                return self._library(path[len("/api/library/"):], body)
            if path.startswith("/api/setups/"):
                return self._setups(path[len("/api/setups/"):], body)
            if path == "/api/run/new":
                sim = APP.new_run(body.get("settings", {}))
                if body.get("start", True):
                    sim.start()
                return self._json({"ok": True, "run_id": sim.run_id})
            sim = APP.sim
            if sim is None:
                return self._error(409, "no run; create one first")
            if path == "/api/run/brushes":
                sim.settings["wind_brushes"] = list(body.get("brushes") or [])
                sim._apply_boundaries()          # the air masses again, with the brushes' winds
                return self._json({"ok": True})
            if path == "/api/run/fire_area":
                box = body.get("box")
                sim.settings["fire_area"] = [float(v) for v in box] if box else None
                return self._json({"ok": True})
            if path == "/api/run/start":
                sim.start()
                sim.resume()
            elif path == "/api/run/pause":
                sim.pause()
            elif path == "/api/run/resume":
                sim.resume()
            elif path == "/api/run/stop":
                sim.stop()
            elif path == "/api/run/storm":
                sim.command("add_storm", **body)
            elif path == "/api/run/cyclone":
                sim.command("add_cyclone", **body)
            elif path == "/api/run/radar":
                sim.command("add_radar", **body)
            elif path == "/api/run/radar/remove":
                sim.command("remove_radar", **body)
            elif path == "/api/run/zone":
                sim.command("add_zone", **body)
            elif path == "/api/run/zone/remove":
                sim.command("remove_zone", **body)
            elif path == "/api/run/patch/remove":
                sim.command("remove_patch", **body)
            elif path == "/api/run/patch/move":
                sim.command("move_patch", **body)
            elif path == "/api/run/patch/release":
                sim.command("release_patch", **body)
            elif path == "/api/run/nest/move":
                sim.command("move_nest", **body)
            elif path == "/api/run/nest/add":
                sim.command("add_nest", **body)
            elif path == "/api/run/tornado":
                sim.command("tornado_box", **body)
            elif path == "/api/run/tornado/detail":
                sim.command("tornado_detail", **body)
            elif path == "/api/run/tornado/remove":
                sim.command("tornado_clear")
            elif path == "/api/run/nudger":
                sim.command("nudger", **body)
            elif path == "/api/run/nudger/plan":
                sim.command("plan_nudger", **body)
            elif path == "/api/run/nudger/remove":
                sim.command("remove_nudger_plan", **body)
            elif path == "/api/run/boundary":
                sim.command("add_boundary", **body)
            elif path == "/api/run/boundary/update":
                sim.command("update_boundary", **body)
            elif path == "/api/run/boundary/remove":
                sim.command("remove_boundary", **body)
            elif path == "/api/run/low":
                sim.command("set_low", **body)
            elif path == "/api/run/auto":
                sim.command("set_auto", **body)
            elif path == "/api/run/speed":
                sim.command("speed", **body)
            elif path == "/api/run/refine":
                sim.command("radar_refine", **body)
            elif path == "/api/run/grid":
                sim.command("grid_motion", **body)
            elif path == "/api/run/display":
                sim.display_radar = body.get("icao")
            else:
                return self._error(404, "not found")
            return self._json({"ok": True})
        except Exception as exc:
            log.exception("POST %s", self.path)
            return self._error(500, f"{type(exc).__name__}: {exc}")


_SUITES = None


def _suites():
    global _SUITES
    if _SUITES is None:
        try:
            _SUITES = physics_suites(menu=True)
        except Exception:
            log.exception("ArWen physics suites unavailable")
            _SUITES = []
    return _SUITES


def _gpu_notice() -> dict:
    """The GPU fault that made the server restart, if it was recent."""
    import time as _time
    from .sim import GPU_FAULT_FILE
    try:
        d = json.loads(GPU_FAULT_FILE.read_text())
    except (OSError, ValueError):
        return {}
    if _time.time() - float(d.get("time", 0)) > 900:
        return {}
    return {"notice": f"The viewer restarted after a GPU error ({d.get('error', '')}); "
                      f"run {d.get('run_id', '')} stopped at +{float(d.get('model_t', 0)) / 60:.0f} min."}


def _parent_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259              # STILL_ACTIVE


def _orphan_guard() -> None:
    """Exit when the supervisor is gone (window closed, process killed)."""
    import time as _time
    try:
        pid = int(os.environ.get("ARHTMODELVIEWER_PARENT", "0"))
    except ValueError:
        return
    if pid <= 0:
        return
    while True:
        _time.sleep(3.0)
        if not _parent_alive(pid):
            os._exit(0)


def _watchdog() -> None:
    """Under the supervisor: a run whose loop has not come round for 5 minutes
    is stuck inside the GPU; restart the server process."""
    import time as _time
    from .sim import GPU_FAULT_FILE, RESTART_CODE
    while True:
        _time.sleep(30.0)
        sim = APP.sim
        if sim is None or sim.status != "running":
            continue
        if _time.time() - getattr(sim, "heartbeat", _time.time()) < 300.0:
            continue
        log.error("run %s has not advanced for 5 minutes: restarting the server", sim.run_id)
        try:
            GPU_FAULT_FILE.write_text(json.dumps({"time": _time.time(), "run_id": sim.run_id,
                                                  "error": "the GPU engine stopped responding",
                                                  "model_t": getattr(sim.engine, "t", 0.0)}))
        except OSError:
            pass
        os._exit(RESTART_CODE)


def _library_post(action: str, body: dict):
    """Library folders, the computer scan and imported files (library.py)."""
    if action == "folder":
        if body.get("remove"):
            LIBRARY.remove_folder(str(body.get("path", "")))
            return LIBRARY.summary()
        LIBRARY.add_folder(str(body.get("path", "")), bool(body.get("recursive", True)))
        return LIBRARY.summary()
    if action == "browse":
        return {"path": pick_folder()}
    if action == "scan":
        if body.get("stop"):
            LIBRARY.stop()
        else:
            LIBRARY.scan_computer()
        return LIBRARY.summary()
    if action == "refresh":
        LIBRARY.refresh()
        return LIBRARY.summary()
    if action == "import":
        kind = body.get("kind", "")
        out = LIBRARY.import_file(kind, str(body.get("name", "imported")), str(body.get("text", "")))
        if kind == "palette":
            out["palette"] = palettes.parse_pal(str(body.get("text", "")))
        return out
    raise LookupError(action)


class _NoWind:
    """Environment stand-in for drawing air masses (their temperature and moisture only)."""
    u_ground = v_ground = np.zeros(1)


def _airmass_map(body: dict) -> dict:
    """Surface temperature (K) and moisture (fraction) anomalies of a list of
    boundaries on a coarse grid over the view, at model time t."""
    bs, ds = [], []
    for i, d in enumerate(body.get("boundaries") or []):
        try:
            b = boundaries.boundary_from_json(d)
        except Exception:
            continue
        ds.append(d)
        b.seed = int(d.get("seed", d.get("id", i + 1)) or i + 1)
        at = float(d.get("at", 0.0) or 0.0)
        if at:
            vx, vy = b.velocity()
            b.points = [[x - vx * at, y - vy * at] for x, y in b.points]
        bs.append(b)
    nx = int(min(max(int(body.get("nx", 120)), 8), 300))
    ny = int(min(max(int(body.get("ny", 90)), 8), 300))
    x0, x1 = float(body.get("x0", -3e5)), float(body.get("x1", 3e5))
    y0, y1 = float(body.get("y0", -2e5)), float(body.get("y1", 2e5))
    t = float(body.get("t", 0.0))
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    out = {"nx": nx, "ny": ny, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "t": t}
    if not bs:
        out.update(dT=[], dq=[])
        return out
    if not any(b.vel for b in bs):
        # a setup's boundaries: joined where they meet, as the run will join them
        boundaries.join_at_lows(bs, 0.0, boundaries.low_motion(body.get("low")))
    am = boundaries.AirMasses(bs, _NoWind())
    m = am.maps(X, Y, t)
    out["dT"] = np.round(m["dT"], 2).ravel().tolist()
    out["dq"] = np.round(m["dq"], 3).ravel().tolist()
    out["fronts"] = [{"id": d.get("id"), "points": b.points_at(t).tolist(), "kind": b.kind, "side": b.side}
                     for d, b in zip(ds, bs)]
    return out


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class _Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second viewer bind the same port and share
    # its requests; bind exclusively so a second copy fails instead.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True

    def server_bind(self):
        import socket
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(port: int = 8830, open_browser: bool = True, lan: bool = False):
    """``lan``: listen on every network interface, so that other computers can poll the radars
    (only /poll/ answers them; Windows asks once whether Python may use the network)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    url = f"http://127.0.0.1:{port}/"
    try:
        httpd = _Server(("0.0.0.0" if lan else "127.0.0.1", port), Handler)
    except OSError:
        print(f"Port {port} is already in use - the viewer is probably running already: {url}", flush=True)
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        return
    _NET["port"] = port
    _NET["lan"] = _lan_addresses() if lan else []
    threading.Thread(target=_synoptic_bank_warmup, name="synoptic-bank", daemon=True).start()
    links = _poll_links()
    print(f"Atmos-WRF on {url}  (GR2Analyst polling: {links['local']})", flush=True)
    if lan:
        print("  other computers poll " + (" or ".join(links["lan"]) or f"http://<this PC's address>:{port}/poll/")
              + "  (the polling feed only)", flush=True)
    LIBRARY.refresh()            # soundings and color tables on this computer (background)
    if os.environ.get("ARHTMODELVIEWER_SUPERVISED") == "1":
        threading.Thread(target=_watchdog, name="arhtmodelviewer-watchdog", daemon=True).start()
        threading.Thread(target=_orphan_guard, name="arhtmodelviewer-orphan-guard", daemon=True).start()
    if open_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if APP.sim is not None:
            APP.sim.stop()
