"""OpenStreetMap from OpenFreeMap's vector tiles (first) or the Overpass API (when they fail), for the 3-D view's and chase mode's roads (landscape.py) and chase mode's
buildings (chase.py): cells fetched in the background and kept on disk by their owners.

Overpass is a shared, often busy service (two queries at a time per address, servers that stall for minutes), so:
  - a queue per kind of cell that always fetches the cell asked for most recently (where the players are now),
    and forgets cells nobody has asked for in FORGET s;
  - short timeouts, a server that fails rests a while, and when every server has failed the whole thing waits
    (BACKOFF, doubling to BACKOFF_MAX) instead of chewing through the queue;
  - status(): how many cells are waiting and whether Overpass is answering, for the page to say so."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .tiles import UA

SERVERS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter")
TIMEOUT = 100.0                  # s: a query's longest wait for its answer
FORGET = 600.0                   # s: a cell nobody asked for since is dropped from the queue
BACKOFF, BACKOFF_MAX = 45.0, 300.0
_SLOTS = threading.BoundedSemaphore(2)          # Overpass gives an address two queries at a time
_G = threading.Lock()
_STATE = {"rest": {}, "down_until": 0.0, "backoff": BACKOFF, "ok_at": 0.0, "fail_at": 0.0, "error": ""}


def overpass(query: str) -> dict:
    """The answer to an Overpass QL query (JSON), from the first server that is not resting; raises when none
    answers (and then the whole service rests)."""
    with _SLOTS:
        err = None
        now = time.time()
        with _G:
            order = sorted(SERVERS, key=lambda s: _STATE["rest"].get(s, 0.0) > now)
        for url in order:
            with _G:
                if _STATE["rest"].get(url, 0.0) > time.time():
                    continue
            try:
                req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    d = json.load(r)
                if d.get("remark") and "runtime error" in str(d["remark"]).lower():
                    raise RuntimeError(d["remark"][:200])
                with _G:
                    _STATE.update(ok_at=time.time(), backoff=BACKOFF, error="")
                return d
            except Exception as e:                       # busy, refusing or down: rest it, the next server
                err = e
                code = e.code if isinstance(e, urllib.error.HTTPError) else None
                with _G:
                    _STATE["rest"][url] = time.time() + (120.0 if code in (429, 503, 504) else 60.0)
        with _G:
            _STATE["down_until"] = time.time() + _STATE["backoff"]
            _STATE["backoff"] = min(_STATE["backoff"] * 2, BACKOFF_MAX)
            _STATE.update(fail_at=time.time(), error=str(err)[:200])
        raise RuntimeError(f"Overpass: {err}")


def busy() -> bool:
    """Overpass is not answering right now (every server failed lately)."""
    with _G:
        return _STATE["fail_at"] > _STATE["ok_at"] and time.time() - _STATE["fail_at"] < 600.0


class CellQueue:
    """Cells to fetch with fetch(*cell) (which writes its file), newest wish first, one worker thread."""

    def __init__(self, name: str, fetch):
        self.name, self.fetch = name, fetch
        self.wanted: dict = {}                     # cell -> when it was last asked for
        self.running: set = set()
        self.failed: dict = {}                     # cell -> not before (a cell that failed waits its turn)
        self.lock = threading.Lock()
        self.worker = None

    def want(self, cell) -> bool:
        """Ask for a cell (not on disk): True while it is waiting or being fetched."""
        with self.lock:
            self.wanted[cell] = time.time()
            if self.worker is None:
                self.worker = threading.Thread(target=self._work, name=f"osm-{self.name}", daemon=True)
                self.worker.start()
        return True

    def _work(self):
        while True:
            with self.lock:
                now = time.time()
                for c in [c for c, t in self.wanted.items() if now - t > FORGET]:
                    self.wanted.pop(c)
                todo = [c for c in self.wanted if c not in self.running and self.failed.get(c, 0.0) <= now]
                if not todo and self.wanted:            # only cells that just failed: a short nap, then again
                    nap = min(self.failed.get(c, now) for c in self.wanted) - now
                    if nap > 0:
                        self.lock.release()
                        try:
                            time.sleep(min(nap, 10.0))
                        finally:
                            self.lock.acquire()
                        continue
                if not todo:
                    self.worker = None
                    return
                cell = max(todo, key=lambda c: self.wanted[c])
                self.running.add(cell)
            wait = _STATE["down_until"] - time.time()
            if wait > 0:                               # Overpass is resting: so do we
                time.sleep(min(wait, 15.0))
                with self.lock:
                    self.running.discard(cell)
                continue
            try:
                self.fetch(*cell)
                with self.lock:
                    self.wanted.pop(cell, None)
                    self.failed.pop(cell, None)
            except Exception:
                with self.lock:                         # stays wanted: again after the others, in a minute or so
                    self.failed[cell] = time.time() + 75.0
            with self.lock:
                self.running.discard(cell)


# ---------------------------------------------------------------- OpenFreeMap (the first choice)
# OpenFreeMap serves OpenStreetMap as vector tiles (OpenMapTiles' schema) from a CDN: no key, a tile in a fraction
# of a second, where Overpass can take minutes or be down.  A cell is cut from the z14 tiles over it (z10 for the
# main roads' 1-degree cells): roads clipped to the cell (no road twice), each building kept by the tile and
# the cell its middle is in.  Its files look like Overpass's, so the rest does not know which one filled them.
import math  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from . import mvt  # noqa: E402

OFM_TILEJSON = "https://tiles.openfreemap.org/planet"
_OFM = {"template": None, "at": 0.0}
# OpenMapTiles' transportation classes -> landscape.ROAD_CLASSES' index (service roads, paths, rail: not roads here)
_OMT_ROAD = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4, "minor": 6, "track": 7}


def _ofm_template() -> str:
    with _G:
        if _OFM["template"] and time.time() - _OFM["at"] < 6 * 3600:
            return _OFM["template"]
    req = urllib.request.Request(OFM_TILEJSON, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        tj = json.load(r)
    with _G:
        _OFM.update(template=tj["tiles"][0], at=time.time())
    return tj["tiles"][0]


def _xy(lat: float, lon: float, z: int):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _lonlat(z: int, fx: float, fy: float):
    """(lat, lon) of a point in tile units at zoom z."""
    n = 2 ** z
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * fy / n)))), fx / n * 360.0 - 180.0


def _tile(template: str, z: int, x: int, y: int) -> bytes:
    url = template.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read() if r.status == 200 else b""
    except urllib.error.HTTPError as e:
        if e.code in (204, 404):                       # nothing there (sea, empty land)
            return b""
        raise


def _cell_tiles(la: float, lo: float, step: float, z: int, layers):
    """The decoded tiles over the cell (lat la..la+step, lon lo..lo+step): [(x, y, layers, extent, the cell's
    box in the tile's pixels)]."""
    template = _ofm_template()
    x0, y1 = _xy(la, lo, z)
    x1, y0 = _xy(la + step, lo + step, z)
    jobs = [(x, y) for x in range(int(math.floor(x0)), int(math.floor(x1)) + 1)
            for y in range(int(math.floor(y0)), int(math.floor(y1)) + 1)]
    with ThreadPoolExecutor(8) as ex:
        datas = list(ex.map(lambda j: _tile(template, z, *j), jobs))
    out = []
    for (x, y), d in zip(jobs, datas):
        if not d:
            continue
        L = mvt.read(d, layers)
        ext = next((l["extent"] for l in L.values()), 4096)
        box = (max(0.0, (x0 - x) * ext), max(0.0, (y0 - y) * ext), min(float(ext), (x1 - x) * ext),
               min(float(ext), (y1 - y) * ext))
        out.append((x, y, L, ext, box))
    return out


def _clip_line(pts, box):
    """A polyline cut to an axis-aligned box: the pieces inside (Liang-Barsky per segment)."""
    bx0, by0, bx1, by1 = box
    pieces, cur = [], []
    for (ax, ay), (cx, cy) in zip(pts, pts[1:]):
        dx, dy, t0, t1, ok = cx - ax, cy - ay, 0.0, 1.0, True
        for p, q in ((-dx, ax - bx0), (dx, bx1 - ax), (-dy, ay - by0), (dy, by1 - ay)):
            if p == 0:
                if q < 0:
                    ok = False
                    break
            else:
                r = q / p
                if p < 0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
        if not ok or t0 > t1:
            if len(cur) > 1:
                pieces.append(cur)
            cur = []
            continue
        a = (ax + t0 * dx, ay + t0 * dy)
        b = (ax + t1 * dx, ay + t1 * dy)
        if not cur or abs(cur[-1][0] - a[0]) > 1e-6 or abs(cur[-1][1] - a[1]) > 1e-6:
            if len(cur) > 1:
                pieces.append(cur)
            cur = [a]
        cur.append(b)
        if t1 < 1.0:
            pieces.append(cur)
            cur = []
    if len(cur) > 1:
        pieces.append(cur)
    return pieces


def ofm_roads(la: float, lo: float, step: float, max_class: int) -> list:
    """The roads of a cell as landscape's files have them: [[class, paved, [lat ...], [lon ...]], ...]."""
    z = 14 if step <= 0.25 else 10
    ways = []
    for x, y, L, ext, box in _cell_tiles(la, lo, step, z, {"transportation"}):
        for f in L.get("transportation", {}).get("features", []):
            t = f["tags"]
            k = _OMT_ROAD.get(t.get("class"))
            if f["type"] != 2 or k is None or k >= max_class or t.get("brunnel") == "tunnel":
                continue
            surf = t.get("surface")
            paved = 0 if (surf == "unpaved" or (surf is None and k == 7)) else 1
            for part in f["parts"]:
                for piece in _clip_line(part, box):
                    ll = [_lonlat(z, x + px / ext, y + py / ext) for px, py in piece]
                    ways.append([k, paved, [round(a, 6) for a, _ in ll], [round(b, 6) for _, b in ll]])
    return ways


def ofm_buildings(la: float, lo: float, step: float, kind_of) -> list:
    """The buildings of a cell as chase's files have them: [[height m, kind, [lat ...], [lon ...]], ...];
    kind_of(area m^2, height m) picks the kind (OpenMapTiles has no building types)."""
    z = 14
    out = []
    for x, y, L, ext, box in _cell_tiles(la, lo, step, z, {"building"}):
        lat_mid = _lonlat(z, x + 0.5, y + 0.5)[0]
        mpp = 40075016.686 * math.cos(math.radians(lat_mid)) / (2 ** z) / ext
        for f in L.get("building", {}).get("features", []):
            if f["type"] != 3:
                continue
            h = float(f["tags"].get("render_height") or 5.0) - float(f["tags"].get("render_min_height") or 0.0)
            for ring in f["parts"]:
                a = mvt.ring_area(ring)
                if a <= 0 or len(ring) < 4:
                    continue                                # a hole (courtyard)
                cx = sum(p[0] for p in ring[:-1]) / (len(ring) - 1)
                cy = sum(p[1] for p in ring[:-1]) / (len(ring) - 1)
                if not (0 <= cx < ext and 0 <= cy < ext and box[0] <= cx < box[2] and box[1] <= cy < box[3]):
                    continue                                # its middle is in another tile or cell
                ll = [_lonlat(z, x + px / ext, y + py / ext) for px, py in ring]
                out.append([round(min(max(h, 2.5), 300.0), 1), kind_of(a * mpp * mpp, h),
                            [round(p[0], 6) for p in ll], [round(p[1], 6) for p in ll]])
    return out
