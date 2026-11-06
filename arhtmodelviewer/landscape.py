"""The 3-D view's landscape from real data: what covers the ground (ESA WorldCover 2021, 10 m: trees, grass,
crops, towns, water ...) and the roads (OpenStreetMap, through the Overpass API).

Land cover: WorldCover's 3 x 3 degree cloud-optimized GeoTIFFs (Deflate, 1024 px tiles, overviews down to ~640 m)
are read in place with HTTP range requests, only the tiles a box needs, at the overview whose pixels are about
the picture's; their headers and tiles are kept in output/cache/landcover for good (the map does not change).
A file that does not exist is open ocean.

Roads: Overpass is slow (tens of seconds for a 10 km square) and sometimes busy, so roads come in cells
(ROAD_CELL degrees for every road down to tracks, MAJOR_CELL for the main ones over a whole box), each fetched
once by a background thread and kept in output/cache/osm; a request returns the cells there are and how many
are still on their way, and the page asks again."""
from __future__ import annotations

import io
import logging
import json
import math
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

import numpy as np

from . import PROJECT
from . import osmfetch
from .tiles import UA

log = logging.getLogger(__name__)
LC_CACHE = PROJECT / "output" / "cache" / "landcover"
OSM_CACHE = PROJECT / "output" / "cache" / "osm"
WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{name}_Map.tif")
# WorldCover's classes, by their codes
CLASSES = {10: "trees", 20: "shrubs", 30: "grass", 40: "crops", 50: "built-up", 60: "bare", 70: "snow and ice",
           80: "water", 90: "wetland", 95: "mangroves", 100: "moss and lichen"}
OCEAN = 80

_TYPES = {1: "B", 2: "B", 3: "H", 4: "I", 5: "II", 7: "B", 11: "f", 12: "d", 16: "Q", 17: "q"}


def _range(url: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()


class _Cog:
    """One WorldCover file: its levels (full resolution, then the overviews) and their tiles, read in place."""

    def __init__(self, name: str):
        self.name, self.url = name, WC_URL.format(name=name)
        self.dir = LC_CACHE / name
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "levels.json"
        if meta.is_file():
            self.levels = json.loads(meta.read_text())
        else:
            self.levels = self._read_levels()
            meta.write_text(json.dumps(self.levels))
        self._lock = threading.Lock()

    def _read_levels(self) -> list:
        head = _range(self.url, 0, 1 << 18)

        def get(a, n):                       # bytes a .. a+n of the file, from the header when it holds them
            return head[a:a + n] if a + n <= len(head) else _range(self.url, a, a + n)
        big = head[2:4] == b"\x2b\x00"
        off = struct.unpack_from("<Q" if big else "<I", head, 8 if big else 4)[0]
        levels = []
        while off:
            n = struct.unpack_from("<Q" if big else "<H", get(off, 8), 0)[0]
            es = 20 if big else 12
            ent = get(off + (8 if big else 2), n * es + 8)
            tags = {}
            for i in range(n):
                e = i * es
                tag, typ = struct.unpack_from("<HH", ent, e)
                cnt = struct.unpack_from("<Q" if big else "<I", ent, e + 4)[0]
                fmt = _TYPES.get(typ, "B")
                size = struct.calcsize("<" + fmt) * cnt
                inline = 8 if big else 4
                vo = e + (12 if big else 8)
                raw = ent[vo:vo + size] if size <= inline else get(
                    struct.unpack_from("<Q" if big else "<I", ent, vo)[0], size)
                tags[tag] = struct.unpack_from("<%d%s" % (cnt * len(fmt), fmt[0]), raw, 0)
            if tags.get(259, (1,))[0] != 8:
                raise ValueError("WorldCover file not Deflate-compressed")
            levels.append({"w": tags[256][0], "h": tags[257][0], "tw": tags[322][0], "th": tags[323][0],
                           "off": list(tags[324]), "cnt": list(tags[325])})
            off = struct.unpack_from("<Q" if big else "<I", ent, n * es)[0]
        return levels

    def tile(self, lv: int, tx: int, ty: int) -> np.ndarray:
        L = self.levels[lv]
        f = self.dir / f"{lv}_{tx}_{ty}.z"
        if f.is_file():
            data = f.read_bytes()
        else:
            k = ty * ((L["w"] + L["tw"] - 1) // L["tw"]) + tx
            data = _range(self.url, L["off"][k], L["off"][k] + L["cnt"][k])
            part = f.with_name(f.name + ".part")
            part.write_bytes(data)
            part.replace(f)
        return np.frombuffer(zlib.decompress(data), np.uint8).reshape(L["th"], L["tw"])


_COGS: dict = {}
_COG_GUARD = threading.Lock()


def _cog(lat0: int, lon0: int):
    """The file whose south-west corner is lat0, lon0 (multiples of 3), or None where there is none (ocean)."""
    name = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
    with _COG_GUARD:
        if name in _COGS:
            return _COGS[name]
    none = LC_CACHE / name / "none"
    c = None
    if not none.is_file():
        try:
            c = _Cog(name)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):             # no file: sea
                none.parent.mkdir(parents=True, exist_ok=True)
                none.write_text("")
            else:
                raise
    with _COG_GUARD:
        _COGS[name] = c
    return c


def landcover(proj, box_km, px: int = 1024) -> tuple:
    """WorldCover's class at each pixel of the box (x0, y0, x1, y1 world km, the model's plane), ``px`` on its
    long side, row 0 at the north edge: (W, H, uint8 codes)."""
    x0, y0, x1, y1 = (float(v) for v in box_km)
    w_km, h_km = max(x1 - x0, 0.05), max(y1 - y0, 0.05)
    W = int(px if w_km >= h_km else max(16, round(px * w_km / h_km)))
    H = int(px if h_km >= w_km else max(16, round(px * h_km / w_km)))
    X = x0 + (np.arange(W) + 0.5) / W * w_km
    Y = y1 - (np.arange(H) + 0.5) / H * h_km
    XX, YY = np.meshgrid(X * 1000.0, Y * 1000.0)
    lat, lon = proj.inv(XX.ravel(), YY.ravel())
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    m_px = w_km * 1000.0 / W
    lv = int(np.clip(math.floor(math.log2(max(m_px, 10.0) / 10.0)), 0, 6))
    out = np.full(lat.shape, OCEAN, np.uint8)
    flat, flon = (np.floor(lat / 3.0) * 3).astype(int), (np.floor(lon / 3.0) * 3).astype(int)
    for a, b in sorted(set(zip(flat.tolist(), flon.tolist()))):
        c = _cog(a, b)
        sel = (flat == a) & (flon == b)
        if c is None:
            continue
        L = c.levels[min(lv, len(c.levels) - 1)]
        res = 3.0 / L["w"]
        col = np.clip(((lon[sel] - b) / res).astype(int), 0, L["w"] - 1)
        row = np.clip(((a + 3 - lat[sel]) / res).astype(int), 0, L["h"] - 1)
        tx, ty = col // L["tw"], row // L["th"]
        vals = np.zeros(col.shape, np.uint8)
        for (i, j) in sorted(set(zip(tx.tolist(), ty.tolist()))):
            m = (tx == i) & (ty == j)
            t = c.tile(min(lv, len(c.levels) - 1), i, j)
            vals[m] = t[row[m] - j * L["th"], col[m] - i * L["tw"]]
        vals[vals == 0] = OCEAN                  # no data: WorldCover leaves the open sea empty
        out[sel] = vals
    return W, H, out.reshape(H, W)


# ---------------------------------------------------------------- roads
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter")
ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "track")
MAJOR = 4                        # classes below this one are the main roads (drawn over a whole box)
ROAD_CELL = 0.1                  # degrees: the cells of every road
MAJOR_CELL = 1.0                 # degrees: the cells of the main roads
UNPAVED = {"unpaved", "gravel", "dirt", "ground", "compacted", "fine_gravel", "grass", "sand", "earth", "mud"}
def _cell_file(major: bool, la: float, lo: float):
    return OSM_CACHE / ("major" if major else "all") / f"{la:+08.2f}_{lo:+08.2f}.json"


def _fetch_cell(major: bool, la: float, lo: float) -> None:
    step = MAJOR_CELL if major else ROAD_CELL
    cls = ROAD_CLASSES[:MAJOR] if major else ROAD_CLASSES
    q = (f'[out:json][timeout:120];way["highway"~"^({"|".join(cls)})(_link)?$"]'
         f'({la:.4f},{lo:.4f},{la + step:.4f},{lo + step:.4f});out geom tags;')
    try:                                         # OpenFreeMap's tiles first (fast, a CDN), Overpass if it fails
        ways = osmfetch.ofm_roads(la, lo, step, len(cls))
    except Exception as e:
        log.warning("OpenFreeMap roads %s %s: %s -- trying Overpass", la, lo, e)
        ways = None
    if ways is not None:
        f = _cell_file(major, la, lo)
        f.parent.mkdir(parents=True, exist_ok=True)
        part = f.with_name(f.name + ".part")
        part.write_text(json.dumps(ways))
        part.replace(f)
        return
    d = osmfetch.overpass(q)
    ways = []
    for w in d.get("elements", []):
        g, tags = w.get("geometry"), w.get("tags", {})
        if not g or len(g) < 2:
            continue
        hw = tags.get("highway", "").replace("_link", "")
        if hw not in ROAD_CLASSES:
            continue
        k = ROAD_CLASSES.index(hw)
        surf = tags.get("surface")
        paved = 0 if (surf in UNPAVED or (surf is None and hw == "track")) else 1
        ways.append([k, paved, [round(p["lat"], 6) for p in g], [round(p["lon"], 6) for p in g]])
    f = _cell_file(major, la, lo)
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f.name + ".part")
    part.write_text(json.dumps(ways))
    part.replace(f)


_ROADS = osmfetch.CellQueue("roads", _fetch_cell)


def roads(proj, box_km, major: bool) -> dict:
    """The roads in the box (world km): {"roads": [[class, paved, [x, y, x, y, ...] km], ...], "pending": n,
    "busy": Overpass not answering} -- every road (major False) or the main ones (True); cells not fetched yet are
    queued (osmfetch: the newest asked for first) and counted."""
    x0, y0, x1, y1 = (float(v) for v in box_km)
    xs = np.array([x0, x1, x0, x1, (x0 + x1) / 2, (x0 + x1) / 2, x0, x1]) * 1000.0
    ys = np.array([y0, y0, y1, y1, y0, y1, (y0 + y1) / 2, (y0 + y1) / 2]) * 1000.0
    lat, lon = proj.inv(xs, ys)
    lat, lon = np.asarray(lat), np.asarray(lon)
    step = MAJOR_CELL if major else ROAD_CELL
    cells = [(major, round(a * step, 2), round(b * step, 2))
             for a in range(int(math.floor(lat.min() / step)), int(math.floor(lat.max() / step)) + 1)
             for b in range(int(math.floor(lon.min() / step)), int(math.floor(lon.max() / step)) + 1)]
    if len(cells) > 400:
        raise ValueError("the box is too big for the roads")
    out, pending = [], 0
    for c in cells:
        f = _cell_file(*c)
        if not f.is_file():
            pending += _ROADS.want(c)
            continue
        for k, paved, la, lo in json.loads(f.read_text()):
            x, y = proj.fwd(np.asarray(la), np.asarray(lo))
            xy = np.empty(2 * len(la))
            xy[0::2], xy[1::2] = np.asarray(x) / 1000.0, np.asarray(y) / 1000.0
            if xy[0::2].max() < x0 or xy[0::2].min() > x1 or xy[1::2].max() < y0 or xy[1::2].min() > y1:
                continue
            out.append([k, paved, np.round(xy, 4).tolist()])
    return {"roads": out, "pending": pending, "busy": bool(pending) and osmfetch.busy(), "classes": ROAD_CLASSES}
