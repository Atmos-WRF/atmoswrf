"""Map tiles under the map and the 3-D view: OpenStreetMap (and CARTO's dark map of OSM data) and satellite imagery.

The page asks /tiles/<source>/<z>/<x>/<y>; this fetches the tile from its provider once, with an identifying
User-Agent as the providers' tile policies ask, and keeps it in output/cache/tiles for TILE_DAYS (maps also work
offline where they have been looked at).  The map warps each Web-Mercator tile onto the model's Lambert plane
itself (map.js); the 3-D view gets one picture of its box, reprojected here (ground()).  Every source's
attribution is drawn on the map, as its terms require."""
from __future__ import annotations

import io
import math
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import PROJECT

CACHE = PROJECT / "output" / "cache" / "tiles"
TILE_DAYS = 30
UA = "Atmos-WRF/1.0 (storm model viewer; +https://github.com/Atmos-WRF/atmoswrf)"
# id: (URL template, max zoom, attribution, tile size)
SOURCES = {
    "osm": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png", 19, "© OpenStreetMap contributors"),
    "sat": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 19,
            "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community"),
    # over the satellite picture: place names and borders, and roads (Esri's reference layers, transparent)
    "places": ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/"
               "tile/{z}/{y}/{x}", 19, "Labels © Esri"),
    "roads": ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/"
              "tile/{z}/{y}/{x}", 19, "Roads © Esri"),
}
# (CARTO's OSM basemaps need an API key since 2025: their tiles came back stamped "API KEY REQUIRED".  The dark
# map is OSM's own tiles darkened in the page instead, map.js.)
_LOCKS: dict = {}
_GUARD = threading.Lock()


def info() -> dict:
    return {k: {"maxz": v[1], "attribution": v[2]} for k, v in SOURCES.items()}


def tile(src: str, z: int, x: int, y: int) -> bytes:
    """One tile's image (PNG or JPEG bytes), from the cache or its provider."""
    if src not in SOURCES:
        raise ValueError("unknown map source")
    url, maxz, _ = SOURCES[src]
    if not (0 <= z <= maxz and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise ValueError("no such tile")
    f = CACHE / src / str(z) / str(x) / f"{y}"
    if f.is_file() and time.time() - f.stat().st_mtime < TILE_DAYS * 86400:
        return f.read_bytes()
    with _GUARD:
        lock = _LOCKS.setdefault((src, z, x, y), threading.Lock())
    with lock:
        if f.is_file() and time.time() - f.stat().st_mtime < TILE_DAYS * 86400:
            return f.read_bytes()
        req = urllib.request.Request(url.format(z=z, x=x, y=y), headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
        except Exception:
            if f.is_file():                          # offline or refused: the old copy rather than nothing
                return f.read_bytes()
            raise
        f.parent.mkdir(parents=True, exist_ok=True)
        part = f.with_name(f.name + ".part")
        part.write_bytes(data)
        part.replace(f)
    with _GUARD:
        _LOCKS.pop((src, z, x, y), None)
    return data


def _merc(lat, lon, z):
    n = 2.0 ** z
    lat = np.clip(lat, -85.0511, 85.0511)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.log(np.tan(np.radians(lat)) + 1.0 / np.cos(np.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def ground(proj, box_km, srcs=("sat",), px: int = 1024) -> bytes:
    """A PNG of the box (x0, y0, x1, y1 world km, north-up in the model's plane), ``px`` on its long side, from
    ``srcs`` drawn in order (a base map, then labels): the 3-D view's ground."""
    from PIL import Image
    x0, y0, x1, y1 = (float(v) for v in box_km)
    w_km, h_km = max(x1 - x0, 0.1), max(y1 - y0, 0.1)
    W = int(px if w_km >= h_km else max(64, round(px * w_km / h_km)))
    H = int(px if h_km >= w_km else max(64, round(px * h_km / w_km)))
    X = x0 + (np.arange(W) + 0.5) / W * w_km
    Y = y1 - (np.arange(H) + 0.5) / H * h_km                     # row 0 at the north edge
    XX, YY = np.meshgrid(X * 1000.0, Y * 1000.0)
    lat, lon = proj.inv(XX.ravel(), YY.ravel())
    lat = np.asarray(lat, np.float64).reshape(XX.shape)
    lon = np.asarray(lon, np.float64).reshape(XX.shape)
    # the zoom whose pixels are about the picture's
    m_px = w_km * 1000.0 / W
    clat = float(np.mean(lat))
    out = np.zeros((H, W, 4), np.float32)
    for src in srcs:
        maxz = SOURCES[src][1]
        z = int(np.clip(round(math.log2(156543.03 * math.cos(math.radians(clat)) / max(m_px, 0.5))), 2, min(maxz, 17)))
        mx, my = _merc(lat, lon, z)
        tx, ty = np.floor(mx).astype(int), np.floor(my).astype(int)
        need = sorted(set(zip(tx.ravel().tolist(), ty.ravel().tolist())))
        if len(need) > 400:
            raise ValueError("the box is too big for the map tiles")
        layer = np.zeros((H, W, 4), np.float32)
        for (a, b) in need:
            try:
                im = Image.open(io.BytesIO(tile(src, z, a, b))).convert("RGBA")
            except Exception:
                continue
            arr = np.asarray(im, np.float32)
            m = (tx == a) & (ty == b)
            u = np.clip(((mx[m] - a) * arr.shape[1]).astype(int), 0, arr.shape[1] - 1)
            v = np.clip(((my[m] - b) * arr.shape[0]).astype(int), 0, arr.shape[0] - 1)
            layer[m] = arr[v, u]
        al = layer[..., 3:4] / 255.0
        out[..., :3] = out[..., :3] * (1 - al) + layer[..., :3] * al
        out[..., 3:4] = np.maximum(out[..., 3:4], layer[..., 3:4])
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
