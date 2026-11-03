"""GR2Analyst / Supercell-Wx placefiles of a run's run-max swaths: storm tracks (2-5 km updraft
helicity) and wind swaths, the page's ``uhtrack`` and ``windtrack``.

The swath is one PNG on the viewer's map grid (Lambert), drawn by the placefile's Image statement
as a mesh of triangles whose corners carry their latitude and longitude, so the picture follows the
viewer's own projection; a label marks the run's strongest UH.  GR2Analyst 3 reads each Image vertex
as "%f,%f,%f,%f" (no space before a comma), needs a Font for a Text, and re-reads the placefile
every RefreshSeconds; each new swath gets a new image URL, so a picture it cached is never shown
again.  The same viewer that polls the radars (127.0.0.1, or another computer with --lan) reads it
from /poll/<name>.txt.
"""
from __future__ import annotations

import math
import struct
import zlib

import numpy as np

from .products import CODECS, TRACK_TILE

# the page's palettes (web/js/colors.js: uh, wspd), linear between stops, transparent below the first
STOPS = {
    "uhtrack": [(0, (0, 0, 0, 0)), (25, (90, 90, 120, 140)), (75, (60, 150, 230, 255)), (150, (60, 220, 90, 255)),
                (300, (245, 230, 40, 255)), (500, (245, 130, 30, 255)), (800, (230, 30, 30, 255)),
                (1200, (250, 110, 240, 255)), (2000, (255, 255, 255, 255))],
    "windtrack": [(0, (20, 20, 40, 0)), (5, (50, 70, 120, 160)), (10, (40, 140, 210, 255)), (15, (50, 200, 120, 255)),
                  (20, (240, 230, 40, 255)), (25, (245, 150, 30, 255)), (32, (230, 30, 30, 255)),
                  (40, (250, 100, 240, 255)), (55, (255, 255, 255, 255))],
}
TITLES = {"uhtrack": "run-max 2-5 km updraft helicity (m2/s2)", "windtrack": "run-max lowest-level wind (m/s)"}
# transparent below: over the radar, the faint low end (on the page the patches' ambient wind, weak
# UH under every updraft) would only veil the echo
VISIBLE = {"uhtrack": 25.0, "windtrack": 15.0}
MAX_SIDE = 4096               # px: a larger swath is max-pooled down (viewers' texture limit)
MESH_KM = 20.0                # km between the mesh's corners (the projections differ by far less inside one)


def lut(name: str) -> np.ndarray:
    """RGBA (256, 4) for a swath's codes; code 0 (no data) transparent."""
    scale, offset, _, _ = CODECS[name]
    vals = (np.arange(256) - offset) / scale
    v = np.array([s[0] for s in STOPS[name]], float)
    c = np.array([s[1] for s in STOPS[name]], float)
    out = np.stack([np.interp(vals, v, c[:, k]) for k in range(4)], axis=1)
    out[vals < max(v[0], VISIBLE[name])] = 0.0
    out[0] = 0.0
    return np.rint(out).astype(np.uint8)


def first_code(name: str) -> int:
    """The lowest code drawn (VISIBLE)."""
    scale, offset, _, _ = CODECS[name]
    return int(math.ceil(VISIBLE[name] * scale + offset))


def png(rgba: np.ndarray) -> bytes:
    """An RGBA image (h, w, 4) uint8 as PNG."""
    h, w = rgba.shape[:2]
    raw = np.zeros((h, 1 + 4 * w), np.uint8)          # filter byte 0 (none) on every row
    raw[:, 1:] = rgba.reshape(h, 4 * w)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))


def mosaic(store, name: str, dx: float, min_code: int | None = None):
    """A swath's tiles pieced together and cropped to where it reaches ``min_code`` (default: where
    it is drawn, VISIBLE): (codes (ny, nx)
    uint8, row 0 the north edge, max-pooled to at most MAX_SIDE a side; (x0, y0, x1, y1) in map
    metres), or None if it has none yet."""
    T = TRACK_TILE
    min_code = first_code(name) if min_code is None else min_code
    items = [((ti, tj), a) for (n, ti, tj), a in list(store.tiles.items()) if n == name]
    items = [(k, a) for k, a in items if int(a.max()) >= min_code]
    if not items:
        return None
    ti0 = min(k[0] for k, _ in items)
    tj0 = min(k[1] for k, _ in items)
    nx = (max(k[0] for k, _ in items) - ti0 + 1) * T
    ny = (max(k[1] for k, _ in items) - tj0 + 1) * T
    full = np.zeros((ny, nx), np.uint8)
    for (ti, tj), a in items:
        i, j = (ti - ti0) * T, (tj - tj0) * T
        np.maximum(full[j:j + T, i:i + T], a, out=full[j:j + T, i:i + T])
    jj, ii = np.nonzero(full >= min_code)
    j0, j1 = max(int(jj.min()) - 1, 0), min(int(jj.max()) + 2, ny)
    i0, i1 = max(int(ii.min()) - 1, 0), min(int(ii.max()) + 2, nx)
    codes = full[j0:j1, i0:i1]
    f = max(1, math.ceil(max(codes.shape) / MAX_SIDE))
    if f > 1:
        h, w = -(-codes.shape[0] // f) * f, -(-codes.shape[1] // f) * f
        pad = np.zeros((h, w), np.uint8)
        pad[:codes.shape[0], :codes.shape[1]] = codes
        codes = pad.reshape(h // f, f, w // f, f).max(axis=(1, 3))
        j1, i1 = j0 + h, i0 + w
    x0, y0 = (ti0 * T + i0) * dx, (tj0 * T + j0) * dx
    return codes[::-1].copy(), (x0, y0, (ti0 * T + i1) * dx, (tj0 * T + j1) * dx)


def image(store, name: str, dx: float):
    """(PNG bytes, (x0, y0, x1, y1)) of a swath, or None."""
    m = mosaic(store, name, dx)
    if m is None:
        return None
    codes, box = m
    return png(lut(name)[codes]), box


def text(name: str, box, proj, image_url: str | None, title: str, refresh: int = 30, peak: dict | None = None,
         label: str = "") -> str:
    """The placefile: the swath's image over ``box`` (map metres, None: no swath yet) and, for the
    storm tracks, a label on the run's strongest UH (``peak``: value, x, y, hover text in ``label``)."""
    lines = [f"; Atmos-WRF: {TITLES[name]}", f"Title: {title}", f"RefreshSeconds: {int(refresh)}",
             "Threshold: 999"]
    if box is not None and image_url:
        x0, y0, x1, y1 = box
        nqx = min(max(1, math.ceil((x1 - x0) / (MESH_KM * 1000.0))), 48)
        nqy = min(max(1, math.ceil((y1 - y0) / (MESH_KM * 1000.0))), 48)
        tu = np.linspace(0.0, 1.0, nqx + 1)
        tv = np.linspace(0.0, 1.0, nqy + 1)
        X, Y = np.meshgrid(x0 + tu * (x1 - x0), y1 - tv * (y1 - y0))       # tv 0 at the top (north)
        lat, lon = proj.inv(X, Y)
        v = lambda a, b: f"{lat[b, a]:.5f},{lon[b, a]:.5f},{tu[a]:.5f},{tv[b]:.5f}"
        lines.append(f"Image: {image_url}")
        for b in range(nqy):
            for a in range(nqx):
                lines += [v(a, b), v(a + 1, b), v(a + 1, b + 1), v(a, b), v(a + 1, b + 1), v(a, b + 1)]
        lines.append("End:")
    if peak:
        la, lo = proj.inv(peak["x"], peak["y"])
        uh = round(peak["value"])
        lines += ['Font: 1, 13, 1, "Arial"', "Color: 255 255 255",
                  f'Text: {float(la):.5f}, {float(lo):.5f}, 1, "UH {uh}", "{label}"']
    return "\n".join(lines) + "\n"
