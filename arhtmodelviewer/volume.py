"""3-D views of the storms (the page's volume window, web/js/volume.js), like GR2Analyst's volume explorer: a box of
the run's latest radar fields (radar.PatchRadarFields: reflectivity, the winds) resampled onto an even grid, one byte
a point (over terrain, then the ground under each column, uint16 m; heights are above the model's zero, the
sounding's ground).  Patches (and nests) that overlap are combined by their largest value.  "cloud" (the page's cloud mode): two
bytes a point, the cloud (cloud water, ice, snow) and the precipitation (rain, graupel, hail) in g/kg, log-coded."""
from __future__ import annotations

import json
import math
import struct

import numpy as np

MAX_SIDE = 200                   # points along the box's longer side
NZ = 72                          # levels, evenly from the ground to ZTOP
ZTOP = 18000.0                   # m
# field: (what is in the fields, byte = clip(value * scale + offset, 1, 255), 0 = nothing there, units)
FIELDS = {
    "dbz": (2.0, 66.0, "dBZ"),           # -32.5 .. 94.5 dBZ in half-dB steps
    "w": (2.0, 128.0, "m/s"),            # -63.5 .. 63.5 m/s
    "zeta": (2000.0, 128.0, "1/s"),      # vertical vorticity, -0.0635 .. 0.0635 /s
    "cloud": (40.0, 121.0, "log10 g/kg"),  # byte 1 + 40 (log10 q + 3): 0.001 .. 31 g/kg
    "pp": (2.0, 200.0, "hPa"),           # pressure perturbation (the tornado view): -99.5 .. 27.5 hPa
}
LIVE_NZ = 96                     # the tornado view's levels, ground to its top
# nests in the ordinary view: up to INSET_MAX of the finest grids in the box ride along at their own spacing
# (INSET_SIDE points a side at most), so a tornado zoom's 55 m grid is not smeared onto the box's ~500 m one
INSET_MAX = 4
INSET_SIDE = 160
# chase mode's moving clouds (extract(..., wind=True)): the ground-relative wind on every WIND_STRIDE-th point of the
# box and of each inset, three bytes a point (east, north, up: clip(rint(m/s * WIND_SCALE + 128), 1, 255); 128 is calm
# and "no data"), so the page can carry the cloud along the wind between two volumes instead of fading one into the next
WIND_STRIDE = 2
WIND_SCALE = 2.0                 # 0.5 m/s steps, +-63.5 m/s


def levels_for(dx: float) -> tuple:
    """(top m, levels) of a grid of spacing dx m in the 3-D view: the box's own levels for storm-scale grids,
    finer ones low down for the fine nests (where the tornado and its funnel are)."""
    if dx >= 400.0:
        return ZTOP, NZ
    if dx >= 150.0:
        return 8000.0, 96
    return 5000.0, 112
CLOUD_SPECIES = ("qc", "qi", "qs")       # what the cloud is made of; the rest falls out of it
PRECIP_SPECIES = ("qr", "qg", "qh")


def _live_patch(patches, pid):
    """Patch ``pid`` of ``patches`` if it has a state to read now, else None (the radar fields alone): a nest being
    re-placed has none for a moment, and a volume asked for then failed ('NoneType' object has no attribute 'p')."""
    p = (patches or {}).get(pid)
    return p if p is not None and getattr(p, "state", None) is not None else None


def _species(patch, names):
    """Sum of the state's mixing ratios ``names`` that exist, in g/kg."""
    st = patch.state
    a = None
    for n in names:
        q = getattr(st, n, None)
        if q is not None and q.ndim == 3:
            a = q.astype("float32") if a is None else a + q
    return None if a is None else 1000.0 * a


def ground(f, X, Y):
    """The model's ground (m above its zero) under ground points X, Y from radar fields ``f`` (cupy, bilinear;
    NaN outside the patch), or None on a flat run."""
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    if getattr(f, "hgt", None) is None:
        return None
    gi = cp.asarray((X - f.x0) / f.dx - 0.5, cp.float32)
    gj = cp.asarray((Y - f.y0) / f.dx - 0.5, cp.float32)
    v = cnd.map_coordinates(f.hgt, cp.stack([gj, gi]), order=1, mode="nearest")
    return cp.where((gi >= 0) & (gi <= f.nx - 1) & (gj >= 0) & (gj <= f.ny - 1), v, cp.nan)


def _sample(f, name: str, X, Y, zv, patch=None):
    """``name`` of radar fields ``f`` at ground points X, Y (2-D, m) and heights zv (m): trilinear, NaN outside.
    The cloud's "cloud" / "precip" come from the patch's state."""
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    if name == "dbz":
        a = f.dbz
    elif name == "w":
        a = f.wc
    elif name in ("cloud", "precip"):
        a = _species(patch, CLOUD_SPECIES if name == "cloud" else PRECIP_SPECIES) if patch is not None else None
        if a is None:
            return None
    elif name in ("wu", "wv", "ww"):
        # the wind, ground-relative: the patch's state now (the cloud's own moment), else the radar fields'
        a = _wind(patch, name) if patch is not None else {"wu": f.uc, "wv": f.vc, "ww": f.wc}[name]
    else:
        a = cp.gradient(f.vc, f.dx, axis=2) - cp.gradient(f.uc, f.dx, axis=1)
    zh = np.asarray(cp.asnumpy(f.z_h) if hasattr(f.z_h, "device") else f.z_h, np.float64)
    gi = (X - f.x0) / f.dx - 0.5
    gj = (Y - f.y0) / f.dx - 0.5
    inside = (gi >= 0) & (gi <= f.nx - 1) & (gj >= 0) & (gj <= f.ny - 1)
    if not inside.any():
        return None
    nz, (ny, nx) = len(zv), X.shape
    hg = ground(f, X, Y)
    if hg is None:
        gk = np.interp(zv, zh, np.arange(len(zh), dtype=np.float64), left=0.0, right=np.nan)
        K = cp.asarray(np.broadcast_to(gk[:, None, None], (nz, ny, nx)), dtype=cp.float32)
    else:                         # over terrain: each column's levels squeezed over its ground (as sample_sweep)
        h = (cp.asarray(zv, cp.float32)[:, None, None] - hg[None]) / (1.0 - hg[None] / f.ztop)
        K = cp.interp(h.ravel(), cp.asarray(zh, cp.float32), cp.arange(len(zh), dtype=cp.float32),
                      left=0.0, right=cp.nan).reshape(nz, ny, nx)
        K = cp.where(h < 0, cp.nan, K)
    J = cp.asarray(np.broadcast_to(gj[None], (nz, ny, nx)), dtype=cp.float32)
    I = cp.asarray(np.broadcast_to(gi[None], (nz, ny, nx)), dtype=cp.float32)
    v = cnd.map_coordinates(a.astype(cp.float32), cp.stack([cp.nan_to_num(K, nan=0.0), J, I]), order=1,
                            mode="nearest")
    ok = cp.asarray(inside)[None] & ~cp.isnan(K)
    return cp.where(ok, v, cp.nan)


def _wind(patch, name):
    """The patch's ground-relative wind now on its mass points: "wu" east, "wv" north (+ the grid's motion), "ww" up."""
    st = patch.state
    if name == "ww":
        return 0.5 * (st.w[:-1] + st.w[1:])
    mx, my = patch.env.motion
    if name == "wu":
        return 0.5 * (st.u[:, :, :-1] + st.u[:, :, 1:]) + mx
    return 0.5 * (st.v[:, :-1, :] + st.v[:, 1:, :]) + my


def _wind_codes(fields, patches, box, nx, ny, ztop, nz):
    """The wind (WIND_STRIDE) under a grid of nx, ny, nz points over box to ztop: (bytes, [nx, ny, nz]) of the coarser
    grid, the finest field holding each point (a coarse grid's wind would smear a nest's vortex), calm elsewhere."""
    import cupy as cp
    x0, y0, x1, y1 = box
    wx, wy, wz = (max(2, int(math.ceil(n / WIND_STRIDE))) for n in (nx, ny, nz))
    xs = x0 + (np.arange(wx) + 0.5) * (x1 - x0) / wx
    ys = y0 + (np.arange(wy) + 0.5) * (y1 - y0) / wy
    X, Y = np.meshgrid(xs, ys)
    zv = (np.arange(wz) + 0.5) * ztop / wz
    best = [cp.full((wz, wy, wx), cp.nan, cp.float32) for _ in range(3)]
    for f in sorted(fields, key=lambda f: -f.dx):                  # coarse first: the finer ones overwrite
        if f.x0 > x1 or f.x0 + f.nx * f.dx < x0 or f.y0 > y1 or f.y0 + f.ny * f.dx < y0:
            continue
        p = _live_patch(patches, f.patch_id)
        for c, ch in enumerate(("wu", "wv", "ww")):
            v = _sample(f, ch, X, Y, zv, p)
            if v is not None:
                best[c] = cp.where(cp.isnan(v), best[c], v)
    codes = [cp.where(cp.isnan(b), 128, cp.clip(cp.rint(b * WIND_SCALE + 128.0), 1, 255)).astype(cp.uint8) for b in best]
    return cp.asnumpy(cp.stack(codes, axis=-1)).tobytes(), [wx, wy, wz]


def _live_field(patch, name: str, win):
    """``name`` on the patch's mass points in its local cells ``win`` (j0, j1, i0, i1), from its state now."""
    import cupy as cp
    st = patch.state
    j0, j1, i0, i1 = win
    if name in ("cloud", "precip"):
        a = _species(patch, CLOUD_SPECIES if name == "cloud" else PRECIP_SPECIES)
        return None if a is None else a[:, j0:j1, i0:i1]
    if name == "w":
        return 0.5 * (st.w[:-1, j0:j1, i0:i1] + st.w[1:, j0:j1, i0:i1])
    if name == "pp":                  # from each level's mean over the window: the vortex's own drop, in hPa
        p = st.p[:, j0:j1, i0:i1].astype(cp.float32)
        return (p - p.mean(axis=(1, 2), keepdims=True)) / 100.0
    if name == "zeta":
        dx = patch.env.grid.dx
        u = 0.5 * (st.u[:, j0:j1, i0:i1] + st.u[:, j0:j1, i0 + 1:i1 + 1])
        v = 0.5 * (st.v[:, j0:j1, i0:i1] + st.v[:, j0 + 1:j1 + 1, i0:i1])
        return cp.gradient(v, dx, axis=2) - cp.gradient(u, dx, axis=1)
    raise ValueError(f"the tornado view has no field {name!r}")


def extract_live(patches, box, name: str = "cloud", ztop: float = 6000.0) -> bytes | None:
    """Like extract, from the patches' (and nests') state now instead of the radar fields, the finest grid wherever
    one holds a point (not the largest value: a coarse grid's smoothed vortex would hide the fine one's), on
    LIVE_NZ levels from the ground to ztop m (heights above the ground, the model's levels as if flat)."""
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    if name not in FIELDS or name == "dbz":
        raise ValueError(f"the tornado view has no field {name!r}")
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    hold = [p for p in patches if p.state is not None
            and p.world_x0() < x1 and p.world_x0() + p.rect.nx * p.env.grid.dx > x0
            and p.world_y0() < y1 and p.world_y0() + p.rect.ny * p.env.grid.dx > y0]
    if not hold:
        return None
    hold.sort(key=lambda p: -p.env.grid.dx)                  # coarse first: the finer ones overwrite
    res = max(min(p.env.grid.dx for p in hold) / 2.0, max(x1 - x0, y1 - y0) / MAX_SIDE)
    nx = max(2, int(math.ceil((x1 - x0) / res)))
    ny = max(2, int(math.ceil((y1 - y0) / res)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys)
    zv = (np.arange(LIVE_NZ) + 0.5) * ztop / LIVE_NZ
    channels = ("cloud", "precip") if name == "cloud" else (name,)
    best = [None] * len(channels)
    for p in hold:
        dx = p.env.grid.dx
        gi = (X - p.world_x0()) / dx - 0.5                   # mass points of the whole patch
        gj = (Y - p.world_y0()) / dx - 0.5
        inside = (gi >= 0) & (gi <= p.rect.nx - 1) & (gj >= 0) & (gj <= p.rect.ny - 1)
        if not inside.any():
            continue
        # only the patch's cells under the box (+1 for the stencils and the interpolation)
        i0 = max(0, int(math.floor(gi[inside].min())) - 1)
        i1 = min(p.rect.nx, int(math.ceil(gi[inside].max())) + 2)
        j0 = max(0, int(math.floor(gj[inside].min())) - 1)
        j1 = min(p.rect.ny, int(math.ceil(gj[inside].max())) + 2)
        gk =np.interp(zv, np.asarray(p.env.z_h, np.float64), np.arange(len(p.env.z_h), dtype=np.float64),
                       left=0.0, right=np.nan)
        nz = len(zv)
        K = np.broadcast_to(gk[:, None, None], (nz, ny, nx))
        J = np.broadcast_to((gj - j0)[None], (nz, ny, nx))
        I = np.broadcast_to((gi - i0)[None], (nz, ny, nx))
        coords = cp.asarray(np.stack([np.nan_to_num(K, nan=0.0), J, I]), dtype=cp.float32)
        ok = cp.asarray(inside[None] & ~np.isnan(K))
        for c, ch in enumerate(channels):
            a = _live_field(p, ch, (j0, j1, i0, i1))
            if a is None:
                continue
            v = cnd.map_coordinates(a.astype(cp.float32), coords, order=1, mode="nearest")
            prev = best[c] if best[c] is not None else cp.full((nz, ny, nx), cp.nan, cp.float32)
            best[c] = cp.where(ok, v, prev)
    if best[0] is None:
        return None
    scale, offset, units = FIELDS[name]
    codes = []
    for b in best:
        if b is None:
            codes.append(cp.zeros((LIVE_NZ, ny, nx), cp.uint8))
        elif name == "cloud":
            codes.append(cp.where(cp.isnan(b) | (b < 1e-3), 0, cp.clip(cp.rint(
                (cp.log10(cp.maximum(b, 1e-3)) + 3.0) * scale + 1.0), 1, 255)).astype(cp.uint8))
        else:
            codes.append(cp.where(cp.isnan(b), 0, cp.clip(cp.rint(b * scale + offset), 1, 255)).astype(cp.uint8))
    data = cp.stack(codes, axis=-1) if len(codes) > 1 else codes[0]
    head = json.dumps({"field": name, "units": units, "nx": nx, "ny": ny, "nz": LIVE_NZ, "box": [x0, y0, x1, y1],
                       "ztop": ztop, "scale": scale, "offset": offset, "channels": len(codes), "ground": False,
                       "live": True, "dx": min(p.env.grid.dx for p in hold)}).encode()
    return struct.pack("<I", len(head)) + head + cp.asnumpy(data).tobytes()


def _plan_insets(entries, box, res, crop=False):
    """The insets of the box: of ``entries`` [(key, x0, y0, x1, y1, dx) m], the finest (up to INSET_MAX) finer
    than the box's spacing ``res`` that overlap it, finest first, each its overlap with the box at its own
    spacing (or INSET_SIDE points a side): [(key, (x0, y0, x1, y1), spacing, nx, ny)].  ``crop``: a nest too wide
    for INSET_SIDE points at its own spacing is cut down to its middle INSET_SIDE points instead of thinned
    (chase mode's 100 km box took a tornado zoom's whole 37 m nest at 72 m, which thinned the funnel; the nest
    is centred on the tornado, and its parent nest covers the rest)."""
    x0, y0, x1, y1 = box
    out = []
    for key, a, b, c, d, dx in sorted(entries, key=lambda e: e[5]):
        if dx >= 0.7 * res or a >= x1 or c <= x0 or b >= y1 or d <= y0:
            continue
        if crop:
            half = INSET_SIDE * dx / 2
            mx, my = (a + c) / 2, (b + d) / 2
            a, b, c, d = max(a, mx - half), max(b, my - half), min(c, mx + half), min(d, my + half)
            if a >= x1 or c <= x0 or b >= y1 or d <= y0:
                continue
        ib = (max(x0, a), max(y0, b), min(x1, c), min(y1, d))
        w, h = ib[2] - ib[0], ib[3] - ib[1]
        r = max(dx, max(w, h) / INSET_SIDE)
        if r >= 0.7 * res or min(w, h) < 3 * r:
            continue
        out.append((key, ib, r, max(2, int(math.ceil(w / r))), max(2, int(math.ceil(h / r)))))
        if len(out) >= INSET_MAX:
            break
    return out


def _pack(name, head: dict, main: bytes, insets, winds: bytes = b"") -> bytes:
    """Header (with the insets' boxes, sizes and levels) then the box's points, its ground, the insets' points (and,
    with the wind, the box's wind and each inset's)."""
    head["insets"] = [m for m, _ in insets]
    hb = json.dumps(head).encode()
    return struct.pack("<I", len(hb)) + hb + main + b"".join(d for _, d in insets) + winds


def _stack(codes):
    import cupy as cp
    return cp.asnumpy(cp.stack(codes, axis=-1) if len(codes) > 1 else codes[0]).tobytes()


def _insets_live(fields, box, name, res, patches, winds=None):
    """The live insets (_plan_insets) from the radar fields of the nests (and their patches' cloud); ``winds`` (a
    list): each one's wind (_wind_codes) appended to it, its size in the inset's "wind" -- chase mode's, whose
    nests are cropped to their own spacing (_plan_insets crop)."""
    import cupy as cp
    by = {f.patch_id: f for f in fields.values()}
    ent = [(f.patch_id, f.x0, f.y0, f.x0 + f.nx * f.dx, f.y0 + f.ny * f.dx, f.dx) for f in fields.values()]
    channels = ("cloud", "precip") if name == "cloud" else (name,)
    out = []
    for key, ib, r, nx, ny in _plan_insets(ent, box, res, crop=winds is not None):
        if winds is not None:
            ztop_, nz_ = levels_for(by[key].dx)
            wb, wd = _wind_codes(list(fields.values()), patches, ib, nx, ny, ztop_, nz_)
            winds.append((wd, wb))
        f = by[key]
        ztop, nz = levels_for(f.dx)
        xs = ib[0] + (np.arange(nx) + 0.5) * (ib[2] - ib[0]) / nx
        ys = ib[1] + (np.arange(ny) + 0.5) * (ib[3] - ib[1]) / ny
        X, Y = np.meshgrid(xs, ys)
        zv = (np.arange(nz) + 0.5) * ztop / nz
        codes = []
        for ch in channels:
            v = _sample(f, ch, X, Y, zv, _live_patch(patches, f.patch_id))
            codes.append(_code(ch, v) if v is not None else cp.zeros((nz, ny, nx), cp.uint8))
        meta = {"box": list(ib), "nx": nx, "ny": ny, "nz": nz, "ztop": ztop, "dx": f.dx}
        if winds is not None:
            meta["wind"] = winds[-1][0]
        out.append((meta, _stack(codes)))
    return out


def extract(fields: dict, box, name: str = "dbz", patches: dict | None = None, wind: bool = False,
            t: float | None = None, drift=None, max_side: int = MAX_SIDE, nz: int = NZ,
            insets: bool = True) -> bytes | None:
    """The volume of ``name`` in box (x0, y0, x1, y1, ground m) from the run's radar fields (and, for "cloud",
    ``patches``: the patches by id): a JSON header (dims, box, heights, scale, channels) then nz * ny * nx points
    (x fastest) of ``channels`` bytes.  None if no patch overlaps the box.  ``wind``: the wind too (_wind_codes;
    the header's "wind": its size, scale), with the model time t s and the storm's motion ``drift`` (m/s) in the
    header.  ``max_side``, ``nz``:
    the grid's points along the longer side and its levels; ``insets`` False: no nests (chase mode's far cloud,
    extract_far), and no grid finer than the box's own spacing read (its parent has the cloud at that scale)."""
    import cupy as cp
    if name not in FIELDS:
        raise ValueError(f"unknown field {name!r}")
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    dxmin = min((f.dx for f in fields.values()), default=1000.0)
    res = max(dxmin, max(x1 - x0, y1 - y0) / max_side)
    nx = max(2, int(math.ceil((x1 - x0) / res)))
    ny = max(2, int(math.ceil((y1 - y0) / res)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys)
    zv = (np.arange(nz) + 0.5) * ZTOP / nz
    channels = ("cloud", "precip") if name == "cloud" else (name,)
    best = [None] * len(channels)
    gnd = None
    coarsest = max((f.dx for f in fields.values()), default=1000.0)
    for f in fields.values():
        if f.x0 > x1 or f.x0 + f.nx * f.dx < x0 or f.y0 > y1 or f.y0 + f.ny * f.dx < y0:
            continue
        if not insets and f.dx < min(res / 4.0, coarsest):
            continue                          # a fine nest seen from far: its parent's grid shows it
        g = ground(f, X, Y)
        if g is not None:
            gnd = g if gnd is None else cp.fmax(gnd, g)
        p = _live_patch(patches, f.patch_id)
        for c, ch in enumerate(channels):
            v = _sample(f, ch, X, Y, zv, p)
            if v is None:
                continue
            best[c] = v if best[c] is None else cp.fmax(best[c], v)
    if best[0] is None:
        return None
    scale, offset, units = FIELDS[name]
    codes = []
    for b in best:
        if b is None:
            codes.append(cp.zeros((nz, ny, nx), cp.uint8))
            continue
        if name == "cloud":                   # log-coded; nothing below 0.001 g/kg
            code = cp.where(cp.isnan(b) | (b < 1e-3), 0,
                            cp.clip(cp.rint((cp.log10(cp.maximum(b, 1e-3)) + 3.0) * scale + 1.0), 1, 255))
        else:
            code = cp.where(cp.isnan(b), 0, cp.clip(cp.rint(b * scale + offset), 1, 255))
        codes.append(code.astype(cp.uint8))
    data = cp.stack(codes, axis=-1) if len(codes) > 1 else codes[0]
    head = {"field": name, "units": units, "nx": nx, "ny": ny, "nz": nz, "box": [x0, y0, x1, y1],
            "ztop": ZTOP, "scale": scale, "offset": offset, "channels": len(codes), "ground": gnd is not None}
    main = cp.asnumpy(data).tobytes()
    if gnd is not None:          # then the ground under each point, uint16 m above the model's zero
        main += cp.asnumpy(cp.clip(cp.rint(cp.nan_to_num(gnd, nan=0.0)), 0, 65535).astype(cp.uint16)).tobytes()
    if not insets:
        return _pack(name, head, main, [])
    if not wind:
        return _pack(name, head, main, _insets_live(fields, (x0, y0, x1, y1), name, res, patches))
    wb, wd = _wind_codes(list(fields.values()), patches, (x0, y0, x1, y1), nx, ny, ZTOP, NZ)
    iw: list = []
    insets = _insets_live(fields, (x0, y0, x1, y1), name, res, patches, iw)
    head.update({"wind": {"dims": wd, "scale": WIND_SCALE, "offset": 128.0}, "t": t,
                 "drift": list(drift or (0.0, 0.0))})
    return _pack(name, head, main, insets, wb + b"".join(b for _, b in iw))


# ---------------------------------------------------------------- the history (the page's 3-D replay)
# Every radar frame (at most every HISTORY_EVERY s of model time) the run keeps each patch's volume on an even grid
# (HISTORY_SIDE points a side at most, the NZ levels to ZTOP), coded as the page gets it, in run_dir/vol3d/
# t<seconds>.npz; extract_past cuts a box out of the one nearest a time like extract does from the live fields.
HISTORY_EVERY = 60.0             # s of model time
HISTORY_SIDE = 256               # points along a patch's longer side
HISTORY_MAX_MB = 1500            # a run's snapshots past this: kept half as often (sim._vol_history)
HISTORY_CHANNELS = ("dbz", "w", "zeta", "cloud", "precip")
_PAST: dict = {}                 # path -> the loaded snapshot (a few, newest used last)


def _code(name, b):
    import cupy as cp
    if name in ("cloud", "precip"):
        scale = FIELDS["cloud"][0]
        return cp.where(cp.isnan(b) | (b < 1e-3), 0,
                        cp.clip(cp.rint((cp.log10(cp.maximum(b, 1e-3)) + 3.0) * scale + 1.0), 1, 255)).astype(cp.uint8)
    scale, offset, _ = FIELDS[name]
    return cp.where(cp.isnan(b), 0, cp.clip(cp.rint(b * scale + offset), 1, 255)).astype(cp.uint8)


def snapshot(fields: dict, patches: dict | None = None) -> dict:
    """The volumes of every patch's radar fields (and, for the cloud, ``patches``' state) now, as numpy arrays
    ready for save_snapshot: per patch "<k>_meta" (first point x, y m, spacing m, nx, ny, top m, levels) and "<k>_<channel>"
    (levels, ny, nx) uint8 codes (levels_for: fine nests on finer levels low down), "<k>_ground" uint16 m over
    terrain."""
    import cupy as cp
    out = {}
    for k, f in enumerate(fields.values()):
        ztop, nz = levels_for(f.dx)
        zv = (np.arange(nz) + 0.5) * ztop / nz
        s = max(1, int(math.ceil(max(f.nx, f.ny) / HISTORY_SIDE)))
        nx, ny = (f.nx + s - 1) // s, (f.ny + s - 1) // s
        xs = f.x0 + (np.arange(nx) * s + 0.5) * f.dx
        ys = f.y0 + (np.arange(ny) * s + 0.5) * f.dx
        X, Y = np.meshgrid(xs, ys)
        p = _live_patch(patches, f.patch_id)
        out[f"{k}_meta"] = np.array([xs[0], ys[0], s * f.dx, nx, ny, ztop, nz], np.float64)
        for ch in HISTORY_CHANNELS:
            v = _sample(f, ch, X, Y, zv, p)
            out[f"{k}_{ch}"] = (cp.asnumpy(_code(ch, v)) if v is not None else np.zeros((nz, ny, nx), np.uint8))
        g = ground(f, X, Y)
        if g is not None:
            out[f"{k}_ground"] = cp.asnumpy(cp.clip(cp.rint(cp.nan_to_num(g, nan=0.0)), 0, 65535).astype(cp.uint16))
    return out


def save_snapshot(folder, t: float, snap: dict) -> int:
    """Write a snapshot (snapshot()) of model time t s to folder/t<s>.npz (written aside, then renamed); its bytes."""
    import os
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"t{int(round(t)):07d}.npz"
    tmp = folder / (".part_" + name)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **snap)
    os.replace(tmp, folder / name)
    return (folder / name).stat().st_size


def history_times(folder) -> list:
    """Model times (s) of the snapshots in folder, oldest first."""
    from pathlib import Path
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(int(p.stem[1:]) for p in folder.glob("t*.npz") if p.stem[1:].isdigit())


def _load(path):
    if path in _PAST:
        d = _PAST.pop(path)
    else:
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        while len(_PAST) >= 4:
            _PAST.pop(next(iter(_PAST)))
    _PAST[path] = d
    return d


def extract_past(folder, t: float, box, name: str = "dbz"):
    """Like extract, from the snapshot in folder nearest model time t (s): (the volume's bytes or None, its time)."""
    from pathlib import Path
    try:
        import cupy as xp
        from cupyx.scipy import ndimage as nd
    except ImportError:                 # no GPU here: the same on the CPU
        xp = np
        from scipy import ndimage as nd
    if name not in FIELDS or name == "pp":
        raise ValueError(f"the 3-D replay has no field {name!r}")
    times = history_times(folder)
    if not times:
        return None, None
    tt = min(times, key=lambda s: abs(s - t))
    snap = _load(str(Path(folder) / f"t{tt:07d}.npz"))
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    keys = sorted({k.split("_")[0] for k in snap})
    metas = {k: snap[f"{k}_meta"] for k in keys}
    over = [k for k in keys if metas[k][0] - metas[k][2] < x1 and metas[k][0] + metas[k][3] * metas[k][2] > x0
            and metas[k][1] - metas[k][2] < y1 and metas[k][1] + metas[k][4] * metas[k][2] > y0]
    if not over:
        return None, tt
    res = max(min(metas[k][2] for k in over), max(x1 - x0, y1 - y0) / MAX_SIDE)
    nx = max(2, int(math.ceil((x1 - x0) / res)))
    ny = max(2, int(math.ceil((y1 - y0) / res)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys)
    channels = ("cloud", "precip") if name == "cloud" else (name,)
    best = [None] * len(channels)
    gnd = None
    zv = (np.arange(NZ) + 0.5) * ZTOP / NZ
    for k in over:
        sx, sy, d, mx, my = metas[k][:5]
        etop, enz = (metas[k][5], int(metas[k][6])) if len(metas[k]) > 6 else (ZTOP, NZ)
        gi, gj = (X - sx) / d, (Y - sy) / d
        inside = (gi >= -0.5) & (gi <= mx - 0.5) & (gj >= -0.5) & (gj <= my - 0.5)
        if not inside.any():
            continue
        J = np.broadcast_to(np.clip(gj, 0, my - 1)[None], (NZ, ny, nx))
        I = np.broadcast_to(np.clip(gi, 0, mx - 1)[None], (NZ, ny, nx))
        kz = zv / etop * enz - 0.5                        # the box's levels on the snapshot's own
        Kz = np.broadcast_to(np.clip(kz, 0, enz - 1)[:, None, None], (NZ, ny, nx))
        coords = xp.asarray(np.stack([Kz, J, I]), dtype=xp.float32)
        ok = xp.asarray(np.broadcast_to(inside[None] & (zv <= etop)[:, None, None], (NZ, ny, nx)))
        for c, ch in enumerate(channels):
            a = xp.asarray(snap[f"{k}_{ch}"], dtype=xp.float32)
            v = nd.map_coordinates(a, coords, order=1, mode="nearest")
            v = xp.where(ok, v, xp.nan)
            best[c] = v if best[c] is None else xp.fmax(best[c], v)
        if f"{k}_ground" in snap:
            g = nd.map_coordinates(xp.asarray(snap[f"{k}_ground"], dtype=xp.float32), coords[1:, 0], order=1,
                                   mode="nearest")
            g = xp.where(xp.asarray(inside), g, xp.nan)
            gnd = g if gnd is None else xp.fmax(gnd, g)
    if best[0] is None:
        return None, tt
    scale, offset, units = FIELDS[name]
    data = xp.stack([xp.where(xp.isnan(b), 0, xp.clip(xp.rint(b), 0, 255)).astype(xp.uint8) for b in best], axis=-1)
    if len(best) == 1:
        data = data[..., 0]
    head = {"field": name, "units": units, "nx": nx, "ny": ny, "nz": NZ, "box": [x0, y0, x1, y1],
            "ztop": ZTOP, "scale": scale, "offset": offset, "channels": len(best), "ground": gnd is not None, "t": tt}
    host = (lambda a: a.get()) if xp is not np else (lambda a: a)
    main = host(data).tobytes()
    if gnd is not None:
        main += host(xp.clip(xp.rint(xp.nan_to_num(gnd, nan=0.0)), 0, 65535).astype(xp.uint16)).tobytes()
    # the nests' insets, on the snapshot's own levels
    ent = [(k, m[0] - m[2] / 2, m[1] - m[2] / 2, m[0] + (m[3] - 0.5) * m[2], m[1] + (m[4] - 0.5) * m[2], m[2])
           for k, m in metas.items()]
    insets = []
    for k, ib, r, inx, iny in _plan_insets(ent, (x0, y0, x1, y1), res):
        sx, sy, d, mx, my = metas[k][:5]
        etop, enz = (metas[k][5], int(metas[k][6])) if len(metas[k]) > 6 else (ZTOP, NZ)
        xs = ib[0] + (np.arange(inx) + 0.5) * (ib[2] - ib[0]) / inx
        ys = ib[1] + (np.arange(iny) + 0.5) * (ib[3] - ib[1]) / iny
        IX, IY = np.meshgrid(xs, ys)
        J = np.broadcast_to(np.clip((IY - sy) / d, 0, my - 1)[None], (enz, iny, inx))
        I = np.broadcast_to(np.clip((IX - sx) / d, 0, mx - 1)[None], (enz, iny, inx))
        Kz = np.broadcast_to(np.arange(enz, dtype=np.float64)[:, None, None], (enz, iny, inx))
        coords = xp.asarray(np.stack([Kz, J, I]), dtype=xp.float32)
        codes = [xp.clip(xp.rint(nd.map_coordinates(xp.asarray(snap[f"{k}_{ch}"], dtype=xp.float32), coords, order=1,
                                                    mode="nearest")), 0, 255).astype(xp.uint8) for ch in channels]
        dat = xp.stack(codes, axis=-1) if len(codes) > 1 else codes[0]
        insets.append(({"box": list(ib), "nx": inx, "ny": iny, "nz": enz, "ztop": float(etop), "dx": float(d)},
                       host(dat).tobytes()))
    return _pack(name, head, main, insets), tt
