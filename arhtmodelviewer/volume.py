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
CLOUD_SPECIES = ("qc", "qi", "qs")       # what the cloud is made of; the rest falls out of it
PRECIP_SPECIES = ("qr", "qg", "qh")


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


HAZE_RH = 0.95                   # the tornado view: relative humidity where air starts to look like cloud
HAZE_GKG = 0.25                  # g/kg of "cloud" it shows as at saturation


def _near_saturation(patch, win, cloud):
    """The tornado view's cloud plus the air that is nearly saturated: a vortex's pressure drop cools the air
    it draws in toward saturation, which is how a real funnel and wall cloud condense below the storm's base.
    The model only condenses at 100 %, and a ~100 m grid's vortex rarely gets there, so the air from HAZE_RH
    up is drawn as thin cloud (HAZE_GKG at saturation): the lowering and the funnel where the vortex is."""
    import cupy as cp
    st = patch.state
    j0, j1, i0, i1 = win
    thb = st.thb if st.thb.ndim == 3 else st.thb[:, None, None]
    if thb.ndim == 3 and thb.shape[-1] > 1:
        thb = thb[:, j0:j1, i0:i1]
    p = st.p[:, j0:j1, i0:i1].astype(cp.float32)
    T = (thb + st.thp[:, j0:j1, i0:i1]).astype(cp.float32) * (p / 1.0e5) ** 0.2857
    es = 611.2 * cp.exp(17.67 * (T - 273.15) / (T - 29.65))              # Bolton, over water
    qs = 0.622 * es / cp.maximum(p - es, 1.0)
    rh = st.qv[:, j0:j1, i0:i1].astype(cp.float32) / qs
    haze = HAZE_GKG * cp.clip((rh - HAZE_RH) / (1.0 - HAZE_RH), 0.0, 1.0) ** 2
    return haze if cloud is None else cloud + haze


def _live_field(patch, name: str, win):
    """``name`` on the patch's mass points in its local cells ``win`` (j0, j1, i0, i1), from its state now."""
    import cupy as cp
    st = patch.state
    j0, j1, i0, i1 = win
    if name in ("cloud", "precip"):
        a = _species(patch, CLOUD_SPECIES if name == "cloud" else PRECIP_SPECIES)
        a = None if a is None else a[:, j0:j1, i0:i1]
        if name == "cloud":
            a = _near_saturation(patch, (j0, j1, i0, i1), a)
        return a
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


def extract(fields: dict, box, name: str = "dbz", patches: dict | None = None) -> bytes | None:
    """The volume of ``name`` in box (x0, y0, x1, y1, ground m) from the run's radar fields (and, for "cloud",
    ``patches``: the patches by id): a JSON header (dims, box, heights, scale, channels) then nz * ny * nx points
    (x fastest) of ``channels`` bytes.  None if no patch overlaps the box."""
    import cupy as cp
    if name not in FIELDS:
        raise ValueError(f"unknown field {name!r}")
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    dxmin = min((f.dx for f in fields.values()), default=1000.0)
    res = max(dxmin, max(x1 - x0, y1 - y0) / MAX_SIDE)
    nx = max(2, int(math.ceil((x1 - x0) / res)))
    ny = max(2, int(math.ceil((y1 - y0) / res)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys)
    zv = (np.arange(NZ) + 0.5) * ZTOP / NZ
    channels = ("cloud", "precip") if name == "cloud" else (name,)
    best = [None] * len(channels)
    gnd = None
    for f in fields.values():
        if f.x0 > x1 or f.x0 + f.nx * f.dx < x0 or f.y0 > y1 or f.y0 + f.ny * f.dx < y0:
            continue
        g = ground(f, X, Y)
        if g is not None:
            gnd = g if gnd is None else cp.fmax(gnd, g)
        p = (patches or {}).get(f.patch_id)
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
            codes.append(cp.zeros((NZ, ny, nx), cp.uint8))
            continue
        if name == "cloud":                   # log-coded; nothing below 0.001 g/kg
            code = cp.where(cp.isnan(b) | (b < 1e-3), 0,
                            cp.clip(cp.rint((cp.log10(cp.maximum(b, 1e-3)) + 3.0) * scale + 1.0), 1, 255))
        else:
            code = cp.where(cp.isnan(b), 0, cp.clip(cp.rint(b * scale + offset), 1, 255))
        codes.append(code.astype(cp.uint8))
    data = cp.stack(codes, axis=-1) if len(codes) > 1 else codes[0]
    head = json.dumps({"field": name, "units": units, "nx": nx, "ny": ny, "nz": NZ, "box": [x0, y0, x1, y1],
                       "ztop": ZTOP, "scale": scale, "offset": offset, "channels": len(codes),
                       "ground": gnd is not None}).encode()
    out = struct.pack("<I", len(head)) + head + cp.asnumpy(data).tobytes()
    if gnd is not None:          # then the ground under each point, uint16 m above the model's zero
        out += cp.asnumpy(cp.clip(cp.rint(cp.nan_to_num(gnd, nan=0.0)), 0, 65535).astype(cp.uint16)).tobytes()
    return out
