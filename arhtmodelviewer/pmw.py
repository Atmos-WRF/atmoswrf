"""The live feed for the PMWeather (Minecraft) addon PMWeatherAtmos: the model's 3-D winds, reflectivity, cloud and
pressure drop around one storm, from the patches' (and nests') state now, the finest grid wherever there is one.

Two grids: "storm", a box ``storm_km`` a side at a few hundred metres, and, when a nest finer than it covers the
target (the tornado zoom), "fine", a box ``fine_km`` a side at that nest's spacing.  One frame is a little-endian
uint32 header length, a JSON header, then each grid's fields one after another, nz * ny * nx codes each (x fastest),
value = (code - offset) / scale; "missing" (no grid there) is code 0 for the unsigned fields (dbz, cloud and precip:
log10 g/kg) and the type's lowest value (-128, -32768) for the signed ones.  Heights are above the ground (the model's levels as if flat), x east, y north, m.
"""
from __future__ import annotations

import json
import math
import struct

import numpy as np

from . import volume

FORMAT = 1
STORM_Z = (10, 40, 80, 150, 250, 400, 600, 850, 1100, 1400, 1800, 2300, 2900, 3600, 4500, 5500, 6600, 7800,
           9000, 10500, 12000, 13500, 15000)
FINE_Z = (5, 12, 20, 30, 45, 60, 80, 105, 135, 170, 210, 260, 320, 400, 500, 620, 760, 920, 1100, 1300, 1500)
STORM_SIDE = 128                 # points along the storm box's side at most
STORM_RES_MIN = 250.0            # m
FINE_SIDE = 160
# the cloud volumes for drawing (even levels, so the mod can use them as 3-D textures as they are)
CLOUD3D_DZ, CLOUD3D_NZ = 250.0, 60        # storm box: ground to 15 km
FINE3D_DZ, FINE3D_NZ = 50.0, 32           # tornado box: ground to 1.6 km (the lowering and the funnel)
TORNADO_MS = 29.0                # ground wind (m/s) that counts as the tornado's, for its width
# field: (dtype, scale, offset); dtype codes "i1" "u1" "i2"
CODES = {
    "storm": {"u": ("i1", 1.0, 0.0), "v": ("i1", 1.0, 0.0), "w": ("i1", 2.0, 0.0),
              "dbz": ("u1", 2.0, 66.0), "cloud": ("u1", 40.0, 121.0), "precip": ("u1", 40.0, 121.0)},
    "cloud3d": {"cloud": ("u1", 40.0, 121.0), "precip": ("u1", 40.0, 121.0)},
    "fine3d": {"cloud": ("u1", 40.0, 121.0)},
    "fine": {"u": ("i2", 50.0, 0.0), "v": ("i2", 50.0, 0.0), "w": ("i2", 50.0, 0.0),
             "pp": ("i2", 100.0, 0.0), "cloud": ("u1", 40.0, 121.0)},
}


def _live(patch, name, win):
    """``name`` of the patch at its mass points in cells ``win``: the winds here, the rest from volume.py."""
    import cupy as cp
    st = patch.state
    j0, j1, i0, i1 = win
    if name == "u":
        return 0.5 * (st.u[:, j0:j1, i0:i1] + st.u[:, j0:j1, i0 + 1:i1 + 1])
    if name == "v":
        return 0.5 * (st.v[:, j0:j1, i0:i1] + st.v[:, j0 + 1:j1 + 1, i0:i1])
    a = volume._live_field(patch, name, win)
    return None if a is None else a.astype(cp.float32)


def sample(patches, X, Y, zv, names):
    """``names`` at ground points X, Y (2-D, m) and heights zv (m above the ground) from the patches' state now,
    the finest grid wherever one holds a point: {name: cupy (nz, ny, nx), NaN where none does}."""
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    x0, x1, y0, y1 = float(X.min()), float(X.max()), float(Y.min()), float(Y.max())
    hold = [p for p in patches if p.state is not None
            and p.world_x0() <= x1 and p.world_x0() + p.rect.nx * p.env.grid.dx >= x0
            and p.world_y0() <= y1 and p.world_y0() + p.rect.ny * p.env.grid.dx >= y0]
    hold.sort(key=lambda p: -p.env.grid.dx)                  # coarse first: the finer ones overwrite
    nz, (ny, nx) = len(zv), X.shape
    out = {n: None for n in names}
    for p in hold:
        dx = p.env.grid.dx
        gi = (X - p.world_x0()) / dx - 0.5
        gj = (Y - p.world_y0()) / dx - 0.5
        inside = (gi >= 0) & (gi <= p.rect.nx - 1) & (gj >= 0) & (gj <= p.rect.ny - 1)
        if not inside.any():
            continue
        i0 = max(0, int(math.floor(gi[inside].min())) - 1)
        i1 = min(p.rect.nx, int(math.ceil(gi[inside].max())) + 2)
        j0 = max(0, int(math.floor(gj[inside].min())) - 1)
        j1 = min(p.rect.ny, int(math.ceil(gj[inside].max())) + 2)
        gk = np.interp(np.asarray(zv, np.float64), np.asarray(p.env.z_h, np.float64),
                       np.arange(len(p.env.z_h), dtype=np.float64), left=0.0, right=np.nan)
        K = np.broadcast_to(gk[:, None, None], (nz, ny, nx))
        coords = cp.asarray(np.stack([np.nan_to_num(K, nan=0.0),
                                      np.broadcast_to((gj - j0)[None], (nz, ny, nx)),
                                      np.broadcast_to((gi - i0)[None], (nz, ny, nx))]), dtype=cp.float32)
        ok = cp.asarray(inside[None] & ~np.isnan(K))
        for n in names:
            a = _live(p, n, (j0, j1, i0, i1))
            if a is None:
                continue
            v = cnd.map_coordinates(a, coords, order=1, mode="nearest")
            prev = out[n] if out[n] is not None else cp.full((nz, ny, nx), cp.nan, cp.float32)
            out[n] = cp.where(ok, v, prev)
    return out


def _dbz(fields, X, Y, zv):
    """Reflectivity at the points from the run's latest radar fields (the largest where they overlap)."""
    import cupy as cp
    best = None
    for f in (fields or {}).values():
        v = volume._sample(f, "dbz", X, Y, np.asarray(zv, np.float64))
        if v is not None:
            best = v if best is None else cp.fmax(best, v)
    return best


def _encode(kind, name, a, shape):
    import cupy as cp
    dt, scale, offset = CODES[kind][name]
    if a is None:
        return np.zeros(shape, dt).tobytes()
    if name in ("cloud", "precip"):
        return cp.asnumpy(volume._code("cloud", a)).tobytes()
    if name == "dbz":
        return cp.asnumpy(volume._code("dbz", a)).tobytes()
    lim = 127 if dt == "i1" else 32767
    code = cp.where(cp.isnan(a), -lim - 1, cp.clip(cp.rint(cp.nan_to_num(a, nan=0.0) * scale + offset), -lim, lim))
    return cp.asnumpy(code.astype(dt)).tobytes()


def _grid(cx, cy, side, res, cap):
    n = max(2, min(cap, int(math.ceil(side / res))))
    step = side / n
    xs = cx - side / 2 + (np.arange(n) + 0.5) * step
    ys = cy - side / 2 + (np.arange(n) + 0.5) * step
    X, Y = np.meshgrid(xs, ys)
    return X, Y, step, [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]


def _tornado(X, Y, step, u, v):
    """The strongest ground wind in the grid, where it is and the width of the area at TORNADO_MS or more."""
    import cupy as cp
    if u is None or v is None:
        return None
    s = cp.nan_to_num(cp.hypot(u[0], v[0]), nan=0.0)
    k = int(cp.argmax(s))
    vmax = float(s.ravel()[k])
    if vmax < TORNADO_MS:
        return None
    j, i = divmod(k, s.shape[1])
    area = float(cp.count_nonzero(s >= TORNADO_MS)) * step * step
    return {"x": float(X[j, i]), "y": float(Y[j, i]), "vmax": round(vmax, 1),
            "width": round(2.0 * math.sqrt(area / math.pi), 1)}


def targets(sim) -> dict:
    """What the feed can follow: the tornado zoom, the confirmed tornadoes, the storm patches (world m)."""
    eng = sim.engine
    out = {"t": float(eng.t), "zoom": None, "tornadoes": [], "patches": []}
    tz = getattr(eng, "tornado", None)
    if tz is not None:
        fp = tz.finest()
        out["zoom"] = {"x": tz.x, "y": tz.y, "dx": tz.chain[-1].env.grid.dx, "placed": fp is not None}
    from . import radar as rd
    for c in getattr(getattr(sim, "tornadoes", None), "items", []):
        if c.found and c.seen >= rd.TORNADO_CONFIRM:
            out["tornadoes"].append({"id": c.id, "x": float(c.x), "y": float(c.y),
                                     "wind": None if c.ve is None else float(c.ve)})
    for p in eng.patches:
        c = p.storm_center or (p.world_x0() + 0.5 * p.rect.nx * p.env.grid.dx,
                               p.world_y0() + 0.5 * p.rect.ny * p.env.grid.dx)
        out["patches"].append({"id": p.id, "x": float(c[0]), "y": float(c[1]), "dx": p.env.grid.dx,
                               "vx": float(p.storm_velocity[0]), "vy": float(p.storm_velocity[1]),
                               "w": p.rect.nx * p.env.grid.dx, "h": p.rect.ny * p.env.grid.dx})
    return out


def pick(tg: dict):
    """The feed's centre when the mod does not give one: the tornado zoom, else a tornado, else the first patch."""
    if tg["zoom"]:
        return tg["zoom"]["x"], tg["zoom"]["y"]
    if tg["tornadoes"]:
        t = max(tg["tornadoes"], key=lambda d: d["wind"] or 0.0)
        return t["x"], t["y"]
    if tg["patches"]:
        return tg["patches"][0]["x"], tg["patches"][0]["y"]
    return None


def frame(sim, cx=None, cy=None, storm_km: float = 40.0, fine_km: float = 3.0, fine: bool = True,
          cloud: bool = True) -> bytes | None:
    """One frame of the feed around (cx, cy) (world m; None: pick()), or None with no storm patch there.  ``cloud``:
    also the cloud volumes for drawing ("cloud3d", "fine3d": the storm's and the tornado box's grids, even levels)."""
    eng = sim.engine
    patches = list(eng.patches) + list(eng.nest_patches())
    tg = targets(sim)
    if cx is None or cy is None:
        c = pick(tg)
        if c is None:
            return None
        cx, cy = c
    storm_side = max(4000.0, min(120000.0, 1000.0 * float(storm_km)))
    X, Y, step, box = _grid(cx, cy, storm_side, max(STORM_RES_MIN, storm_side / STORM_SIDE), STORM_SIDE)
    zs = np.asarray(STORM_Z, np.float64)
    got = sample(patches, X, Y, zs, ("u", "v", "w", "cloud", "precip"))
    if got["u"] is None:
        return None
    vf = getattr(sim, "vol_fields", None)
    got["dbz"] = _dbz(vf[0] if vf else None, X, Y, zs)
    shape = (len(zs),) + X.shape
    grids, blobs = [], []
    names = ("u", "v", "w", "dbz", "cloud", "precip")
    grids.append({"name": "storm", "box": box, "nx": X.shape[1], "ny": X.shape[0], "dx": step, "z": list(STORM_Z),
                  "fields": [[n] + list(CODES["storm"][n]) for n in names]})
    blobs += [_encode("storm", n, got[n], shape) for n in names]
    tor = _tornado(X, Y, step, got["u"], got["v"])
    if cloud:
        zc = (np.arange(CLOUD3D_NZ) + 0.5) * CLOUD3D_DZ
        gc = sample(patches, X, Y, zc, ("cloud", "precip"))
        grids.append({"name": "cloud3d", "box": box, "nx": X.shape[1], "ny": X.shape[0], "dx": step,
                      "z": [float(z) for z in zc], "fields": [[n] + list(CODES["cloud3d"][n]) for n in ("cloud", "precip")]})
        blobs += [_encode("cloud3d", n, gc[n], (CLOUD3D_NZ,) + X.shape) for n in ("cloud", "precip")]
    if fine:
        dxf = min((p.env.grid.dx for p in patches if p.state is not None
                   and p.world_x0() <= cx <= p.world_x0() + p.rect.nx * p.env.grid.dx
                   and p.world_y0() <= cy <= p.world_y0() + p.rect.ny * p.env.grid.dx), default=None)
        if dxf is not None and dxf < 0.7 * step:
            fside = max(500.0, min(FINE_SIDE * dxf, 1000.0 * float(fine_km)))
            Xf, Yf, stepf, boxf = _grid(cx, cy, fside, dxf, FINE_SIDE)
            zf = np.asarray(FINE_Z, np.float64)
            gf = sample(patches, Xf, Yf, zf, ("u", "v", "w", "pp", "cloud"))
            fnames = ("u", "v", "w", "pp", "cloud")
            shp = (len(zf),) + Xf.shape
            grids.append({"name": "fine", "box": boxf, "nx": Xf.shape[1], "ny": Xf.shape[0], "dx": stepf,
                          "z": list(FINE_Z), "model_dx": dxf,
                          "fields": [[n] + list(CODES["fine"][n]) for n in fnames]})
            blobs += [_encode("fine", n, gf[n], shp) for n in fnames]
            if cloud:
                zc = (np.arange(FINE3D_NZ) + 0.5) * FINE3D_DZ
                gc = sample(patches, Xf, Yf, zc, ("cloud",))
                grids.append({"name": "fine3d", "box": boxf, "nx": Xf.shape[1], "ny": Xf.shape[0], "dx": stepf,
                              "z": [float(z) for z in zc], "fields": [["cloud"] + list(CODES["fine3d"]["cloud"])]})
                blobs.append(_encode("fine3d", "cloud", gc["cloud"], (FINE3D_NZ,) + Xf.shape))
            tor = _tornado(Xf, Yf, stepf, gf["u"], gf["v"]) or tor
    motion = (0.0, 0.0)
    near = min(tg["patches"], key=lambda d: math.hypot(d["x"] - cx, d["y"] - cy), default=None)
    if near is not None:
        motion = (near["vx"], near["vy"])
    head = {"format": FORMAT, "t": float(eng.t), "center": [float(cx), float(cy)], "motion": list(motion),
            "tornado": tor, "targets": tg, "grids": grids}
    hb = json.dumps(head).encode()
    return struct.pack("<I", len(hb)) + hb + b"".join(blobs)
