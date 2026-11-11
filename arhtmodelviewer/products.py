"""Map products of the ArWen patches, packed as 8-bit codes for the browser.

Every product is a (ny, nx) uint8 array per patch; ``CODECS`` gives the
value of a code (value = (code - offset) / scale, code 0 = no data), the
same convention the Level II moments use.
"""
from __future__ import annotations

import zlib

import numpy as np

# name: (scale, offset, units, label)
CODECS = {
    "cref": (2.0, 66.0, "dBZ", "Composite reflectivity"),
    "ref1km": (2.0, 66.0, "dBZ", "1 km AGL reflectivity"),
    "reflow": (2.0, 66.0, "dBZ", "Lowest-level reflectivity"),
    "thp": (8.0, 128.0, "K", "Surface theta' (cold pools)"),
    "wmax": (2.0, 1.0, "m/s", "Max updraft"),
    "uh25": (0.1, 1.0, "m2/s2", "2-5 km updraft helicity"),
    "wspd": (2.0, 1.0, "m/s", "Lowest-level wind speed"),
    "vort": (500.0, 128.0, "1/s", "Lowest-level vertical vorticity"),    # +-0.254: tornado-like vortices fit
    "rain": (8.0, 1.0, "mm", "Accumulated rain (log2 scale)"),
    "uhtrack": (0.1, 1.0, "m2/s2", "Run-max 2-5 km UH (storm tracks)"),
    "windtrack": (2.0, 1.0, "m/s", "Run-max lowest-level wind (wind swaths)"),
    "efswath": (2.0, 1.0, "m/s", "EF swath: the tornadoes' run-max wind (grids of 200 m or finer)"),
    # hurricane runs (hurricane.py)
    "ir": (1.5, 166.0, "C", "Simulated IR satellite (10.3 um)"),
    "sst": (7.0, 1.0, "C", "Sea-surface temperature"),
    "slp": (1.6, -1407.0, "hPa", "Sea-level pressure"),
    "wind10": (2.0, 1.0, "m/s", "10 m wind (1-min sustained)"),
}

TRACK_TILE = 64


class TrackStore:
    """Run-max swaths on world-anchored tiles (they outlive the patches)."""

    def __init__(self):
        self.tiles = {}   # (name, ti, tj) -> uint8 (T, T)

    def update(self, name: str, rect, codes: np.ndarray) -> None:
        T = TRACK_TILE
        i0, j0, ny, nx = rect.i0, rect.j0, codes.shape[0], codes.shape[1]
        for tj in range(j0 // T, (j0 + ny - 1) // T + 1):
            for ti in range(i0 // T, (i0 + nx - 1) // T + 1):
                a0, a1 = max(ti * T, i0), min((ti + 1) * T, i0 + nx)
                b0, b1 = max(tj * T, j0), min((tj + 1) * T, j0 + ny)
                sub = codes[b0 - j0:b1 - j0, a0 - i0:a1 - i0]
                if not sub.any():
                    continue
                key = (name, ti, tj)
                tile = self.tiles.get(key)
                if tile is None:
                    tile = self.tiles[key] = np.zeros((T, T), np.uint8)
                view = tile[b0 - tj * T:b1 - tj * T, a0 - ti * T:a1 - ti * T]
                np.maximum(view, sub, out=view)

    def pack(self, t: float, name: str, dx: float) -> bytes | None:
        T = TRACK_TILE
        items = [((ti, tj), a) for (n, ti, tj), a in self.tiles.items() if n == name]
        if not items:
            return None
        lst = [({"x0": ti * T * dx, "y0": tj * T * dx, "x1": (ti + 1) * T * dx, "y1": (tj + 1) * T * dx,
                 "nx": T, "ny": T}, a.copy()) for (ti, tj), a in items]
        return pack_frame_product(t, name, lst)


def _encode(val, name, mask=None):
    import cupy as cp
    scale, offset, _, _ = CODECS[name]
    code = cp.clip(cp.rint(val * scale + offset), 1, 255)
    if mask is not None:
        code = cp.where(mask, code, 0)
    return cp.asnumpy(code.astype(cp.uint8))


def updraft_helicity(patch):
    """2-5 km updraft helicity (m2/s2, WRF's definition: the integral of w * zeta, >= 0), (ny, nx)."""
    import cupy as cp
    env, st = patch.env, patch.state
    zh, dx = env.z_h, env.grid.dx
    k2 = int(np.searchsorted(zh, 2000.0))
    k5 = int(np.searchsorted(zh, 5000.0))
    dz = cp.asarray(np.diff(env.z_w)[k2:k5], dtype=cp.float32)[:, None, None]
    u = st.u[k2:k5]
    v = st.v[k2:k5]
    # vorticity at cell centres from the C-grid winds
    um = 0.5 * (u[:, :, 1:] + u[:, :, :-1])
    vm = 0.5 * (v[:, 1:, :] + v[:, :-1, :])
    zeta = getattr(env, "cyc", 1.0) * (cp.gradient(vm, dx, axis=2) - cp.gradient(um, dx, axis=1))   # cyclonic +
    wm = 0.5 * (st.w[k2:k5] + st.w[k2 + 1:k5 + 1])
    return cp.maximum((wm * zeta * dz).sum(axis=0), 0.0)


def low_wind(patch):
    """Lowest-level ground-relative wind speed (m/s) and the cell-centre grid-relative winds."""
    import cupy as cp
    st = patch.state
    u0 = 0.5 * (st.u[0, :, 1:] + st.u[0, :, :-1])
    v0 = 0.5 * (st.v[0, 1:, :] + st.v[0, :-1, :])
    mx, my = patch.env.motion
    return cp.sqrt((u0 + mx) ** 2 + (v0 + my) ** 2), u0, v0


def swath_codes(patch):
    """One patch's part of the run-max swaths now: ({"uhtrack": codes, "windtrack": codes}, the
    2-5 km UH itself, on the GPU)."""
    uh = updraft_helicity(patch)
    return {"uhtrack": _encode(uh, "uhtrack"), "windtrack": _encode(low_wind(patch)[0], "windtrack")}, uh


def patch_products(patch, fields=None) -> dict:
    """{name: uint8 (ny, nx)} for one patch.  ``fields`` (radar.PatchRadarFields)
    supplies the ArWen reflectivity if it is already computed."""
    import cupy as cp
    from .radar import PatchRadarFields

    env, st = patch.env, patch.state
    if fields is None or fields.patch_id != patch.id or fields.nx != patch.rect.nx:
        fields = PatchRadarFields(patch)
    dbz = fields.dbz
    out = {}
    out["cref"] = _encode(dbz.max(axis=0), "cref", dbz.max(axis=0) > -30)
    zh = env.z_h
    k1 = int(np.clip(np.searchsorted(zh, 1000.0) - 1, 0, len(zh) - 2))
    t1 = float((1000.0 - zh[k1]) / (zh[k1 + 1] - zh[k1]))
    d1 = dbz[k1] * (1 - t1) + dbz[k1 + 1] * t1
    out["ref1km"] = _encode(d1, "ref1km", d1 > -30)
    out["reflow"] = _encode(dbz[0], "reflow", dbz[0] > -30)
    out["thp"] = _encode(st.thp[0], "thp")
    out["wmax"] = _encode(cp.maximum(st.w.max(axis=0), 0.0), "wmax")
    out["uh25"] = _encode(updraft_helicity(patch), "uh25")
    dx = env.grid.dx
    wspd, u0, v0 = low_wind(patch)
    out["wspd"] = _encode(wspd, "wspd")
    z0 = cp.gradient(v0, dx, axis=1) - cp.gradient(u0, dx, axis=0)
    out["vort"] = _encode(z0, "vort")
    rain = st._scratch.get("mp_rainnc") if hasattr(st, "_scratch") else None
    if rain is not None and rain.shape == (patch.rect.ny, patch.rect.nx):
        out["rain"] = _encode(cp.log2(1.0 + rain), "rain", rain > 0.05)
    ocean = getattr(env, "ocean", None)
    if ocean is not None:
        from . import hurricane
        out["ir"] = _encode(hurricane.ir_brightness(patch), "ir")
        sst, water = ocean.maps(patch, env.clock)
        out["sst"] = _encode(sst - 273.15, "sst", water)
        out["slp"] = _encode(hurricane.surface_pressure(env, st) / 100.0, "slp")
        out["wind10"] = _encode(hurricane.ten_metre_wind(env, st), "wind10")
    return out


def pack_frame_product(t: float, name: str, patches: list) -> bytes:
    """Binary payload: u32 header length, JSON header, then each patch's codes.
    ``patches``: list of (rect_json, uint8 array)."""
    import json
    header = {"t": t, "product": name, "codec": CODECS[name][:3], "patches": []}
    blobs = []
    off = 0
    for meta, arr in patches:
        m = dict(meta)
        m["offset"] = off
        header["patches"].append(m)
        b = np.ascontiguousarray(arr).tobytes()
        blobs.append(b)
        off += len(b)
    hj = json.dumps(header).encode()
    pad = (-(4 + len(hj))) % 4
    return len(hj + b" " * pad).to_bytes(4, "little") + hj + b" " * pad + b"".join(blobs)


def compress(b: bytes) -> bytes:
    return zlib.compress(b, 1)
