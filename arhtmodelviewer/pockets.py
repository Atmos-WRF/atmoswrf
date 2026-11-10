"""Sounding pockets: more soundings placed on the map, each eased into the run's own over its radius -- a pocket
of stronger shear, weaker lapse rates, a moister boundary layer, wherever it is put.

Each pocket's sounding is the difference from the run's sounding, level by level on the model's half levels:
theta (added), vapour (a ratio) and the wind (added).  At a point the pockets are weighed by distance -- all of it
within INNER x the radius, easing out to nothing at the radius -- and where pockets overlap they share the point
(their weights normalised), the run's sounding taking whatever is left.  The air-mass machinery carries it from
there (boundaries.AirMasses, environment.Environment): new patches and their lateral boundaries start from the
blended columns, and the patches are held to them (forcing.relax_target3d, apply_wind_hold) gently enough that
storms keep their own air."""
from __future__ import annotations

import numpy as np

INNER = 0.35          # x the radius: the pocket's own sounding, unblended
KEEP_WIND = 0.5       # the pocket's winds held at this share of the air-mass keep rate (Engine airmass_keep_tau)
KEEP_THERMO = 0.25    # ... and its theta / vapour (storms keep their own heating and outflow)
KEEP_DEPTH = 6000.0   # m: held up to about this height (forcing's taper ends ~1.3 km above it)
# the pocket's own air only in the troposphere: its theta and vapour differences fade out over THERMO_FADE (m AGL),
# its winds over WIND_FADE.  Two soundings' tropopauses and stratospheres differ by kilometres and 10-20 K (CBN
# 27 April 2011 vs OUN 3 May 1999: +12 K at 11 km, +17 K at 14 km); blended over a pocket's edge that was a
# front aloft no atmosphere has, held there, and it drove rings of updrafts round the pocket (test run 2026-11-10)
THERMO_FADE = (9000.0, 12000.0)
WIND_FADE = (12000.0, 15000.0)
WARN_DTH = 2.0        # K below 8 km: a contrast that will act as a front across the pocket's edge (noted in the Log)


def _smoothstep(a, b, x, xp=np):
    t = xp.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class SoundingPockets:
    """``items``: [{"x", "y" (world m), "radius" (m), "sounding": snd.Sounding (grid-relative winds)}]."""

    def __init__(self, items: list, env):
        self.items = []
        z = np.asarray(env.z_h, dtype=np.float64)
        ztop = env.grid.ztop + 5000.0
        th0, q0 = env.theta_f(z), np.maximum(env.qv_f(z), 1e-7)
        u0, v0 = np.asarray(env.u_ground, np.float64), np.asarray(env.v_ground, np.float64)
        ft = 1.0 - _smoothstep(*THERMO_FADE, z)
        fw = 1.0 - _smoothstep(*WIND_FADE, z)
        low = z < 8000.0
        for it in items:
            th, qv, uf, vf = it["sounding"].profile_functions(ztop=ztop)
            dth = (th(z) - th0) * ft
            self.items.append({
                "x": float(it["x"]), "y": float(it["y"]), "r": max(float(it["radius"]), 2000.0),
                "title": getattr(it["sounding"], "title", "") or "",
                "dth": dth,
                "qr": 1.0 + (np.clip(np.maximum(qv(z), 1e-7) / q0, 0.05, 6.0) - 1.0) * ft,
                "du": (uf(z) - u0) * fw, "dv": (vf(z) - v0) * fw,
                "dth_max": float(np.max(np.abs(dth[low]))) if low.any() else 0.0,
            })
        self._dev = {}

    def __bool__(self):
        return bool(self.items)

    def weights(self, X, Y, xp=np):
        """(K, ...) each pocket's share of ground points (X, Y); their sum is at most 1."""
        X = xp.asarray(X, dtype=xp.float64)
        Y = xp.asarray(Y, dtype=xp.float64)
        w = xp.stack([1.0 - _smoothstep(INNER * p["r"], p["r"], xp.hypot(X - p["x"], Y - p["y"]), xp)
                      for p in self.items])
        return w / xp.maximum(w.sum(axis=0), 1.0)[None]

    def cover(self, X, Y, xp=np):
        """0..1: how much of the pockets' air is at (X, Y)."""
        return self.weights(X, Y, xp).sum(axis=0)

    def _profiles(self, xp):
        key = xp.__name__
        if key not in self._dev:
            self._dev[key] = {n: xp.asarray(np.stack([p[n] for p in self.items])) for n in ("dth", "qr", "du", "dv")}
        return self._dev[key]

    def fields(self, X, Y, xp=np, levels: slice | None = None):
        """theta' to add, vapour factor, du, dv (nz, ...) at ground points (X, Y) (``levels``: of the half levels)."""
        w = self.weights(X, Y, xp)                               # (K, ...)
        P = self._profiles(xp)
        sl = levels if levels is not None else slice(None)
        out = []
        for n in ("dth", "qr", "du", "dv"):
            prof = P[n][:, sl]                                    # (K, nz)
            f = xp.tensordot(prof, w, axes=([0], [0]))            # (nz, ...)
            if n == "qr":
                f = f + (1.0 - w.sum(axis=0))[None]               # the run's own vapour where no pocket is
            out.append(f)
        return tuple(out)

    def to_json(self) -> list:
        return [{"x": p["x"], "y": p["y"], "radius": p["r"], "title": p["title"]} for p in self.items]


def from_settings(specs: list, env, load, turn=None) -> SoundingPockets | None:
    """The run's ``sounding_pockets`` settings ([{"x_km", "y_km", "radius_km", "sounding": a sounding spec}]) as
    pockets; ``load`` reads a spec (sim.load_sounding), ``turn`` turns a sounding's winds onto the grid."""
    items = []
    for d in specs or []:
        spec = d.get("sounding") or {}
        if not spec:
            continue
        s = load(spec)
        if turn is not None:
            s = turn(s)
        items.append({"x": float(d.get("x_km", 0.0)) * 1000.0, "y": float(d.get("y_km", 0.0)) * 1000.0,
                      "radius": float(d.get("radius_km", 60.0)) * 1000.0, "sounding": s})
    return SoundingPockets(items, env) if items else None


def keep_maps(pockets: SoundingPockets, X, Y) -> tuple:
    """The pockets' keep on coarse columns: (thermo weight, wind weight, depth m) for forcing.relax_target3d."""
    c = pockets.cover(X, Y)
    return KEEP_THERMO * c, KEEP_WIND * c, np.where(c > 0.0, KEEP_DEPTH, 100.0)

