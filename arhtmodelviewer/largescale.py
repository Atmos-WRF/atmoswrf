"""Large-scale ascent during a run: the slow synoptic lift ahead of a trough (QG ascent, a few cm/s peaking in
the mid troposphere) that keeps destabilizing a cool-season warm sector for hours -- why 1000-1500 J/kg days
with strong forcing keep their storms going.  A single-sounding run held its sounding fixed (steady
environment), so a storm lived only on the instability it started with.

Cloud-resolving models' large-scale forcing (e.g. the GATE / ARM case set-ups): the environment's theta and
qv change at -w_ls(z) d(theta)/dz and -w_ls(z) dqv/dz, from its own (lifted) profile, horizontally uniform:
the mid levels cool (the stable profile lifted), the moisture deepens, caps wear away from above the
boundary layer.  w_ls(z) = W sin(pi z / H) below H.

Everything that holds a run to its environment follows the lifted one: every patch and nest gets the same
tendency each step, the steady-environment reference column too (its target profile with it, so the drift
correction keeps the lift), the rim and air-mass relaxation aim at it (forcing.env_targets), new patches start
in it and the patches' edge tables are renewed from it every REFRESH s (Environment.ls_dth / ls_dqv)."""
from __future__ import annotations

import numpy as np

REFRESH = 300.0          # s of model time between renewals of the patches' edge tables
# The lifted environment is kept below RH_MAX: a layer the ascent brought to saturation everywhere at once
# (a 1-1.4 km cloud deck at 101-104 % on a GFS sounding) condensed in every column together, and the uniform
# latent heating above a CIN-free boundary layer broke into rows of popcorn cells across whole patches.  The
# excess leaves as if it rained out (the deck a real ascent would make, not a sheet of storms).
RH_MAX = 0.99


class LargeScaleAscent:
    def __init__(self, env, w_cms: float, top_km: float = 10.0):
        self.env = env
        self.w_peak = float(w_cms) / 100.0
        self.top = float(top_km) * 1000.0
        zh = np.asarray(env.z_h, dtype=np.float64)
        self.w = np.where(zh < self.top, self.w_peak * np.sin(np.pi * np.clip(zh / self.top, 0, 1)), 0.0)
        nz = len(zh)
        env.ls_dth = np.zeros(nz)
        env.ls_dqv = np.zeros(nz)
        self._next_refresh = REFRESH
        self.lift_m = 0.0          # how far the peak level has been lifted (m)
        self._bal = self._balanced()

    def _balanced(self):
        """(mu', phi' (nz+1,)) of the environment column as lifted so far, hydrostatically balanced -- what
        Environment._lifted gives a new patch, so the patches it is added to keep the same pressure."""
        env = self.env
        qv0 = np.asarray(getattr(env, "qv_env_h", env.qv_f(env.z_h)), dtype=np.float64)
        mup, php = env.balance(np.asarray(env.ls_dth, dtype=np.float64)[:, None],
                               np.maximum(qv0 + env.ls_dqv, 1e-7)[:, None])
        return float(mup[0]), np.asarray(php[:, 0], dtype=np.float64)

    def tendencies(self):
        """(dtheta/dt, dqv/dt) per half level (K/s, kg/kg/s) from the environment as lifted so far."""
        env = self.env
        zh = np.asarray(env.z_h, dtype=np.float64)
        th = np.asarray(env.base.thb, dtype=np.float64) + env.ls_dth
        qv = np.asarray(getattr(env, "qv_env_h", env.qv_f(env.z_h)), dtype=np.float64) + env.ls_dqv
        return -self.w * np.gradient(th, zh), -self.w * np.gradient(qv, zh)

    def advance(self, dt: float):
        """One step: the environment's offsets moved on, and the step's increments returned (device): theta'
        and qv (nz, 1, 1), mu' (scalar) and phi' (nz+1, 1, 1) -- the column kept in hydrostatic balance.  Without
        the mu' / phi' part a patch's air drifted out of balance with the air of a patch grown or merged later
        (built balanced): the seam's pressure jump lifted an uncapped warm sector into rows of popcorn cells."""
        import cupy as cp
        env = self.env
        tth, tq = self.tendencies()
        dth, dq = tth * dt, tq * dt
        qv_env = np.asarray(getattr(env, "qv_env_h", env.qv_f(env.z_h)), dtype=np.float64)
        dq = np.maximum(dq, -(qv_env + env.ls_dqv))          # never below dry
        env.ls_dth = env.ls_dth + dth
        # never above RH_MAX: the excess vapour out (as rain), from the column as lifted now
        from .sounding import qv_sat
        pb = np.asarray(env.base.pb, dtype=np.float64)
        T = (np.asarray(env.base.thb, dtype=np.float64) + env.ls_dth) * (pb / 1.0e5) ** (287.0 / 1004.5)
        cap = RH_MAX * qv_sat(pb, T) - (qv_env + env.ls_dqv)
        dq = np.minimum(dq, cap)
        env.ls_dqv = env.ls_dqv + dq
        self.lift_m += self.w_peak * dt
        cache = env.__dict__.setdefault("_dev_cache", {})
        cache.pop("ls_thp", None)
        cache.pop("ls_qv", None)
        mup0, php0 = self._bal
        self._bal = self._balanced()
        return (cp.asarray(dth, dtype=cp.float32)[:, None, None], cp.asarray(dq, dtype=cp.float32)[:, None, None],
                float(self._bal[0] - mup0), cp.asarray(self._bal[1] - php0, dtype=cp.float32)[:, None, None])

    def reset_balance(self) -> None:
        """After the offsets were put back (a rollback): the balanced column from them."""
        self._bal = self._balanced()

    def due_refresh(self, t: float) -> bool:
        if t >= self._next_refresh:
            self._next_refresh = t + REFRESH
            self.env._bsnap = None           # the edge tables' column, from the lifted environment
            return True
        return False

    def describe(self) -> dict:
        k5 = int(np.argmin(np.abs(np.asarray(self.env.z_h) - 5000.0)))
        return {"w_cms": self.w_peak * 100.0, "top_km": self.top / 1000.0, "lift_m": round(self.lift_m),
                "dth_5km": round(float(self.env.ls_dth[k5]), 2)}
