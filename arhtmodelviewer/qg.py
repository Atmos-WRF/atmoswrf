"""The large-scale flow of the synoptic mode: a two-layer quasi-geostrophic model of the
mid-latitude westerlies (Phillips 1956; Vallis 2017 ch. 9), pseudo-spectral and doubly periodic.

Layer 1 is the upper troposphere (~250 hPa, 10 km), layer 2 the lower (~750 hPa, 2.5 km).
Radiation relaxes the temperature (the layers' thickness, psi1 - psi2) toward a jet-stream
profile; the jet is baroclinically unstable, so troughs, ridges and surface cyclones grow out of
it, deepen, occlude and decay on their own, 4000-6000 km apart, moving east at 8-15 m/s.
Friction drags the lower layer.  Beyond the jet a sponge holds the flow to its radiative state
(the doubly periodic domain needs easterlies somewhere; they live there, where nothing grows).

Everything is in metres, seconds and m^2/s (streamfunction); x east, y north.
"""
from __future__ import annotations

import math

import numpy as np

F0 = 1.0e-4                 # Coriolis parameter (s^-1), ~43 N
BETA = 1.6e-11              # its northward gradient (1/m/s)
LD = 550e3                  # deformation radius 1/k_d (m): the most unstable waves ~5000 km long
F = 1.0 / (2.0 * LD * LD)   # layer coupling (equal depths: k_d^2 = 2F)
P1, P2 = 25000.0, 75000.0   # layer pressures (Pa)
Z1, Z2 = 10000.0, 2500.0    # their heights (m)
RD = 287.0
G = 9.81
SURFACE_A = 0.1             # surface_psi: psi2 - SURFACE_A (psi1 - psi2)
SURFACE_ZONAL_DAMP = 0.7    # ... less this much of its zonal mean


class QGModel:
    """Two-layer QG on an NX x NY doubly periodic grid of spacing dx (m)."""

    # du_jet 48 m/s, 8-day radiation, 3-day friction: zonal-mean jet ~30 m/s at 250 hPa with streaks to
    # ~80 m/s, surface cyclones -16 hPa on average and -28 at their deepest (tested 2026-10-26: 40 / 10 / 4
    # gave lows of -10 hPa; 55 / 6 / 2.5 broke the time step).  dt 900 s keeps RK4 well inside its
    # advective limit for an 80 m/s streak on the 75 km grid.
    def __init__(self, nx: int = 128, ny: int = 112, dx: float = 75e3, du_jet: float = 48.0,
                 jet_width: float = 900e3, tau_rad: float = 8 * 86400.0, tau_drag: float = 3 * 86400.0,
                 dt: float = 900.0):
        self.nx, self.ny, self.dx = nx, ny, dx
        self.Lx, self.Ly = nx * dx, ny * dx
        self.dt = dt
        self.tau_rad, self.r = tau_rad, 1.0 / tau_drag
        kx = 2 * np.pi * np.fft.rfftfreq(nx, dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, dx)
        self.kx, self.ky = np.meshgrid(kx, ky)                   # (ny, nx//2+1)
        self.K2 = self.kx ** 2 + self.ky ** 2
        self.ikx, self.iky = 1j * self.kx, 1j * self.ky
        inv = np.zeros_like(self.K2)
        inv[self.K2 > 0] = 1.0 / self.K2[self.K2 > 0]
        self.invK2 = inv
        self.invK2F = 1.0 / (self.K2 + 2.0 * F)
        # exponential cut-off filter (pyqg): the smallest scales are dissipated every step
        ks = np.sqrt((self.kx * dx) ** 2 + (self.ky * dx) ** 2)
        kc = 0.65 * np.pi
        self.filt = np.where(ks > kc, np.exp(-23.6 * (ks - kc) ** 4), 1.0)
        # the jet: radiative-equilibrium thickness psi_c = (psi1 - psi2) / 2, the jet in the middle of the
        # domain; the easterly shear that balances it (a periodic psi) lives in the sponge at the edges
        y = (np.arange(ny) + 0.5) * dx
        self.y = y
        yj = 0.5 * self.Ly
        jet = du_jet / np.cosh((y - yj) / jet_width) ** 2
        far = np.minimum(np.abs(y - yj), self.Ly - np.abs(y - yj))     # distance from the jet axis
        back = np.exp(-((self.Ly / 2 - far) / 900e3) ** 2)            # a bump far from it
        back *= jet.sum() / back.sum()
        dU = jet - back                                                 # U1 - U2 (m/s), zero mean
        # psi_c = -(1/2) int dU dy, periodic (zero-mean dU)
        pc = -0.5 * np.concatenate([[0.0], np.cumsum(0.5 * (dU[1:] + dU[:-1]) * dx)])
        pc -= pc.mean()
        self.psic_eq_y = pc
        self.psic_ref = float(np.interp(yj, y, pc))                     # at the jet axis
        self.psic_eq = np.broadcast_to(pc[:, None], (ny, nx)).copy()
        self.psic_eq_h = np.fft.rfft2(self.psic_eq)
        # the sponge: eddies damped (and the flow relaxed to the radiative state) over a day
        # where |y - yj| > 2800 km, nothing within it
        sp = np.clip((far - 2800e3) / 800e3, 0.0, 1.0)
        self.sponge = (sp * sp * (3 - 2 * sp) / 86400.0)[:, None]
        self.t = 0.0
        self.q1h = None
        self.q2h = None

    # ---------------------------------------------------------------- state
    def init_rest(self, seed: int = 1, noise: float = 1e6) -> None:
        """The radiative-equilibrium jet plus small random waves (longer than ~1500 km)."""
        r = np.random.default_rng(seed)

        def waves():
            h = np.fft.rfft2(r.standard_normal((self.ny, self.nx)))
            h *= np.sqrt(self.K2) < 2 * np.pi / 1500e3
            a = np.fft.irfft2(h, (self.ny, self.nx))
            return noise * a / max(a.std(), 1e-12)

        pc = self.psic_eq + waves()
        pb = waves()
        self.set_psi(pb + pc, pb - pc)

    def set_psi(self, psi1, psi2) -> None:
        p1h, p2h = np.fft.rfft2(psi1), np.fft.rfft2(psi2)
        self.q1h = -self.K2 * p1h + F * (p2h - p1h)
        self.q2h = -self.K2 * p2h + F * (p1h - p2h)

    def invert(self, q1h, q2h):
        qb, qc = 0.5 * (q1h + q2h), 0.5 * (q1h - q2h)
        pb = -qb * self.invK2
        pc = -qc * self.invK2F
        return pb + pc, pb - pc

    def copy_state(self):
        return self.t, self.q1h.copy(), self.q2h.copy()

    def set_state(self, st) -> None:
        self.t, self.q1h, self.q2h = st[0], st[1].copy(), st[2].copy()

    def translate(self, dx: float, dy: float = 0.0) -> None:
        """Move the whole flow by (dx, dy) m: exact for a periodic spectral field."""
        ph = np.exp(-1j * (self.kx * dx + self.ky * dy))
        self.q1h = self.q1h * ph
        self.q2h = self.q2h * ph

    # ---------------------------------------------------------------- dynamics
    def _rhs(self, q1h, q2h):
        p1h, p2h = self.invert(q1h, q2h)
        irf = np.fft.irfft2
        s = (self.ny, self.nx)
        out = []
        for ph, qh in ((p1h, q1h), (p2h, q2h)):
            u = irf(-self.iky * ph, s)
            v = irf(self.ikx * ph, s)
            qx = irf(self.ikx * qh, s)
            qy = irf(self.iky * qh, s)
            adv = np.fft.rfft2(u * qx + v * qy)
            out.append(-adv - BETA * self.ikx * ph)
        d1, d2 = out
        # radiation: the thickness relaxes to the jet's
        excess = (p1h - p2h) - 2.0 * self.psic_eq_h
        d1 = d1 + F / self.tau_rad * excess
        d2 = d2 - F / self.tau_rad * excess
        # friction on the lower layer: d(zeta2)/dt = -r zeta2
        d2 = d2 + self.r * self.K2 * p2h
        return d1, d2

    def step(self, n: int = 1) -> None:
        dt = self.dt
        for _ in range(n):
            q1, q2 = self.q1h, self.q2h
            a1, a2 = self._rhs(q1, q2)
            b1, b2 = self._rhs(q1 + 0.5 * dt * a1, q2 + 0.5 * dt * a2)
            c1, c2 = self._rhs(q1 + 0.5 * dt * b1, q2 + 0.5 * dt * b2)
            e1, e2 = self._rhs(q1 + dt * c1, q2 + dt * c2)
            q1 = q1 + dt / 6.0 * (a1 + 2 * b1 + 2 * c1 + e1)
            q2 = q2 + dt / 6.0 * (a2 + 2 * b2 + 2 * c2 + e2)
            # sponge: toward the radiative jet (barotropic flow at rest), implicit
            p1h, p2h = self.invert(q1, q2)
            pe1h, pe2h = self.psic_eq_h, -self.psic_eq_h
            qe1 = -self.K2 * pe1h + F * (pe2h - pe1h)
            qe2 = -self.K2 * pe2h + F * (pe1h - pe2h)
            irf, s = np.fft.irfft2, (self.ny, self.nx)
            w = 1.0 - 1.0 / (1.0 + dt * self.sponge)
            if w.max() > 0:
                d1 = irf(q1 - qe1, s) * w
                d2 = irf(q2 - qe2, s) * w
                q1 = q1 - np.fft.rfft2(d1)
                q2 = q2 - np.fft.rfft2(d2)
            self.q1h = q1 * self.filt
            self.q2h = q2 * self.filt
            self.t += dt

    def heat(self, q_k: np.ndarray, dt: float) -> None:
        """Diabatic heating of the mid-troposphere, ``q_k`` (K/s, ny x nx; latent heat of the rain) over dt:
        the layers' thickness grows by R ln(p2/p1) / f0 per kelvin -- the lower-layer PV falls and the upper
        rises, so heating under a trough's ascent deepens the low below it and builds the ridge above."""
        d = np.fft.rfft2(q_k * (RD * math.log(P2 / P1) / F0)) * dt      # d(psi1 - psi2)
        self.q1h = self.q1h - F * d
        self.q2h = self.q2h + F * d

    # ---------------------------------------------------------------- fields
    def fields(self) -> dict:
        """Physical fields (ny, nx): psi1, psi2, u1, v1, u2, v2 and the mid-level temperature
        anomaly tmid (K, relative to the jet axis) from the thickness."""
        p1h, p2h = self.invert(self.q1h, self.q2h)
        irf, s = np.fft.irfft2, (self.ny, self.nx)
        out = {"psi1": irf(p1h, s), "psi2": irf(p2h, s),
               "u1": irf(-self.iky * p1h, s), "v1": irf(self.ikx * p1h, s),
               "u2": irf(-self.iky * p2h, s), "v2": irf(self.ikx * p2h, s)}
        out["tmid"] = F0 * (out["psi1"] - out["psi2"] - 2.0 * self.psic_ref) / (RD * math.log(P2 / P1))
        return out

    def surface_psi(self, f: dict | None = None):
        """Streamfunction at the ground: the lower layer's, less SURFACE_A of the thermal wind above it.
        (Linear extrapolation from 2.5 and 10 km, a = 1/3, took a third of the jet stream's shear off
        the ground and left ~2 m/s in warm sectors under a 60 m/s jet; most of that shear is the upper
        troposphere's, not the boundary layer's.)"""
        f = f or self.fields()
        ps = (1.0 + SURFACE_A) * f["psi2"] - SURFACE_A * f["psi1"]
        # friction holds the mean surface westerlies to ~30 % of the lower layer's (~4 m/s, not 13), so the
        # cyclones are closed lows where they are, not troughs in a steep north-south pressure slope
        return ps - SURFACE_ZONAL_DAMP * ps.mean(axis=1, keepdims=True)


def find_lows(anom: np.ndarray, dx: float, radius: float = 700e3, min_depth: float = 5e6):
    """Cyclones: local minima (within ``radius``) of a periodic surface-streamfunction anomaly that are at
    least ``min_depth`` (m^2/s; 1e7 is ~12 hPa at rho f0 = 1.2e-4) below zero: [(i, j, depth), ...].
    (Measured against the zonal mean, not the surroundings: these cyclones are ~3000 km across, so a
    1000 km neighbourhood is nearly as low as their centre.)"""
    n = max(1, int(round(radius / dx)))
    m = anom.copy()
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            if di * di + dj * dj <= n * n:
                m = np.minimum(m, np.roll(np.roll(anom, di, axis=1), dj, axis=0))
    return [(int(i), int(j), float(-anom[j, i])) for j, i in np.argwhere((anom <= m + 1e-6) & (anom <= -min_depth))]
