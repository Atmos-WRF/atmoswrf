"""Hurricanes: a procedural sea-surface temperature map, the tropical sounding it supports, the ocean
surface (sea-state drag, evaporation, sensible heat, dissipative heating and a mixed-layer cold wake), a
balanced warm-core seed vortex (the map's "surface low"), the cyclone tracker (lowest pressure, maximum
1-min sustained wind, Saffir-Simpson category) and the simulated infrared satellite picture.

What OutbreakLab's hurricane mode taught (js/hurricane.js) carries over: a free troposphere anchored to the
observed hurricane-season mean (Jordan 1958) instead of a moist adiabat from the warm boundary layer (that was
5-7 K too warm aloft and capped storms at Category 1), a cold-point tropopause near -80 C at 90 hPa, a floor
on the upper-tropospheric stability, a Rankine-type seed vortex whose angular momentum never falls outward
(inertially stable), a cold wake as the intensity brake, and IR brightness from ice / snow / cloud water
absorption so the dense overcast of a strong storm reads black-grey-white.
"""
from __future__ import annotations

import math

import numpy as np

from .sounding import CP, G, KAPPA, P0

RD = 287.04
KT = 0.514444
LV = 2.5e6

# ------------------------------------------------------------------ the sea surface
# Late-summer zonal-mean SST (C) of the hurricane basins by latitude (either hemisphere): the deep tropics
# 28-29, the Gulf / Caribbean / subtropical warm pool 29-29.5 at 20-26 deg, then the sharp poleward fall
# (26-27 near 33 deg, 21-22 at 40, 12-13 at 50).
SST_LAT = np.array([0.0, 8.0, 15.0, 20.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0, 44.0, 50.0, 60.0, 70.0])
SST_VAL = np.array([28.0, 28.5, 28.9, 29.3, 29.4, 28.9, 28.0, 26.6, 24.6, 21.8, 17.6, 12.5, 6.0, 1.0])
# the anomaly octaves on top: (wavelength km, standard deviation K, plane waves): basin-scale warm and cool
# regions, warm pools and tongues, then the mesoscale eddies and rings of the ocean's weather
SST_OCTAVES = ((2600.0, 0.55, 10), (1100.0, 0.45, 14), (450.0, 0.35, 18), (180.0, 0.22, 22), (70.0, 0.10, 24))
R_EARTH = 6.371e6


class ProceduralSST:
    """A seeded sea-surface temperature map: the zonal mean (SST_LAT) plus random plane waves on the sphere
    (SST_OCTAVES), so the same seed is the same ocean everywhere and every run; ``offset`` (C) shifts it all."""

    def __init__(self, seed: int = 1, offset: float = 0.0):
        self.seed, self.offset = int(seed), float(offset)
        rng = np.random.default_rng(self.seed)
        k, a, ph = [], [], []
        for lam, sd, n in SST_OCTAVES:
            d = rng.normal(size=(n, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            kk = 2.0 * math.pi / (lam * 1000.0) * rng.uniform(0.8, 1.25, n)
            k.append(d * kk[:, None])
            a.append(np.full(n, sd * math.sqrt(2.0 / n)))
            ph.append(rng.uniform(0.0, 2.0 * math.pi, n))
        self._k = np.concatenate(k)
        self._a = np.concatenate(a)
        self._ph = np.concatenate(ph)

    def celsius(self, lat, lon) -> np.ndarray:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        base = np.interp(np.abs(lat), SST_LAT, SST_VAL)
        cl, sl = np.cos(np.radians(lat)), np.sin(np.radians(lat))
        P = np.stack([R_EARTH * cl * np.cos(np.radians(lon)), R_EARTH * cl * np.sin(np.radians(lon)),
                      R_EARTH * sl], axis=-1)
        anom = np.zeros(lat.shape)
        for i in range(0, len(self._a), 16):              # in chunks: no (points x waves) array
            ph = P @ self._k[i:i + 16].T + self._ph[i:i + 16]
            anom += np.cos(ph) @ self._a[i:i + 16]
        # the anomalies are weaker where the water is cool (and the open ocean's poleward gradient rules)
        return base + anom * np.clip((base - 10.0) / 15.0, 0.3, 1.0) + self.offset


_LAND: dict = {}


def land_mask():
    """is_land(lat, lon) -> bool array from the global Natural Earth mask (data/landmask, 0.05 deg), or None
    without it."""
    if "fn" not in _LAND:
        import io
        import lzma
        from . import PROJECT
        path = PROJECT / "data" / "landmask" / "ne_land_0p05.npz.xz"
        fn = None
        if path.is_file():
            z = np.load(io.BytesIO(lzma.decompress(path.read_bytes())))
            shape = tuple(int(v) for v in z["shape"])
            mask = np.unpackbits(z["bits"], axis=1)[:, :shape[1]].astype(bool)
            res = float(z["res"])

            def fn(lat, lon):
                lat = np.asarray(lat, dtype=np.float64)
                lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0
                j = np.clip(((90.0 - lat) / res).astype(np.int64), 0, shape[0] - 1)
                i = np.clip((lon / res).astype(np.int64), 0, shape[1] - 1)
                return mask[j, i]
        _LAND["fn"] = fn
    return _LAND["fn"]


# ------------------------------------------------------------------ the tropical sounding
# Jordan (1958) mean hurricane-season sounding (West Indies), C by hPa, with the cold-point tropopause of the
# tropics (-80 C near 90 hPa) above Jordan's 100 hPa mean (OutbreakLab: with a -76 C lid a Category 5's
# overcast never got colder than a Category 3's)
JORDAN = ((1000, 26.0), (950, 22.6), (900, 19.8), (850, 17.4), (800, 14.7), (750, 12.2), (700, 9.6), (600, 2.4),
          (500, -6.9), (400, -17.3), (300, -32.4), (250, -41.6), (200, -53.6), (175, -59.8), (150, -65.6),
          (125, -71.5), (100, -77.0), (90, -80.0), (70, -72.0), (50, -63.0))


def _jordan(p: float) -> float:
    if p >= JORDAN[0][0]:
        return JORDAN[0][1]
    for (pa, ta), (pb, tb) in zip(JORDAN[:-1], JORDAN[1:]):
        if pb <= p <= pa:
            return ta + (tb - ta) * math.log(pa / p) / math.log(pa / pb)
    return JORDAN[-1][1]


def _es(tc):
    """Saturation vapour pressure over water (hPa), Bolton (1980)."""
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def _ws(p, tc):
    e = _es(tc)
    return 0.622 * e / np.maximum(p - e, 1.0)


def _moist_t(p0: float, t0: float, p1: float) -> float:
    """Temperature (K) at p1 of a saturated parcel lifted pseudo-adiabatically from (p0 hPa, t0 K)."""
    t, p = t0, p0
    n = max(4, int(abs(p0 - p1) / 5.0))
    dp = (p1 - p0) / n
    for _ in range(n):
        w = float(_ws(p, t - 273.15))
        gam = (RD * t + LV * w) / (CP + LV * LV * w * 0.622 / (RD * t * t)) / p     # dT/dp (K/hPa)
        t += gam * dp
        p += dp
    return t


def _td_from_rh(tc: float, rh: float) -> float:
    e = max(float(_es(tc)) * max(rh, 1e-4), 1e-6)
    a = math.log(e / 6.112)
    return 243.5 * a / (17.67 - a)


def tropical_sounding(sst: float = 29.0, rh_mid: float = 70.0, steer_kt: float = 10.0, steer_dir: float = 280.0,
                      shear_kt: float = 5.0, shear_from: float = 270.0, trop_c: float = -80.0,
                      instab: float = 3.0, psfc: float = 1012.0) -> str:
    """A hurricane-season sounding over sea at ``sst`` (C), as SHARPpy text: a well-mixed 80-85 % RH boundary
    layer 1 K cooler than the sea to 960 hPa, then Jordan's mean free troposphere shifted by half the SST's
    departure from 28.5 C (and ``instab`` - 3 K: 3 is Jordan's own), moist (``rh_mid`` %) mid levels, at least
    2 -> 3.5 K/km of stability above 6 km, the tropopause at ``trop_c``.  Winds: the steering flow
    ``steer_kt`` toward ``steer_dir`` (deg), with ``shear_kt`` of 0-12 km shear from ``shear_from``."""
    pB, T0 = 960.0, sst - 1.0 + 273.15
    ps = [psfc] + [p for p in _levels() if p < psfc - 2.0]
    th0 = T0 * (1000.0 / psfc) ** KAPPA
    TB = th0 * (pB / 1000.0) ** KAPPA
    rm = min(max(rh_mid / 100.0, 0.1), 0.95)

    def rh_at(p):
        if p >= 950:
            return 0.80 + 0.05 * (psfc - p) / (psfc - 950.0)
        if p >= 850:
            return 0.85
        if p >= 700:
            return 0.85 + (rm - 0.85) * (850 - p) / 150.0
        if p >= 500:
            return rm
        # the outflow layer near ice saturation (60-75 % over ice at 300-150 hPa, Jordan's own upper levels
        # were too dry to hold a cirrus shield: it sublimated within ~100 km)
        up = max(0.45, rm - 0.1)
        if p >= 300:
            return rm + (up - rm) * (500 - p) / 200.0
        if p >= 200:
            return up + (0.45 - up) * (300 - p) / 100.0
        if p >= 130:
            return 0.45 - 0.1 * (200 - p) / 70.0
        return 0.1

    def smooth(a, b, x):
        t = min(max((x - a) / (b - a), 0.0), 1.0)
        return t * t * (3 - 2 * t)

    sd = math.radians(steer_dir)
    su, sv = steer_kt * KT * math.sin(sd), steer_kt * KT * math.cos(sd)
    hd = math.radians(shear_from)
    hu, hv = -shear_kt * KT * math.sin(hd), -shear_kt * KT * math.cos(hd)

    def wind_at(z):
        f = min(max((z - 1500.0) / 10500.0, 0.0), 1.0) - 0.5
        if z > 12000.0:
            f = 0.5 * (1.0 - smooth(12000.0, 17000.0, z))
        lo = 0.85 + 0.15 * smooth(0.0, 1500.0, z) if z < 1500.0 else 1.0
        return (su + hu * f) * lo, (sv + hv * f) * lo

    rows = []
    z, p_prev, tv_prev, th_prev, z_prev, tropo, z90 = 0.0, psfc, 0.0, 0.0, 0.0, False, None
    for i, p in enumerate(ps):
        if p >= pB:
            T = th0 * (p / 1000.0) ** KAPPA - 273.15
        else:
            tj = (_jordan(p) + 0.5 * (sst - 28.5) + (3.0 - instab)
                  + (trop_c + 80.0) * min(max(math.log(250.0 / p) / math.log(250.0 / 90.0), 0.0), 1.0))
            T = (_moist_t(pB, TB, p) - 273.15) * (1.0 - smooth(pB, 880.0, p)) + tj * smooth(pB, 880.0, p)
            z_est = z + RD / G * tv_prev * math.log(p_prev / p)
            need = 0.0
            if z_est > 6000.0:
                need = 2e-3 + (3.5e-3 - 2e-3) * smooth(6000.0, 12000.0, z_est)
            if need > 0.0:
                T = max(T, (th_prev + need * (z_est - z_prev)) * (p / 1000.0) ** KAPPA - 273.15)
        if T <= trop_c or tropo:
            tropo, T = True, trop_c
        if p <= 90.0:
            if z90 is None:
                z90 = z
            T = trop_c + 0.0022 * max(0.0, z - z90)
        td = _td_from_rh(T, rh_at(p))
        if p < 120.0:
            td = min(td, T - 25.0, -88.0)
        tv = (T + 273.15) * (1.0 + 0.61 * float(_ws(p, td)))
        if i > 0:
            z += RD / G * 0.5 * (tv + tv_prev) * math.log(p_prev / p)
        p_prev, tv_prev, z_prev = p, tv, z
        th_prev = (T + 273.15) * (1000.0 / p) ** KAPPA
        u, v = wind_at(z)
        spd = math.hypot(u, v) / KT
        d = 0.0 if spd < 0.05 else (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0
        rows.append((p, z, T, td, d, spd))
    f = lambda v, n: f"{v:10.{n}f}"
    body = "\n".join(",".join((f(r[0], 2), f(r[1], 0), f(r[2], 2), f(r[3], 2), f(r[4], 0), f(r[5], 1))) for r in rows)
    return ("%TITLE%\n TROP   000000/0000  SST " + f"{sst:.1f}C\n\n"
            "   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD\n"
            "-------------------------------------------------------------------\n%RAW%\n" + body + "\n%END%")


def _levels():
    out, p = [], 1000
    while p >= 50:
        out.append(float(p))
        p -= 10 if p > 850 else 25 if p > 200 else 10
    return out


# ------------------------------------------------------------------ the ocean surface
CK10 = 1.15e-3          # 10 m enthalpy exchange coefficient: ~constant with wind (CBLAST, Haus et al. 2010)
CD_CAP = 2.0e-3         # 10 m drag coefficient's ceiling: it levels off above ~30 m/s (Powell et al. 2003)
GUST = 1.0              # m/s of sub-grid gustiness under the fluxes
SALT = 0.98             # sea water's saturation vapour pressure against fresh water's
WAKE_RATE = 3.2e-10     # K/s per (m/s)^3 of 10 m wind, for a 50 m mixed layer (OutbreakLab's, x 50/h)
WAKE_EVERY = 60.0       # s of model time between cold-wake updates
WAKE_TILE = 64


def cd10(u10):
    """10 m drag coefficient over the sea (Large & Pond at light winds rising to Powell's ceiling): ~1.0e-3 at
    5 m/s, 1.5e-3 at 12, CD_CAP from about 19 m/s on."""
    import cupy as cp
    xp = cp.get_array_module(u10)
    return xp.clip((0.75 + 0.067 * u10) * 1.0e-3, 0.9e-3, CD_CAP)


class WakeStore:
    """The cold wake (K below the SST map) on ground-anchored tiles at the grid spacing: it stays where the
    storm stirred the mixed layer while the grid and the domain move on."""

    def __init__(self, dx: float):
        self.dx = float(dx)
        self.tiles: dict = {}

    def _keys(self, I, J):
        T = WAKE_TILE
        return np.unique(np.stack([(I // T).ravel(), (J // T).ravel()], 1), axis=0)

    def get(self, X, Y) -> np.ndarray:
        I = np.floor(X / self.dx).astype(np.int64)
        J = np.floor(Y / self.dx).astype(np.int64)
        out = np.zeros(I.shape, np.float32)
        T = WAKE_TILE
        for ti, tj in self._keys(I, J):
            tile = self.tiles.get((int(ti), int(tj)))
            if tile is None:
                continue
            m = (I // T == ti) & (J // T == tj)
            out[m] = tile[J[m] - tj * T, I[m] - ti * T]
        return out

    def add(self, X, Y, dT, cap: float) -> None:
        I = np.floor(X / self.dx).astype(np.int64)
        J = np.floor(Y / self.dx).astype(np.int64)
        T = WAKE_TILE
        for ti, tj in self._keys(I, J):
            m = (I // T == ti) & (J // T == tj) & (dT > 0)
            if not m.any():
                continue
            tile = self.tiles.setdefault((int(ti), int(tj)), np.zeros((T, T), np.float32))
            jj, ii = J[m] - tj * T, I[m] - ti * T
            np.maximum.at(tile, (jj, ii), np.minimum(tile[jj, ii] + dT[m], cap))


class Ocean:
    """The sea under a hurricane run: its SST map (ProceduralSST minus the cold wake), which columns are
    water (the land cover's land stays land, with its own friction and no fluxes), and the surface exchange
    each step (sea_surface)."""

    REFRESH = 60.0       # s between updates of a patch's SST / water maps (the grid moves)

    def __init__(self, env, sst: ProceduralSST, latlon, land_fn=None, wake: bool = True, mld: float = 50.0,
                 ref_sst_c: float | None = None):
        self.env, self.sst, self.latlon, self.land_fn = env, sst, latlon, land_fn
        self.wake = WakeStore(env.grid.dx) if wake else None
        self.mld = max(10.0, float(mld))
        self.wake_cap = 4.0 * (50.0 / self.mld) ** 0.5 + 1.0          # 5 K for 50 m, 3.8 K for 100 m, 9 K for 12 m
        self.ref_sst_k = 273.15 + float(ref_sst_c if ref_sst_c is not None else 29.0)
        self.ambient = 0.0             # share of the open sea's fuel the environment keeps (Engine._ambient_keep)
        self._next_wake = 0.0
        self.u10 = {}                                                  # patch id -> last 10 m wind (cupy)

    def sst_c(self, X, Y) -> np.ndarray:
        lat, lon = self.latlon(X, Y)
        s = self.sst.celsius(lat, lon)
        if self.wake is not None:
            s = s - self.wake.get(X, Y)
        return s

    def water(self, X, Y) -> np.ndarray:
        if self.land_fn is None:
            return np.ones(np.shape(X), bool)
        lat, lon = self.latlon(X, Y)
        return ~self.land_fn(lat, lon)

    def maps(self, p, t: float):
        """(SST K, water mask) of patch p's columns now, cached for REFRESH s (cupy)."""
        import cupy as cp
        key = (p.rect.i0, p.rect.j0, p.rect.nx, p.rect.ny)
        c = getattr(p, "_ocean", None)
        if c is None or c[0] != key or t - c[1] >= self.REFRESH or t < c[1]:
            X, Y = self.env.ground_xy(p.rect, t=t)
            p._ocean = (key, t, cp.asarray(self.sst_c(X, Y) + 273.15, dtype=cp.float32),
                        cp.asarray(self.water(X, Y)))
        return p._ocean[2], p._ocean[3]

    def stir(self, patches, t: float) -> None:
        """The cold wake: every WAKE_EVERY s, each water column cools by WAKE_RATE (50/h) U10^3 over the time
        since the last stir, up to wake_cap below the map (the mixed layer entrains cold thermocline water)."""
        if self.wake is None or t < self._next_wake:
            return
        span = WAKE_EVERY if self._next_wake == 0.0 else WAKE_EVERY + max(0.0, t - self._next_wake)
        self._next_wake = t + WAKE_EVERY
        import cupy as cp
        for p in patches:
            u10 = self.u10.get(p.id)
            if u10 is None or u10.shape != (p.rect.ny, p.rect.nx):
                continue
            _, water = self.maps(p, t)
            dT = cp.asnumpy(cp.where(water, WAKE_RATE * (50.0 / self.mld) * u10 ** 3 * span, 0.0))
            if float(dT.max()) < 1e-3:
                continue
            X, Y = self.env.ground_xy(p.rect, t=t)
            self.wake.add(X, Y, dT, self.wake_cap)
            p._ocean = None                                            # the SST under it changed


def sea_surface(env, st, dt: float, sst_k, water, cd_land):
    """One step of the surface over sea and land on state ``st``: bulk drag with the sea-state drag
    coefficient over water (cd10, scaled to the lowest level by the log law) and ``cd_land`` (the land
    cover's, or one number) over land, applied implicitly by the engine; over water, implicit bulk sensible
    and latent heat fluxes into the lowest layer toward the sea's theta and saturation humidity (Ck10 x the
    same log-law factor), and the frictional (dissipative) heating of the surface layer.  Returns
    (per-column drag coefficient of the lowest level, 10 m wind) on the GPU."""
    import cupy as cp
    mx, my = env.motion
    u, v = st.u[0], st.v[0]
    ug = 0.5 * (u[:, 1:] + u[:, :-1]) + mx
    vg = 0.5 * (v[1:, :] + v[:-1, :]) + my
    spd = cp.sqrt(ug * ug + vg * vg)
    z1 = float(env.z_h[0])
    dz1 = float(env.z_w[1] - env.z_w[0])
    # the 10 m wind and roughness, one fixed-point pass: z0 from the 10 m drag coefficient
    c10 = cd10(spd * 0.9)
    z0 = 10.0 * cp.exp(-0.4 / cp.sqrt(c10))
    f10 = cp.clip(cp.log(10.0 / z0) / cp.log(z1 / z0), 0.5, 1.3)
    u10 = spd * f10
    c10 = cd10(u10)
    z0 = 10.0 * cp.exp(-0.4 / cp.sqrt(c10))
    f10 = cp.clip(cp.log(10.0 / z0) / cp.log(z1 / z0), 0.5, 1.3)
    u10 = spd * f10
    cd_sea = c10 * f10 * f10
    ck = CK10 * f10 * f10
    thb0 = float(env.base.thb[0])
    p1 = st.p[0]
    th1 = thb0 + st.thp[0]
    q1 = st.qv[0]
    t1 = th1 * (p1 / P0) ** KAPPA
    psfc = p1 * cp.exp(G * z1 / (RD * t1 * (1.0 + 0.61 * q1)))
    ths = sst_k * (P0 / psfc) ** KAPPA
    es = 611.2 * cp.exp(17.67 * (sst_k - 273.15) / (sst_k - 29.65)) * SALT
    qs = 0.622 * es / (psfc - es)
    a = cp.where(water, dt * ck * cp.sqrt(spd * spd + GUST * GUST) / dz1, 0.0)
    pi1 = (p1 / P0) ** KAPPA
    heat = cp.where(water, dt * cd_sea * spd ** 3 / (CP * dz1) / pi1, 0.0)
    st.thp[0] = ((th1 + a * ths) / (1.0 + a) + heat - thb0).astype(st.thp.dtype)
    st.qv[0] = ((q1 + a * qs) / (1.0 + a)).astype(st.qv.dtype)
    cd = cp.where(water, cd_sea, cd_land).astype(cp.float32)
    return cd, u10


# ------------------------------------------------------------------ the seed vortex
def vortex_profile(r, vmax: float, rmw: float, r0: float, f: float):
    """Tangential wind (m/s) of the seed vortex at radii r (m): solid body inside the radius of maximum wind,
    a Rankine-type r^-0.6 tail beyond it tapered smoothly to nothing by r0, with its absolute angular
    momentum forced never to fall outward (an inertially stable vortex: OutbreakLab's x e^(1-x) profile was
    unstable beyond 2 RMW)."""
    r = np.asarray(r, dtype=np.float64)
    x = r / rmw
    v = np.where(x < 1.0, vmax * x, vmax * np.maximum(x, 1e-9) ** -0.6)
    taper = np.clip((r0 - r) / (0.6 * (r0 - rmw)), 0.0, 1.0)
    v = v * taper * taper * (3.0 - 2.0 * taper)
    M = np.maximum.accumulate(r * v + 0.5 * f * r * r)
    return np.where(r > 0, (M - 0.5 * f * r * r) / np.maximum(r, 1e-9), 0.0)


def vortex_tables(env, vmax: float, rmw: float, r0: float, depth: float = 11000.0, f: float | None = None):
    """Radial tables of the seed vortex on the model levels, in gradient-wind and hydrostatic balance:
    (r (nr,), tangential wind (nz, nr), theta' (nz, nr)).  The pressure perturbation follows from
    cp theta_v d(pi')/dr = v^2 / r + f v (integrated inward from 1.5 r0, where it is zero), the warm core from
    hydrostatic balance on it: theta_v' = cp theta_v^2 / g d(pi')/dz."""
    f = env.f if f is None else f
    zh = np.asarray(env.z_h, dtype=np.float64)
    r = np.linspace(0.0, 1.5 * r0, 600)
    vr = vortex_profile(r, vmax, rmw, r0, f)
    zfull = 1000.0
    F = np.where(zh <= zfull, 1.0, 0.5 * (1.0 + np.cos(np.pi * np.clip((zh - zfull) / (depth - zfull), 0.0, 1.0))))
    V = F[:, None] * vr[None, :]
    thb = np.asarray(env.base.thb, dtype=np.float64)
    qv0 = np.asarray(env.qv_f(zh), dtype=np.float64)
    thv = thb * (1.0 + 0.61 * qv0)
    dpdr = (V * V / np.maximum(r[None, :], 1.0) + f * V) / (CP * thv[:, None])
    dr = np.diff(r)
    pi = np.zeros_like(V)
    seg = 0.5 * (dpdr[:, 1:] + dpdr[:, :-1]) * dr[None, :]
    pi[:, :-1] = -np.cumsum(seg[:, ::-1], axis=1)[:, ::-1]
    dpdz = np.gradient(pi, zh, axis=0)
    th = CP * thv[:, None] ** 2 / G * dpdz / (1.0 + 0.61 * qv0[:, None])
    return r, V, th


def apply_vortex(patch, x: float, y: float, vmax: float, rmw: float, r0: float, rh: float = 0.9) -> float:
    """Add the seed vortex centred at ground (x, y) to a patch: its winds (cyclonic for the hemisphere of
    env.f), its warm core and the mass that balances it (the surface low), and a moist envelope (relative
    humidity raised to ``rh`` from 1 to 8 km inside 0.6 r0, as a pre-genesis disturbance has).  Returns the
    central pressure fall (hPa)."""
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import DTYPE
    env, st = patch.env, patch.state
    sign = 1.0 if env.f >= 0.0 else -1.0
    r, V, TH = vortex_tables(env, vmax, rmw, r0)
    nz = env.grid.nz
    rt = cp.asarray(r)

    def radial(table, Xg, Yg):
        R = cp.asarray(np.hypot(Xg - x, Yg - y))
        out = cp.empty((nz,) + R.shape, dtype=cp.float64)
        for k in range(nz):
            out[k] = cp.interp(R, rt, cp.asarray(table[k]))
        return out, R

    X, Y = env.ground_xy(patch.rect)
    th_old = st.thp.astype(cp.float64)
    q_old = st.qv.astype(cp.float64)
    dth, R = radial(TH, X, Y)
    th_new = th_old + dth
    # the moist envelope
    zh = cp.asarray(np.asarray(env.z_h))[:, None, None]
    thb = cp.asarray(np.asarray(env.base.thb, dtype=np.float64))[:, None, None]
    T = (thb + th_new) * (st.p.astype(cp.float64) / P0) ** KAPPA
    es = 611.2 * cp.exp(17.67 * (T - 273.15) / (T - 29.65))
    qsat = 0.622 * es / (st.p - es)
    wz = cp.clip((zh - 500.0) / 500.0, 0.0, 1.0) * cp.clip((9000.0 - zh) / 1000.0, 0.0, 1.0)
    wr = cp.clip((0.6 * r0 - R) / (0.3 * r0), 0.0, 1.0)[None]
    q_new = cp.maximum(q_old, q_old + wz * wr * (rh * qsat - q_old))
    m0, f0 = env.balance(th_old, q_old)
    m1, f1 = env.balance(th_new, q_new)
    st.thp[...] = th_new.astype(DTYPE)
    st.qv[...] = q_new.astype(DTYPE)
    st.mup += (m1 - m0).astype(DTYPE)
    st.php += (f1 - f0).astype(DTYPE)
    for name, sx, sy in (("u", 1, 0), ("v", 0, 1)):
        Xf, Yf = env.ground_xy(patch.rect, sx, sy)
        vt, Rf = radial(V, Xf, Yf)
        Rf = cp.maximum(Rf, 1.0)
        comp = -(cp.asarray(Yf) - y) / Rf if name == "u" else (cp.asarray(Xf) - x) / Rf
        getattr(st, name)[...] += (sign * vt * comp[None]).astype(DTYPE)
    update_diagnostics(st)
    mu_drop = float(cp.asnumpy((m1 - m0).min()))
    return -mu_drop / 100.0


# ------------------------------------------------------------------ diagnostics
SS_KT = ((137, 5), (113, 4), (96, 3), (83, 2), (64, 1))


def category(vmax_kt: float) -> tuple[str, str]:
    """(short code, name) on the Saffir-Simpson scale from the maximum 1-min sustained wind (kt)."""
    for lim, c in SS_KT:
        if vmax_kt >= lim:
            return f"C{c}", f"Category {c} hurricane"
    if vmax_kt >= 34:
        return "TS", "Tropical storm"
    return "TD", "Tropical depression"


def surface_pressure(env, st):
    """Surface (= sea-level, the ground is at 0 m) pressure (Pa) of each column, hydrostatically from the
    lowest model level."""
    import cupy as cp
    thb0 = float(env.base.thb[0])
    z1 = float(env.z_h[0])
    p1 = st.p[0]
    t1 = (thb0 + st.thp[0]) * (p1 / P0) ** KAPPA
    return p1 * cp.exp(G * z1 / (RD * t1 * (1.0 + 0.61 * st.qv[0])))


def ten_metre_wind(env, st):
    """10 m wind speed (m/s) from the lowest level by the sea's log law (the 1-min sustained wind: the model's
    resolved wind at a 2-3 km grid is about a 1-min mean)."""
    import cupy as cp
    mx, my = env.motion
    ug = 0.5 * (st.u[0, :, 1:] + st.u[0, :, :-1]) + mx
    vg = 0.5 * (st.v[0, 1:, :] + st.v[0, :-1, :]) + my
    spd = cp.sqrt(ug * ug + vg * vg)
    z1 = float(env.z_h[0])
    u10 = spd * 0.9
    for _ in range(2):                                   # as sea_surface: z0 from the 10 m drag, twice
        z0 = 10.0 * cp.exp(-0.4 / cp.sqrt(cd10(u10)))
        u10 = spd * cp.clip(cp.log(10.0 / z0) / cp.log(z1 / z0), 0.5, 1.3)
    return u10


# infrared absorption (m2/kg) of each hydrometeor in the 10.3 um window (OutbreakLab's, tuned to SatPainter
# pictures): thin cirrus already reads cold, a Category 5's canopy black-grey-white
IR_KAPPA = {"qc": 90.0, "qi": 120.0, "qs": 35.0, "qg": 5.0, "qr": 1.0, "qh": 3.0}
IR_VAPOUR = 0.004


def ir_brightness(patch):
    """Simulated window-channel infrared brightness temperature (C) of each column: emission from every layer
    by its hydrometeor (and a little vapour) optical depth, seen from above, over the sea surface (or the
    lowest air over land)."""
    import cupy as cp
    env, st = patch.env, patch.state
    thb = cp.asarray(np.asarray(env.base.thb, dtype=np.float32))[:, None, None]
    T = (thb + st.thp) * (st.p / P0) ** KAPPA
    rho = 1.0 / st.alt
    dz = cp.asarray(np.diff(np.asarray(env.z_w)).astype(np.float32))[:, None, None]
    kq = IR_VAPOUR * st.qv
    for name, k in IR_KAPPA.items():
        q = getattr(st, name, None)
        if q is not None:
            kq = kq + k * cp.maximum(q, 0.0)
    tau = kq * rho * dz                                    # (nz, ny, nx), bottom to top
    above = cp.cumsum(tau[::-1], axis=0)[::-1] - tau       # optical depth above each layer
    w = cp.exp(-above) * (1.0 - cp.exp(-tau))
    tb = (w * T).sum(axis=0)
    total = above[0] + tau[0]
    ocean = getattr(env, "ocean", None)
    if ocean is not None:
        sst, water = ocean.maps(patch, env.clock)
        ts = cp.where(water, sst, T[0] + 2.0)
    else:
        ts = T[0] + 1.0
    tb = tb + cp.exp(-total) * ts
    return tb - 273.15


def cyclone_fix(env, patches, x: float, y: float, search: float = 150000.0, ocean=None):
    """The cyclone near ground (x, y): the lowest smoothed surface pressure within ``search`` m (its centre),
    the lowest pressure there (hPa), the maximum 10 m (1-min) wind within 250 km and its radius.  None if no
    patch holds that point."""
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter
    best = None
    dx = env.grid.dx
    for p in patches:
        x0, y0 = p.world_x0(), p.world_y0()
        x1, y1 = x0 + p.rect.nx * dx, y0 + p.rect.ny * dx
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        ps = surface_pressure(env, p.state).astype(cp.float32)
        # a weak, ragged low (under 8 hPa deep) is found on a 40 km mean -- its lowest cell hops from one
        # convective burst to the next -- a mature one on a 12 km mean
        sm = uniform_filter(ps, size=max(3, int(round(12000.0 / dx)) | 1), mode="nearest")
        if float(cp.median(ps)) - float(sm.min()) < 800.0:
            sm = uniform_filter(ps, size=max(3, int(round(40000.0 / dx)) | 1), mode="nearest")
        X = x0 + (cp.arange(p.rect.nx) + 0.5) * dx
        Y = y0 + (cp.arange(p.rect.ny) + 0.5) * dx
        D = cp.hypot(X[None, :] - x, Y[:, None] - y)
        k = int(cp.argmin(cp.where(D <= search, sm, cp.inf)))
        j, i = divmod(k, p.rect.nx)
        cx, cy = float(X[i]), float(Y[j])
        Dc = cp.hypot(X[None, :] - cx, Y[:, None] - cy)
        pmin = float(cp.min(cp.where(Dc <= 30000.0, ps, cp.inf))) / 100.0
        u10 = (ocean.u10.get(p.id) if ocean is not None else None)
        if u10 is None or u10.shape != ps.shape:
            u10 = ten_metre_wind(env, p.state)
        near = Dc <= 250000.0
        kv = int(cp.argmax(cp.where(near, u10, -1.0)))
        vmax = float(u10.ravel()[kv])
        # the radius of maximum wind: of the azimuthal-mean wind (the single strongest cell is anywhere in a
        # ragged storm)
        bw = max(dx, 4000.0)
        kb = cp.minimum(Dc / bw, 63.0).astype(cp.int32).ravel()
        num = cp.bincount(kb, weights=u10.ravel().astype(cp.float64), minlength=64)
        cnt = cp.bincount(kb, minlength=64)
        prof = cp.asnumpy(num[:62] / cp.maximum(cnt[:62], 1))
        rmw = (int(np.argmax(prof)) + 0.5) * bw
        edge = min(cx - x0, x1 - cx, cy - y0, y1 - cy)
        best = {"x": cx, "y": cy, "pmin": pmin, "vmax": vmax, "rmw": rmw, "edge_km": edge / 1000.0}
        break
    return best
