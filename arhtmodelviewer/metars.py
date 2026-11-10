"""Simulated METARs at the real ASOS / AWOS sites (data/metar/stations.csv) around the run.

Every SAMPLE_EVERY s of model time each station reads its lowest model level: the patch cell it stands in,
else the environment (the reference column plus the air masses of the run's fronts / synoptic weather).
The wind is brought down to 10 m by the log law over the ground's roughness (the sea's in hurricane runs);
an instantaneous gust adds the unresolved turbulence (GUST_SIGMA standard deviations of the lowest level's
TKE, never under GUST_MIN of the wind).  Rain falls into a gauge at the Marshall-Palmer rate of the
lowest-level reflectivity.  Every OBS_EVERY s each station issues an observation the way an ASOS does: the
2-min mean wind, the 10-min peak as a gust when it beats the mean by 10 kt, PK WND for the hour's peak over
25 kt, visibility from the rain rate, thunder from 40 dBZ within 5 / 10 statute miles, a ceilometer that
sees no higher than 12,000 ft, the temperature, dewpoint, sea-level pressure and altimeter.  The pressure is
reduced from the model's ground (the sounding's elevation, where the whole run stands) to sea level.  Over
terrain each station stands on the model's ground under it (Station.hg m above the model's zero): its lowest
level, cloud bases and pressure reduction are from there, and outside the patches it reads the reference
column at its own height.

The page draws them (/api/metars, /api/metar) and GR2Analyst / Supercell-Wx read /poll/metars.txt, whose
wind barbs come from one icon sheet drawn here (/poll/metar-barbs.png).
"""
from __future__ import annotations

import csv
import math
import struct
import zlib
from collections import deque
from pathlib import Path

import numpy as np

from .sounding import G, KAPPA, P0

RD = 287.04
KT = 0.514444
STATIONS_CSV = Path(__file__).resolve().parent.parent / "data" / "metar" / "stations.csv"

OBS_EVERY = 600.0          # s between observations
SAMPLE_EVERY = 10.0        # s between samples of the wind and the rain
RADAR_REACH = 250e3        # stations within this of a radar of the run ...
PATCH_REACH = 150e3        # ... or of a patch take part
MAX_ACTIVE = 1500
HISTORY = 144              # observations kept per station (24 h at 10 min)
GUST_SIGMA = 2.5           # gust = 10 m wind + this many standard deviations of the turbulent wind
GUST_MIN = 0.25            # ... and at least this share above the wind (3-s gust over a ~1-min resolved wind)
CLOUD_Q = 1e-5             # kg/kg of cloud water / ice / snow that a ceilometer sees as cloud
CLOUD_KM = 4.0             # radius (km) of the sky a station reports
TS_NEAR_KM, TS_VC_KM = 9.0, 16.0     # thunder at the station / in the vicinity (5 and 10 statute miles)
CEILOMETER_FT = 12000.0
SEA_Z0 = 2e-4

_STATIONS = None


def stations() -> dict:
    """The METAR sites: {"icao", "name", "country": lists, "lat", "lon", "elev": arrays}."""
    global _STATIONS
    if _STATIONS is None:
        rows = list(csv.DictReader(open(STATIONS_CSV, encoding="utf-8")))
        _STATIONS = {"icao": [r["icao"] for r in rows], "name": [r["name"] for r in rows],
                     "country": [r["country"] for r in rows],
                     "lat": np.array([float(r["lat"]) for r in rows]), "lon": np.array([float(r["lon"]) for r in rows]),
                     "elev": np.array([float(r["elev"]) for r in rows])}
    return _STATIONS


def dewpoint(T: float, qv: float, p: float) -> float:
    """Dewpoint (C) of air at T (K), mixing ratio qv, pressure p (Pa)."""
    e = max(qv * p / (0.622 + qv), 1.0) / 100.0
    a = math.log(e / 6.112)
    return 243.5 * a / (17.67 - a)


def rel_humidity(t_c: float, td_c: float) -> float:
    es = lambda x: 6.112 * math.exp(17.67 * x / (x + 243.5))
    return min(100.0, 100.0 * es(td_c) / es(t_c))


def rain_rate(dbz: float) -> float:
    """Rain rate (mm/h) of a reflectivity, Marshall-Palmer, nothing under 15 dBZ, hail capped."""
    if not math.isfinite(dbz) or dbz < 15.0:
        return 0.0
    return min((10.0 ** (min(dbz, 56.0) / 10.0) / 200.0) ** (1.0 / 1.6), 150.0)


def vis_sm(rate: float, snow: bool) -> float:
    """Visibility (statute miles) in rain / snow of this liquid rate (mm/h)."""
    if rate <= 0.0:
        return 10.0
    km = (2.5 if snow else 12.0) / math.sqrt(rate)
    return min(10.0, km / 1.609344)


def _vis_text(v: float) -> str:
    if v >= 10.0:
        return "10SM"
    if v >= 3.0:
        return f"{int(v)}SM"
    for lim, txt in ((2.5, "2 1/2"), (2.0, "2"), (1.75, "1 3/4"), (1.5, "1 1/2"), (1.25, "1 1/4"), (1.0, "1"),
                     (0.75, "3/4"), (0.5, "1/2"), (0.25, "1/4")):
        if v >= lim:
            return txt + "SM"
    return "M1/4SM"


def _tc(x: float) -> str:
    r = int(round(x))
    return ("M" if r < 0 else "") + f"{abs(r):02d}"


class _Station:
    __slots__ = ("i", "icao", "x", "y", "z0", "hg", "samples", "rain", "rain_total", "rain_t", "best", "hist", "now",
                 "_north")

    def __init__(self, i, icao, x, y, z0, hg=0.0):
        self.i, self.icao, self.x, self.y, self.z0 = i, icao, x, y, z0
        self.hg = hg               # the model's ground here, m above its zero (terrain runs; 0 when flat)
        self._north = None         # true north's angle from the grid's here (MetarNet._north), once asked
        self.samples = deque()     # (t, u10, v10, gust m/s, gust direction) of the last hour
        self.rain = deque()        # (t, mm) of the last hour
        self.rain_total = 0.0
        self.rain_t = None
        self.best = None           # (gust m/s, direction, t): the run's peak gust here
        self.hist = deque(maxlen=HISTORY)
        self.now = None            # the latest sample's (T K, qv, p Pa, dbz, z1 m)


class MetarNet:
    def __init__(self, sim):
        self.sim = sim
        self.st = {}               # icao -> _Station taking part
        self.snaps = []            # (t, [obs, ...]) of every observation time
        self.next_sample = None
        self.next_obs = None
        self.version = 0
        self._fields = None

    def _north(self, s) -> float:
        """Degrees true north lies clockwise of the model grid's north at station ``s``: the model's winds are along its
        map grid (Lambert), a wind direction is from true north (geo.Lambert.north_angle)."""
        a = getattr(s, "_north", None)
        if a is None:
            proj = getattr(self.sim, "proj", None)
            if proj is None:
                a = 0.0
            else:
                lon = float(np.ravel(proj.inv(np.array([s.x]), np.array([s.y]))[1])[0])
                a = float(proj.north_angle(lon))
            s._north = a
        return a

    # ------------------------------------------------------------ which stations
    def _reach(self):
        """The stations near the run now: within RADAR_REACH of a radar or PATCH_REACH of a patch."""
        sim, S = self.sim, stations()
        X, Y = sim.proj.fwd(S["lat"], S["lon"])
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        keep = np.zeros(len(X), bool)
        for r in list(sim.radars):
            keep |= np.hypot(X - r.x, Y - r.y) <= RADAR_REACH
        eng = sim.engine
        for p in (eng.patches if eng else []):
            m = p.to_json()
            dx = np.maximum(np.maximum(m["x0"] - X, X - m["x1"]), 0.0)
            dy = np.maximum(np.maximum(m["y0"] - Y, Y - m["y1"]), 0.0)
            keep |= np.hypot(dx, dy) <= PATCH_REACH
        idx = np.nonzero(keep)[0]
        if len(idx) > MAX_ACTIVE:
            cx, cy = np.mean([r.x for r in sim.radars] or [0.0]), np.mean([r.y for r in sim.radars] or [0.0])
            idx = idx[np.argsort(np.hypot(X[idx] - cx, Y[idx] - cy))[:MAX_ACTIVE]]
        return idx, X, Y

    def _refresh_stations(self) -> None:
        idx, X, Y = self._reach()
        S, env = stations(), self.sim.env
        new = [i for i in idx if S["icao"][i] not in self.st]
        if new:
            nx, ny = X[new], Y[new]
            z0 = np.full(len(new), float(env.physics.z0))
            if env.roughness is not None:
                try:
                    z0 = np.asarray(env.roughness(nx, ny), float) * np.ones(len(new))
                except Exception:
                    pass
            ocean = getattr(env, "ocean", None)
            if ocean is not None:
                z0 = np.where(np.asarray(ocean.water(nx, ny), bool), SEA_Z0, z0)
            hg = np.zeros(len(new))
            if getattr(env, "terrain", None) is not None:
                try:
                    hg = np.asarray(env.terrain.height(nx, ny), float) * np.ones(len(new))
                except Exception:
                    pass
            for k, i in enumerate(new):
                self.st[S["icao"][i]] = _Station(int(i), S["icao"][i], float(X[i]), float(Y[i]), max(float(z0[k]), 1e-4),
                                                 float(hg[k]))

    # ------------------------------------------------------------ sampling
    def _environment(self, sts):
        """(u, v ground-relative, theta K, qv, p Pa, tke) of the lowest level in the environment at stations
        outside the patches (numpy arrays)."""
        import cupy as cp
        eng, env = self.sim.engine, self.sim.env
        ref = eng._ref
        mx, my = env.motion
        n = len(sts)
        zh = np.asarray(env.z_h, float)
        prof = lambda a: cp.asnumpy(a[:len(zh)].mean(axis=(1, 2))).astype(float)
        # at each station's lowest level: z_h[0] above its ground, the column's own heights over terrain
        zs = np.array([s.hg for s in sts], float) + zh[0]
        at = lambda a: np.interp(zs, zh, a)
        u = at(prof(0.5 * (ref.u[:, :, :-1] + ref.u[:, :, 1:]))) + mx
        v = at(prof(0.5 * (ref.v[:, :-1, :] + ref.v[:, 1:, :]))) + my
        th = at(np.asarray(env.base.thb, float)[:len(zh)] + prof(ref.thp))
        qv = at(prof(ref.qv))
        p = np.exp(at(np.log(prof(ref.p))))
        tk = getattr(ref, "tke", None)
        tke = at(prof(tk)) if tk is not None and tk.shape[0] >= len(zh) else np.full(n, -1.0)
        if env.airmass is not None and n:
            X = np.array([s.x for s in sts])
            Y = np.array([s.y for s in sts])
            try:
                m = env.airmass.maps(X, Y, env.clock)
                dT, qf, du, dv = env.airmass.fields_3d(m, env.z_h[:1])     # (its own heights are flat)
                th += np.asarray(dT[0], float)
                qv *= np.asarray(qf[0], float)
                u += np.asarray(du[0], float)
                v += np.asarray(dv[0], float)
                pk = env._pockets(X, Y, levels=slice(0, 1))
                if pk is not None:
                    th += np.asarray(pk[0][0], float)
                    qv *= np.asarray(pk[1][0], float)
                    u += np.asarray(pk[2][0], float)
                    v += np.asarray(pk[3][0], float)
            except Exception:
                pass
        return u, v, th, qv, p, tke

    def _gather(self, p, jj, ii, fields):
        """(u, v ground-relative, theta K, qv, p Pa, tke, dbz) of patch p's lowest level at cells (jj, ii)."""
        import cupy as cp
        st, env = p.state, p.env
        J, I = cp.asarray(jj), cp.asarray(ii)
        mx, my = env.motion
        cols = [0.5 * (st.u[0][J, I] + st.u[0][J, I + 1]) + mx, 0.5 * (st.v[0][J, I] + st.v[0][J + 1, I]) + my,
                st.thp[0][J, I] + (st.thb[0][J, I] if getattr(st, "thb", None) is not None and st.thb.ndim == 3
                                   else float(env.base.thb[0])), st.qv[0][J, I], st.p[0][J, I]]
        tk = getattr(st, "tke", None)
        cols.append(tk[0][J, I] if tk is not None else cp.full(len(jj), -1.0))
        f = fields.get(p.id) if fields else None
        ok = f is not None and getattr(f, "patch_id", None) == p.id and f.dbz.shape[1:] == (p.rect.ny, p.rect.nx)
        cols.append(f.dbz[0][J, I] if ok else cp.full(len(jj), -99.0))
        return cp.asnumpy(cp.stack([cp.asarray(c, dtype=cp.float64) for c in cols]))

    def _where(self, sts):
        """{station: (patch, j, i)} of the stations inside a patch (a nest wins over its parent)."""
        eng = self.sim.engine
        out = {}
        for p in list(eng.patches) + list(eng.nest_patches()):
            dx = p.env.grid.dx
            x0, y0 = p.world_x0(), p.world_y0()
            for s in sts:
                i, j = int((s.x - x0) // dx), int((s.y - y0) // dx)
                if 0 <= i < p.rect.nx and 0 <= j < p.rect.ny:
                    out[s.icao] = (p, j, i)
        return out

    def _sample(self, t: float, fields) -> None:
        sts = list(self.st.values())
        if not sts:
            return
        where = self._where(sts)
        vals = {}
        by_patch = {}
        for s in sts:
            if s.icao in where:
                p, j, i = where[s.icao]
                by_patch.setdefault(id(p), (p, []))[1].append((s, j, i))
        for p, lst in by_patch.values():
            a = self._gather(p, np.array([q[1] for q in lst]), np.array([q[2] for q in lst]), self._fields)
            z1, zt = float(p.env.z_h[0]), float(p.env.z_w[-1])
            for k, (s, _, _) in enumerate(lst):
                vals[s.icao] = tuple(a[:, k]) + (z1 * (1.0 - s.hg / zt),)        # squeezed over terrain
        out = [s for s in sts if s.icao not in vals]
        if out:
            u, v, th, qv, pp, tke = self._environment(out)
            z1 = float(self.sim.env.z_h[0])
            for k, s in enumerate(out):
                vals[s.icao] = (u[k], v[k], th[k], qv[k], pp[k], tke[k], -99.0, z1)
        for s in sts:
            u, v, th, qv, pp, tke, dbz, z1 = vals[s.icao]
            fac = 1.0 if z1 <= 10.0 else min(max(math.log(10.0 / s.z0) / math.log(z1 / s.z0), 0.4), 1.0)
            u10, v10 = u * fac, v * fac
            spd = math.hypot(u10, v10)
            sig = math.sqrt(max(tke, 0.0) * 2.0 / 3.0) * fac if tke >= 0 else 0.0
            gust = spd + max(GUST_SIGMA * sig, GUST_MIN * spd)
            wdir = (math.degrees(math.atan2(-u10, -v10)) - self._north(s) + 360.0) % 360.0
            s.samples.append((t, u10, v10, gust, wdir))
            while s.samples and s.samples[0][0] < t - 3600.0:
                s.samples.popleft()
            if s.best is None or gust > s.best[0]:
                s.best = (gust, wdir, t)
            T = th * (pp / P0) ** KAPPA
            rate = rain_rate(dbz)
            if s.rain_t is not None and rate > 0.0:
                mm = rate * min(max(t - s.rain_t, 0.0), 60.0) / 3600.0
                s.rain.append((t, mm))
                s.rain_total += mm
            s.rain_t = t
            while s.rain and s.rain[0][0] < t - 3600.0:
                s.rain.popleft()
            s.now = (T, qv, pp, dbz, z1, rate)

    # ------------------------------------------------------------ observing
    def _sky_and_thunder(self, t: float):
        """{icao: (layers [(cover, ft AGL)], tops ft, cref within TS_NEAR_KM, within TS_VC_KM, hail at the ground)}
        for the stations inside patches."""
        import cupy as cp
        out = {}
        where = self._where(list(self.st.values()))
        groups = {}
        for icao, (p, j, i) in where.items():
            groups.setdefault(id(p), (p, []))[1].append((icao, j, i))
        for p, lst in groups.values():
            st, env = p.state, p.env
            dx = env.grid.dx
            q = None
            for name in ("qc", "qi", "qs"):
                a = getattr(st, name, None)
                if a is not None:
                    q = a if q is None else q + a
            zh = cp.asarray(np.asarray(env.z_h, dtype=np.float32))
            if q is not None:
                cloudy = q >= CLOUD_Q
                anyc = cloudy.any(axis=0)
                base = cp.where(anyc, zh[cp.argmax(cloudy, axis=0)], cp.nan)
                top = cp.where(anyc, zh[len(env.z_h) - 1 - cp.argmax(cloudy[::-1], axis=0)], cp.nan)
            f = self._fields.get(p.id) if self._fields else None
            ok = f is not None and getattr(f, "patch_id", None) == p.id and f.dbz.shape[1:] == (p.rect.ny, p.rect.nx)
            cref = f.dbz.max(axis=0) if ok else None
            hail = None
            for name in ("qh", "qg"):
                a = getattr(st, name, None)
                if a is not None:
                    hail = a[0] if hail is None else cp.maximum(hail, a[0])
            rc, rn, rv = max(1, int(round(CLOUD_KM * 1000 / dx))), int(round(TS_NEAR_KM * 1000 / dx)), int(round(TS_VC_KM * 1000 / dx))
            ny, nx = p.rect.ny, p.rect.nx
            zt = float(env.z_w[-1])
            for icao, j, i in lst:
                win = lambda a, r: a[max(j - r, 0):min(j + r + 1, ny), max(i - r, 0):min(i + r + 1, nx)]
                layers, tops = [], None
                if q is not None:
                    sq = 1.0 - self.st[icao].hg / zt              # over terrain the levels squeeze over the ground
                    b = cp.asnumpy(win(base, rc)).ravel() * sq
                    tp = cp.asnumpy(win(top, rc)).ravel() * sq
                    n = b.size
                    good = np.isfinite(b)
                    if good.any():
                        tops = float(np.nanmax(tp)) * 3.28084
                    # two decks at most: low (base under 3 km) and the rest, each by the share of sky it covers
                    for lo, hi in ((0.0, 3000.0), (3000.0, 1e9)):
                        sel = good & (b >= lo) & (b < hi)
                        frac = sel.sum() / max(n, 1)
                        if frac < 0.05:
                            continue
                        ft = float(np.percentile(b[sel], 25)) * 3.28084
                        if ft > CEILOMETER_FT:
                            continue
                        cover = "FEW" if frac < 0.25 else "SCT" if frac < 0.5 else "BKN" if frac < 0.875 else "OVC"
                        layers.append((cover, ft))
                near = vc = -99.0
                if cref is not None:
                    near = float(cp.asnumpy(win(cref, rn).max()))
                    vc = float(cp.asnumpy(win(cref, rv).max()))
                gr = float(cp.asnumpy(hail[j, i])) if hail is not None else 0.0
                out[icao] = (layers, tops, near, vc, gr)
        return out

    def _env_sky(self, hg: float = 0.0):
        """The sky outside the patches: a deck where the reference column is saturated, clear otherwise (heights
        above a station on ground hg m above the model's zero)."""
        import cupy as cp
        eng, env = self.sim.engine, self.sim.env
        ref = eng._ref
        th = np.asarray(env.base.thb, float) + cp.asnumpy(ref.thp.mean(axis=(1, 2)))
        pp = cp.asnumpy(ref.p.mean(axis=(1, 2)))
        qv = cp.asnumpy(ref.qv.mean(axis=(1, 2)))
        T = th * (pp / P0) ** KAPPA
        for k, z in enumerate(env.z_h):
            if z < hg:
                continue
            ft = float(z - hg) * 3.28084
            if ft > CEILOMETER_FT:
                break
            rh = rel_humidity(T[k] - 273.15, dewpoint(T[k], qv[k], pp[k]))
            if rh >= 98.0:
                return [("BKN", ft)]
            if rh >= 93.0:
                return [("SCT", ft)]
        return []

    def _observe(self, t: float) -> None:
        sim = self.sim
        from . import level2
        when = level2.utc(sim.start_ms + t * 1000.0)
        stamp = when.strftime("%d%H%MZ")
        sky_in = self._sky_and_thunder(t)
        sky_env = {}
        tors = [c for c in getattr(sim.tornadoes, "items", []) if getattr(c, "found", False)]
        zsfc = float(sim.env.sounding.zsfc)
        obs_list = []
        for s in self.st.values():
            if s.now is None or not s.samples:
                continue
            hs = zsfc + s.hg                                   # the station's elevation on the model's ground
            T, qv, pp, dbz, z1, rate = s.now
            two = [q for q in s.samples if q[0] > t - 120.0] or [s.samples[-1]]
            ten = [q for q in s.samples if q[0] > t - 600.0] or [s.samples[-1]]
            hour = list(s.samples)
            um = sum(q[1] for q in two) / len(two)
            vm = sum(q[2] for q in two) / len(two)
            spd_kt = sum(math.hypot(q[1], q[2]) for q in two) / len(two) / KT
            wdir = (math.degrees(math.atan2(-um, -vm)) - self._north(s) + 360.0) % 360.0
            g10 = max(ten, key=lambda q: q[3])
            gust_kt = g10[3] / KT
            pk = max(hour, key=lambda q: q[3])
            t_c = T - 273.15 - 0.0098 * (2.0 - z1)            # lowest level -> 2 m, dry adiabatic
            td_c = min(dewpoint(T, qv, pp), t_c)
            tv = T * (1.0 + 0.61 * qv)
            psfc = pp * math.exp(G * z1 / (RD * tv))           # the model's ground
            slp = psfc * math.exp(G * hs / (RD * (tv + 0.00325 * hs))) / 100.0
            alt = ((psfc / 100.0 - 0.3) ** 0.190284 + 8.4228807e-5 * hs) ** (1.0 / 0.190284) / 33.8639
            snow = t_c <= 0.5
            if s.icao in sky_in:
                layers, tops, near, vc, gr = sky_in[s.icao]
            else:
                key = round(s.hg, -1)
                if key not in sky_env:
                    sky_env[key] = self._env_sky(s.hg)
                layers, tops, near, vc, gr = sky_env[key], None, -99.0, -99.0, 0.0
            vis = vis_sm(rate, snow)
            wx = []
            ts = near >= 40.0
            precip = "SN" if snow else "RA"
            if rate > 0.0:
                inten = "-" if dbz < 30.0 else "+" if dbz >= 45.0 else ""
                hail = "GR" if (gr > 1e-3 and dbz >= 55.0) else ""
                wx.append(inten + ("TS" if ts else "") + precip + hail)
            elif ts:
                wx.append("TS")
            elif vc >= 40.0:
                wx.append("VCTS")
            if not rate and rel_humidity(t_c, td_c) >= 97.0:
                vis = min(vis, 5.0)
                wx.append("BR")
            if rate > 0.0 and tops:
                # under a raining cloud the ceilometer hits the scud and the rain shaft near the inflow's LCL,
                # not the model's cloud base (in the downdraft it is often above the 12,000 ft limit)
                scud = min(max(125.0 * (t_c - td_c) * 3.28084 * 0.8, 300.0), CEILOMETER_FT)
                if not any(f <= scud for _, f in layers):
                    layers = [("OVC" if dbz >= 35.0 else "BKN", scud)] + [(c, f) for c, f in layers if f > scud][:1]
            # the wind group
            if spd_kt < 2.5 and gust_kt < 10.0:
                wtxt = "00000KT"
                rdir, rspd = 0, 0
            else:
                rdir = int(round(wdir / 10.0)) * 10 or 360
                rspd = int(round(spd_kt))
                wtxt = f"{rdir:03d}{rspd:02d}"
                if gust_kt - spd_kt >= 10.0:
                    wtxt += f"G{int(round(gust_kt)):02d}"
                wtxt += "KT"
            sky = " ".join(f"{c}{min(int(round(f / 100.0)), 999):03d}" for c, f in layers) or "CLR"     # AUTO: no CB
            p1h = sum(mm for _, mm in s.rain) / 25.4
            rmk = ["AO2"]
            if pk[3] / KT > 25.0:
                pw = level2.utc(sim.start_ms + pk[0] * 1000.0)
                rmk.append(f"PK WND {int(round(pk[4] / 10.0)) * 10 or 360:03d}{int(round(pk[3] / KT)):02d}/"
                           f"{pw.strftime('%H%M') if pw.hour != when.hour else pw.strftime('%M')}")
            if ts:
                rmk.append("TS OHD MOV UNKN" if near >= 45.0 else "TS ALQDS")
            rmk.append(f"SLP{int(round(slp * 10.0)) % 1000:03d}")
            if p1h > 0.0:
                rmk.append(f"P{max(int(round(p1h * 100.0)), 1):04d}")
            tg = lambda x: ("1" if x < 0 else "0") + f"{int(round(abs(x) * 10.0)):03d}"
            rmk.append(f"T{tg(t_c)}{tg(td_c)}")
            metar = (f"METAR {s.icao} {stamp} AUTO {wtxt} {_vis_text(vis)} " + (" ".join(wx) + " " if wx else "")
                     + f"{sky} {_tc(t_c)}/{_tc(td_c)} A{int(round(alt * 100.0)):04d} RMK " + " ".join(rmk))
            near_tor = None
            for c in tors:
                d = math.hypot(c.x - s.x, c.y - s.y) / 1000.0
                if near_tor is None or d < near_tor:
                    near_tor = d
            o = {"t": t, "icao": s.icao, "dir": rdir, "kt": rspd, "gust": int(round(gust_kt)) if gust_kt - spd_kt >= 10.0 else 0,
                 "peak10": round(gust_kt, 1), "peak10_dir": round(g10[4]), "pk": [round(pk[3] / KT, 1), round(pk[4]), pk[0]],
                 "best": [round(s.best[0] / KT, 1), round(s.best[1]), s.best[2]],
                 "t_c": round(float(t_c), 1), "td_c": round(float(td_c), 1), "rh": round(rel_humidity(t_c, td_c)), "slp": round(slp, 1),
                 "alt": round(alt, 2), "vis": round(vis, 2), "wx": " ".join(wx), "sky": [[c, round(f)] for c, f in layers],
                 "tops_ft": round(tops) if tops else None, "dbz": round(dbz, 1) if dbz > -90 else None,
                 "rate": round(rate, 1), "p1h": round(p1h, 2), "ptot": round(s.rain_total / 25.4, 2),
                 "tor_km": round(near_tor, 1) if near_tor is not None else None, "in_storm": s.icao in sky_in,
                 "metar": metar}
            s.hist.append(o)
            obs_list.append(o)
        self.snaps.append((t, obs_list))
        while len(self.snaps) > HISTORY:
            self.snaps.pop(0)
        self.version += 1

    def step(self, fields, t: float) -> None:
        """Called by the run every step (under the run's lock)."""
        if self.sim.engine is None or getattr(self.sim.engine, "_ref", None) is None:
            return
        if self.next_obs is not None and t < self.next_sample - SAMPLE_EVERY - 1.0:       # a rollback
            self.snaps = [s for s in self.snaps if s[0] <= t]
            for s in self.st.values():
                while s.hist and s.hist[-1]["t"] > t:
                    s.hist.pop()
                while s.samples and s.samples[-1][0] > t:
                    s.samples.pop()
            self.next_sample = t
            self.next_obs = (t // OBS_EVERY + 1) * OBS_EVERY
        self._fields = fields
        if self.next_obs is None:
            self._refresh_stations()
            self.next_sample = t
            self.next_obs = (t // OBS_EVERY + 1) * OBS_EVERY
        if t >= self.next_sample:
            self._sample(t, fields)
            self.next_sample = t + SAMPLE_EVERY
        if t >= self.next_obs:
            self._observe(t)
            self.next_obs = (t // OBS_EVERY + 1) * OBS_EVERY
            self._refresh_stations()        # radars or patches may have come since

    # ------------------------------------------------------------ for the page and the placefile
    def snapshot(self, t: float | None = None):
        """(t, [obs]) of the latest observation at or before t (None: the latest)."""
        snaps = list(self.snaps)
        if not snaps:
            return None
        if t is None:
            return snaps[-1]
        best = None
        for s in snaps:
            if s[0] <= t + 1.0:
                best = s
        return best

    def to_json(self, t: float | None = None) -> dict:
        snap = self.snapshot(t)
        S = stations()
        if snap is None:
            return {"t": None, "stations": []}
        out = []
        for o in snap[1]:
            s = self.st.get(o["icao"])
            if s is None:
                continue
            out.append({"icao": o["icao"], "name": S["name"][s.i], "us": S["country"][s.i] == "US", "x": s.x, "y": s.y,
                        "dir": o["dir"], "kt": o["kt"], "gust": o["gust"], "t_c": o["t_c"], "td_c": o["td_c"],
                        "wx": o["wx"], "sky": o["sky"][0][0] if o["sky"] else "CLR"})
        return {"t": snap[0], "stations": out}

    def detail(self, icao: str, t: float | None = None) -> dict | None:
        s = self.st.get(icao)
        if s is None:
            return None
        S = stations()
        tt = 1e18 if t is None else t + 1.0
        hist = [o for o in list(s.hist) if o["t"] <= tt]
        if not hist:
            return None
        best = max(hist, key=lambda o: o["best"][0])["best"]
        return {"icao": icao, "name": S["name"][s.i], "lat": float(S["lat"][s.i]), "lon": float(S["lon"][s.i]),
                "elev": float(S["elev"][s.i]), "us": S["country"][s.i] == "US", "model_ground": float(self.sim.env.sounding.zsfc) + s.hg,
                "start_ms": self.sim.start_ms, "latest": hist[-1], "history": hist[::-1][:36], "best": best}

    def placefile(self, icon_url: str, refresh: int = 10) -> str:
        snap = self.snapshot()
        lines = _header(icon_url, refresh)
        if snap is None:
            return "\n".join(lines) + "\n"
        S = stations()
        for o in snap[1]:
            s = self.st.get(o["icao"])
            if s is None:
                continue
            us = S["country"][s.i] == "US"
            f = (lambda c: c * 9.0 / 5.0 + 32.0) if us else (lambda c: c)
            hover = o["metar"].replace('"', "'") + "\\n" + S["name"][s.i].replace('"', "'")
            if o["best"][0] >= 1:
                hover += f"\\nRun-max gust {o['best'][0]:.0f} kt"
            k = 1 if o["kt"] == 0 and not o["gust"] else min(2 + int(round(o["kt"] / 5.0)), N_ICONS)
            lines += [f"Object: {S['lat'][s.i]:.4f},{S['lon'][s.i]:.4f}", "Threshold: 250",
                      f'Icon: 0,0,{o["dir"] % 360},1,{k},"{hover}"',
                      "Color: 255 90 90", f'Text: -14, 9, 1, "{f(o["t_c"]):.0f}"',
                      "Color: 90 230 90", f'Text: -14, -9, 1, "{f(o["td_c"]):.0f}"']
            if o["gust"]:
                lines += ["Color: 255 190 60", f'Text: 16, 9, 1, "G{o["gust"]}"']
            if o["wx"]:
                lines += ["Color: 120 200 255", f'Text: -30, 0, 1, "{o["wx"].split()[0]}"']
            lines.append("End:")
        return "\n".join(lines) + "\n"


def _header(icon_url: str, refresh: int) -> list:
    return ["; Atmos-WRF: simulated METARs (wind barbs, temperature, dewpoint)",
            "Title: Atmos-WRF METARs", f"RefreshSeconds: {int(refresh)}", "Threshold: 999",
            f'IconFile: 1, {ICON}, {ICON}, {ICON // 2}, {ICON // 2}, "{icon_url}"', 'Font: 1, 11, 1, "Arial"']


def empty_placefile(icon_url: str, refresh: int = 10) -> str:
    return "\n".join(_header(icon_url, refresh)) + "\n"


# ------------------------------------------------------------------ the wind barb icons
ICON = 64
N_ICONS = 42          # 1: calm, 2: 0-2 kt (a bare staff), k: 5 (k - 2) kt


def _segment(img, a, b, w, val):
    """Anti-aliased segment of width w (px) from a to b on a supersampled canvas."""
    h, wd = img.shape
    yy, xx = np.mgrid[0:h, 0:wd] + 0.5
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = max(dx * dx + dy * dy, 1e-9)
    tt = np.clip(((xx - ax) * dx + (yy - ay) * dy) / L2, 0.0, 1.0)
    d = np.hypot(xx - (ax + tt * dx), yy - (ay + tt * dy))
    np.maximum(img, np.clip(w / 2.0 + 0.5 - d, 0.0, 1.0) * val, out=img)


def _triangle(img, p, val):
    h, wd = img.shape
    yy, xx = np.mgrid[0:h, 0:wd] + 0.5
    (x1, y1), (x2, y2), (x3, y3) = p
    den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    l1 = ((y2 - y3) * (xx - x3) + (x3 - x2) * (yy - y3)) / den
    l2 = ((y3 - y1) * (xx - x3) + (x1 - x3) * (yy - y3)) / den
    inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
    img[inside] = np.maximum(img[inside], val)


def _barb(kt: int, calm: bool, ss: int = 4) -> np.ndarray:
    """One icon (ICON x ICON RGBA): a white barb with a dark edge pointing north from the centre (the
    placefile turns it by the wind's direction), feathers on the clockwise side as in the northern
    hemisphere's station plots: a pennant 50 kt, a feather 10, a half feather 5."""
    n = ICON * ss
    c = n / 2.0
    staff, gap, feather, pen_h = 24 * ss, 4.5 * ss, 12 * ss, 6 * ss
    lean = math.radians(65)                      # the feathers' angle off the staff
    fx, fy = math.sin(lean), -math.cos(lean)

    def paint(img, wid):
        yy, xx = np.mgrid[0:n, 0:n] + 0.5
        if calm:
            r = np.hypot(xx - c, yy - c)
            np.maximum(img, np.clip(wid / 2 + 0.5 - abs(r - 5 * ss), 0, 1), out=img)
            return
        top = c - staff
        _segment(img, (c, c), (c, top), wid, 1.0)
        np.maximum(img, np.clip(2.2 * ss + wid / 2 - np.hypot(xx - c, yy - c), 0, 1), out=img)
        left = int(round(kt / 5.0)) * 5
        y = top
        while left >= 50:
            tip = (c + feather * fx, y + feather * fy * 0.15)
            _triangle(img, [(c, y), (c, y + pen_h), tip], 1.0)
            _segment(img, (c, y), tip, wid, 1.0)
            _segment(img, (c, y + pen_h), tip, wid, 1.0)
            y += pen_h + 2 * ss
            left -= 50
        while left >= 10:
            _segment(img, (c, y), (c + feather * fx, y + feather * fy), wid, 1.0)
            y += gap
            left -= 10
        if left >= 5:
            if y == top:                         # a lone half feather sits one step down the staff
                y += gap
            _segment(img, (c, y), (c + 0.55 * feather * fx, y + 0.55 * feather * fy), wid, 1.0)

    edge = np.zeros((n, n))
    ink = np.zeros((n, n))
    paint(edge, 4.0 * ss)
    paint(ink, 1.7 * ss)
    a = np.maximum(edge * 0.85, ink)
    rgb = np.repeat(ink[..., None] * 255.0 + (1 - ink[..., None]) * 15.0, 3, axis=2)
    out = np.concatenate([rgb, a[..., None] * 255.0], axis=2)
    out = out.reshape(ICON, ss, ICON, ss, 4).mean(axis=(1, 3))
    return np.rint(out).astype(np.uint8)


_SHEET = None


def barb_sheet() -> bytes:
    """The placefile's icon sheet (PNG): icon 1 calm, icon k >= 2 the barb of 5 (k - 2) kt, 8 a row."""
    global _SHEET
    if _SHEET is None:
        cols = 8
        rows = -(-N_ICONS // cols)
        sheet = np.zeros((rows * ICON, cols * ICON, 4), np.uint8)
        for k in range(1, N_ICONS + 1):
            ic = _barb(5 * (k - 2), calm=(k == 1))
            r, q = divmod(k - 1, cols)
            sheet[r * ICON:(r + 1) * ICON, q * ICON:(q + 1) * ICON] = ic
        h, w = sheet.shape[:2]
        raw = np.zeros((h, 1 + 4 * w), np.uint8)
        raw[:, 1:] = sheet.reshape(h, 4 * w)

        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        _SHEET = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                  + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))
    return _SHEET
