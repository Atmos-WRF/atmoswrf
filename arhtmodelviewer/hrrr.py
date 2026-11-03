"""HRRR as the weather: NOAA's High-Resolution Rapid Refresh (3 km, hourly) supplies the environment, and
ArWen makes the storms.

What comes from the HRRR: the air everywhere and its evolution hour by hour -- temperature, moisture and wind
on 25 pressure levels from 1000 to 100 hPa, the surface (2 m temperature and dewpoint, 10 m wind, surface
and sea-level pressure) and the 700 hPa vertical motion -- so its fronts, drylines, outflow boundaries,
moisture return, cap and low-level jet are the run's.  What does not: its reflectivity, rain and clouds.
Storms form where its environment lets them (the same initiation as the synoptic mode: CAPE, CIN and lift
from its fronts, convergence and ascent) and the storm patches simulate them with the run's own
microphysics.  Everything else of the synoptic mode works the same on it: maps, fronts, soundings, the loop,
the storm patches' edges and their gentle low-level hold.

The data: NOAA's open archive on AWS (noaa-hrrr-bdp-pds; every cycle since 2014, the newest within the
hour), only the records needed (the .idx index gives each one's bytes; ~70 MB a forecast hour), decoded with
ECMWF's eccodes, kept every 3rd point (9 km) and cached on disk per cycle and hour.

The viewer's world is flat: the HRRR's terrain is flattened.  Each column keeps its temperature and
relative humidity height for height above its own ground (what CAPE, the cap, the cloud base and the shear
depend on) and the storm patches get them on their own pressure levels.
"""
from __future__ import annotations

import datetime as _dt
import math
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import qg
from .sounding import EPS, G, KAPPA, P0, RD, dewpoint_from_qv, qv_sat, qv_sat_ice
from .synoptic import (CACHE, DEFAULTS, DT, DX, RH_MAX, SEASON, SynopticWorld, ZC, _Bilinear, _sample, _smooth, _ss)

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HCACHE = CACHE / "hrrr"
LEVELS = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
          650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100)
LVARS = ("HGT", "TMP", "SPFH", "UGRD", "VGRD")
SFC = (("PRES", "surface"), ("HGT", "surface"), ("TMP", "2 m above ground"), ("DPT", "2 m above ground"),
       ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"), ("MSLMA", "mean sea level"), ("VVEL", "700 mb"),
       # where the HRRR's own storms are (its strongest updraft in the hour; none at F00) and its
       # most-unstable CAPE (elevated storms): initiation only, never drawn
       ("MAXUVV", "100-1000 mb above ground"), ("CAPE", "255-0 mb above ground"))
STRIDE = 3                       # every 3rd point: 9 km
_HLOCK = threading.Lock()        # the forecast hours in memory (shared by a world and its copies)
_DECODE = threading.Lock()       # eccodes, one message at a time
R_HRRR = 6371229.0               # the HRRR's sphere (GRIB shape of the earth 6)


def url(cycle: _dt.datetime, fh: int) -> str:
    return f"{BUCKET}/hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfprsf{fh:02d}.grib2"


def max_hours(cycle: _dt.datetime) -> int:
    """Forecast hours a cycle has: 48 at 00, 06, 12 and 18 UTC, else 18."""
    return 48 if cycle.hour % 6 == 0 else 18


def _get(u: str, rng: tuple | None = None, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(u, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def available(cycle: _dt.datetime, fh: int = 0) -> bool:
    try:
        _get(url(cycle, fh) + ".idx", timeout=15)
        return True
    except Exception:
        return False


def latest_cycle(now: _dt.datetime | None = None) -> _dt.datetime:
    """The newest cycle whose analysis is in the archive (they arrive ~1 h after their time)."""
    now = (now or _dt.datetime.now(_dt.timezone.utc)).replace(minute=0, second=0, microsecond=0)
    for back in range(1, 8):
        c = now - _dt.timedelta(hours=back)
        if available(c, 0):
            return c
    raise LookupError("no recent HRRR cycle found in NOAA's archive (no internet?)")


def cycle_for(valid: _dt.datetime, now: _dt.datetime | None = None) -> tuple[_dt.datetime, int]:
    """(cycle, forecast hour) for a valid time: a past time from its own analysis, a future one (or one not
    in yet) from the newest run whose forecast reaches it and is out already."""
    now = (now or _dt.datetime.now(_dt.timezone.utc)).replace(minute=0, second=0, microsecond=0)
    valid = valid.replace(minute=0, second=0, microsecond=0)
    if valid <= now - _dt.timedelta(hours=2) and available(valid, 0):
        return valid, 0
    c = min(valid, now)
    oldest = valid - _dt.timedelta(hours=48)
    while c >= oldest:
        fh = int((valid - c).total_seconds() // 3600)
        if fh <= max_hours(c) and available(c, fh):
            return c, fh
        c -= _dt.timedelta(hours=1)
    raise LookupError(f"no HRRR run reaches {valid:%Y-%m-%d %HZ} yet (the newest reach about 48 h ahead)")


def _maxfilter(a: np.ndarray, r: int) -> np.ndarray:
    """The largest value within r points (a square), edges included."""
    p = np.pad(a, r, mode="edge")
    out = a.copy()
    ny, nx = a.shape
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            np.maximum(out, p[dy:dy + ny, dx:dx + nx], out=out)
    return out


class Lambert:
    """The HRRR's Lambert conformal grid (one standard parallel), forward lat/lon -> m."""

    def __init__(self, lad: float, lov: float, latin: float):
        self.n = math.sin(math.radians(latin))
        self.F = math.cos(math.radians(latin)) * math.tan(math.pi / 4 + math.radians(latin) / 2) ** self.n / self.n
        self.rho0 = R_HRRR * self.F / math.tan(math.pi / 4 + math.radians(lad) / 2) ** self.n
        self.lov = lov

    def fwd(self, lat, lon):
        lat = np.radians(np.asarray(lat, np.float64))
        dl = (np.asarray(lon, np.float64) - self.lov + 180.0) % 360.0 - 180.0
        rho = R_HRRR * self.F / np.tan(np.pi / 4 + lat / 2) ** self.n
        th = self.n * np.radians(dl)
        return rho * np.sin(th), self.rho0 - rho * np.cos(th)


# ------------------------------------------------------------------ one forecast hour
def fetch_hour(cycle: _dt.datetime, fh: int, progress=None) -> dict:
    """One forecast hour: {"levels": (len(LVARS), len(LEVELS), ny, nx), "sfc": (len(SFC), ny, nx), "grid":
    {...}} on every STRIDE-th point, float32 (from the disk cache when there)."""
    path = HCACHE / f"{cycle:%Y%m%d%H}" / f"f{fh:02d}.npz"
    if path.is_file():
        try:
            with np.load(path) as z:
                if z["sfc"].shape[0] != len(SFC):          # cached by an older version: fetch it again
                    raise ValueError
                return {"levels": z["levels"], "sfc": z["sfc"], "grid": dict(zip(z["gk"].tolist(), z["gv"].tolist()))}
        except Exception:
            pass
    try:
        import eccodes
    except Exception as exc:
        raise RuntimeError("the HRRR decoder (eccodes) is not installed: run Setup.cmd once more") from exc
    u = url(cycle, fh)
    idx = _get(u + ".idx", timeout=30).decode().splitlines()
    recs = [line.split(":") for line in idx if line.strip()]
    want = {}
    for k, r in enumerate(recs):
        name, lev = r[3], r[4]
        if lev.endswith(" mb") and name in LVARS and lev.split()[0].isdigit() and int(lev.split()[0]) in LEVELS:
            want[k] = ("L", LVARS.index(name), LEVELS.index(int(lev.split()[0])))
        elif (name, lev) in SFC:
            want[k] = ("S", SFC.index((name, lev)), 0)
    # neighbouring records in one request
    ranges, run = [], []
    for k in sorted(want):
        if run and k != run[-1] + 1:
            ranges.append(run)
            run = []
        run.append(k)
    if run:
        ranges.append(run)
    done = [0]
    lock = threading.Lock()
    out_l = None
    out_s = None
    grid = {}

    def work(ks):
        nonlocal out_l, out_s
        start = int(recs[ks[0]][1])
        end = int(recs[ks[-1] + 1][1]) - 1 if ks[-1] + 1 < len(recs) else ""
        blob = None
        for attempt in range(4):
            try:
                blob = _get(u, (start, end))
                break
            except Exception:
                if attempt == 3:
                    raise
        for k in ks:
            a = int(recs[k][1]) - start
            b = (int(recs[k + 1][1]) - start) if k + 1 < len(recs) else len(blob)
            with _DECODE, lock:                       # eccodes is not thread-safe: downloads in parallel, decodes one by one
                h = eccodes.codes_new_from_message(blob[a:b])
                try:
                    nx, ny = eccodes.codes_get(h, "Nx"), eccodes.codes_get(h, "Ny")
                    v = eccodes.codes_get_values(h).reshape(ny, nx)[::STRIDE, ::STRIDE].astype(np.float32)
                    if not grid:
                        for key in ("Nx", "Ny", "DxInMetres", "latitudeOfFirstGridPointInDegrees",
                                    "longitudeOfFirstGridPointInDegrees", "LaDInDegrees", "LoVInDegrees",
                                    "Latin1InDegrees"):
                            grid[key] = float(eccodes.codes_get(h, key))
                finally:
                    eccodes.codes_release(h)
                if out_l is None:
                    out_l = np.full((len(LVARS), len(LEVELS)) + v.shape, np.nan, np.float32)
                    out_s = np.full((len(SFC),) + v.shape, np.nan, np.float32)
                kind, i, j = want[k]
                if kind == "L":
                    out_l[i, j] = v
                else:
                    out_s[i] = v
        with lock:
            done[0] += len(ks)
            if progress:
                progress(done[0] / len(want), f"HRRR {cycle:%Y-%m-%d %HZ} F{fh:02d}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, ranges))
    if out_l is None or np.isnan(out_s).all():
        raise RuntimeError(f"HRRR {cycle:%Y-%m-%d %HZ} F{fh:02d}: no data decoded")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = path.with_suffix(".part.npz")
        np.savez_compressed(tmp, levels=out_l, sfc=out_s, gk=np.array(list(grid)), gv=np.array(list(grid.values())))
        tmp.replace(path)
    except OSError:
        pass
    return {"levels": out_l, "sfc": out_s, "grid": grid}


# ------------------------------------------------------------------ the world
class HrrrWorld(SynopticWorld):
    """SynopticWorld's maps, fronts, diagnostics, initiation field and storm-patch interface on HRRR data."""

    KEEP_HOURS = 5                 # forecast hours held in memory (~110 MB each; a run reads 2 h ahead of itself)
    DIURNAL_IN_ANALYSIS = False    # the HRRR's 2 m temperature is the hour's own
    _SHARED = ("_hours", "_hsurf", "_hlock", "_s25", "hp", "_geo")     # read-only: shared by the preview's and the run's copies

    def __init__(self, proj, start: _dt.datetime, cfg: dict | None = None, seed: int = 1, progress=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        h = self.cfg.get("hrrr") or {}
        self.cycle = _dt.datetime.fromisoformat(str(h["cycle"]).replace("Z", "+00:00"))
        if self.cycle.tzinfo is None:
            self.cycle = self.cycle.replace(tzinfo=_dt.timezone.utc)
        self.start_fh = int(h.get("start_fh", 0))
        self.fh_last = min(self.start_fh + int(h.get("hours", 12)), max_hours(self.cycle))
        self.proj = proj
        self.start = self.cycle + _dt.timedelta(hours=self.start_fh)
        self.seed = int(seed)
        self.lock = threading.RLock()
        self.progress = progress or (lambda f, msg: None)
        (self.jet_lat, self.sst_gulf, self.dtheta, self.eml_src, self.t500_jet) = SEASON[self.start.month]
        self.keep_hourly = True
        self._snaps, self._diag, self._anal, self.notes = {}, {}, {}, []
        self.q = qg.QGModel()                         # its grid sizes only (the surface grid is placed from it)
        self._grids()
        self._hours = {}                              # fh -> data dict
        self._hsurf = {}                              # fh -> the 25 km snapshot stack for the maps
        self._hlock = _HLOCK
        self.n = 0
        self.max_hour = self.fh_last - self.start_fh
        self.scenario_note = (f"HRRR {self.cycle:%Y-%m-%d %HZ} (NOAA), F{self.start_fh:02d} to F{self.fh_last:02d}: "
                              f"the HRRR's environment, the run's own storms")
        first = self._hour(self.start_fh)             # the grid's geometry comes with the first hour
        self._setup_hrrr_grid(first["grid"])
        self.progress(1.0, "ready")
        # the rest of the hours in the background, in order
        threading.Thread(target=self._prefetch, name="hrrr-prefetch", daemon=True).start()

    # ------------------------------------------------------------ data
    def _prefetch(self):
        for fh in range(self.start_fh + 1, self.fh_last + 1):
            try:
                self._hour(fh, keep=False)
            except Exception:
                return

    def _hour(self, fh: int, keep: bool = True) -> dict:
        fh = int(min(max(fh, self.start_fh), self.fh_last))
        with self._hlock:
            if fh in self._hours:
                return self._hours[fh]
        d = fetch_hour(self.cycle, fh, self.progress)
        if not keep:
            return d
        with self._hlock:
            self._hours[fh] = d
            while len(self._hours) > self.KEEP_HOURS:
                old = min(self._hours, key=lambda k: abs(k - fh) * 1000 - k)
                if old == fh:
                    break
                del self._hours[old]
                self._hsurf.pop(old, None)
        return d

    def _setup_hrrr_grid(self, g: dict):
        self.hp = Lambert(g["LaDInDegrees"], g["LoVInDegrees"], g["Latin1InDegrees"])
        self.hx0, self.hy0 = self.hp.fwd(g["latitudeOfFirstGridPointInDegrees"],
                                         (g["longitudeOfFirstGridPointInDegrees"] + 180.0) % 360.0 - 180.0)
        self.hdx = g["DxInMetres"] * STRIDE
        # the HRRR's points of the 25 km grid (the maps), once
        fi, fj = self._hidx(self.X, self.Y)
        first = self._hours[self.start_fh]["sfc"]
        self._s25 = _Bilinear(first.shape[-2:], fi, fj, periodic=False)
        hny, hnx = first.shape[-2:]
        self._inside = ((fi >= 0) & (fi <= hnx - 1) & (fj >= 0) & (fj <= hny - 1)).astype(np.float64)
        self._zs25 = self._s25(first[SFC.index(("HGT", "surface"))].astype(np.float64))     # the ground (m)

    def _hidx(self, X, Y):
        lat, lon = self.proj.inv(X, Y)
        x, y = self.hp.fwd(lat, lon)
        return (x - self.hx0) / self.hdx, (y - self.hy0) / self.hdx

    def _rotation(self, X, Y):
        """(cos, sin) turning the HRRR's grid-relative winds into the world grid's at ground points (X, Y):
        the angle between the two grids' north, from both projections numerically."""
        lat, lon = self.proj.inv(X, Y)
        hx, hy = self.hp.fwd(lat, lon)
        hx2, hy2 = self.hp.fwd(lat + 0.02, lon)
        wx2, wy2 = self.proj.fwd(lat + 0.02, lon)
        ah = np.arctan2(hx2 - hx, hy2 - hy)            # north's angle from the HRRR grid's +y (clockwise +)
        aw = np.arctan2(wx2 - X, wy2 - Y)              # ... from the world grid's +y
        d = ah - aw
        return np.cos(d), np.sin(d)

    def _turn(self, u, v, c, s):
        """Grid-relative (HRRR) -> world grid-relative: rotate by the angle between the grids."""
        return u * c - v * s, u * s + v * c

    # ------------------------------------------------------------ time
    def calendar(self, t: float) -> _dt.datetime:
        return self.start + _dt.timedelta(seconds=float(t))

    def _fh_at(self, t: float):
        a = self.start_fh + t / 3600.0
        a = min(max(a, self.start_fh), self.fh_last)
        h0 = int(math.floor(a))
        return h0, min(h0 + 1, self.fh_last), a - h0

    def advance_to(self, t: float) -> None:
        h0, h1, _ = self._fh_at(t)
        self._hour(h0)
        self._hour(h1)

    def _surf(self, fh: int) -> np.ndarray:
        """The 25 km snapshot stack (SynopticWorld.FIELDS) of a forecast hour: what the maps and the fronts read."""
        if fh in self._hsurf:
            return self._hsurf[fh]
        d = self._hour(fh)
        s = self._s25(d["sfc"].astype(np.float64))
        si = dict(zip([f"{a}:{b}" for a, b in SFC], s))
        ps = si["PRES:surface"]
        T2 = si["TMP:2 m above ground"]
        Td = si["DPT:2 m above ground"]
        e = 611.2 * np.exp(17.67 * (Td - 273.15) / (Td - 29.65))
        q2 = EPS * e / np.maximum(ps - e, 1.0)
        c, sn = self._rotation(self.X, self.Y)
        us, vs = self._turn(si["UGRD:10 m above ground"], si["VGRD:10 m above ground"], c, sn)
        w700 = -si["VVEL:700 mb"] / (0.9 * G)                       # omega (Pa/s) -> w (m/s)
        zero = np.zeros(self.X.shape)
        vals = {"th": T2 * (P0 / ps) ** KAPPA, "qv": q2, "ps": si["MSLMA:mean sea level"], "us": us, "vs": vs,
                "usf": us, "vsf": vs, "w": np.clip(w700, -0.5, 0.5), "ctt": T2, "tmid": zero, "te": zero}
        stack = np.stack([np.asarray(vals.get(k, zero), np.float32) for k in self.FIELDS])
        # outside the HRRR's domain (the oceans' far corners): the nearest values are already there (clamped)
        self._hsurf[fh] = stack
        return stack

    def _fields_at(self, t: float):
        h0, h1, w = self._fh_at(t)
        s = self._surf(h0) if w < 1e-6 else (1 - w) * self._surf(h0) + w * self._surf(h1)
        s = s.copy()
        k = self.FIELDS.index("prc")
        s[k] = self.fb_rain                           # the run's own storms' rain (ingest_storms)
        return s

    # ------------------------------------------------------------ columns
    def _point_fields(self, X, Y, t):
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        h0, h1, w = self._fh_at(t)
        fi, fj = self._hidx(X, Y)

        def sample(fh):
            d = self._hour(fh)
            return _sample(d["levels"], fi, fj).astype(np.float64), _sample(d["sfc"], fi, fj).astype(np.float64)

        L0, S0 = sample(h0)
        if w > 1e-6 and h1 != h0:
            L1, S1 = sample(h1)
            L0, S0 = (1 - w) * L0 + w * L1, (1 - w) * S0 + w * S1
        c, s = self._rotation(X, Y)
        P = {"L": L0, "S": S0, "rot": (c, s)}
        geo = self.__dict__.get("_geo")
        if geo is None:
            geo = self._geo = np.stack([self.lon, self.lat, self.land, self.desert, self.plains, self.east])
        g = _sample(geo, (X - self.x0) / DX, (Y - self.y0) / DX)
        P.update(lon=g[0], lat=g[1], land=g[2], desert=g[3], plains=g[4], east=g[5])
        return P

    def _build(self, P, t, zc=ZC, pb=None):
        """Columns on heights zc (m above the HRRR's own ground): its levels above ground, from its 2 m / 10 m
        surface up, interpolated in height; above 100 hPa, isothermal.  ``pb`` (pressure on zc): the storm
        patches' own pressure -- temperature and relative humidity kept, theta and qv for that pressure."""
        L, S = P["L"], P["S"]                          # (5, 25, N), (8, N)
        c, s = P["rot"]
        hgt, tmp, sph, uu, vv = L
        si = {f"{a}:{b}": S[k] for k, (a, b) in enumerate(SFC)}
        psfc, zsfc = si["PRES:surface"], si["HGT:surface"]
        T2, Td2 = si["TMP:2 m above ground"], si["DPT:2 m above ground"]
        e2 = 611.2 * np.exp(17.67 * (Td2 - 273.15) / (Td2 - 29.65))
        q2 = EPS * e2 / np.maximum(psfc - e2, 1.0)
        u10, v10 = self._turn(si["UGRD:10 m above ground"], si["VGRD:10 m above ground"], c, s)
        ul, vl = self._turn(uu, vv, c, s)
        plev = np.asarray(LEVELS, np.float64)[:, None] * 100.0
        zl = hgt - zsfc[None, :]                       # (25, N) above ground
        above = (plev < psfc[None, :]) & (zl > 5.0)
        zl = np.where(above, zl, -1.0)                 # below ground: out of the way (sorted first)
        qlev = sph / np.maximum(1.0 - sph, 1e-6)
        # surface + levels, sorted by height per column
        Z = np.vstack([np.zeros((1, zl.shape[1])), zl]).T                  # (N, 26)
        Tz = np.vstack([T2[None], tmp]).T
        Qz = np.vstack([q2[None], qlev]).T
        Uz = np.vstack([u10[None], ul]).T
        Vz = np.vstack([v10[None], vl]).T
        Pz = np.vstack([psfc[None], np.broadcast_to(plev, zl.shape)]).T
        order = np.argsort(Z, axis=1, kind="stable")
        Z, Tz, Qz, Uz, Vz, Pz = (np.take_along_axis(a, order, 1) for a in (Z, Tz, Qz, Uz, Vz, Pz))
        N, M = Z.shape[0], len(zc)
        zq = np.asarray(zc, np.float64)
        kk = (Z[:, :, None] <= zq[None, None, :]).sum(1) - 1               # (N, M): the level below
        kk = np.clip(kk, 0, Z.shape[1] - 2)
        rows = np.arange(N)[:, None]
        za, zb = Z[rows, kk], Z[rows, kk + 1]
        f = np.clip((zq[None, :] - za) / np.maximum(zb - za, 1.0), 0.0, 1.0)
        lin = lambda A: A[rows, kk] * (1 - f) + A[rows, kk + 1] * f
        T, qv, u, v = lin(Tz), lin(Qz), lin(Uz), lin(Vz)
        p = np.exp(np.log(Pz[rows, kk]) * (1 - f) + np.log(Pz[rows, kk + 1]) * f)
        # above the top level: isothermal, the pressure falling hydrostatically, winds held
        ztop, Ttop, ptop = Z[:, -1:], Tz[:, -1:], Pz[:, -1:]
        hi = zq[None, :] > ztop
        T = np.where(hi, Ttop, T)
        p = np.where(hi, ptop * np.exp(-G * (zq[None, :] - ztop) / (RD * Ttop)), p)
        qv = np.where(hi, np.minimum(Qz[:, -1:], 3e-6), qv)
        rh = np.clip(qv / qv_sat(p, T), 0.0, 1.0)
        if pb is not None:
            # the storm patches' own pressure (the world is flat): T and RH height for height, and never near
            # saturation above the boundary layer (the patches get the air, not its clouds)
            p = np.broadcast_to(np.asarray(pb, np.float64)[None, :], T.shape)
            cap = np.where(zq[None, :] > 1500.0, RH_MAX, 0.95)
            rh = np.minimum(rh, cap)
            qv = rh * qv_sat(p, T)
            qv = np.where(T < 273.15, np.minimum(qv, RH_MAX * qv_sat_ice(p, T)), qv)
        qv = np.maximum(qv, 1e-7)
        th = T * (P0 / p) ** KAPPA
        return {"p": np.array(p), "T": T, "th": th, "qv": qv, "u": u, "v": v, "h": np.full(N, 1000.0),
                "z_trop": np.full(N, 12000.0)}

    def columns_for_model(self, X, Y, t: float, z, pb) -> dict:
        """Columns for the storm patches: on their levels z, with their pressure pb (see _build)."""
        X = np.asarray(X, np.float64).ravel()
        Y = np.asarray(Y, np.float64).ravel()
        P = self._point_fields(X, Y, t)
        C = self._build(P, t, np.asarray(z, np.float64), pb)
        return {k: C[k] for k in ("p", "T", "th", "qv", "u", "v")}

    def columns(self, X, Y, t: float, z=None) -> dict:
        X = np.asarray(X, np.float64).ravel()
        Y = np.asarray(Y, np.float64).ravel()
        P = self._point_fields(X, Y, t)
        zz = ZC if z is None else np.asarray(z, np.float64)
        C = self._build(P, t, zz)
        return dict(C, z=zz)

    # ------------------------------------------------------------ maps
    def diagnostics(self, t: float) -> dict:
        """SynopticWorld's, with the HRRR's own 500 hPa height and wind (the columns run above the ground, so
        theirs would show the terrain)."""
        out = super().diagnostics(t)
        if out.get("_z500") is None:
            Xd, Yd = self.X[self.dsl], self.Y[self.dsl]
            h0, h1, w = self._fh_at(round(t / DT) * DT)
            fi, fj = self._hidx(Xd.ravel(), Yd.ravel())
            k = LEVELS.index(500)

            def at(fh):
                L = self._hour(fh)["levels"]
                return _sample(np.stack([L[0, k], L[3, k], L[4, k]]), fi, fj).astype(np.float64)

            v = at(h0) if w < 1e-6 or h1 == h0 else (1 - w) * at(h0) + w * at(h1)
            u5, v5 = self._turn(v[1], v[2], *self._rotation(Xd.ravel(), Yd.ravel()))
            out.update(z500=v[0].reshape(Xd.shape), u500=u5.reshape(Xd.shape), v500=v5.reshape(Xd.shape), _z500=True)
            # the HRRR's own storms in the coming hour (its hour-max updraft, in the next hour's file, within
            # ~27 km) and the most-unstable CAPE right here: a storm starts in unstable air near the HRRR's,
            # not in the rain-cooled air under it, where it would die
            s = self._hour(h1 if h1 != h0 else h0)["sfc"]
            uvv = _maxfilter(np.nan_to_num(s[SFC.index(("MAXUVV", "100-1000 mb above ground"))]), 3)
            mu = np.nan_to_num(s[SFC.index(("CAPE", "255-0 mb above ground"))])
            g = _sample(np.stack([uvv, mu]), fi, fj).astype(np.float64)
            out["hrrr_uvv"] = g[0].reshape(Xd.shape)
            out["mucape"] = g[1].reshape(Xd.shape)
            out["ci"] = self._ci(out)
        return out

    def _ci(self, d):
        """SynopticWorld's initiation, and where the HRRR itself makes storms in the coming hour (resolved
        updrafts over ~3 m/s) with CAPE around: ours may start there too (they grow by the run's own physics)."""
        base = super()._ci(d)
        uvv = d.get("hrrr_uvv")
        if uvv is None:
            return base
        cape = np.maximum(d["mlcape"], d.get("mucape", 0.0))
        hrrr = (2.4 * np.clip((uvv - 2.0) / 6.0, 0.0, 1.0) * _ss((cape - 100.0) / 800.0)
                * (d["land"] + (1 - d["land"]) * 0.4))           # over water, fewer (they cost a patch too)
        return np.maximum(base, hrrr)

    def analysis(self, t: float) -> dict:
        """SynopticWorld's, without the 'fronts' that are only the terrain: on the high West's slopes the 2 m
        theta jumps with the ground's height."""
        an = super().analysis(t)
        if not an.get("_terrain"):
            keep = []
            for fr in an["fronts"]:
                P = np.asarray(fr["points"], np.float64) * 1000.0
                z = _sample(self._zs25[None], (P[:, 0] - self.x0) / DX, (P[:, 1] - self.y0) / DX)[0]
                if (z > 1400.0).mean() < 0.35:
                    keep.append(fr)
            an["fronts"] = keep
            an["_terrain"] = True
        return an

    # ------------------------------------------------------------ the run's storms
    def ingest_storms(self, X, Y, th0, q0, rain, z0: float, t: float) -> None:
        """The HRRR's air is fixed: the storms' cold pools stay in their patches; their rain shows on the maps."""
        X = np.asarray(X, np.float64).ravel()
        Y = np.asarray(Y, np.float64).ravel()
        i = np.clip(np.rint((X - self.x0) / DX).astype(int), 0, self.nx - 1)
        j = np.clip(np.rint((Y - self.y0) / DX).astype(int), 0, self.ny - 1)
        self.fb_rain *= 0.7
        self.fb_rain[j, i] = np.maximum(self.fb_rain[j, i], np.asarray(rain, np.float64).ravel())
        self.fb_w[j, i] = 1.0
        self._pay = {}                                 # the maps show the new rain

    def _step(self):                                   # nothing to step: the HRRR is the weather
        self.n += 1


def from_settings(proj, start, cfg, seed, progress=None) -> HrrrWorld:
    return HrrrWorld(proj, start, cfg, seed, progress=progress)
