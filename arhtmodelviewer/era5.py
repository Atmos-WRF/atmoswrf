"""ERA5 as the weather: ECMWF's reanalysis (0.25 deg, hourly, 1940 to a few months ago) supplies the environment
of any past day, and ArWen makes the storms -- past outbreaks seen through the run's own physics.

The same world as the HRRR's (hrrr.HrrrWorld: its maps, fronts, initiation, soundings and storm-patch columns),
on ERA5's data instead: temperature, moisture, wind and height on its pressure levels from 1000 to 100 hPa and
the surface (pressure, ground height, 2 m temperature and dewpoint, 10 m wind, sea-level pressure).  What ERA5
does not carry: storms of its own (a 30 km model's convection is parameterized), so storms start only where
the environment lets them (CAPE, CIN and lift from its fronts, convergence and ascent); and the 700 hPa
vertical motion the initiation reads is the kinematic one (the continuity equation on its own winds).

The data: Google's public ERA5 (ARCO Zarr on Cloud Storage: no account), read with gpuwm's native Zarr reader,
~30 s an hour for the whole map, kept on disk per hour.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import math
import tempfile
import threading
from pathlib import Path

import numpy as np

from . import qg
from .hrrr import HrrrWorld, LEVELS, SFC
from .synoptic import CACHE, DEFAULTS, SEASON, _Bilinear

ECACHE = CACHE / "era5"
FIRST = _dt.datetime(1940, 1, 1, tzinfo=_dt.timezone.utc)
G0 = 9.80665
R_EARTH = 6371229.0
PRESSURE = {"geopotential": ("Z", "m2 s-2"), "temperature": ("T", "K"), "specific_humidity": ("SPFH", "kg kg-1"),
            "u_component_of_wind": ("U", "m s-1"), "v_component_of_wind": ("V", "m s-1")}
SURFACE = {"surface_pressure": ("PSFC", "Pa"), "geopotential_at_surface": ("SOILGEO", "m2 s-2"),
           "2m_temperature": ("T2", "K"), "2m_dewpoint_temperature": ("D2", "K"),
           "10m_u_component_of_wind": ("U10", "m s-1"), "10m_v_component_of_wind": ("V10", "m s-1"),
           "mean_sea_level_pressure": ("PMSL", "Pa")}
_FETCH = threading.Lock()        # one download at a time (the run and the prefetch ask for the same hours)


def _levels_from(src_hpa: np.ndarray, a: np.ndarray) -> np.ndarray:
    """(n_src, ny, nx) on ERA5's levels -> (len(LEVELS), ny, nx) on the HRRR world's (725 hPa, which ERA5 lacks:
    linear in log p between 700 and 750)."""
    lp = np.log(src_hpa)
    out = np.empty((len(LEVELS),) + a.shape[1:], np.float32)
    for k, p in enumerate(LEVELS):
        hit = np.where(src_hpa == p)[0]
        if len(hit):
            out[k] = a[hit[0]]
            continue
        lo = np.where(src_hpa > p)[0]
        hi = np.where(src_hpa < p)[0]
        i0 = lo[np.argmin(src_hpa[lo])]
        i1 = hi[np.argmax(src_hpa[hi])]
        f = (math.log(p) - lp[i0]) / (lp[i1] - lp[i0])
        out[k] = (1 - f) * a[i0] + f * a[i1]
    return out


def _omega700(lat: np.ndarray, lon: np.ndarray, p_hpa: np.ndarray, u: np.ndarray, v: np.ndarray,
              psfc: np.ndarray) -> np.ndarray:
    """The kinematic 700 hPa omega (Pa/s): the horizontal divergence summed from the ground up (continuity, no
    vertical motion at the ground), levels under the ground left out.  Negative = rising."""
    dlam = math.radians(float(lon[1] - lon[0]))
    dphi = math.radians(float(lat[1] - lat[0]))
    cphi = np.cos(np.radians(lat))[None, :, None]
    dudx = np.gradient(u, axis=2) / (R_EARTH * cphi * dlam)
    dvdy = np.gradient(v * cphi, axis=1) / (R_EARTH * cphi * dphi)
    div = dudx + dvdy                                     # (n, ny, nx), 1/s
    om = np.zeros(psfc.shape)
    order = np.argsort(-p_hpa)                            # from the bottom up
    for a, b in zip(order[:-1], order[1:]):
        pa, pb = p_hpa[a] * 100.0, p_hpa[b] * 100.0
        if p_hpa[a] < 700.0:
            break
        top = max(pb, 70000.0)
        # the part of the layer above the ground
        lo = np.minimum(pa, psfc)
        dp = np.clip(lo - top, 0.0, None)
        om += 0.5 * (div[a] + div[b]) * dp
    # the 0.25 deg divergence is noisy: a light smoothing (3 x 3)
    p = np.pad(om, 1, mode="edge")
    return sum(p[i:i + om.shape[0], j:j + om.shape[1]] for i in range(3) for j in range(3)) / 9.0


def fetch_hour(valid: _dt.datetime, area: tuple, progress=None) -> dict:
    """One hour of ERA5 over area (south, west, north, east): {"levels": (5, len(LEVELS), ny, nx), "sfc":
    (len(SFC), ny, nx), "grid": {"lat0", "lon0", "d"}} in the HRRR world's layout, latitudes increasing, float32
    (from the disk cache when there)."""
    valid = valid.astimezone(_dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    tag = hashlib.sha1(repr(tuple(round(a, 2) for a in area)).encode()).hexdigest()[:8]
    path = ECACHE / f"{valid:%Y%m%d%H}_{tag}.npz"
    with _FETCH:
        if path.is_file():
            try:
                with np.load(path) as z:
                    return {"levels": z["levels"], "sfc": z["sfc"], "grid": dict(zip(z["gk"].tolist(), z["gv"].tolist()))}
            except Exception:
                pass
        from netCDF4 import Dataset
        from gpuwm import fetch
        from gpuwm.era5_arco import ARCO_STORE
        from gpuwm.zarr_bridge import extract_regular_zarr
        if progress:
            progress(0.1, f"ERA5 {valid:%Y-%m-%d %HZ}: downloading")
        req = {"store": ARCO_STORE, "times": [valid.isoformat() + "Z"], "area": list(area),
               "expected_levels_hpa": list(fetch.ERA5_PRESSURE_LEVELS_HPA), "specific_humidity_authority": True,
               "fields": [{"source": k, "output": o, "units": u, "pressure": pr}
                          for m, pr in ((PRESSURE, True), (SURFACE, False)) for k, (o, u) in m.items()],
               "derived": []}
        ECACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".era5-", dir=ECACHE) as tmp:
            nc = Path(tmp) / "hour.nc"
            try:
                extract_regular_zarr(req, request_path=Path(tmp) / "request.json", output=nc, progress=lambda m: None)
            except Exception as exc:
                raise RuntimeError(f"ERA5 {valid:%Y-%m-%d %HZ} could not be read from Google's archive (no internet, "
                                   f"or a time it does not hold yet: ERA5 runs from 1940 to a few months ago): {exc}") from exc
            with Dataset(nc) as ds:
                lat = np.asarray(ds["latitude"][:], np.float64)
                lon = np.asarray(ds["longitude"][:], np.float64)
                lev = np.asarray(ds["level"][:], np.float64)
                rd = lambda k: np.asarray(ds[k][0], np.float64)
                f = {k: rd(k) for k in ("Z", "T", "SPFH", "U", "V", "PSFC", "SOILGEO", "T2", "D2", "U10", "V10", "PMSL")}
        if lat[0] > lat[-1]:                              # rows south to north, like the HRRR's
            lat = lat[::-1]
            f = {k: (a[:, ::-1] if a.ndim == 3 else a[::-1]) for k, a in f.items()}
        lon = (lon + 180.0) % 360.0 - 180.0
        if progress:
            progress(0.8, f"ERA5 {valid:%Y-%m-%d %HZ}: decoding")
        levels = np.stack([_levels_from(lev, f["Z"] / G0), _levels_from(lev, f["T"]), _levels_from(lev, f["SPFH"]),
                           _levels_from(lev, f["U"]), _levels_from(lev, f["V"])])
        sv = {("PRES", "surface"): f["PSFC"], ("HGT", "surface"): f["SOILGEO"] / G0,
              ("TMP", "2 m above ground"): f["T2"], ("DPT", "2 m above ground"): f["D2"],
              ("UGRD", "10 m above ground"): f["U10"], ("VGRD", "10 m above ground"): f["V10"],
              ("MSLMA", "mean sea level"): f["PMSL"],
              ("VVEL", "700 mb"): _omega700(lat, lon, lev, f["U"], f["V"], f["PSFC"])}
        # no storms of its own (MAXUVV) and no CAPE field: initiation from the environment alone
        sfc = np.stack([np.asarray(sv.get(k, np.zeros_like(f["PSFC"])), np.float32) for k in SFC])
        grid = {"lat0": float(lat[0]), "lon0": float(lon[0]), "d": float(lat[1] - lat[0])}
        tmpz = path.with_suffix(".part.npz")
        np.savez_compressed(tmpz, levels=levels, sfc=sfc, gk=np.array(list(grid)), gv=np.array(list(grid.values())))
        tmpz.replace(path)
        return {"levels": levels, "sfc": sfc, "grid": grid}


class Era5World(HrrrWorld):
    """HrrrWorld on ERA5: hour 0 is the start the page picked, any hour of any day since 1940."""

    def __init__(self, proj, start: _dt.datetime, cfg: dict | None = None, seed: int = 1, progress=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        h = self.cfg.get("hrrr") or {}
        self.cycle = _dt.datetime.fromisoformat(str(h["cycle"]).replace("Z", "+00:00"))
        if self.cycle.tzinfo is None:
            self.cycle = self.cycle.replace(tzinfo=_dt.timezone.utc)
        self.start_fh = int(h.get("start_fh", 0))
        self.fh_last = self.start_fh + max(1, int(h.get("hours", 12)))
        self.proj = proj
        self.start = self.cycle + _dt.timedelta(hours=self.start_fh)
        self.seed = int(seed)
        self.lock = threading.RLock()
        self.progress = progress or (lambda f, msg: None)
        (self.jet_lat, self.sst_gulf, self.dtheta, self.eml_src, self.t500_jet) = SEASON[self.start.month]
        self.keep_hourly = True
        self._snaps, self._diag, self._anal, self.notes = {}, {}, {}, []
        self.q = qg.QGModel()
        self._grids()
        # the map's area, a degree around it, on ERA5's quarter-degree lines
        q4 = lambda v, up: (math.ceil(v * 4) if up else math.floor(v * 4)) / 4.0
        self.area = (q4(float(self.lat.min()) - 1.0, False), q4(float(self.lon.min()) - 1.0, False),
                     q4(float(self.lat.max()) + 1.0, True), q4(float(self.lon.max()) + 1.0, True))
        self._hours = {}
        self._hsurf = {}
        self._hlock = threading.Lock()
        self.n = 0
        self.max_hour = self.fh_last - self.start_fh
        self.scenario_note = (f"ERA5 {self.start:%Y-%m-%d %HZ} (ECMWF reanalysis), {self.max_hour} h: "
                              f"ERA5's environment, the run's own storms")
        first = self._hour(self.start_fh)
        self._setup_hrrr_grid(first["grid"])
        self.progress(1.0, "ready")
        threading.Thread(target=self._prefetch, name="era5-prefetch", daemon=True).start()

    def _hour(self, fh: int, keep: bool = True) -> dict:
        fh = int(min(max(fh, self.start_fh), self.fh_last))
        with self._hlock:
            if fh in self._hours:
                return self._hours[fh]
        d = fetch_hour(self.cycle + _dt.timedelta(hours=fh), self.area, self.progress)
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
        self.glat0, self.glon0, self.gd = g["lat0"], g["lon0"], g["d"]
        fi, fj = self._hidx(self.X, self.Y)
        first = self._hours[self.start_fh]["sfc"]
        self._s25 = _Bilinear(first.shape[-2:], fi, fj, periodic=False)
        hny, hnx = first.shape[-2:]
        self._inside = ((fi >= 0) & (fi <= hnx - 1) & (fj >= 0) & (fj <= hny - 1)).astype(np.float64)
        self._zs25 = self._s25(first[SFC.index(("HGT", "surface"))].astype(np.float64))

    def _hidx(self, X, Y):
        lat, lon = self.proj.inv(X, Y)
        lon = (np.asarray(lon, np.float64) + 180.0) % 360.0 - 180.0
        return (lon - self.glon0) / self.gd, (np.asarray(lat, np.float64) - self.glat0) / self.gd

    def _rotation(self, X, Y):
        """(cos, sin) turning ERA5's east/north winds into the world grid's: the world grid's north angle."""
        lat, lon = self.proj.inv(X, Y)
        wx2, wy2 = self.proj.fwd(lat + 0.02, lon)
        aw = np.arctan2(wx2 - X, wy2 - Y)
        return np.cos(-aw), np.sin(-aw)
