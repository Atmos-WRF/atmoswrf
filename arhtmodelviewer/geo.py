"""Geography: the viewer's Lambert conformal map, real radar sites, and map
lines.

The model world is a flat plane in Lambert conformal metres (WRF's sphere,
R = 6370 km) centred on a reference point the user picks; world (0, 0) is
that point.  Everything comes from the viewer's own data folder, built from
public-domain sources by data/build_geodata.py (data/basemap/SOURCES.md):
map lines (states, counties, coastlines, lakes, borders) from Natural Earth
10m and the US Census 5m counties, radar sites from NOAA NCEI's WSR-88D/TDWR
list, cities from the Census Gazetteer and population estimates.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

R_EARTH = 6370000.0

PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / "output" / "cache"
DATA = PROJECT / "data"

BASEMAP = DATA / "basemap" / "basemap.npz"
RADARS = DATA / "radars.csv"
CITIES = DATA / "cities.csv"


class Lambert:
    """Lambert conformal conic on a sphere, origin at (lat0, lon0)."""

    def __init__(self, lat0: float, lon0: float, truelat1: float | None = None,
                 truelat2: float | None = None):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        if truelat1 is None:
            truelat1 = lat0 - 5.0 if lat0 >= 0 else lat0 + 5.0
        if truelat2 is None:
            truelat2 = lat0 + 5.0 if lat0 >= 0 else lat0 - 5.0
        self.t1, self.t2 = float(truelat1), float(truelat2)
        p1, p2 = math.radians(self.t1), math.radians(self.t2)
        if abs(p1 - p2) < 1e-9:
            n = math.sin(p1)
        else:
            n = (math.log(math.cos(p1) / math.cos(p2))
                 / math.log(math.tan(math.pi / 4 + p2 / 2) / math.tan(math.pi / 4 + p1 / 2)))
        self.n = n
        self.F = math.cos(p1) * math.tan(math.pi / 4 + p1 / 2) ** n / n
        self.rho0 = R_EARTH * self.F / math.tan(math.pi / 4 + math.radians(self.lat0) / 2) ** n

    def fwd(self, lat, lon):
        lat = np.radians(np.asarray(lat, dtype=np.float64))
        dlon = (np.asarray(lon, dtype=np.float64) - self.lon0 + 180.0) % 360.0 - 180.0
        rho = R_EARTH * self.F / np.tan(np.pi / 4 + lat / 2) ** self.n
        th = self.n * np.radians(dlon)
        return rho * np.sin(th), self.rho0 - rho * np.cos(th)

    def inv(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        dy = self.rho0 - y
        sgn = 1.0 if self.n > 0 else -1.0
        rho = sgn * np.hypot(x, dy)
        th = np.arctan2(sgn * x, sgn * dy)
        lat = 2.0 * np.arctan((R_EARTH * self.F / rho) ** (1.0 / self.n)) - np.pi / 2
        lon = self.lon0 + np.degrees(th / self.n)
        return np.degrees(lat), lon

    def convergence(self, lon) -> np.ndarray:
        """Angle (deg) from true north to grid north at longitude ``lon``."""
        return self.n * ((np.asarray(lon) - self.lon0 + 180.0) % 360.0 - 180.0)

    def north_angle(self, lon) -> np.ndarray:
        """Angle (deg) of TRUE north clockwise from grid north (+y) at longitude ``lon``: 0 on the central meridian,
        ~13 deg in California on a map centred on Oklahoma, ~70 over Japan (checked against fwd: 2026-11-09)."""
        return -self.convergence(lon)

    def to_grid(self, u, v, lon):
        """Earth-relative wind (u east, v north) -> grid-relative (along +x, +y) at longitude ``lon``."""
        a = np.radians(self.north_angle(lon))
        return u * np.cos(a) + v * np.sin(a), -u * np.sin(a) + v * np.cos(a)

    def to_earth(self, u, v, lon):
        """Grid-relative wind -> earth-relative (u east, v north) at longitude ``lon``."""
        a = np.radians(self.north_angle(lon))
        return u * np.cos(a) - v * np.sin(a), u * np.sin(a) + v * np.cos(a)

    def to_json(self) -> dict:
        return {"lat0": self.lat0, "lon0": self.lon0, "truelat1": self.t1,
                "truelat2": self.t2, "n": self.n, "F": self.F, "rho0": self.rho0,
                "R": R_EARTH}


# ------------------------------------------------------------- radar sites
def _read_csv(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def radar_sites() -> list[dict]:
    """Every radar site with lat/lon/elevation (m): WSR-88D (type 1), TDWR (3), and the world's other
    weather radars (4 = S band, 5 = C band; from the nexradar app's tables, data/radars.csv)."""
    out = []
    for r in _read_csv(RADARS):
        try:
            out.append({"icao": r["icao"].upper(), "lat": float(r["lat"]), "lon": float(r["lon"]),
                        "elev": float(r["elev_m"]), "type": int(r["type"]),
                        "state": r["state"], "name": r["name"]})
        except (KeyError, ValueError):
            continue
    return out


def cities() -> list[dict]:
    """US cities and towns (incorporated places) with their population."""
    out = []
    for r in _read_csv(CITIES):
        try:
            out.append({"name": r["name"], "state": r["state"], "lat": float(r["lat"]),
                        "lon": float(r["lon"]), "pop": float(r["pop"])})
        except (KeyError, ValueError):
            continue
    return out


# --------------------------------------------------------------- map lines
# (layer, thinning tolerance km)
_LAYERS = [("counties", 0.35), ("states", 0.5), ("borders", 0.8), ("coast", 0.6), ("lakes", 0.6)]


def _basemap() -> dict:
    """{layer: (lon/lat vertices (N, 2) deg, part starts (P + 1,), part bboxes
    (P, 4) lon0, lat0, lon1, lat1)} from data/basemap/basemap.npz."""
    out = {}
    try:
        z = np.load(BASEMAP)
    except OSError:
        return out
    with z:
        scale = float(z["scale"])
        for name, _ in _LAYERS:
            if name + "_d" not in z:
                continue
            ll = np.cumsum(z[name + "_d"], axis=0, dtype=np.int64) * scale
            starts = np.concatenate([[0], np.cumsum(z[name + "_n"], dtype=np.int64)])
            lo = np.minimum.reduceat(ll, starts[:-1], axis=0)
            hi = np.maximum.reduceat(ll, starts[:-1], axis=0)
            out[name] = (ll, starts, np.hstack([lo, hi]))
    return out


def _simplify(xy: np.ndarray, tol: float) -> np.ndarray:
    """Radial-distance thinning (fast, keeps shape at map scales)."""
    if len(xy) <= 2:
        return xy
    keep = [0]
    last = xy[0]
    d2 = tol * tol
    for k in range(1, len(xy) - 1):
        dxy = xy[k] - last
        if dxy[0] * dxy[0] + dxy[1] * dxy[1] >= d2:
            keep.append(k)
            last = xy[k]
    keep.append(len(xy) - 1)
    return xy[keep]


def map_lines(proj: Lambert, half_width_km: float = 2600.0) -> dict:
    """{layer: [[x0, y0, x1, y1, ...] km, ...]} for every layer, clipped to a
    square of ``half_width_km`` around the projection origin.  Cached."""
    try:
        st = BASEMAP.stat()
        stamp = f"{st.st_size:x}{int(st.st_mtime):x}"
    except OSError:
        stamp = "none"
    key = f"lines_{proj.lat0:.3f}_{proj.lon0:.3f}_{int(half_width_km)}_{stamp}.json"
    cache = CACHE / key
    if cache.is_file():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    lat_c, lon_c = proj.lat0, proj.lon0
    dlat = half_width_km / 111.0 + 2.0
    dlon = half_width_km / (111.0 * max(math.cos(math.radians(lat_c)), 0.2)) + 2.0
    box = (lon_c - dlon, lat_c - dlat, lon_c + dlon, lat_c + dlat)
    layers = _basemap()
    out = {}
    for name, tol in _LAYERS:
        if name not in layers:
            out[name] = []
            continue
        ll, starts, bb = layers[name]
        near = (bb[:, 3] >= box[1]) & (bb[:, 1] <= box[3])
        across = np.zeros(len(bb), dtype=bool)
        for shift in (-360.0, 0.0, 360.0):          # the box may cross the dateline
            across |= (bb[:, 2] + shift >= box[0]) & (bb[:, 0] + shift <= box[2])
        lines = []
        for k in np.flatnonzero(near & across):
            seg = ll[starts[k]:starts[k + 1]]
            x, y = proj.fwd(seg[:, 1], seg[:, 0])
            xy = np.stack([x, y], axis=1) / 1000.0
            inside = (np.abs(xy[:, 0]) < half_width_km) & (np.abs(xy[:, 1]) < half_width_km)
            if not inside.any():
                continue
            xy = _simplify(xy, tol)
            lines.append(np.round(xy.ravel(), 2).tolist())
        out[name] = lines
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, separators=(",", ":")))
    except OSError:
        pass
    return out
