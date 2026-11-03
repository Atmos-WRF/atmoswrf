"""The ground under the storms: GMTED2010 at 30 arc-seconds over the lower 48 (data/terrain), or an idealized ridge
or hill for experiments, as model ground height.

The model's z = 0 is the run sounding's ground (its station elevation, or ``zero_m``): the atmosphere is the
sounding's everywhere at the same height above sea level, so a column over a 1000 m plateau starts 1000 m up the
sounding -- the moist boundary layer of a low-level station is under the plateau's ground.  Ground lower than the
model's zero is held at zero (the sounding says nothing below its ground).

settings "terrain": {"on": true, "kind": "real" | "ridge" | "hill", "zero_m": (default: the sounding's elevation),
                     "scale": 1.0 (x the real heights),
                     ridge / hill: "x_km", "y_km" (centre, ground coordinates), "height_m", "half_width_km",
                     "length_km" (ridge: its length, 0 = endless), "angle_deg" (ridge: the direction its crest runs,
                     degrees from north)}
"""
from __future__ import annotations

import io
import lzma
import math
from pathlib import Path

import numpy as np

from . import PROJECT

PATH = PROJECT / "data" / "terrain" / "gmted2010_us_30s.npz.xz"
_DEM = None


def _dem():
    global _DEM
    if _DEM is None:
        z = np.load(io.BytesIO(lzma.decompress(PATH.read_bytes())))
        _DEM = (z["height"].astype(np.float32), float(z["lat0"]), float(z["lon0"]), float(z["d"]))
    return _DEM


def dem_height(lat, lon) -> np.ndarray:
    """GMTED2010 ground height (m MSL) at (lat, lon), bilinear; 0 outside the data."""
    h, lat0, lon0, d = _dem()
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    gy, gx = (lat - lat0) / d, (lon - lon0) / d
    ny, nx = h.shape
    ok = (gy >= 0) & (gy <= ny - 1) & (gx >= 0) & (gx <= nx - 1)
    gy, gx = np.clip(gy, 0, ny - 1.001), np.clip(gx, 0, nx - 1.001)
    j, i = gy.astype(np.int64), gx.astype(np.int64)
    ty, tx = gy - j, gx - i
    v = ((h[j, i] * (1 - tx) + h[j, i + 1] * tx) * (1 - ty)
         + (h[j + 1, i] * (1 - tx) + h[j + 1, i + 1] * tx) * ty)
    return np.where(ok, v, 0.0)


class Terrain:
    def __init__(self, spec: dict, latlon, zero_m: float, ztop: float):
        self.spec = dict(spec)
        self.kind = str(spec.get("kind", "real"))
        self.latlon = latlon
        self.zero = float(spec["zero_m"]) if spec.get("zero_m") is not None else float(zero_m)
        self.scale = float(spec.get("scale", 1.0))
        self.top = 0.6 * float(ztop)            # gpuwm needs ground below the model top; keep a deep column

    def msl(self, X, Y) -> np.ndarray:
        """Ground height (m MSL) at ground points (X, Y) (m)."""
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        if self.kind == "real":
            lat, lon = self.latlon(X, Y)
            return self.zero + self.scale * (dem_height(lat, lon) - self.zero)
        s = self.spec
        x0, y0 = 1000.0 * float(s.get("x_km", 0.0)), 1000.0 * float(s.get("y_km", 0.0))
        a = 1000.0 * max(float(s.get("half_width_km", 10.0)), 0.5)
        H = float(s.get("height_m", 800.0))
        dx, dy = X - x0, Y - y0
        if self.kind == "hill":
            r2 = (dx * dx + dy * dy) / (a * a)
            return self.zero + H / (1.0 + r2) ** 1.5                     # 3-D witch of Agnesi
        th = math.radians(float(s.get("angle_deg", 0.0)))
        along = dx * math.sin(th) + dy * math.cos(th)                  # along the crest
        across = dx * math.cos(th) - dy * math.sin(th)
        h = H / (1.0 + (across / a) ** 2)                               # witch of Agnesi ridge
        L = 1000.0 * float(s.get("length_km", 0.0))
        if L > 0:
            over = np.maximum(np.abs(along) - 0.5 * L, 0.0)
            h = h * np.exp(-(over / a) ** 2)
        return self.zero + h

    def height(self, X, Y) -> np.ndarray:
        """Model ground height (m above the model's zero, >= 0) at ground points (X, Y)."""
        return np.clip(self.msl(X, Y) - self.zero, 0.0, self.top)

    def describe(self) -> str:
        if self.kind == "real":
            return (f"real terrain (GMTED2010, 30 arc-seconds{'' if self.scale == 1 else f' x {self.scale:g}'}) "
                    f"on the sounding's ground at {self.zero:.0f} m")
        s = self.spec
        return (f"idealized {self.kind}: {float(s.get('height_m', 800)):.0f} m high, half width "
                f"{float(s.get('half_width_km', 10)):.0f} km at ({float(s.get('x_km', 0)):.0f}, "
                f"{float(s.get('y_km', 0)):.0f}) km")
