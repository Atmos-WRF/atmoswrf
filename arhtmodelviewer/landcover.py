"""What lies under a tornado: MODIS land use at 30 arc-seconds (~1 km) over the
United States (data/landcover, cut from WRF's WPS geography by
tools/extract_landcover.py), as the debris a tornado can loft there.

The 21 Noah-modified IGBP categories get a debris load: forests give the most
(trees, branches, leaves: the biggest tornado debris signatures of 27 April 2011
were over the forests of Alabama and Mississippi), built-up land almost as much,
woody savanna and farm-and-woodland mosaics a little less, crops, grass and bare
soil less, water next to nothing.  Towns from the population data come on top
(radar.add_debris).
"""
from __future__ import annotations

import io
import lzma
from pathlib import Path

import numpy as np

from . import PROJECT

PATH = PROJECT / "data" / "landcover" / "modis_landuse_us_30s.npz.xz"

NAMES = {1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest", 3: "deciduous needleleaf forest",
         4: "deciduous broadleaf forest", 5: "mixed forest", 6: "closed shrubland", 7: "open shrubland",
         8: "woody savanna", 9: "savanna", 10: "grassland", 11: "wetland", 12: "cropland", 13: "urban",
         14: "cropland / woodland mosaic", 15: "snow and ice", 16: "barren", 17: "water", 18: "wooded tundra",
         19: "mixed tundra", 20: "barren tundra", 21: "lake"}
LOAD = np.full(256, 0.5, np.float32)          # outside the data: what the viewer used everywhere before
LOAD[[1, 2, 3, 4, 5]] = 0.95
LOAD[13] = 0.9
LOAD[[8, 14]] = 0.75
LOAD[[6, 9, 18]] = 0.6
LOAD[[7, 12, 16]] = 0.45
LOAD[[10, 11, 19, 20]] = 0.4
LOAD[15] = 0.15
LOAD[[17, 21]] = 0.08
# Surface roughness length z0 (m) per category: WRF's LANDUSE.TBL for MODIS, summer values (forests
# 0.5, built-up land 0.8, crops 0.15, grass 0.12, water 0.0001).  Outside the data: 0.1, what the
# viewer's single z0 used to be everywhere.
Z0 = np.full(256, 0.1, np.float32)
Z0[[1, 2, 3, 4, 5]] = 0.5
Z0[6] = 0.10
Z0[7] = 0.06
Z0[8] = 0.05
Z0[9] = 0.15
Z0[10] = 0.12
Z0[11] = 0.30
Z0[12] = 0.15
Z0[13] = 0.80
Z0[14] = 0.14
Z0[15] = 0.001
Z0[16] = 0.01
Z0[[17, 21]] = 0.0001
Z0[18] = 0.30
Z0[19] = 0.15
Z0[20] = 0.10


class LandCover:
    def __init__(self, path: Path = PATH):
        z = np.load(io.BytesIO(lzma.decompress(path.read_bytes())))
        self.cat = z["category"]
        self.lat0, self.lon0, self.d = float(z["lat0"]), float(z["lon0"]), float(z["d"])

    def category(self, lat, lon) -> np.ndarray:
        """Category at each point (0 outside the data)."""
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        j = np.rint((lat - self.lat0) / self.d).astype(np.int64)
        i = np.rint((lon - self.lon0) / self.d).astype(np.int64)
        ny, nx = self.cat.shape
        ok = (j >= 0) & (j < ny) & (i >= 0) & (i < nx)
        out = np.zeros(lat.shape, np.uint8)
        out[ok] = self.cat[j[ok], i[ok]]
        return out

    def debris_load(self, lat, lon) -> np.ndarray:
        """How much debris a tornado lofts at each point (0.08 water .. 0.95 forest)."""
        return LOAD[self.category(lat, lon)]

    def roughness(self, lat, lon) -> np.ndarray:
        """Surface roughness length z0 (m) at each point (Z0)."""
        return Z0[self.category(lat, lon)]


_LOADED: dict = {}


def load() -> LandCover | None:
    """The land cover, if the viewer has it (None otherwise: debris then knows only towns)."""
    if "lc" not in _LOADED:
        _LOADED["lc"] = LandCover() if PATH.is_file() else None
    return _LOADED["lc"]
