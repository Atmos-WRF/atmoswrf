# Land cover

`modis_landuse_us_30s.npz.xz` (an xz-compressed npz): the lower 48 states and their borders (20–55° N, 130–60° W) cut by
`tools\extract_landcover.py` from WRF's static geography `modis_landuse_20class_30s_with_lakes`
(WPS_GEOG, as distributed by NCAR/UCAR for WRF; here from ArWen Desktop's copy): the Noah-modified
21-category IGBP land use from MODIS, 30 arc-seconds (~1 km), categories 1–21 (17 water, 21 lakes,
13 urban). MODIS land cover is NASA data, free of restrictions. The viewer's radar reads it for the
debris a tornado lofts (`arhtmodelviewer\landcover.py`).
