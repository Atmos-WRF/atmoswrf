# Land / sea mask

`ne_land_0p05.npz.xz`: global land mask at 0.05 degrees (plate carree, row 0 at 90 N, column 0 at 180 W),
rasterized from Natural Earth 1:10m land polygons (public domain, naturalearthdata.com) as packaged in
SatPainter's map data. Built by `dev files (removed from AtmosphereSandbox)/tools/build_landmask.py`.
Hurricane mode uses it for where the sea is (the MODIS land cover stops at 20 N).
