# Atmos-WRF 1.1

**New:** 3-D view upgrades: free camera and a from-the-ground chaser camera, a much more realistic cloud renderer (smooth billowy towers, rain curtains, proper sunlight and shadows, Detail and Rain sliders), replay (go back in time in 3-D like the radar, and it follows the map's replay bar), and the real landscape (3-D trees from ESA WorldCover, real roads from OpenStreetMap, real land cover or sharp satellite ground around the camera)

**Fixed:**
- the 3-D view (and the page) taking forever to load while a tornado zoom is running

# Atmos-WRF 1.0

**New:** tornado list (intensity, path length, etc) with damage surveys (KML, CSV and GR2 placefile), loop maker, phone remote (app you can run on phone to both view and manage the sim), DOW data. OSM and satellite maps, under the radar and on the 3-D view’s ground, Apache-2.0 license and credits.

**Changed:** physics suite, Coriolis and steady environment on by default, the classic storm starts back, and weaker cold pools

**Fixed:**
- GR2 lag and skipped volumes
- suite growth seams
- cold pools with no storm over them, and the double surface heating
- the GLW stale error
- the fixed domain not moving
