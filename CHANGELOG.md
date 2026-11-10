# Atmos-WRF 2.0

**New:** Chase mode: driving under your run's live storms, car handling, SUV / sedan / truck, Basic map, Detailed map (OpenDrive world), model-driven 3-D clouds, far cloud, rain and snow, wind pushes and rolls the car, swaying trees, on foot, gas stations with interiors, route to the nearest gas station, other players on foot, night, lightning and thunder, mesonet probe, building damage, debris and dust, sound, minimap with radar, multiplayer, Cloudflare internet link, Chase tab

**Changed:** map-projection wind fix, true-north METAR winds, nests centred on near-ground rotation, model air at a point, FEMA building footprints, OpenFreeMap roads and buildings, smarter OpenStreetMap fetching

**Fixed:**
- teleport spam in low country
- jumps while the detailed map loads
- getting thrown out of buildings
- knocked off the road (old collisions back)
- detailed map not loading
- missed driving keys
- Chase tab internet link

# Atmos-WRF 1.2

**New:** Follow tornado in the 3-D view (locks onto a tornado, or the strongest low-level rotation, and keeps it in the middle through playback, the map's replay and the ground camera), Export loop (save the 3-D view as a WebM / MP4 video, GIF or PNG frames over any range of the saved volumes), 3-D nests at their own resolution (insets, live and in replay), 1990s (legacy) WSR-88D radars (Radar tab, Era), and a live feed for the PMWeather (Minecraft) addon with cloud volumes for drawing

**Fixed:**
- the steady environment's runaway cold, foggy surface layer (the "random huge cold pool")
- the 3-D ground (land cover / satellite) staying stale or black after the box moved while roads were still loading

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
