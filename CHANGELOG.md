# Update log

## Atmos-WRF 1.0

The first GitHub release. Before this it was shared as "Atmos-WRF 3.0.zip", and before that as ARHT-MODEL
Viewer (last version 2.1.7). This release has everything from 3.0 (below), plus:

- **Real maps.** A dropdown on the map bar picks Plain, OpenStreetMap, OpenStreetMap dark, Satellite, or
  Satellite + labels (roads and place names). The map sits under the radar and on the 3-D view's ground.
  It needs the internet the first time you look at an area; tiles are then kept for 30 days.
- Version numbering starts over at 1.0 under the Atmos-WRF name.
- Licensed under Apache-2.0; credits in NOTICE (ArWen / gpuwm, WRF, map and terrain data, map tiles).

## Atmos-WRF 3.0

The rename from ARHT-MODEL Viewer to Atmos-WRF (launcher `Atmos-WRF.cmd`), on top of 2.1.7.

### New

- **Tornado list and damage surveys.** Every tornado of a run, named after the biggest town it hit and rated
  strongest first. Each track gets EF contours from the model's winds, exported as Google Earth KML, CSV and a
  GR2Analyst placefile (also live at `/poll/survey.txt`). The list opens by itself when a run with tornadoes ends.
- **Loop maker.** Animated GIFs of any radar's BR / BV / CC / ZDR / SW at any tilt, over any range of volumes,
  framed on the map view, in any color table from your library.
- **Phone remote.** Watch a run from a phone or tablet on your Wi-Fi (Radar tab -> Phone remote). A watch PIN
  to share (live radar, status, tornadoes, warnings, log) and a control PIN that can also pause, resume and stop.
- **Doppler on Wheels.** Deploy a DOW anywhere, pick it up and move it, polled to GR2Analyst like the real
  trucks (X band, 0.93 deg beam, 25-75 m gates, 12 tilts about every 1.5 min). DOW data has no debris balls
  and no attenuation; vortex holes show on the DOW only.

### Changed

- New runs default to the ArWen physics suite (Thompson, YSU, MM5, Noah, RRTMG), with Coriolis and the
  steady environment on.
- The classic storm starts are back: the single round updraft push from the older versions, which grew
  stronger low-level rotation than the newer starts.
- Weaker, more realistic cold pools: Thompson's rain evaporation is scaled down (x0.6).

### Fixed

- GR2Analyst falling up to an hour behind, and skipping volumes in bursts of three. The feed now thins fast
  runs evenly and never lags more than about 90 s.
- Rectangle "seams" at the old edges of growing storm areas when a physics suite runs.
- Cold pools with no storm over them (steady environment + a physics suite), and the whole domain convecting
  when a sounding's surface heating and a land-surface suite heated the ground twice.
- "GLW is stale: its producer (radiation_scheme)..." errors on grids whose time step changes.
- A fixed-size domain that never moved with "Fixed to the ground" grid motion, so storms drove out of it.
