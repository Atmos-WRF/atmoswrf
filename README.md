Hi, thank you for downloading my model thingy!

Open setup first, then start Atmos-WRF.

Updating: just unzip the new version and start it. Every version shares one setup per PC
(%LOCALAPPDATA%\ARHT-MODEL\runtime), so the 3+ GB is downloaded once; a new version only installs
packages it changed. The first time, the setup of an older version you already have is reused.
This works best on NVIDIA GPUs, but it also has a CPU mode for AMD or no GPU PCs.

New: the Weather tab. Tick "Simulate the atmosphere" and the whole country's weather is simulated
(troughs, cyclones, fronts, drylines, Gulf moisture, the cap, the low-level jet); storms form on their
own where it makes them, no sounding needed. The first preview builds the model's climate once (~3 min).

New in 1.4.6: real days from NOAA's HRRR. In the Atmosphere tab set "Storm environment" to NOAA HRRR and pick
a start time (past, now, or up to ~2 days ahead from the newest run) or a run yourself: the HRRR's air
(fronts, drylines, moisture, cap, jet) is the run's weather, and Atmos-WRF's own storms form in it (the
HRRR's radar is not used). "Drag out the storm domain" picks where storms may form while you look at
STP, CAPE or "+ 500 hPa" over any map. Needs internet (~70 MB per forecast hour, cached). The first
start of this version adds the small HRRR decoder to the shared setup by itself (a few seconds).

New in 3.0:
- Tornado list (map toolbar, and by itself when a run with tornadoes ends): every tornado named after the
  biggest town it hit and rated like an NWS office before a survey, strongest first. With it the damage
  survey: each track's EF contours from the model's winds, as Google Earth KML, a CSV, and a GR2Analyst
  placefile (live: Placefile Manager -> + -> http://127.0.0.1:8830/poll/survey.txt). Saved in the run's
  "survey" folder too.
- Loop maker (map toolbar): an animated GIF of any radar's BR / BV / CC / ZDR / SW at any tilt, over any
  range of its volumes, framed on the map in view, in any color table of your library. GIFs go to output\loops.
- Phone remote (Radar tab -> Phone remote): open the address it shows on a phone or tablet on your Wi-Fi and
  enter a PIN. The watch PIN (to share) shows the live radar, the run's status, tornadoes, warnings and log;
  the control PIN can also pause, resume and stop. "Add to Home Screen" gives it an app icon. Away from home:
  Tailscale on the PC and the phone.
- Doppler on Wheels (Radar tab): park a DOW anywhere, pick it up, move it; X band, 0.93 deg beam, 75 m gates,
  12 tilts every ~1.5 min, polled to GR2Analyst (restart GR2Analyst after each move).
- Less brutal outflow: Thompson's rain evaporation is scaled down (cold pools under storms were too cold).
- A sounding's "Surface heating" is skipped when a physics package with a land surface runs without the steady
  environment (it heats the ground itself; both together fired the whole domain).

New in 2.1: the tornado zoom. During a run, click "Tornado zoom" (map toolbar) and drag a small box (2-10 km)
around a tornado or a low-level mesocyclone: nests down to about 100 m go on it and follow its vortex by
themselves. The map shows the box and the surface pressure minimum's track (orange). "Tornado 3-D" opens it in
the 3-D window from the finest grid: the cloud lit by the sun (Relief lights its lumps and towers), the
pressure drop, the spin and the updrafts, with the track and the air parcels going into the vortex. It costs
speed (the fine grids take many short steps); "x zoom" removes it. The box snaps onto the low-level mesocyclone
near where you drop it and follows it until a tornado's vortex forms, then the vortex.

Detail (next to the button): the zoom's finest grid gets its own vertical levels, set up like CM1 research runs
of tornadoes: layers from 8-25 m at the ground growing slowly upward, and LES turbulence (no boundary-layer
scheme). High: ~110 m. Ultra: ~55 m, 11 m layers at the ground. Extreme: ~37 m, 8 m layers: about real time
on an RTX 4060 for one storm. The run's own grid still matters: storms and their mesocyclones come from it.

Also in 2.1: fewer stalls between model steps (a re-fitted storm area is built ~15x faster, the radar's clear-air
and clutter layer ~5x faster), and the "x real time" at the top is the run's real speed over the last ~20 s (the
radar sweeps, frames and warnings included; the model steps' own speed is in its tooltip). QLCS tornadoes: a
circulation along a line of storms can now be a tornado (before, only one under a storm followed for 25 min).

Also in 2.1: runs stay inside the graphics card's own memory. The viewer measures how much memory each grid
cell really takes in your run and stops storm areas from growing past what the card holds, instead of
Windows spilling the run into normal RAM (several times slower on 8 GB cards). For much more speed, use the
2 km grid: about 8x less work than 1 km.

New in 2.1.7: two lighter tornado-zoom details for weaker cards, Lite (~350 m, a ~20 km box) and Medium (~200 m,
~14 km), on the run's own levels. Change a running tornado zoom's detail from the dropdown (High / Ultra / Extreme / PC explosion)
without redrawing it: it stays on the vortex it follows, keeps its track, and keeps the levels the two details
share (the rest are rebuilt from them); the box keeps its size against each detail's usual one, made smaller if
the GPU can't hold it. The extra map frames a running zoom makes never take more than a fifth of the run's time
(a ~10 m zoom spent half its time on them).

New in 2.1.6: a tornado-zoom box too big for the GPU memory left is made smaller when you place it, so all its
levels fit (the Log says by how much), instead of sitting on "placing..." for good; if a level still has to wait
for memory, the map says so and how much it needs.

New in 2.1.5: radar cores that look like radar cores. With Morrison, WRF's reflectivity saw graupel and hail as
dry ice, so nothing went past ~55 dBZ and storms showed a flat red plateau; wet (melting, or wet-growing just
above the freezing level) graupel and hail now read like water, giving 60-70 dBZ hail cores. Nests take rain,
hail, graupel, snow and cloud in through their edges from the grid around them (only water vapour came in
before), so small tornado-zoom boxes no longer go near echo-free in the middle of a core. The tornado zoom shows
on the map right after you place it (and every 20 s of real time while it runs), and it no longer loses its
levels the first time it follows the vortex on an 8 GB card. Tip: a radar added with "Measurement noise" off
shows the model's clean values (no gate-to-gate grain); the radar list marks it "clean".

New in 2.1.4: the map shows nests at their own resolution. Inside a 500 m, 150 m or 50 m nest (the tornado
zoom's too) every map product is drawn on the nest's cells instead of the patch's, and the readout under the
cursor reads the finest grid there. Running out of GPU memory no longer ends the run: a nest or tornado-zoom
level is only built when the card has room for it, and if the card does fill up anyway the biggest nest is
removed (the run rolls back to its last checkpoint if it was mid-step), a "memory" line says so in the Log, and
the run goes on.
Nest sizes are free now: the tornado-zoom box is whatever you drag (0.05-60 km), the automatic nests 4-400 km
(inner 2-200 km), up to 12 at once. Two new tornado-zoom details, "PC explosion": ~10 m (suction vortices, a
1-2 km box) and ~1 m (debris scale, a box of a few hundred metres; its lowest layer is about 1 m deep). They get
there in 5:1 steps (350 m grid: 70, 14, 7 m and 70, 14, 2.8, 0.93 m). Both are far slower than real time, and the
1 m one takes many GB: a 0.3 km box on a 350 m grid needs ~9 GB, 0.15 km ~2.3 GB. A level that doesn't fit is
not built (the Log says so).

New in 2.1.3: the Move tool (map toolbar). Drag a patch to where you want it: it stays there 15 min, then follows
its storms again (hold Shift when you drop it: locked there until you click Follow in its list). Drag a nest onto
a storm: it follows that storm from then on. Click a storm: a nest is added on it (up to 3 by hand).
The automatic nests and patches got smarter too: storms are ranked by their updraft AND how hard they rotate
(0-3 km updraft helicity), so the supercell or the QLCS mesovortex gets the nest over a stronger plain updraft;
a much more rotating storm takes over the weakest nest; nests sit where their storm is heading (4 min ahead), are
checked every minute instead of every two, and pick up a storm again when its track splits. A patch at its size
limit keeps its nests and the tornado zoom inside it first, then its best ranked storms, instead of the one storm
nearest its centre.

PS: arhtland is goated.
