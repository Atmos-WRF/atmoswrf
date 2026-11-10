"""A model run: ArWen engine + storm starts + radars + frames + Level II.

Everything that touches the GPU happens on the run's own thread; the web
server talks to it through a command queue and reads finished frames and
Level II files, which are plain bytes.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import logging
import math
import os
import queue
import random
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import PROJECT, geo, level2, products, radar as rd, sounding as snd, synoptic as syn
from . import microtweaks

microtweaks.install()                   # Thompson's rain evaporation scaled (the cold-pool fix), before any kernel compiles
from .boundaries import (AirMasses, Boundary, WarmSectorTexture, boundary_from_json, hot_spots, join_at_lows,
                         signed_distance,
                         low_motion)
from .environment import Environment, GRID_PRESETS, PhysicsOptions
from .forcing import StormStart
from .patch import Rect

log = logging.getLogger(__name__)

OUTPUT = PROJECT / "output"

DEFAULT_SETTINGS = {
    "sounding": {"file": "data/soundings/OUN 1999-05-03 2350Z classic Plains outbreak.txt"},
    "grid": "1km",
    # real ground friction: bulk drag on the lowest level over the land cover's own roughness (forest,
    # towns, crops, grass, water: landcover.Z0; "roughness": "uniform" uses z0 everywhere); no Coriolis
    # ArWen's best suite, Thompson + YSU + MM5 + Noah + RRTMG (the user's default, 2026-11-01; was the built-in core
    # with Thompson aerosol-aware); without its Thompson tables or on the CPU backend a run takes the built-in core
    # Coriolis on and the steady environment on (the user's defaults, 2026-11-03)
    "physics": {"coriolis": True, "surface_drag": True, "roughness": "landcover", "z0": 0.1, "mp_physics": 28,
                "suite": "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1", "les": False, "soil_moisture": 0.25,
                "vegetation": 50.0},
    "map": {"lat0": 35.2, "lon0": -97.6},
    "start_time": "1999-05-03T21:30:00Z",
    "speed_limit": 0.0,
    "frame_interval": 60.0,
    "radars": [{"icao": "KTLX", "vcp": 212, "sails": 1, "dealias": True, "volume_min": 2.5, "floor": -5.0, "noise": True,
                "export": True}],
    # every radar: hooks joined to their tips and the echo sharpened (radar.connect_hooks, sharpen_echo)
    "radar_refine": True,
    # "rm", "lm", "mean", "none", [umove, vmove] (m/s), or "follow": from Bunkers right, then the
    # storms' measured motion, checked every grid_follow_min
    "grid_motion": "rm",
    "grid_follow_min": 10.0,
    # storm-following refinement nests (nest.py): up to "max" grids "ratio" x finer (2-4; 0 = none), "size_km"
    # a side, each on one of the strongest storms, two-way (the parent carries what the nest resolves)
    # "inner_ratio": a second level inside each nest, "inner_km" a side, on its low-level mesocyclone (UH03)
    "refine": {"ratio": 0, "size_km": 40.0, "max": 2, "inner_ratio": 0, "inner_km": 16.0},
    # large-scale ascent during the run (largescale.py): peak w_cms cm/s at half of top_km, 0 = none; single-sounding
    # runs only (the synoptic / ERA5 / HRRR weather brings its own)
    "large_scale": {"w_cms": 0.0, "top_km": 10.0},
    # unlimited: patches grow, shift and merge around every storm, each at most max_km a side (0: no
    # limit); fixed: one size_km square that shifts with its storms but never grows, so the GPU
    # memory a run takes is known from the start
    "domain": {"mode": "unlimited", "size_km": 160.0, "max_km": 0.0},
    "storms": [],
    # nudger cells sent at the strongest storm during the run (interactions.py): at at_min, a cell
    # moving with ``motion`` that meets its updraft lead_min later, aim_km ahead (+) or behind (-) it
    "nudgers": [],
    "zones": [],
    "boundaries": [],
    # where boundaries meet (a surface low): moves with its fronts, or at a set speed / heading
    "low": {"auto": True, "speed": 10.0, "heading": 45.0},
    # OutbreakLab mode: storms form on their own -- at random in the open warm sector of a square
    # area around the map centre (per 100,000 km2 per hour), and along the fronts at their own
    # rates -- each started as a cluster of updraft pushes; ArWen, with the microphysics and
    # physics chosen in Setup, decides what becomes of them
    # every storm that starts on its own (warm sector, fronts, initiation areas) gets its own size,
    # push and motion, satellite towers ("towers") and 0-3 later cells sent into it ("mergers");
    # "texture" scales the open warm sector's small variations (boundaries.WarmSectorTexture, 0 = none)
    "auto_storms": {"on": False, "rate": 1.2, "area_km": 400.0, "start_min": 15.0, "end_min": 270.0,
                    "push": 10.0, "radius_km": 12.0, "towers": True, "mergers": True, "texture": 1.0},
    # synoptic mode (synoptic.py): the whole country's weather is simulated -- troughs, cyclones, fronts,
    # drylines, moisture return, the cap, the diurnal cycle and the low-level jet -- instead of one sounding;
    # storms form where it makes them, inside a square of area_km around the map centre, at most max_storms
    # at a time; "scenario" picks the day (synoptic.SCENARIOS) with its cyclone peak_h hours after the start
    "synoptic": dict(syn.DEFAULTS),
    # hurricane mode (hurricane.py): the sea under the run -- a procedural SST map (seed, offset C), the sea's
    # drag, evaporation and heat, a mixed-layer cold wake (mld m) -- in a fixed domain_km x domain_km_y domain on
    # the cyclone; "core_pushes": updraft pushes near the core every push_every_min until push_until_kt
    # (genesis sooner); "cyclones": the surface lows (seed vortices) placed for the next run
    "hurricane": {"on": False, "sst_seed": 1, "sst_offset": 0.0, "wake": True, "mld": 50.0, "domain_km": 900.0,
                  "domain_km_y": 900.0,
                  "core_pushes": True, "push_every_min": 10.0, "push_until_kt": 64.0, "ambient": 0.3},
    "cyclones": [],
    "engine": {"steady_environment": True},
}
KT = 0.514444
CYCLONE_EVERY = 60.0        # s of model time between cyclone fixes (hurricane mode)
EYEWALL_NO_TORNADO = 80000.0   # m from a cyclone's centre (or 4 x its radius of maximum wind) with no tornado detection
# QLCS tornadoes (Simulation._tornado_ok): a circulation in a line of storms -- QLCS_UPDRAFTS live deep updrafts within
# QLCS_REACH m, the nearest (within 15 km) followed QLCS_MIN_AGE s -- may be a tornado though no updraft there has
# lived radar.TORNADO_MIN_AGE
QLCS_UPDRAFTS = 3
QLCS_REACH = 40000.0
QLCS_MIN_AGE = 600.0

SYN_CI_EVERY = 300.0       # s of model time between synoptic storm-initiation draws
SYN_FRONTS_EVERY = 900.0
SYN_LINE_SPACING = (15000.0, 22000.0)   # m between the cells of a squall line the simulated weather fires   # s between analyses of the simulated fronts (latching, squall lines, the map)


# test hook: ARHTMODELVIEWER_DEBUG_GPU_FAULT_AT=<model s> makes the run trigger a real CUDA
# illegal-address fault then (tests the restart path); never set in normal use
_DEBUG_FAULT_AT = (float(__import__("os").environ["ARHTMODELVIEWER_DEBUG_GPU_FAULT_AT"])
                   if __import__("os").environ.get("ARHTMODELVIEWER_DEBUG_GPU_FAULT_AT") else None)
_DEBUG_TIMING = bool(__import__("os").environ.get("ARHTMODELVIEWER_DEBUG_TIMING"))
# model times (s, comma separated) at which the run asks for more GPU memory than any card has: Engine.recover_memory
_DEBUG_OOM_AT = (sorted(float(v) for v in __import__("os").environ["ARHTMODELVIEWER_DEBUG_OOM_AT"].split(","))
                 if __import__("os").environ.get("ARHTMODELVIEWER_DEBUG_OOM_AT") else None)


def _debug_gpu_fault():
    import cupy as cp
    k = cp.RawKernel('extern "C" __global__ void bad(float* p) { p[(long long)1 << 40] = 1.0f; }', "bad")
    a = cp.zeros(1, cp.float32)
    k((1,), (1,), (a,))
    cp.cuda.Device().synchronize()


# s of model time between updates of the run-max swaths (storm tracks, wind swaths), whatever the
# map's frame interval: ~200 m of a storm's travel, under a grid cell at every grid
SWATH_EVERY = 10.0
# s of model time between rebuilds of the radars' 3-D fields (reflectivity, dual-pol, debris): a WSR-88D's
# lowest sweep takes 20-30 s anyway, and a storm moves ~300 m in 20 s; at 10 s the rebuild cost ~12 % of a run
RADAR_FIELDS_EVERY = 20.0
# squall lines on boundaries (_line_chance / _fire_line): starts every 7-10 km over 60-140 km, at most one
# line per boundary every 30 min; "auto" mode leans linear when the 0-6 km shear across the boundary is
# under ~10 m/s or the storms move off it slower than 4 m/s (Dial et al. 2010)
LINE_SPACING = (7000.0, 10000.0)
LINE_LENGTH = (60000.0, 140000.0)
LINE_EVERY = 1800.0
LINE_ACROSS = 10.0
LINE_AWAY = 4.0
# storms latching onto boundaries (_latch_step): a deep updraft within LATCH_AHEAD ahead of a boundary (or just
# behind it) gets a low-level push every LATCH_EVERY s on the boundary's convergence zone, a little downstream
# along it, so it rides the boundary and keeps feeding on the backed, sheared inflow there (Markowski et al.
# 1998: most significant tornadoes of VORTEX-95 were within ~10 km of a boundary)
LATCH_AHEAD = 25000.0
LATCH_EVERY = 120.0
LATCH_W = 12.0
# Only established storms latch, close to the boundary: on a synoptic / ERA5 front (a wide zone) every 12 m/s
# updraft near it got pushes, each push's new updraft got its own, and the warm sector filled with small pulse
# cells (54 of a run's 67 storm starts were latches).  An updraft born where a push went (LATCH_CHILD_KM,
# LATCH_CHILD_S after it) is that push's, not a storm of its own, and gets none.
LATCH_W_STORM = 20.0           # m/s: what a storm that latches has
LATCH_AGE = 1200.0             # s it has been followed
LATCH_BEHIND = 10000.0         # m behind the boundary at most (its zone half-width when narrower)
LATCH_CHILD_KM = 10.0
LATCH_CHILD_S = 900.0
AUTOSAVE_EVERY = 120.0          # wall s between the run's state.json saves (_autosave)
# Low-CAPE storms: the storm thresholds (a followed deep updraft 12 m/s, a latching one 20, a tornado under a
# storm followed 25 min, a radar-indicated warning under 20 m/s) were tuned on big days.  A 500-1000 J/kg storm
# resolves at 7-14 m/s on a 1 km grid, so high-shear low-CAPE storms were never followed, never latched and never
# allowed tornadoes.  Each now scales by storm_scale: the w a storm there can reach (STORM_W_GAIN x sqrt(2 CAPE)
# of its inflow CAPE) over 12 m/s, between STORM_W_FLOOR / 12 and 1 (from ~1500 J/kg up: as before).
STORM_W_GAIN = 0.22
STORM_W_FLOOR = 6.5            # m/s
# Storm starts in strong shear: a start's lift is sheared apart before a deep updraft takes over when the 0-6 km
# shear is extreme.  The 27 Apr 2011 Cullman sounding (1600 J/kg, 46 m/s) made no deep storm from an 8 m/s,
# 10 km, 25 min start and a 60-70 min right-moving supercell from 14 m/s, 14 km, 45 min.  From START_SHEAR[0] to
# START_SHEAR[1] of 0-6 km shear a start ramps up to START_BOOST: x strength, x radius, + s of push.
START_SHEAR = (20.0, 45.0)     # m/s
START_BOOST = (1.75, 1.4, 1200.0)
# storm starts as a cluster of small thermals on a weak mesoscale lift (_thermal_cluster), on grids this fine
# or finer; coarser grids cannot hold a 2-3 km tower and keep one (softened, ragged) push
CLUSTER_DX = 1500.0
CLUSTER_N = (6, 14)               # thermals per 10 km of start radius
THERMAL_R = (800.0, 1600.0)       # m: thermal radius (at least THERMAL_MIN_DX grid cells)
THERMAL_MIN_DX = 2.5
THERMAL_W = (0.5, 0.9)            # x the start's strength
LIFT_W = 0.25                     # the mesoscale lift under them, x the start's strength
NUDGE_TAU = 30.0                  # s: towers / line cells pull w toward their target over half a minute (was 10 s);
                                  # a coarse grid's single storm push keeps 10 s (softer, its storms die young)
THERMAL_TAU = 15.0                # s: a thermal grips harder (a narrow push is fought by its own pressure field)
LIFT_TAU = 120.0
# the storm-scale core under the thermals: one round, upright push CORE_R of the start's radius at CORE_W of its
# strength, grown over the start's ramp.  Without it the thermals (1.6-3.2 km) never organised one wide updraft and
# the low-level mesocyclone never came: on the KLZK North setup at 1 km, 120 min, 0-3 km UH peaked at 207 (the old
# single push 1408), with a 0.7 R / 0.8 W ragged core 657, with this one 1303 (1 km zeta 0.0139, old 0.0125)
CORE_W = 1.0
CORE_R = 1.0
# the storm starts of 1.6 and before (user, 2026-11-01): one round push per storm that jumps to 40 % of its strength
# and widens from half its radius over the ramp, satellite towers on every grid, no ragged shapes or thermals.  Same
# setup, 500 m: near-ground vorticity 0.20 (1.4) vs 0.075 with the thermal clusters.  False = the 10-31 thermal starts
CLASSIC_STARTS = True

FRAME_WALL_ZOOM = 20.0  # s of real time: the longest a running tornado zoom goes without a new map frame
FRAME_SHARE = 0.2       # ... unless those frames would take more than this share of the run's real time
RESTART_CODE = 75
GPU_FAULT_FILE = OUTPUT / "last_gpu_error.json"


def gpu_fault(exc: BaseException) -> bool:
    """A CUDA error that leaves the process's GPU context unusable (illegal
    address, launch failure, ...), as opposed to running out of memory."""
    name = type(exc).__name__
    text = str(exc)
    if (name not in ("CUDARuntimeError", "CUDADriverError") and "cudaError" not in text
            and "CUDA_ERROR" not in text):
        return False
    return not any(k in text for k in ("MemoryAllocation", "OUT_OF_MEMORY", "out of memory"))


def restart_after_gpu_fault(sim, exc: BaseException) -> None:
    """Record the fault for the next server, then (under the supervisor) exit so a
    fresh process -- and a fresh GPU context -- takes over in a few seconds."""
    import os
    text = str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
    try:
        GPU_FAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        GPU_FAULT_FILE.write_text(json.dumps({"time": time.time(), "run_id": sim.run_id, "error": text,
                                              "model_t": getattr(sim.engine, "t", 0.0)}))
    except OSError:
        pass
    if os.environ.get("ARHTMODELVIEWER_SUPERVISED") == "1":
        sim.error = (f"GPU error: {text}. The GPU engine is restarting (a few seconds); "
                     f"start the run again. Level II files written so far are kept.\n" + (sim.error or ""))
        threading.Timer(3.0, os._exit, args=(RESTART_CODE,)).start()
    else:
        sim.error = (f"GPU error: {text}. Restart Atmos-WRF to use the GPU again "
                     f"(close the window and start Atmos-WRF.cmd).\n" + (sim.error or ""))


def free_gpu() -> None:
    """Return released arrays to the card (cyclic references hold them until gc)."""
    from .patch import release_freed
    try:
        release_freed()
        import cupy as cp
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def gpu_held() -> int:
    """Bytes of GPU memory the viewer's arrays hold now (CuPy's pool, its free blocks too)."""
    try:
        import cupy as cp
        return int(cp.get_default_memory_pool().total_bytes())
    except Exception:
        return 0


def load_sounding(spec: dict) -> snd.Sounding:
    if spec.get("analytic"):
        a = spec["analytic"]
        s = snd.weisman_klemp(theta_sfc=a.get("theta_sfc", 300.0), qv0=a.get("qv0", 14.0),
                              shear_ms=a.get("shear", 35.0), hodograph=a.get("hodograph", "quarter"))
    elif spec.get("text"):
        s = snd.parse_any(spec["text"], spec.get("title"))
    elif spec.get("tropical"):
        from .hurricane import tropical_sounding
        tr = {k: float(v) for k, v in (spec["tropical"] or {}).items()
              if k in ("sst", "rh_mid", "steer_kt", "steer_dir", "shear_kt", "shear_from", "trop_c", "instab")}
        s = snd.parse_any(tropical_sounding(**tr), spec.get("title") or
                          f"Tropical (Jordan 1958 mean, SST {tr.get('sst', 29.0):.1f} C)")
    else:
        path = Path(spec.get("file", ""))
        if not path.is_absolute():
            path = PROJECT / path
        s = snd.parse_file(path)
    heat = spec.get("heating") or {}
    if float(heat.get("hours", 0.0) or 0.0) > 0.0:     # first: the afternoon the other changes act on
        s = s.surface_heated(float(heat.get("flux", 300.0)), float(heat["hours"]), float(heat.get("bowen", 1.0)))
    mix = spec.get("mix") or {}
    if mix.get("depth"):
        s = s.mixed(float(mix["depth"]), float(mix.get("heat", 0.0)))
    moist = spec.get("moist") or {}
    if float(moist.get("dtd", 0.0) or 0.0) > 0.0:       # after the mix, which would average it away
        s = s.moistened(float(moist["dtd"]), float(moist.get("depth", 100.0)))
    mid = spec.get("mid_moist") or {}
    if float(mid.get("rh", 0.0) or 0.0) > 0.0:
        s = s.mid_moistened(float(mid["rh"]), float(mid.get("bottom", 750.0)), float(mid.get("top", 450.0)))
    cap = spec.get("cap") or {}
    if abs(float(cap.get("cin", 0.0) or 0.0)) > 0.0:    # after the mix and moisture, which set the parcel
        s = s.capped(abs(float(cap["cin"])), max(float(cap.get("base", 100.0)), float(mix.get("depth") or 0.0)))
    if spec.get("ice_cap", True):
        s = s.ice_capped()
    scale = float(spec.get("wind_scale", 1.0) or 1.0)
    if scale != 1.0:
        s = s.with_wind_scale(scale)
    jet = spec.get("llj") or {}
    if float(jet.get("speed", 0.0) or 0.0) != 0.0:
        s = s.with_low_level_jet(float(jet["speed"]), float(jet.get("height", 750.0)))
    aloft = spec.get("aloft") or {}
    if float(aloft.get("speed", 0.0) or 0.0) != 0.0:
        s = s.with_winds_aloft(float(aloft["speed"]), float(aloft.get("base", 5000.0)), float(aloft.get("full", 9000.0)))
    return s


def _intersect(a, b):
    """Two rectangles (x0, y0, x1, y1) or None (anywhere) intersected."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def _within(box, area):
    """Rectangle (x0, y0, x1, y1) ``box`` cut to ``area`` (None: anywhere); empty if x1 <= x0 or y1 <= y0."""
    if area is None:
        return box
    return max(box[0], area[0]), max(box[1], area[1]), min(box[2], area[2]), min(box[3], area[3])


def _epoch_ms(iso: str) -> int:
    t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp() * 1000)


@dataclass
class Zone:
    """Random storm initiation in a rectangle (world m)."""
    x0: float
    y0: float
    x1: float
    y1: float
    rate: float = 3.0            # storms per hour
    t_start: float = 0.0
    t_end: float = 3 * 3600.0
    spacing: float = 30000.0     # keep new storms this far from other starts (m)
    strength: float = 10.0
    seed: int = 1
    id: int = 0

    def to_json(self):
        return dict(self.__dict__)


class HandOverLock:
    """The run's lock (re-entrant) that the run loop hands over: Python's locks are not fair, and the loop holds
    this one for a whole step (seconds with a tornado zoom's nests) then takes it back at once, so a page request
    waiting on it (the 3-D view, the state, the phone remote) could wait through step after step.  Threads that
    have to wait are counted, and the loop lets them in first (hand_over) before its next step."""

    def __init__(self):
        self._lock = threading.RLock()
        self._count = threading.Lock()
        self.waiting = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._lock.acquire(blocking=False):
            return True
        if not blocking:
            return False
        with self._count:
            self.waiting += 1
        try:
            return self._lock.acquire(True, timeout)
        finally:
            with self._count:
                self.waiting -= 1

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()

    def hand_over(self, limit: float = 3.0) -> None:
        """Wait (without the lock) until every thread waiting for it has had it, at most ``limit`` s."""
        t0 = time.perf_counter()
        while self.waiting > 0 and time.perf_counter() - t0 < limit:
            time.sleep(0.001)


@dataclass
class Frame:
    t: float
    epoch_ms: int
    patches: list
    products: dict               # name -> compressed payload
    radars: dict = field(default_factory=dict)   # icao -> {moment: compressed payload}
    size: int = 0


class Simulation:
    def __init__(self, settings: dict | None = None):
        self.settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        if settings:
            self.settings.update(settings)
        s = self.settings
        self.run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = OUTPUT / "runs" / self.run_id
        self.status = "building"
        self.error = None
        self.cmds: queue.Queue = queue.Queue()
        self.lock = HandOverLock()
        self.frames: list[Frame] = []
        self.frame_bytes = 0
        self.max_frame_bytes = int(s.get("max_frame_mb", 1500)) * 1024 * 1024
        self.l2_index: dict[str, list] = {}           # icao -> [(name, size)]
        self._l2_pool = ThreadPoolExecutor(max_workers=2)
        self._l2_pending = 0                          # Level II volumes queued or being written
        self.zones: list[Zone] = []
        self.nudger_plans: list[dict] = []      # scheduled nudgers (settings "nudgers"), each with an id
        self._plans_done: set = set()
        self.boundaries: list[Boundary] = []
        self.pockets = None                     # pockets.SoundingPockets: more soundings eased into the run's
        self.lows = []
        self.tracks = products.TrackStore()
        from .efswath import EFSwath
        self.efswath = EFSwath()                 # the tornadoes' EF swath on the fine grids (efswath.py)
        self.swath_version = 0                  # swath updates so far (the placefile's image URL)
        self._next_swath = 0.0
        self.uh_peaks: list[dict] = []          # each new run-max 2-5 km UH: value, t, x, y
        if s.get("seed") is None:
            s["seed"] = random.SystemRandom().randrange(1, 2 ** 31)   # a new outbreak every run (kept in settings.json)
        self._rng = random.Random(int(s["seed"]))
        self.cyclones: list[dict] = []          # hurricane mode: the tracked surface lows (_cyclones_step)
        self._next_cyclone = 0.0
        self._line_last = {}
        self._next_latch = 0.0
        self._latch_sites = []                  # (t, x, y, parent track id) of the latch pushes (_latch_step)
        self.texture = None                     # boundaries.WarmSectorTexture (build)
        self.synoptic = None                    # synoptic.SynopticWorld in synoptic mode (build)
        self.syn_boundaries: list[Boundary] = []   # its fronts and drylines now, as boundaries (latching, lines)
        self._syn_summary = None
        self._syn_next_ci = 0.0
        self._syn_next_fronts = 0.0
        self.syn_progress = None                # (fraction, message) while the weather is being built
        self._mergers: list[dict] = []          # merger cells waiting for their storm (_plan_mergers)
        self.wall_start = None
        self.speed = 0.0
        self.display_radar = None
        self._thread = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self.engine = None
        self.env = None
        self.heartbeat = time.time()
        self.radars: list[rd.Radar] = []
        self.dow_since: dict[str, float] = {}     # DOW id -> wall time of its current deployment (_deploy_dow)
        self.tornadoes = rd.TornadoTracker()    # debris balls grow and fade from radar refresh to refresh
        self.tor_paths: dict = {}               # tornado tracks: id -> points (t, x, y, wind, Vrot, CC, EF)
        from .warnings_auto import WarningCenter
        self.warnings = WarningCenter(self)     # SPS / SVR / TOR polygons on the storms (warnings_auto.py)
        from .metars import MetarNet
        self.metars = MetarNet(self)            # METARs at the real ASOS sites around the run (metars.py)
        self._tor_radar: dict = {}              # id -> the radar's latest Vrot / CC at it (_sample_tornadoes)
        self.start_ms = _epoch_ms(s["start_time"])
        m = s["map"]
        self.proj = geo.Lambert(float(m["lat0"]), float(m["lon0"]))
        self._heating_skipped = self._double_heating(s)
        spec = s["sounding"]
        if self._heating_skipped:
            spec = {k: v for k, v in spec.items() if k != "heating"}
        self.sounding = load_sounding(spec)
        self.diag = snd.diagnostics(self.sounding)

    @staticmethod
    def _double_heating(s: dict) -> bool:
        """The sounding's "Surface heating" would heat the ground twice: a physics suite with a land surface heats
        it itself as the run goes, and without the steady environment nothing takes that back -- the pre-heated
        boundary layer kept warming and the whole domain fired (suspected by the user, 2026-11-03).  With the
        steady environment the suite's heating of each cell is cancelled (Engine._suite_weights), so the
        pre-heating is the only heating and stays."""
        if not float(((s.get("sounding") or {}).get("heating") or {}).get("hours", 0.0) or 0.0) > 0.0:
            return False
        sid = (s.get("physics") or {}).get("suite")
        if not sid or (s.get("engine") or {}).get("steady_environment", False):
            return False
        try:
            from .environment import physics_suites
            suite = next((x for x in physics_suites() if x["id"] == sid), None)
        except Exception:
            return False
        return bool(suite and suite.get("surface"))

    # ------------------------------------------------------------ build
    def build(self) -> None:
        from .engine import SPIN_UP_GROUND_MIN, Engine, EngineOptions
        s = self.settings
        grid = GRID_PRESETS[s["grid"]] if s["grid"] in GRID_PRESETS else GRID_PRESETS["1km"]
        ph = s.get("physics", {})
        self.notes = []
        from . import backend as _backend
        if _backend.note():
            self.notes.append(_backend.note())
        if microtweaks.describe():
            self.notes.append(microtweaks.describe())
        if self._heating_skipped:
            self.notes.append("the sounding's Surface heating is not used: the physics suite's land surface heats the "
                              "ground itself as the run goes (turn on the steady environment to keep the pre-heating "
                              "instead)")
        hc = self._hurricane()
        lat = self.proj.lat0
        if hc.get("on"):
            if ph.get("suite"):
                self.notes.append("hurricane mode runs the built-in core (its own sea surface fluxes): the physics "
                                  "suite is not used")
                ph = {**ph, "suite": ""}
            if s.get("cyclones"):                 # the f-plane at the first surface low
                lat = float(self.proj.inv(float(s["cyclones"][0]["x"]), float(s["cyclones"][0]["y"]))[0])
            if not ph.get("coriolis", False):
                self.notes.append("hurricane mode without Coriolis: a surface low has nothing to spin it up "
                                  "(turn Coriolis on, Physics tab)")
        phys = PhysicsOptions(latitude=lat, coriolis=ph.get("coriolis", False),
                              surface_drag=ph.get("surface_drag", True), z0=float(ph.get("z0", 0.1)),
                              mp_physics=self._microphysics(ph), suite=self._suite(ph), les=bool(ph.get("les")),
                              soil_moisture=float(ph.get("soil_moisture", 0.25)),
                              vegetation=float(ph.get("vegetation", 50.0)))
        # the environment's winds on the model's grid at the storm area (_storm_lon); synoptic runs (the simulated
        # atmosphere, HRRR) make their winds on the grid themselves
        self._turn_lon = None if self._syn_cfg().get("on") else self._storm_lon()
        # the model grid moves with the storms (CM1 umove/vmove); everything shown stays ground-relative
        follow = s.get("grid_motion", "rm") == "follow"
        motion = self._motion("rm" if follow else s.get("grid_motion", "rm"))
        self.ground_fixed = False                # a physics suite holds the grid to the ground
        self._domain_motion = None
        if phys.suite:
            from .environment import physics_suites
            suite = next((x for x in physics_suites() if x["id"] == phys.suite), None)
            if suite is None:
                raise ValueError(f"unknown ArWen physics suite {phys.suite!r}")
            if suite["surface"] and (motion[0] or motion[1]):
                self._domain_motion = motion          # the domain moves with it instead (Engine.domain_motion)
                # ArWen's surface layer and PBL see the model wind: on a moving grid that is
                # not the wind over the ground, so suites with surface physics stay fixed to it
                motion = (0.0, 0.0)
                follow = False
                self.ground_fixed = True
                self.notes.append("the grid is fixed to the ground: the suite's surface layer needs ground-relative winds")
        tr = s.get("terrain") or {}
        self.terrain_on = bool(tr.get("on"))
        if self.terrain_on and (hc.get("on") or self._syn_cfg().get("on")):
            self.terrain_on = False
            self.notes.append("terrain is not used in hurricane or synoptic runs yet: the ground is flat")
        if self.terrain_on and (motion[0] or motion[1]) and not self.ground_fixed:
            # the ground cannot slide under a moving grid: the domain moves over it in steps instead
            self._domain_motion = motion
            motion = (0.0, 0.0)
            follow = False
            self.ground_fixed = True
            self.notes.append("the grid is fixed to the ground: the terrain stays put under it")
        start = level2.utc(self.start_ms)
        if level2.l2_shift_ms(self.start_ms):
            self.notes.append(f"Level II files are dated {level2.utc(self.start_ms + level2.l2_shift_ms(self.start_ms)):%Y} "
                              "(Level II cannot hold a date before 1970; the calendar repeats every 28 years, so the "
                              "weekday and every clock time are right, and GR2Analyst's polling shows them as now)")
        h = self._syn_cfg().get("hrrr") or {}
        if self._syn_cfg().get("on") and self._syn_cfg().get("source") == "hrrr" and h.get("cycle"):
            # the run starts at the HRRR hour it begins from
            cyc = dt.datetime.fromisoformat(str(h["cycle"]).replace("Z", "+00:00"))
            cyc = cyc if cyc.tzinfo else cyc.replace(tzinfo=dt.timezone.utc)
            start = cyc + dt.timedelta(hours=int(h.get("start_fh", 0)))
            self.start_ms = int(start.timestamp() * 1000)
            s["start_time"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        if self._syn_cfg().get("on"):
            self._build_synoptic(start)
            if not self.ground_fixed:           # from the simulated centre column, not the library sounding
                motion = self._motion("rm" if follow else s.get("grid_motion", "rm"))
        if not follow and self._domain_motion is None:
            # a domain at its size limit rides the set motion with the grid instead of chasing updrafts
            # (left splits and far-off cells dragged it away from the right movers)
            self._domain_motion = motion
        grid_snd = self.sounding
        if self._turn_lon is not None and abs(float(self.proj.north_angle(self._turn_lon))) > 0.05:
            import dataclasses
            gu, gv = self.proj.to_grid(np.asarray(self.sounding.u), np.asarray(self.sounding.v), self._turn_lon)
            grid_snd = dataclasses.replace(self.sounding, u=np.asarray(gu), v=np.asarray(gv))
            if abs(float(self.proj.north_angle(self._turn_lon))) >= 0.5:
                self.notes.append(f"the sounding's winds are turned {float(self.proj.north_angle(self._turn_lon)):+.1f} deg "
                                  f"onto the map's grid at the storms' longitude ({self._turn_lon:.1f}): true north there "
                                  "is that far round from the grid's north")
        self.env = Environment(grid_snd, grid, phys, motion=motion)
        self.env.start_time = start
        self._build_pockets(s, hc)
        self.env.latlon = lambda X, Y: self.proj.inv(X, Y)
        self.env.longitude = float(self.proj.lon0)
        if self.terrain_on:
            from .terrain import Terrain
            self.env.terrain = Terrain(tr, self.env.latlon, zero_m=self.sounding.zsfc, ztop=grid.ztop)
            self.notes.append(self.env.terrain.describe())
        self.env.towns = self._towns()          # what a tornado can hit (radar debris signature)
        from .landcover import load as load_landcover
        self.env.landcover = load_landcover()   # and what grows or stands there (forest, town, crops)
        self._set_roughness(ph)
        self.env.tornado_ok = self._tornado_ok     # only mature, unforced storms make tornadoes (radar.TORNADO_COLD)
        self.env.tornado_near = self._tornado_near
        self.env.tornado_forced = self._tornado_forced  # the only test on grids that resolve their tornadoes
        eo = s.get("engine", {}) or {}
        dom = s.get("domain") or {}
        fixed = dom.get("mode") == "fixed"
        fixed_km = float(dom.get("size_km") or 160.0) if fixed else None
        fixed_km_y = min(max(float(dom.get("size_km_y") or fixed_km), 48.0), 1000.0) if fixed else None
        shape = self._domain_shape(dom) if fixed else None
        if hc.get("on"):                          # one domain on the cyclone, the size the Tropics tab sets
            fixed_km = min(max(float(hc.get("domain_km") or 900.0), 200.0), 2400.0)
            fixed_km_y = min(max(float(hc.get("domain_km_y") or fixed_km), 200.0), 2400.0)
        max_km = None if fixed else float(dom.get("max_km") or 0.0) or None
        self.engine = Engine(self.env, EngineOptions(
            drift_correction=eo.get("steady_environment", False),
            rim=eo.get("rim", True), manage=True, follow_storms=follow,
            follow_interval=60.0 * max(1.0, float(s.get("grid_follow_min") or 10.0)),
            persistence=float(s.get("persistence", 0.0)), fixed_km=fixed_km, fixed_km_y=fixed_km_y, max_km=max_km,
            domain_shape=None if hc.get("on") else shape,
            refine_ratio=self._refine_ratio(grid), refine_km=float((s.get("refine") or {}).get("size_km") or 40.0),
            refine_max=max(1, int((s.get("refine") or {}).get("max") or 2)),
            refine_inner_ratio=self._refine_inner(grid), refine_inner_km=float((s.get("refine") or {}).get("inner_km") or 16.0),
            ls_w_cms=self._large_scale_w(), ls_top_km=float((s.get("large_scale") or {}).get("top_km") or 10.0)))
        self.engine.domain_motion = self._domain_motion
        # a physics suite: new air takes on the suite's own day (no growth seams); with the steady environment too
        # (it then holds only the wind steady)
        self.engine.opt.follow_environment = bool(self.env.suite and not hc.get("on") and not self._syn_cfg().get("on"))
        self.engine.track_w = lambda x, y: 12.0 * self.storm_scale(x, y)
        if hc.get("on"):
            self._build_ocean(hc)
        self._syn_cape = None
        for text in self.notes:
            self.engine.note("physics", text)
        self.engine.note("physics", self._physics_summary())
        from .engine import CELL_BYTES
        if self.engine.fixed:
            n, m = self.engine.fixed, self.engine.fixed_ny
            self.engine.note("patch", f"fixed {n * grid.dx / 1000:.0f} x {m * grid.dx / 1000:.0f} km domain ({n}x{m} "
                                       f"cells, about {n * m * grid.nz * CELL_BYTES / 2**30:.1f} GB of GPU memory): it "
                                       "follows the storms but never grows")
        elif self.engine.cap:
            n = self.engine.cap
            self.engine.note("patch", f"patches grow up to {n * grid.dx / 1000:.0f} km ({n}x{n} cells, about "
                                       f"{n * n * grid.nz * CELL_BYTES / 2**30:.1f} GB of GPU memory each), then "
                                       "follow their storms without growing")
        for r in s.get("radars", []):
            self._add_radar(r)
        # air masses first: every patch (the storm starts' too) is built with them
        for b in s.get("boundaries", []) or []:
            self.boundaries.append(self._new_boundary(b))
        if self.synoptic is not None:
            # the simulated atmosphere is the air everywhere: drawn fronts, the warm-sector texture and the
            # hodograph brushes would be a second, conflicting one
            if self.boundaries or s.get("wind_brushes"):
                self.engine.note("synoptic", "synoptic mode: the drawn fronts and hodograph brushes are not used "
                                             "(the simulated weather makes its own)")
            self.env.airmass = syn.SynopticAirMass(self.synoptic, self.env)
            self.engine.note("synoptic", self.synoptic.scenario_note)
            self.engine.note("synoptic", f"base sounding: the simulated column at the map centre at the start "
                                         f"(MLCAPE {self.diag.get('mlcape') or 0:.0f} J/kg, 0-1 km SRH "
                                         f"{self.diag.get('srh1') or 0:.0f} m2/s2)")
        else:
            self.texture = self._new_texture()
            if self.boundaries:
                self.lows = join_at_lows(self.boundaries, 0.0, self._low_motion())
            self.env.airmass = self._airmasses()
            if self.engine.opt.drift_correction:
                self.engine.spin_up()           # before any patch: the wind the model's mixing keeps
            elif not hc.get("on"):
                # the surface layer the ground's friction mixes, everywhere from the start (no edge seams)
                self.engine.spin_up(SPIN_UP_GROUND_MIN, ground=True)
        for st in s.get("storms", []):
            self._add_storm(st)
        for c in s.get("cyclones", []) or []:
            if hc.get("on"):
                self._add_cyclone(c)
        for z in s.get("zones", []):
            self._add_zone(z)
        for n in s.get("nudgers", []) or []:
            self._plan_nudger(n)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "settings.json").write_text(json.dumps(self.settings, indent=1))
        self.status = "ready"

    # Thompson aerosol-aware (28) by default: in side-by-side runs of a classic supercell (OUN 3 May 1999,
    # 1 km, 100 min; docs/PROGRESS.md) it kept the inflow notch clear and the hook a hook, with the strongest
    # low-level mesocyclone; Morrison and P3 filled the notch with forward-flank rain.  It needs its lookup
    # tables (Setup.cmd fetches them); without them a run takes Milbrandt-Yau (9), the best of the
    # schemes that need none.
    MP_FALLBACK = 9

    def _suite(self, ph: dict):
        """The physics suite to run, or None for the built-in core: a suite with Thompson needs its tables, and
        suites are untested on the CPU backend -- without either, the run takes the built-in core (the
        land-cover friction and Thompson-aerosol, or Milbrandt-Yau without tables) and says so."""
        sid = ph.get("suite") or None
        if not sid:
            return None
        from . import backend
        from .environment import physics_suites
        suite = next((x for x in physics_suites() if x["id"] == sid), None)
        why = None
        if backend.is_cpu():
            why = "physics suites are not tested on the CPU backend"
        elif suite is not None and suite["mp_physics"] in (8, 28) and not self._thompson_tables():
            why = "its Thompson lookup tables are not installed (run Setup.cmd again)"
        if why:
            self.notes.append(f"physics suite {suite['label'] if suite else sid} not used: {why}; "
                              "this run uses the built-in core")
            return None
        return sid

    @staticmethod
    def _thompson_tables() -> bool:
        root = Path(os.environ.get("GPUWM_THOMPSON_TABLE_ROOT", ""))
        return root.is_dir() and any(root.iterdir())

    def _microphysics(self, ph: dict) -> int:
        mp = int(ph.get("mp_physics", 28) or 28)
        if mp in (8, 28):
            if not self._thompson_tables():
                self.notes.append("Thompson's lookup tables are not installed (run Setup.cmd again): "
                                  "this run uses Milbrandt-Yau 2-moment instead")
                return self.MP_FALLBACK
        return mp

    def _tornado_near(self, x: float, y: float) -> bool:
        """Is a tornado followed at the last radar refresh (and found then) within reach of (x, y): its
        circulation found again keeps being a tornado though its outflow dips under radar.TORNADO_COLD."""
        reach = rd.TRACK_REACH + 2000.0
        return any(c.found and math.hypot(c.x - x, c.y - y) <= reach for c in self.tornadoes.items)

    def _refine_ratio(self, grid) -> int:
        """The nests' ratio (0: none): 2-4, and no nest finer than 100 m (the finest grid the model is set up for)."""
        r = int((self.settings.get("refine") or {}).get("ratio") or 0)
        if r < 2:
            return 0
        r = min(r, 4)
        while r >= 2 and grid.dx / r < 99.0:
            r -= 1
        if r < 2:
            self.notes.append(f"no refinement nests: a {grid.label} grid is already the finest the model takes")
            return 0
        self.notes.append(f"refinement nests: {grid.dx / r:.0f} m grids ({r}:1) on the strongest storms")
        return r

    def _large_scale_w(self) -> float:
        """Peak large-scale ascent (cm/s) for the run, 0 in synoptic / ERA5 / HRRR runs (their weather lifts itself)."""
        w = float((self.settings.get("large_scale") or {}).get("w_cms") or 0.0)
        if w <= 0.0:
            return 0.0
        if self._syn_cfg().get("on"):
            self.notes.append("large-scale ascent is off: the simulated weather brings its own")
            return 0.0
        top = float((self.settings.get("large_scale") or {}).get("top_km") or 10.0)
        self.notes.append(f"large-scale ascent: {w:g} cm/s at {top / 2:g} km, the environment lifted all run")
        return min(w, 30.0)

    def _refine_inner(self, grid) -> int:
        """The inner nests' ratio (0: none), inside nests of _refine_ratio: no grid finer than 100 m."""
        rf = self.settings.get("refine") or {}
        r1 = self._refine_ratio_quiet(grid)
        r = min(int(rf.get("inner_ratio") or 0), 4)
        if not r1 or r < 2:
            return 0
        while r >= 2 and grid.dx / r1 / r < 99.0:
            r -= 1
        if r < 2:
            self.notes.append("no inner nests: the nests are already the finest grid the model takes")
            return 0
        self.notes.append(f"inner nests: {grid.dx / r1 / r:.0f} m grids ({r}:1) on the nests' low-level mesocyclones")
        return r

    def _refine_ratio_quiet(self, grid) -> int:
        r = min(int((self.settings.get("refine") or {}).get("ratio") or 0), 4)
        while r >= 2 and grid.dx / r < 99.0:
            r -= 1
        return r if r >= 2 else 0

    def inflow_cape(self, x: float, y: float) -> float:
        """J/kg a storm at (x, y) feeds on: the most of ML and MU CAPE within ~50 km on the weather's last
        initiation fields (synoptic / ERA5 / HRRR runs), else the run's sounding."""
        c = getattr(self, "_syn_cape", None)
        if c is not None:
            X, Y, F = c[:3]
            k = int(np.argmin((X - x) ** 2 + (Y - y) ** 2))
            return float(F.ravel()[k])
        d = self.diag or {}
        return float(max(d.get("mlcape") or 0.0, d.get("mucape") or 0.0))

    def start_boost(self, x: float, y: float) -> float:
        """0..1: how much a storm start at (x, y) is strengthened for the 0-6 km shear there (START_SHEAR), from the
        weather's last initiation fields (synoptic / ERA5 / HRRR runs), else the run's sounding."""
        c = getattr(self, "_syn_cape", None)
        if c is not None and len(c) > 3:
            X, Y, _, SH = c
            sh = float(SH.ravel()[int(np.argmin((X - x) ** 2 + (Y - y) ** 2))])
        else:
            sh = float((self.diag or {}).get("shear6") or 0.0)
        return min(max((sh - START_SHEAR[0]) / (START_SHEAR[1] - START_SHEAR[0]), 0.0), 1.0)

    def _boosted(self, x: float, y: float, strength: float, radius: float, duration: float):
        """(strength, radius, duration) of a storm start at (x, y), raised for strong shear (start_boost)."""
        f = self.start_boost(x, y)
        return (strength * (1.0 + (START_BOOST[0] - 1.0) * f), radius * (1.0 + (START_BOOST[1] - 1.0) * f),
                duration + START_BOOST[2] * f)

    def storm_scale(self, x: float, y: float) -> float:
        """STORM_W_FLOOR / 12 .. 1: how far the storm thresholds come down for the CAPE at (x, y)."""
        w = STORM_W_GAIN * math.sqrt(2.0 * max(self.inflow_cape(x, y), 0.0))
        return min(max(w, STORM_W_FLOOR), 12.0) / 12.0

    def _tornado_forced(self, x: float, y: float) -> bool:
        """Is the air at (x, y) not the model's own: a storm-start push still forcing it, or a hurricane's eyewall
        (its spin and small vortices read as tornadoes but are the hurricane's winds; rainband tornadoes farther out
        stay possible).  The only test a grid that resolves its tornadoes keeps (radar.find_vortex_tornadoes)."""
        eng = self.engine
        if eng is None:
            return False
        for c in self.cyclones:
            if c["active"] and math.hypot(x - c["x"], y - c["y"]) < max(EYEWALL_NO_TORNADO,
                                                                       4.0 * min(c.get("rmw") or 0.0, 60000.0)):
                return True
        t = eng.t
        for s in eng.starts:
            if s.kind == "nudge" and s.t0 <= t <= s.t1:
                sx, sy = s.position(t)
                if math.hypot(x - sx, y - sy) < s.radius + 3000.0:
                    return True
        return False

    def _tornado_ok(self, x: float, y: float) -> bool:
        """May a circulation at (x, y) be a tornado (on a storm-scale grid): nothing forcing the air there
        (_tornado_forced), and a deep updraft followed for radar.TORNADO_MIN_AGE is within 15 km of it."""
        eng = self.engine
        if eng is None:
            return True
        if self._tornado_forced(x, y):
            return False
        t = eng.t
        scale = self.storm_scale(x, y)
        age = rd.TORNADO_MIN_AGE * scale
        live = [tr for tr in eng.tracks.live if t - tr.last[0] <= 180.0]
        near = [tr for tr in live if math.hypot(x - tr.last[1], y - tr.last[2]) <= 15000.0]
        if any(t - tr.born >= age for tr in near):
            return True
        # a QLCS: the mesovortices along a line's leading edge spin up under updrafts that come and go (the
        # tracker re-links them along the line, so none lives TORNADO_MIN_AGE), yet they make tornadoes. A
        # circulation with QLCS_UPDRAFTS live updrafts within QLCS_REACH (a line, not a lone cell), the nearest
        # followed QLCS_MIN_AGE, may be one; the outflow test (radar.TORNADO_COLD) still applies
        line = [tr for tr in live if math.hypot(x - tr.last[1], y - tr.last[2]) <= QLCS_REACH]
        return (len(line) >= QLCS_UPDRAFTS
                and any(t - tr.born >= QLCS_MIN_AGE * scale for tr in near))

    def _set_roughness(self, ph: dict) -> None:
        """Ground friction over the land cover: each column's roughness from what is under it, and the
        reference column's drag from the region's (200 km around the map centre) mean coefficient."""
        env, lc = self.env, self.env.landcover
        if not env.cd or lc is None or (ph.get("roughness") or "landcover") != "landcover":
            return
        env.roughness = lambda X, Y: lc.roughness(*self.proj.inv(X, Y))
        g = np.linspace(-100e3, 100e3, 41)
        X, Y = np.meshgrid(g, g)
        env.cd = float(np.mean(env.drag_coefficient(env.roughness(X, Y).astype(np.float64))))

    def _physics_summary(self) -> str:
        from .environment import MICROPHYSICS
        env = self.env
        mp = MICROPHYSICS.get(int(env.physics.mp_physics), str(env.physics.mp_physics))
        if env.physics.suite:
            les = ", LES: its PBL scheme off, 3-D TKE turbulence" if env.physics.les else ""
            return f"ArWen suite {env.physics.suite} (microphysics: {mp}{les})"
        if not env.cd:
            ground = "free-slip ground"
        elif getattr(env, "roughness", None) is not None:
            ground = f"ground friction over the land cover (mean Cd {env.cd:.4f})"
        else:
            ground = f"ground friction (z0 {env.physics.z0:g} m, Cd {env.cd:.4f})"
        return f"built-in core: {mp}, 1.5-order TKE, {ground}"

    # ------------------------------------------------------------ commands
    def command(self, name: str, **kw) -> None:
        self.cmds.put((name, kw))

    # commands that change the run's settings (kept in its settings.json, and what the page offers
    # for the next run)
    SETTINGS_COMMANDS = ("add_radar", "remove_radar", "plan_nudger", "remove_nudger_plan", "add_boundary",
                         "update_boundary", "remove_boundary", "set_auto", "set_low", "speed", "frame_interval",
                         "radar_refine", "grid_motion")

    def _handle(self, name, kw):
        if name == "add_storm":
            self._add_storm(kw)
        elif name == "add_radar":
            self._add_radar(kw)
        elif name == "remove_radar":
            icao = str(kw.get("icao", "")).upper()
            self.radars = [r for r in self.radars if r.icao != icao]
            self.settings["radars"] = [r for r in self.settings.get("radars", []) if r.get("icao", "").upper() != icao]
        elif name == "add_zone":
            self._add_zone(kw)
        elif name == "add_cyclone":
            self._add_cyclone(kw)
        elif name == "remove_patch":
            self.engine.remove_patch(int(kw.get("id", -1)))
        elif name == "move_patch":
            self.engine.move_patch(int(kw.get("id", -1)), 1000.0 * float(kw["x"]), 1000.0 * float(kw["y"]),
                                   bool(kw.get("lock")))
        elif name == "release_patch":
            self.engine.release_patch(int(kw.get("id", -1)))
        elif name == "move_nest":
            self.engine.move_nest(int(kw.get("id", -1)), 1000.0 * float(kw["x"]), 1000.0 * float(kw["y"]))
        elif name == "add_nest":
            self.engine.add_nest(1000.0 * float(kw["x"]), 1000.0 * float(kw["y"]))
        elif name == "tornado_box":
            self.engine.tornado_box(1000.0 * float(kw["x"]), 1000.0 * float(kw["y"]), float(kw.get("size_km", 5.0)),
                                    str(kw.get("detail") or "high"))
            self._frame_now = True            # the map shows the zoom's grid after this step, not the next frame
            self._time_left = 6               # ARHTMODELVIEWER_DEBUG_TIMING: the next steps' timings in the log
        elif name == "tornado_detail":
            if self.engine.tornado is None:
                raise ValueError("no tornado zoom to change")
            self.engine.tornado.change_detail(str(kw.get("detail") or "high"))
            self._frame_now = True
            self._time_left = 6
        elif name == "tornado_clear":
            self.engine.tornado_clear()
        elif name == "remove_zone":
            self.zones = [z for z in self.zones if z.id != kw.get("id")]
        elif name == "nudger":
            self._send_nudger(kw)
        elif name == "plan_nudger":
            self._plan_nudger(kw)
        elif name == "remove_nudger_plan":
            self.nudger_plans = [n for n in self.nudger_plans if n["id"] != kw.get("id")]
            self.settings["nudgers"] = [{k: v for k, v in n.items() if k != "id"} for n in self.nudger_plans]
        elif name == "add_boundary":
            self.boundaries.append(self._new_boundary(kw))
            self._apply_boundaries()
        elif name == "update_boundary":
            self._update_boundary(kw)
        elif name == "remove_boundary":
            self.boundaries = [b for b in self.boundaries if b.id != int(kw.get("id", -1))]
            self._apply_boundaries()
        elif name == "set_auto":
            old = float(self._auto().get("texture", 1.0) or 0.0)
            self.settings["auto_storms"] = {**self._auto(), **{
                k: kw[k] for k in DEFAULT_SETTINGS["auto_storms"] if k in kw}}
            if float(self._auto().get("texture", 1.0) or 0.0) != old:
                self.texture = self._new_texture()
                self._apply_boundaries()
        elif name == "set_low":
            self.settings["low"] = {**(self.settings.get("low") or {}),
                                    **{k: kw[k] for k in ("auto", "speed", "heading") if k in kw}}
            self._apply_boundaries()
        elif name == "speed":
            self.settings["speed_limit"] = float(kw.get("limit", 0.0))
        elif name == "frame_interval":
            self.settings["frame_interval"] = float(kw.get("seconds", 60.0))
        elif name == "radar_refine":
            self.settings["radar_refine"] = bool(kw.get("on", True))
        elif name == "grid_motion":
            self._set_grid_motion(kw)
        if name in self.SETTINGS_COMMANDS:
            self._save_settings()

    def _save_settings(self) -> None:
        try:
            if self.run_dir.is_dir():
                (self.run_dir / "settings.json").write_text(json.dumps(self.settings, indent=1))
        except OSError:
            log.exception("could not save the run's settings")

    def _set_grid_motion(self, kw):
        """The grid's motion (CM1's umove/vmove) changed during the run: a set motion from now on,
        or following the storms' measured motion every ``follow_min``."""
        eng = self.engine
        spec = kw.get("motion", "rm")
        if isinstance(spec, (list, tuple)):
            spec = [float(spec[0]), float(spec[1])]
            if not all(math.isfinite(v) and abs(v) <= 60.0 for v in spec):
                raise ValueError("umove and vmove must be within 60 m/s")
        elif spec not in ("rm", "lm", "mean", "none", "follow"):
            raise ValueError(f"unknown grid motion {spec!r}")
        every = max(1.0, float(kw.get("follow_min") or self.settings.get("grid_follow_min") or 10.0))
        if self.ground_fixed:
            # the grid stays on the ground (its suite's surface layer needs ground-relative winds): the domain
            # moves at the motion asked for instead, or follows its storms ("follow")
            eng.domain_motion = None if spec == "follow" else self._motion(spec)
            eng.note("grid", "the domain follows its storms" if spec == "follow" else
                     f"the domain moves at {math.hypot(*eng.domain_motion):.1f} m/s ({spec}); the grid stays on the ground")
            self.settings["grid_motion"] = spec
            self.settings["grid_follow_min"] = every
            return
        eng.set_follow(spec == "follow", 60.0 * every)
        eng.domain_motion = None if spec == "follow" else self._motion(spec)
        if spec == "follow":
            eng.note("grid", f"following the storms' measured motion, checked every {every:g} min")
        else:
            eng.set_grid_motion(self._motion(spec), reason="set from the page")
        self.settings["grid_motion"] = spec
        self.settings["grid_follow_min"] = every

    def _motion(self, spec):
        """A motion (m/s) on the model's grid: given as a vector (on the grid already), or the sounding's (its Bunkers
        movers, its 0-6 km mean wind: earth-relative, turned onto the grid at the run's storm area, _grid_turn)."""
        d = self.diag
        if isinstance(spec, (list, tuple)):
            return float(spec[0]), float(spec[1])
        u, v = {"rm": tuple(d["rm"]), "lm": tuple(d["lm"]), "mean": tuple(d["mean6"]),
                "none": (0.0, 0.0)}.get(spec or "rm", tuple(d["rm"]))
        return self._to_grid(u, v)

    # The sounding's winds are earth-relative (from true north); the model's are along its map grid (Lambert
    # conformal), whose north tilts from true north by the convergence, ~0.58 deg per degree of longitude from the
    # map's centre.  Taken as grid winds they turned: a storm area 22 deg west of the centre had every wind turned
    # 13 deg anticlockwise, 70 deg over Japan -- eastward storms went north (the user, 2026-11-09).  So a run's
    # environment (its sounding, the Bunkers movers, the grid's motion) is turned onto the grid at its storm area,
    # as WRF turns earth-relative winds onto its grid (sinalpha / cosalpha).  One angle for the run: the domain
    # moves as one frame, and across a 350 km patch the convergence changes ~2 deg.
    def _storm_lon(self) -> float:
        """Longitude of the run's storm area: its storm starts', cyclones' or nudgers' mean, else the map's centre."""
        s = self.settings
        pts = [(float(q["x"]), float(q["y"])) for k in ("storms", "cyclones", "nudgers") for q in (s.get(k) or [])
               if isinstance(q, dict) and "x" in q and "y" in q]
        if not pts:
            return float(self.proj.lon0)
        x, y = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        return float(np.ravel(self.proj.inv(np.array([x]), np.array([y]))[1])[0])

    def _to_grid(self, u, v):
        lon = getattr(self, "_turn_lon", None)
        if lon is None:
            return float(u), float(v)
        gu, gv = self.proj.to_grid(float(u), float(v), lon)
        return float(gu), float(gv)

    def _add_storm(self, kw):
        vx, vy = self._motion(kw.get("motion", "rm"))
        kind = kw.get("kind", "nudge")
        x, y = float(kw["x"]), float(kw["y"])
        wt = float(kw.get("strength", 10.0 if kind == "nudge" else 3.0))
        rh, dur = float(kw.get("radius", 10000.0)), float(kw.get("duration", 900.0))
        if kind == "nudge":                       # strong shear: a stronger, wider, longer push (start_boost)
            bw, br, bd = self._boosted(x, y, wt, rh, dur)
            wt = wt if "strength" in kw else bw
            rh = rh if "radius" in kw else br
            dur = dur if "duration" in kw else bd
        t0 = float(kw.get("t0", self.engine.t if self.engine else 0.0))
        if kind == "nudge" and not kw.get("plain") and not CLASSIC_STARTS:
            ramp = float(kw.get("ramp", min(300.0, 0.4 * dur)))
            self._thermal_cluster(x, y, rh, wt, ramp, dur, vx, vy, "placed", t0=t0)
            return
        s = StormStart(x=x, y=y, t0=t0, kind=kind, radius=rh, strength=wt, duration=dur, vx=vx, vy=vy)
        self.engine.add_start(s)

    # ------------------------------------------------------------ nudgers
    NUDGER_KEYS = ("at_min", "lead_min", "aim_km", "motion", "strength", "radius_km")

    def _plan_nudger(self, kw: dict) -> None:
        """Schedule a nudger at the strongest storm (settings "nudgers")."""
        n = {k: kw[k] for k in self.NUDGER_KEYS if k in kw}
        n.setdefault("at_min", (self.engine.t if self.engine else 0.0) / 60.0)
        n["id"] = 1 + max([q["id"] for q in self.nudger_plans], default=0)
        self.nudger_plans.append(n)
        self.settings["nudgers"] = [{k: v for k, v in q.items() if k != "id"} for q in self.nudger_plans]

    def _send_nudger(self, kw: dict):
        """A nudger now: at the strongest storm, or at the storm nearest kw["target"] {x, y} (m)."""
        tgt = kw.get("target") or "strongest"
        if isinstance(tgt, dict):
            tgt = (float(tgt["x"]), float(tgt["y"]))
        try:
            return self.engine.send_nudger(
                tgt, self._motion(kw.get("motion", "mean")), 60.0 * float(kw.get("lead_min", 20.0)),
                aim=1000.0 * float(kw.get("aim_km", 0.0)), strength=float(kw.get("strength", 8.0)),
                radius=1000.0 * float(kw.get("radius_km", 6.0)))
        except ValueError as exc:
            self.engine.note("nudger", f"nudger not sent: {exc}")
            return None

    def _nudgers_step(self) -> None:
        t = self.engine.t
        for n in self.nudger_plans:
            if n["id"] not in self._plans_done and t >= 60.0 * float(n.get("at_min", 0.0)):
                self._plans_done.add(n["id"])
                self._send_nudger(n)

    def _towns(self) -> np.ndarray:
        """(n, 3) float32: world x, y (m) and population of the towns within 1500 km."""
        cs = geo.cities()
        if not cs:
            return np.zeros((0, 3), np.float32)
        x, y = self.proj.fwd(np.array([c["lat"] for c in cs]), np.array([c["lon"] for c in cs]))
        pop = np.array([c["pop"] for c in cs])
        keep = (np.hypot(x, y) < 1.5e6) & (pop >= 100)
        return np.stack([x[keep], y[keep], pop[keep]], axis=1).astype(np.float32)

    def _add_radar(self, kw):
        icao = kw.get("icao", "KXXX").upper()
        if kw.get("kind") == "dow":
            return self._deploy_dow(kw)
        site = next((x for x in geo.radar_sites() if x["icao"] == icao), None)
        if site is None and ("lat" not in kw or "lon" not in kw):
            raise ValueError(f"{icao} is not a radar site in data/radars.csv (Jackson's radar is KDGX, for example)")
        lat = float(kw.get("lat", site["lat"] if site else self.proj.lat0))
        lon = float(kw.get("lon", site["lon"] if site else self.proj.lon0))
        # the old "alias" key (folded velocities, on by default) is ignored: dealiased is the
        # default now, and 2.5-minute volumes
        # a TDWR (site type 3): C band, its own scans; its velocities folded at its low Nyquist by default,
        # as real TDWR data comes
        tdwr = (site is not None and site.get("type") == 3) or kw.get("kind") == "tdwr"
        # an international site (data/radars.csv type 4 = S band, run as a WSR-88D; 5 = C band)
        cband = not tdwr and ((site is not None and site.get("type") == 5) or kw.get("kind") == "cband")
        # a 1990s WSR-88D (radar.LEGACY_VCPS): a WSR-88D site only
        legacy = kw.get("kind") == "legacy" and not tdwr and not cband
        r = rd.Radar(icao=icao, lat=lat, lon=lon,
                     elev_m=float(self.sounding.zsfc) + 20.0,
                     kind="tdwr" if tdwr else "cband" if cband else "legacy" if legacy else "wsr88d",
                     vcp=int(kw.get("vcp", 11 if legacy else 212)), sails=int(kw.get("sails", 1)),
                     alias=not bool(kw.get("dealias", not tdwr)), volume_s=60.0 * float(kw.get("volume_min", 2.5)),
                     floor_dbz=float(kw.get("floor", -5.0)), noise=bool(kw.get("noise", True)),
                     clear_air=max(0.0, min(3.0, float(kw.get("clear_air", 1.0)))),
                     export=bool(kw.get("export", True)))
        x, y = self.proj.fwd(lat, lon)
        r.x, r.y = float(x), float(y)
        r.vol_start = self.engine.t if self.engine else 0.0
        self.radars = [q for q in self.radars if q.icao != icao] + [r]
        # radars added during the run are part of its settings too (settings.json, the next run)
        spec = dict({k: kw[k] for k in self.RADAR_KEYS if k in kw}, icao=icao)
        self.settings["radars"] = [q for q in self.settings.get("radars", [])
                                   if q.get("icao", "").upper() != icao] + [spec]
        if self.display_radar is None:
            self.display_radar = icao

    RADAR_KEYS = ("icao", "lat", "lon", "vcp", "sails", "dealias", "volume_min", "floor", "noise", "export",
                  "kind", "gate")

    def _deploy_dow(self, kw):
        """A Doppler on Wheels parked at (lat, lon) -- anywhere, during a run or in its setup -- scanning from now
        (radar.DOW_*): one of DOW_IDS, so GR2Analyst's polling list has it; deployed again elsewhere it is the
        same truck moved.  Its GR2Analyst feed lists only the volumes of this deployment (dow_since): earlier
        ones were made from another place, which GR2Analyst would draw at the new one."""
        icao = str(kw.get("icao", "DOW7")).upper()
        if icao not in rd.DOW_IDS:
            raise ValueError(f"{icao} is not a DOW (one of {', '.join(rd.DOW_IDS)})")
        lat, lon = float(kw["lat"]), float(kw["lon"])
        x, y = self.proj.fwd(lat, lon)
        ground = 0.0
        if self.env is not None and getattr(self.env, "terrain", None) is not None:
            try:
                ground = float(np.ravel(self.env.terrain.height(np.array([[float(x)]]), np.array([[float(y)]])))[0])
            except Exception:
                ground = 0.0
        r = rd.Radar(icao=icao, lat=lat, lon=lon, elev_m=float(self.sounding.zsfc) + ground + 4.0, kind="dow",
                     gate_m=float(kw.get("gate", 75.0)), alias=not bool(kw.get("dealias", True)),
                     floor_dbz=float(kw.get("floor", -5.0)), noise=bool(kw.get("noise", True)),
                     clear_air=max(0.0, min(3.0, float(kw.get("clear_air", 1.0)))), export=bool(kw.get("export", True)))
        r.x, r.y = float(x), float(y)
        r.vol_start = self.engine.t if self.engine else 0.0
        self.radars = [q for q in self.radars if q.icao != icao] + [r]
        self.dow_since[icao] = time.time()
        spec = dict({k: kw[k] for k in self.RADAR_KEYS if k in kw}, icao=icao, kind="dow")
        self.settings["radars"] = [q for q in self.settings.get("radars", [])
                                   if q.get("icao", "").upper() != icao] + [spec]
        if self.engine is not None:
            self.engine.note("radar", f"{icao} deployed at {lat:.4f}, {lon:.4f} ({r.cuts[0].gate:.0f} m gates, "
                                      f"{r.volume_seconds:.0f} s volumes)")
        self.display_radar = icao

    def _add_zone(self, kw):
        z = Zone(**{k: kw[k] for k in ("x0", "y0", "x1", "y1") },
                 rate=float(kw.get("rate", 3.0)), t_start=float(kw.get("t_start", 0.0)),
                 t_end=float(kw.get("t_end", 3 * 3600.0)), strength=float(kw.get("strength", 10.0)),
                 spacing=float(kw.get("spacing", 30000.0)), seed=int(kw.get("seed", 1)))
        z.id = 1 + max([q.id for q in self.zones], default=0)
        self.zones.append(z)

    def _work_ahead(self) -> bool:
        """Anything still to come: a storm start not yet over, an initiation area or
        a boundary still starting storms."""
        t = self.engine.t
        if self.synoptic is not None:
            return True                         # the weather goes on, and makes its own storms
        if any(s.t1 + 60.0 >= t for s in self.engine.starts):
            return True
        if any(z.t_end > t for z in self.zones):
            return True
        if any(c.get("active") for c in self.cyclones):
            return True
        ao = self._auto()
        if ao.get("on") and float(ao.get("end_min", 0.0)) * 60.0 > t:
            return True
        return any(b.init_rate > 0 and b.init_end > t for b in self.boundaries)

    # ------------------------------------------------------------ boundaries
    def _new_boundary(self, d: dict) -> Boundary:
        """A boundary from the UI / settings.  ``at`` (s) is the model time its
        points were drawn at; they are stored as where the boundary was at t = 0."""
        b = boundary_from_json(d)
        b.vel = None                    # where it meets other boundaries is worked out again (join_at_lows)
        b.id = 1 + max([q.id for q in self.boundaries], default=0)
        if "seed" not in d:
            b.seed = b.id
        at = float(d.get("at", 0.0) or 0.0)
        if at:
            vx, vy = b.velocity()
            b.points = [[x - vx * at, y - vy * at] for x, y in b.points]
        return b

    def _update_boundary(self, kw: dict) -> None:
        bid = int(kw.get("id", -1))
        for i, old in enumerate(self.boundaries):
            if old.id != bid:
                continue
            t = self.engine.t if self.engine else 0.0
            d = old.to_json()
            d.update({k: v for k, v in kw.items() if k not in ("id", "points", "at")})
            if "points" in kw:
                d["points"], d["at"] = kw["points"], float(kw.get("at", t))
            else:                       # keep it where it is now, whatever its new motion
                d["points"], d["at"] = old.points_at(t).tolist(), t
            d["vel"] = None             # its own motion; _apply_boundaries joins it at its lows again
            nb = boundary_from_json(d)
            vx, vy = nb.velocity()
            nb.points = [[x - vx * d["at"], y - vy * d["at"]] for x, y in nb.points]
            nb.id, nb.seed = old.id, old.seed
            if "side" not in kw:
                nb.side = old.side if old.kind == nb.kind else nb.side
            self.boundaries[i] = nb
            break
        self._apply_boundaries()

    def _low_motion(self):
        return low_motion(self.settings.get("low"))

    def _new_texture(self):
        amp = float(self._auto().get("texture", 1.0) or 0.0)
        if amp <= 0 or self.env is None:
            return None
        low = np.asarray(self.env.z_h) <= 1000.0           # it drifts with the 0-1 km wind
        flow = (float(np.mean(self.env.u_ground[low])), float(np.mean(self.env.v_ground[low])))
        return WarmSectorTexture(int(self.settings["seed"]) * 7919 + 17, min(amp, 3.0), flow)

    def _airmasses(self):
        brushes = self.settings.get("wind_brushes") or []
        if not (self.boundaries or self.texture is not None or brushes or self.pockets):
            return None
        return AirMasses(self.boundaries, self.env, self.texture, brushes, self.pockets)

    @staticmethod
    def _domain_shape(dom: dict):
        """A fixed domain's outline in its box ((fx, fy) fractions), or None for the whole box: an ellipse filling
        it, or an outline drawn on the map (the box then its extent)."""
        kind = dom.get("shape") or "rect"
        if kind == "ellipse":
            a = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
            return tuple((0.5 + 0.5 * math.cos(t), 0.5 + 0.5 * math.sin(t)) for t in a)
        if kind == "custom":
            P = [(min(max(float(p[0]), 0.0), 1.0), min(max(float(p[1]), 0.0), 1.0)) for p in dom.get("polygon") or []]
            return tuple(P) if len(P) >= 3 else None
        return None

    def set_pockets(self) -> None:
        """The running run's sounding pockets changed (server /api/run/pockets): built again and put in."""
        if self.env is None:
            return
        self.pockets = None
        self._build_pockets(self.settings, self._hurricane())
        if self.synoptic is None:
            self._apply_boundaries()

    def _build_pockets(self, s: dict, hc: dict) -> None:
        """The run's sounding pockets (pockets.py): each sounding eased into the run's over its radius, its winds
        turned onto the grid as the run's are."""
        specs = s.get("sounding_pockets") or []
        if not specs:
            return
        if hc.get("on") or self._syn_cfg().get("on"):
            self.notes.append("sounding pockets are not used in hurricane or synoptic runs (their air is their own)")
            return
        import dataclasses
        from . import pockets as pk

        def turn(snd_):
            a = float(self.proj.north_angle(self._turn_lon)) if self._turn_lon is not None else 0.0
            if abs(a) <= 0.05:
                return snd_
            gu, gv = self.proj.to_grid(np.asarray(snd_.u), np.asarray(snd_.v), self._turn_lon)
            return dataclasses.replace(snd_, u=np.asarray(gu), v=np.asarray(gv))

        try:
            self.pockets = pk.from_settings(specs, self.env, load_sounding, turn)
        except Exception as exc:
            self.pockets = None
            self.notes.append(f"sounding pockets left out: {exc}")
            return
        if self.pockets:
            from .sounding import diagnostics
            for d, it in zip([d for d in specs if d.get("sounding")], self.pockets.items):
                try:
                    dg = diagnostics(load_sounding(d["sounding"]))
                    extra = (f": MLCAPE {dg.get('mlcape') or 0:.0f} J/kg, 0-6 km shear "
                             f"{dg.get('shear6') or 0:.0f} m/s, 0-1 km SRH {dg.get('srh1') or 0:.0f} m2/s2")
                except Exception:
                    extra = ""
                self.notes.append(f"sounding pocket {it['title'] or 'sounding'} at ({it['x'] / 1000:.0f}, "
                                  f"{it['y'] / 1000:.0f}) km, {it['r'] / 1000:.0f} km across its blend{extra}")

    def _apply_boundaries(self) -> None:
        t = self.engine.t if self.engine else 0.0
        if self.synoptic is not None:           # kept for the next run; this one's air is the simulated weather
            self.settings["boundaries"] = [b.to_json() for b in self.boundaries]
            return
        self.lows = join_at_lows(self.boundaries, t, self._low_motion()) if self.boundaries else []
        am = self._airmasses()
        if self.engine is not None:
            self.engine.set_airmass(am)
        else:
            self.env.airmass = am
        self.settings["boundaries"] = [b.to_json() for b in self.boundaries]

    @staticmethod
    def _point_on(P: np.ndarray, s: float):
        """Point at arc length s along polyline P, and the unit normal to its left."""
        n = len(P)
        for i in range(n - 1):
            a, b = P[i], P[i + 1]
            L = float(np.hypot(*(b - a)))
            if s <= L or i == n - 2:
                t = min(max(s / max(L, 1.0), 0.0), 1.0)
                d = (b - a) / max(L, 1e-9)
                return a + t * (b - a), np.array([-d[1], d[0]])
            s -= L
        return P[-1], np.array([0.0, 1.0])

    def _boundaries_step(self):
        """Storms fire just ahead of boundaries (on the side away from the air
        mass), mostly at the boundary's hot spots."""
        if self.synoptic is not None:
            return                              # the simulated weather's own fronts fire storms (_synoptic_initiate)
        eng = self.engine
        am = self.env.airmass or None
        area = self._start_box()
        for b in self.boundaries:
            if b.init_rate <= 0 or not (b.init_start <= eng.t <= b.init_end):
                continue
            # its length now: fronts joined at a surface low pivot about it and stretch or shrink
            P = b.points_at(eng.t)
            L = float(np.sum(np.hypot(*np.diff(P, axis=0).T))) if len(P) > 1 else 0.0
            if L < 1000.0 or self._rng.random() >= b.init_rate * (L / 100e3) * eng.dt / 3600.0:
                continue
            if self._rng.random() < self._line_chance(b, P, eng.t):
                if eng.t - self._line_last.get(b.id, -1e9) < LINE_EVERY:
                    continue
                if self._fire_line(b, P, L, area, am):
                    self._line_last[b.id] = eng.t
                continue
            for _ in range(40):
                sa = self._rng.uniform(0.0, L)
                hs = float(hot_spots(b, np.array([sa]), eng.t)[0]) if b.variation > 0 else 1.0
                if self._rng.random() > 0.15 + 0.85 * hs:
                    continue
                pt, nl = self._point_on(P, sa)
                off = self._rng.uniform(0.4, 1.2) * b.hw + 3000.0 - float(b.bulge(sa, eng.t))
                x, y = pt - b.side * nl * off
                if area is not None and not (area[0] <= x <= area[2] and area[1] <= y <= area[3]):
                    continue                    # outside the firing area or the fixed-size domain
                # its warm side must be open warm sector, not another boundary's air mass (a cold
                # front behind a dryline has the dry slot there: storms started in it die)
                if am is not None and float(am.weight(np.array([x]), np.array([y]), eng.t)[0]) > 0.25:
                    continue
                if self._storm_near(x, y, 25000.0, echo=bool(self._auto().get("on"))):
                    continue
                why = f"{b.to_json()['label'].lower()} #{b.id}"
                if self._start_storm(x, y, why, axis=math.atan2(-nl[0], nl[1])) is not None:
                    eng.note("boundary", f"storm fired along {why}")
                break

    def _latch_step(self) -> None:
        """Storms latch onto boundaries: every LATCH_EVERY s each deep updraft (LATCH_W m/s) ahead of a
        boundary within LATCH_AHEAD (or up to one zone half-width behind it) gets a short, shallow lifting
        push (0.5-1.7 km, 5 m/s) where the boundary's convergence is, abeam of it and ~6 km downstream along
        the boundary, moving with the boundary plus the storm's motion along it.  New updraft keeps growing
        on the boundary side, so the storm turns to ride the boundary instead of leaving it, ingesting the
        backed, strongly sheared low-level air there.  The finished pushes are dropped from the run's list.
        In synoptic / ERA5 / HRRR runs only established storms latch (LATCH_W_STORM, LATCH_AGE), strongest first,
        at most the run's max storms at once, never an updraft a push itself set off, and none further than
        LATCH_BEHIND behind the boundary."""
        eng = self.engine
        bounds = self.syn_boundaries if self.synoptic is not None else self.boundaries
        if eng.t < self._next_latch or not bounds:
            return
        self._next_latch = eng.t + LATCH_EVERY
        eng.starts[:] = [s for s in eng.starts if s.why != "latch" or s.t1 > eng.t]
        self._latch_sites = [q for q in self._latch_sites if eng.t - q[0] < LATCH_CHILD_S + 600.0]
        # at most as many pushes at once as the storms the run allows at once (synoptic mode's max storms)
        cap = int(self._syn_cfg().get("max_storms", 10)) if self.synoptic is not None else 10 ** 6
        n = sum(1 for s in eng.starts if s.why == "latch")
        for tr in sorted(eng.tracks.live, key=lambda q: -q.last[3]):
            _, x, y, w = tr.last
            if w < (LATCH_W_STORM if self.synoptic is not None else LATCH_W) * self.storm_scale(x, y) or n >= cap:
                continue
            if eng.t - tr.born < LATCH_AGE and self.synoptic is not None:
                continue
            bt, bx, by = tr.pts[0][:3]
            if any(q[3] != tr.id and 0.0 <= bt - q[0] <= LATCH_CHILD_S
                   and math.hypot(bx - q[1], by - q[2]) <= LATCH_CHILD_KM * 1000.0 for q in self._latch_sites):
                continue                                  # born from a push: no pushes of its own
            best = None
            for b in bounds:
                P = b.points_at(eng.t)
                if len(P) < 2:
                    continue
                d, over, along, nx, ny = signed_distance(P, np.array([x]), np.array([y]), normal=True)
                if over[0] > 0:
                    continue
                ahead = -(d[0] * b.side + float(b.bulge(along[0], eng.t)))
                if -min(b.hw, LATCH_BEHIND) <= ahead <= LATCH_AHEAD and (best is None or abs(ahead) < abs(best[1])):
                    best = (b, ahead, along[0], np.array([nx[0], ny[0]]))
            if best is None:
                continue
            b, ahead, s, nl = best
            tang = np.array([nl[1], -nl[0]])
            sv = np.asarray(tr.velocity(self.env.storm_motion))
            along_v = float(sv @ tang)
            warm = -b.side * nl
            lead = math.copysign(6000.0, along_v) if abs(along_v) > 1.0 else 0.0
            # toward the boundary, at most 8 km from the updraft: it draws the storm in rather than starting another
            px, py = np.array([x, y]) - warm * float(np.clip(ahead - (0.3 * b.hw + 1500.0), -8000.0, 8000.0)) + tang * lead
            vx, vy = np.asarray(b.velocity()) + along_v * tang
            s0 = StormStart(x=float(px), y=float(py), t0=eng.t, kind="nudge", radius=5000.0, strength=5.0,
                            zc=1100.0, depth=600.0, duration=LATCH_EVERY + 60.0, ramp=60.0,
                            vx=float(vx), vy=float(vy), why="latch")
            s0.id = eng._next_start_id
            eng._next_start_id += 1
            eng.starts.append(s0)
            self._latch_sites.append((eng.t, float(px), float(py), tr.id))
            n += 1

    def _line_chance(self, b, P, t) -> float:
        """How likely a boundary's next storms come up as a squall line rather than separate cells.
        Its mode "line" / "cells" says so; "auto" follows Dial et al. (2010) and Bluestein & Weisman
        (2000): cells stay discrete when the 0-6 km shear crosses the boundary and the storms move
        off it into the warm sector, and grow into a line when the shear runs along it (under
        ~10 m/s across it) or the boundary keeps up with them."""
        mode = getattr(b, "mode", "auto") or "auto"
        if mode in ("line", "cells"):
            return 1.0 if mode == "line" else 0.0
        z = np.asarray(self.env.z_h)
        k6 = min(int(np.searchsorted(z, 6000.0)), len(z) - 1)
        shear = np.array([float(self.env.u_ground[k6] - self.env.u_ground[0]),
                          float(self.env.v_ground[k6] - self.env.v_ground[0])])
        d = P[-1] - P[0]
        d = d / max(float(np.hypot(*d)), 1.0)
        warm = -b.side * np.array([-d[1], d[0]])            # toward the warm side
        across = abs(float(shear @ warm))
        away = float((np.asarray(self._motion("rm")) - np.asarray(b.velocity())) @ warm)
        p = min(max((LINE_ACROSS + 5.0 - across) / 10.0, 0.0), 1.0)
        if away < LINE_AWAY:                                # the boundary keeps up with its storms
            p = max(p, min(1.0, 0.5 + 0.1 * (LINE_AWAY - away)))
        return p

    def _fire_line(self, b, P, L, area, am, spacing=LINE_SPACING) -> bool:
        """A squall line: a 60-140 km stretch of the boundary goes up at once, as starts every
        7-10 km just ahead of it (broken-line formation, Bluestein & Jain 1985) whose cold pools
        join before the cells can separate.  They move with the boundary plus the mean wind along
        it, not as right movers, so the line stays on its forcing."""
        eng, r = self.engine, self._rng
        seg = min(L, r.uniform(*LINE_LENGTH))
        s0 = r.uniform(0.0, max(L - seg, 0.0))
        mean, bv = np.asarray(self._motion("mean")), np.asarray(b.velocity())
        ao = self._auto()
        push = float(ao.get("push", 10.0)) if ao.get("on") else 10.0
        why = f"{b.to_json()['label'].lower()} #{b.id} line"
        n, s = 0, s0 + r.uniform(0.0, spacing[0])
        while s <= s0 + seg:
            pt, nl = self._point_on(P, s)
            x, y = pt - b.side * nl * (r.uniform(0.2, 0.6) * b.hw + 2000.0 - float(b.bulge(s, eng.t)))
            s += r.uniform(*spacing)
            if area is not None and not (area[0] <= x <= area[2] and area[1] <= y <= area[3]):
                continue
            if am is not None and float(am.weight(np.array([x]), np.array([y]), eng.t)[0]) > 0.25:
                continue
            if self._storm_near(x, y, 0.8 * spacing[0], echo=False):
                continue
            tang = np.array([nl[1], -nl[0]])
            vx, vy = bv + float(mean @ tang) * tang
            ramp = r.uniform(300, 600)
            zc, dp = ((1200, 1700), (1200, 1600)) if CLASSIC_STARTS else ((1000, 1500), (1000, 1400))
            if eng.add_start(StormStart(x=float(x), y=float(y), t0=eng.t + r.uniform(0, 480), kind="nudge",
                                        radius=r.uniform(5000, 7000), strength=push * r.uniform(0.8, 1.15),
                                        zc=r.uniform(*zc), depth=r.uniform(*dp),
                                        duration=ramp + r.uniform(600, 1100), ramp=ramp,
                                        vx=float(vx), vy=float(vy), why=why,
                                        **self._shape(math.atan2(tang[1], tang[0]), noise=(0.3, 0.6)))) is not None:
                n += 1
        if n:
            eng.note("boundary", f"a squall line ({n} cells over {seg / 1000:.0f} km) fired along {why[:-5]}")
        return n > 0

    # ------------------------------------------------------------ thread
    def start(self):
        if self._thread is not None:
            self._pause.clear()
            return
        self._thread = threading.Thread(target=self._run, name="arhtmodelviewer-sim", daemon=True)
        self._thread.start()

    def pause(self):
        self._pause.set()

    def resume(self):
        self._pause.clear()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def release_gpu(self) -> bool:
        """Drop a stopped run's GPU arrays (its patches, reference column and checkpoints) so
        the next run's memory budget sees them free.  False while the run is still stepping."""
        if self._thread is not None and self._thread.is_alive():
            return False
        self.engine = None
        return True

    def _run(self):
        from .engine import _out_of_memory
        fields_cache = {}
        faulted = False
        oom_times = []
        try:
            if self.engine is None:
                self.build()
            self.status = "running"
            self.wall_start = time.perf_counter()
            model0 = self.engine.t
            next_frame = self.engine.t
            fields_time = -1e9
            self._autosave_at = time.time() + AUTOSAVE_EVERY
            while not self._stop.is_set():
                self.heartbeat = time.time()
                if self.heartbeat >= self._autosave_at:
                    self._autosave_at = self.heartbeat + AUTOSAVE_EVERY
                    self._autosave()
                while True:
                    try:
                        name, kw = self.cmds.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self._handle(name, kw)
                    except Exception as exc:  # a bad command must not kill the run
                        self.engine.note("error", f"{name}: {exc}")
                if self._pause.is_set():
                    self.status = "paused"
                    time.sleep(0.05)
                    self.wall_start = time.perf_counter()
                    model0 = self.engine.t
                    continue
                eng = self.engine
                if not eng.patches and not self._work_ahead():
                    # no storm and nothing scheduled: hold the clock instead of racing through
                    # empty hours (and empty radar volumes) until something is added
                    self.status = "waiting"
                    time.sleep(0.1)
                    self.wall_start = time.perf_counter()
                    model0 = eng.t
                    continue
                self.status = "running"
                if _DEBUG_FAULT_AT is not None and eng.t >= _DEBUG_FAULT_AT:
                    _debug_gpu_fault()
                if not eng.patches:
                    time.sleep(0.01)      # nothing to compute yet: do not spin the GPU / CPU
                self._synoptic_prepare()  # the weather's own work, outside the lock the page reads under
                self.lock.hand_over()     # the page's requests first (the 3-D view, the state), then the next step
                with self.lock:
                    stepping = True
                    timing = _DEBUG_TIMING and getattr(self, "_time_left", 0) > 0
                    if timing:
                        self._time_left -= 1
                        tw0 = time.perf_counter()
                    try:
                        eng.step()
                        self._ef_step()
                        if timing:
                            import cupy as _cp
                            _cp.cuda.Device().synchronize()
                            log.warning("timing: step %.2f s (dt %.2f s, %d nests)", time.perf_counter() - tw0,
                                        eng.dt, len(eng.nest_patches()))
                            tw0 = time.perf_counter()
                        if _DEBUG_OOM_AT and eng.t >= _DEBUG_OOM_AT[0]:
                            _DEBUG_OOM_AT.pop(0)
                            import cupy as cp
                            cp.empty(1 << 45, cp.uint8)          # a real cudaErrorMemoryAllocation, mid-step
                        stepping = False
                        live = {p.id for p in eng.patches + eng.nest_patches()}
                        if set(fields_cache) - live:
                            fields_cache = {k: v for k, v in fields_cache.items() if k in live}
                        if eng.t < fields_time:          # a watchdog rollback: those fields are from the discarded run
                            fields_cache, fields_time = {}, -1.0e9
                        self._zones_step()
                        self._boundaries_step()
                        self._latch_step()
                        self._warm_sector_step()
                        self._nudgers_step()
                        self._mergers_step()
                        self._synoptic_step()
                        self._cyclones_step()
                        if eng.patches and not (self._next_swath - SWATH_EVERY <= eng.t < self._next_swath):
                            self._update_swaths()           # (also when a rollback took the clock back)
                        # radar fields are refreshed at most every ~10 model seconds
                        due = self._radar_due()
                        # also: right after a tornado zoom is placed, and while one runs every FRAME_WALL_ZOOM s of
                        # real time (a 10 m or 1 m zoom takes hours of real time for a 60 s frame interval)
                        # -- but never more than FRAME_SHARE of the run's time: each one rebuilds every grid's radar
                        # fields, and a ~10 m zoom's 82 s steps each got one (twice the run's time)
                        need_frame = (eng.t >= next_frame or getattr(self, "_frame_now", False)
                                      or (eng.tornado is not None
                                          and time.time() - getattr(self, "_frame_wall", 0.0)
                                          >= max(FRAME_WALL_ZOOM, getattr(self, "_frame_cost", 0.0) / FRAME_SHARE)))
                        tf0 = time.perf_counter()
                        if (due or need_frame) and eng.patches:
                            if eng.t - fields_time >= RADAR_FIELDS_EVERY or need_frame or set(fields_cache) != live:
                                self.env.radar_xy = [(r.x, r.y) for r in self.radars]     # fine lines: close in only
                                self.env.radar_clean = any(r.kind == "dow" for r in self.radars)   # DOWs: no debris
                                # the storms' echo as the model makes it: no hook joining or edge sharpening
                                fields_cache = rd.build_fields(eng.patches, self.env, self.tornadoes,
                                                               nests=eng.nest_patches(),
                                                               holes=eng.nests.holes() if eng.nests else ())
                                fields_time = eng.t
                                if timing:
                                    log.warning("timing: radar fields %.2f s", time.perf_counter() - tw0)
                                    tw0 = time.perf_counter()
                                self.vol_fields = (fields_cache, eng.t)      # the page's 3-D view (volume.py)
                                self._record_tornadoes(eng.t)
                                if not self._hurricane().get("on"):      # no SVR / TOR polygons on a hurricane
                                    self.warnings.step(fields_cache, eng.t)
                        try:
                            self.metars.step(fields_cache, eng.t)
                        except Exception as exc:           # an observation must never stop the run
                            if not getattr(self, "_metar_failed", False):
                                self._metar_failed = True
                                log.exception("METARs failed")
                                eng.note("error", f"METARs: {exc}")
                        if due:
                            self._radar_step(list(fields_cache.values()))
                        if need_frame:
                            if timing:
                                tw1 = time.perf_counter()
                            self._make_frame(fields_cache)
                            self._frame_cost = time.perf_counter() - tf0      # with the radar fields it rebuilt
                            self._vol_history(fields_cache, eng.t)
                            if timing:
                                log.warning("timing: frame %.2f s (since the step: %.2f s)",
                                            time.perf_counter() - tw1, time.perf_counter() - tw0)
                            next_frame = eng.t + float(self.settings.get("frame_interval", 60.0))
                    except Exception as exc:
                        if not _out_of_memory(exc):
                            raise
                        # the card is full (nests, the tornado zoom, a radar's fields on top of the patches):
                        # give memory back and go on with less instead of the run ending
                        fields_cache, fields_time = {}, -1.0e9
                        self.vol_fields = None
                        oom_times = [t for t in oom_times if time.time() - t < 120.0] + [time.time()]
                        if len(oom_times) > 4:
                            raise RuntimeError(f"the GPU keeps running out of memory ({exc}): use a coarser grid, "
                                               "smaller patches or fewer nests") from exc
                        what = eng.recover_memory(stepping)
                        log.warning("out of GPU memory (%s): %s", exc, what)
                        eng.note("memory", f"the GPU ran out of memory: {what}; the run goes on")
                        next_frame = min(next_frame, eng.t)
                # Level II encoding (CPU threads) behind fast volumes (30 s, several radars, a quick
                # run): wait for it rather than pile up volumes in memory (~80 MB of sweeps each).
                # Outside the lock, which the writers take when they finish
                while self._l2_pending > max(4, 2 * len(self.radars)) and not self._stop.is_set():
                    time.sleep(0.02)
                # speed limit / measurement
                wall = time.perf_counter() - self.wall_start
                model = eng.t - model0
                self.speed = model / wall if wall > 0 else 0.0
                self._pace(eng.t)
                lim = float(self.settings.get("speed_limit", 0.0) or 0.0)
                if lim > 0 and model / lim > wall:
                    time.sleep(min(model / lim - wall, 0.5))
            self.status = "stopped"
        except Exception as exc:
            self.error = f"{exc}\n{traceback.format_exc()}"
            if type(exc).__name__ in ("MissingTableAssets", "TableAssetError"):
                # gpuwm names its own `gpuwm fetch-tables`, which (run outside the viewer) stages
                # into the user profile, not runtime\tables where the viewer reads them
                self.error = ("This microphysics needs lookup tables that are not installed: run Setup.cmd "
                              "again (TEMPO: \"Setup.cmd tempo\" from a command prompt in the folder), then "
                              f"start the run again.\n{self.error}")
            log.exception("simulation failed")
            faulted = gpu_fault(exc)
            if faulted:
                # the CUDA context is gone for good: every later GPU call in this process
                # fails the same way, so the server restarts (see __main__.supervise)
                restart_after_gpu_fault(self, exc)
            self.status = "error"
        self._autosave(wait=True)
        try:                                    # the tornado list and damage survey (survey.py) in its folder
            from .survey import Survey
            st = json.loads((self.run_dir / "state.json").read_text())
            if st.get("tornado_tracks"):
                Survey(st).write(self.run_dir / "survey")
        except Exception:
            log.exception("could not write the run's tornado list / damage survey")
        # a stopped or failed run cannot go on: give its GPU memory back now instead of when the
        # next run starts (frames, radar pictures and Level II files are on the host and stay)
        fields_cache = None
        if self.engine is not None and not faulted:
            try:
                self.engine.release_gpu()
                free_gpu()
            except Exception:
                log.exception("could not release the run's GPU memory")

    def _autosave(self, wait: bool = False) -> None:
        """The run's state (what the page's /api/state shows: tornado tracks, warnings, storm tracks, events,
        settings, map and radars) to state.json in its folder, so a crash, a power cut or a closed window loses
        at most AUTOSAVE_EVERY of it (the tornado list, Google Earth and loop tools read this file).  Taken under
        the lock, written in the background, replaced in one step (never half a file)."""
        if self.engine is None:
            return
        try:
            with self.lock:
                d = self.info()
        except Exception:
            log.exception("autosave: could not read the run's state")
            return

        def write():
            try:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.run_dir / ".state.json.part"
                tmp.write_text(json.dumps(d, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
                os.replace(tmp, self.run_dir / "state.json")
            except Exception:
                log.exception("autosave: could not write state.json")

        if wait:
            write()
        else:
            threading.Thread(target=write, name="autosave", daemon=True).start()

    # ------------------------------------------------------------ hurricanes
    def _hurricane(self) -> dict:
        return {**DEFAULT_SETTINGS["hurricane"], **(self.settings.get("hurricane") or {})}

    def _build_ocean(self, hc: dict) -> None:
        """Hurricane mode: the sea under the run (hurricane.Ocean): the procedural SST map, water wherever
        Natural Earth has no land, and the reference column's sea at the SST under the first surface low (else
        the map centre)."""
        from . import hurricane as hur
        env = self.env
        sst = hur.ProceduralSST(int(hc.get("sst_seed") or 1), float(hc.get("sst_offset") or 0.0))
        land_fn = hur.land_mask()                 # Natural Earth land (the land cover stops at 20 N)
        cs = self.settings.get("cyclones") or []
        x, y = (float(cs[0]["x"]), float(cs[0]["y"])) if cs else (0.0, 0.0)
        lat, lon = self.proj.inv(x, y)
        ref = float(sst.celsius(lat, lon))
        env.ocean = hur.Ocean(env, sst, env.latlon, land_fn, wake=bool(hc.get("wake", True)),
                              mld=float(hc.get("mld") or 50.0), ref_sst_c=ref)
        env.ocean.ambient = min(max(float(hc.get("ambient", 0.3) or 0.0), 0.0), 1.0)
        self.engine.note("cyclone", f"hurricane mode: procedural SST map (seed {sst.seed}, offset {sst.offset:+.1f} C; "
                                    f"{ref:.1f} C under {'the first surface low' if cs else 'the map centre'}), "
                                    "sea-state drag, evaporation and sensible heat over water"
                                    + (f", a {env.ocean.mld:.0f} m mixed layer's cold wake" if env.ocean.wake else "")
                                    + f"; f-plane at {env.physics.latitude:.1f} deg")

    def _add_cyclone(self, kw: dict) -> None:
        """A surface low: a balanced warm-core seed vortex (hurricane.apply_vortex) of vmax_kt at rmw_km, its
        circulation r0_km across, then tracked (_cyclones_step)."""
        eng, env = self.engine, self.env
        if getattr(env, "ocean", None) is None:
            raise ValueError("surface lows need hurricane mode (Tropics tab), set before the run starts")
        x, y = float(kw["x"]), float(kw["y"])
        vmax = min(max(float(kw.get("vmax_kt", 30.0)), 10.0), 120.0) * KT
        rmw = min(max(float(kw.get("rmw_km", 90.0)), 15.0), 250.0) * 1000.0
        r0 = min(max(float(kw.get("r0_km", 500.0)), 2.0 * rmw / 1000.0), 1400.0) * 1000.0
        if eng.fixed:              # the circulation must fit the domain: its edges hold the plain environment
            half = 0.5 * min(eng.fixed, eng.fixed_ny) * env.grid.dx
            if r0 > 1.1 * half:
                eng.note("cyclone", f"surface low's circulation cut from {r0 / 1000:.0f} to {1.1 * half / 1000:.0f} km "
                                    "to fit the domain (Tropics tab: a bigger domain keeps a large low whole)")
                r0 = max(1.1 * half, 2.0 * rmw)
        vx, vy = self._motion("mean")
        s = StormStart(x=x, y=y, t0=eng.t, kind="vortex", radius=r0, strength=vmax, duration=0.0, vx=vx, vy=vy,
                       why="surface low", rmw=rmw)
        if eng.add_start(s) is None:
            return
        cid = 1 + max([c["id"] for c in self.cyclones], default=0)
        self.cyclones.append({"id": cid, "x": x, "y": y, "vx": vx, "vy": vy, "t": eng.t, "t0": eng.t, "rmw": rmw,
                              "pmin": None, "vmax": vmax, "track": [], "active": True, "cat": "TD",
                              "next_push": eng.t + 120.0, "pushes": 0, "seed_rmw": rmw})
        eng.hold_point = (x, y, 2.5 * rmw)
        if env.f == 0.0:
            eng.note("cyclone", "no Coriolis: the surface low spins down instead of organising (Physics tab)")

    def _cyclones_step(self) -> None:
        """Every CYCLONE_EVERY s: each surface low's centre (lowest smoothed surface pressure near where its
        motion puts it), lowest pressure, maximum 1-min wind and category; the domain kept on the strongest;
        and, until it reaches push_until_kt, an updraft push near its core every push_every_min."""
        eng = self.engine
        if not self.cyclones or getattr(self.env, "ocean", None) is None or eng.t < self._next_cyclone:
            return
        self._next_cyclone = eng.t + CYCLONE_EVERY
        from . import hurricane as hur
        hc = self._hurricane()
        lead = None
        for c in self.cyclones:
            if not c["active"] or eng.t <= c["t0"]:
                continue
            dt_ = eng.t - c["t"]
            px, py = c["x"] + c["vx"] * dt_, c["y"] + c["vy"] * dt_
            fix = hur.cyclone_fix(self.env, eng.patches, px, py, search=100000.0, ocean=self.env.ocean)
            if fix is None:
                c["active"] = False
                eng.note("cyclone", f"cyclone {c['id']} left the domain")
                continue
            tr = c["track"]
            tr.append([eng.t, fix["x"], fix["y"], round(fix["pmin"], 1), round(fix["vmax"] / KT, 1)])
            # its motion: least squares over the last hour of fixes (the centre jitters from fix to fix)
            recent = [q for q in tr if q[0] >= eng.t - 3600.0]
            if len(recent) >= 6:
                a = np.asarray(recent)
                tt = a[:, 0] - a[:, 0].mean()
                den = float((tt * tt).sum())
                if den > 0:
                    c["vx"] = float((tt * (a[:, 1] - a[:, 1].mean())).sum()) / den
                    c["vy"] = float((tt * (a[:, 2] - a[:, 2].mean())).sum()) / den
            c.update(x=fix["x"], y=fix["y"], t=eng.t, pmin=fix["pmin"], vmax=fix["vmax"], rmw=fix["rmw"])
            code, name = hur.category(fix["vmax"] / KT)
            order = ("TD", "TS", "C1", "C2", "C3", "C4", "C5")
            # down a category only once the wind is 4 kt below its threshold
            if (order.index(code) < order.index(c["cat"])
                    and order.index(hur.category(fix["vmax"] / KT + 4.0)[0]) >= order.index(c["cat"])):
                code = c["cat"]
            # a new category is announced once it has held for two fixes (no flapping at a threshold)
            c["cat_new"] = code if code != c["cat"] else None
            c["cat_held"] = c.get("cat_held", 0) + 1 if c["cat_new"] == c.get("cat_was_new") and c["cat_new"] else 0
            c["cat_was_new"] = c["cat_new"]
            if c["cat_new"] and c["cat_held"] >= 1:
                was = c["cat"]
                c["cat"] = code
                c["cat_held"] = 0
                eng.note("cyclone", f"cyclone {c['id']} is now a {name.lower()} ({was} -> {code}): "
                                    f"{fix['vmax'] / KT:.0f} kt, {fix['pmin']:.0f} hPa")
            if (hc.get("core_pushes") and fix["vmax"] / KT < float(hc.get("push_until_kt") or 64.0)
                    and eng.t >= c["next_push"]):
                self._core_push(c)
                c["next_push"] = eng.t + 60.0 * max(2.0, float(hc.get("push_every_min") or 10.0))
            if lead is None or fix["vmax"] > lead["vmax"]:
                lead = c
        if lead is not None:
            eng.hold_point = (lead["x"], lead["y"], 150000.0)

    def _core_push(self, c: dict) -> None:
        """An updraft push in the cyclone (genesis sooner): 6-9 m/s for 15 min, 6-9 km wide, moving with it, on a
        ring tied to the SEED low's radius of maximum wind -- 0.5-2 of it, and one push in four out in a rainband
        at 2-4 of it (at most 350 km).  Pushes placed by the measured radius of maximum wind followed its
        jumps while the storm formed down to the 20 km floor, and forced a pinhole core (R34 46 km for a
        Category 2, 2026-10-30)."""
        r = self._rng
        rmw = float(c.get("seed_rmw") or 75000.0)
        f = r.uniform(2.0, 4.0) if r.random() < 0.25 else r.uniform(0.5, 2.0)
        rr, th = min(rmw * f, 350000.0), r.uniform(0.0, 2.0 * math.pi)
        s = StormStart(x=c["x"] + rr * math.cos(th), y=c["y"] + rr * math.sin(th), t0=self.engine.t, kind="nudge",
                       radius=r.uniform(6000.0, 9000.0), strength=r.uniform(6.0, 9.0), zc=r.uniform(1200.0, 1800.0),
                       depth=r.uniform(1200.0, 1600.0), duration=900.0, ramp=300.0, vx=c["vx"], vy=c["vy"],
                       why="core push")
        if self.engine.add_start(s) is not None:
            c["pushes"] += 1

    def _cyclone_json(self, c: dict) -> dict:
        from .hurricane import category
        kt = (c["vmax"] or 0.0) / KT
        code, name = category(kt)
        return {"id": c["id"], "x": c["x"], "y": c["y"], "active": c["active"], "pmin": c["pmin"],
                "vmax_kt": round(kt, 1), "vmax_mph": round(kt * 1.15078, 0), "cat": code, "name": name,
                "rmw_km": round((c.get("rmw") or 0.0) / 1000.0, 1), "pushes": c.get("pushes", 0),
                "motion": [c["vx"], c["vy"]], "track": c["track"][-1500:]}

    # ------------------------------------------------------------ automatic storms
    def _auto(self) -> dict:
        return self.settings.get("auto_storms") or {}

    def _shape(self, axis: float | None = None, noise=(0.5, 1.0)) -> dict:
        """A random ragged shape for a nudge (StormStart aspect .. seed): stretched up to 2.2 : 1 (along ``axis``,
        rad, +-25 deg, when given), a lumpy outline, a lean downshear (0-3 km shear) of 0.05-0.3, 3-D noise.
        With CLASSIC_STARTS: none (a round push, the old nudging)."""
        if CLASSIC_STARTS:
            return {"aspect": 0.0}
        r = self._rng
        z = np.asarray(self.env.z_h)
        k3 = min(int(np.searchsorted(z, 3000.0)), len(z) - 1)
        sx = float(self.env.u_ground[k3] - self.env.u_ground[0])
        sy = float(self.env.v_ground[k3] - self.env.v_ground[0])
        sm = max(math.hypot(sx, sy), 1e-6)
        lean = r.uniform(0.05, 0.3) * min(sm / 15.0, 1.0)
        ang = (axis + r.uniform(-0.45, 0.45)) if axis is not None else r.uniform(0.0, math.pi)
        return {"aspect": r.uniform(1.0, 2.2), "angle": ang, "lobes": r.uniform(0.15, 0.45),
                "lobe_phase": r.uniform(0.0, 2.0 * math.pi), "tilt_x": lean * sx / sm, "tilt_y": lean * sy / sm,
                "noise": r.uniform(*noise), "seed": r.randrange(1, 2 ** 23), "tau": NUDGE_TAU}

    def _thermal_cluster(self, x: float, y: float, rh: float, wt: float, ramp: float, dur: float, vx: float,
                         vy: float, why: str, t0: float | None = None, axis: float | None = None):
        """A storm start the way the air does it: a weak, ragged mesoscale lift over the start (LIFT_W of the
        strength, the whole radius ``rh``, ``ramp`` / ``dur``) and on it CLUSTER_N thermals per 10 km of radius,
        each 1.6-3.2 km across (or THERMAL_MIN_DX cells), THERMAL_W of the strength, rooted in the boundary layer
        (0.7-1.3 km), with a random ragged shape, going up one after another over the first ~25 min for 4-10 min
        each.  Most die; the model picks the ones that grow and merges them into the storm.  On grids coarser than
        CLUSTER_DX: one softened, ragged push.  Returns the lift (the storm's start for mergers and the map)."""
        eng, r = self.engine, self._rng
        now = eng.t if t0 is None else t0
        dx = self.env.grid.dx
        if CLASSIC_STARTS:                          # 1.6's main push
            return eng.add_start(StormStart(x=float(x), y=float(y), t0=now, kind="nudge", radius=rh, strength=wt,
                                            zc=r.uniform(1200, 1900), depth=r.uniform(1200, 1700), duration=dur,
                                            ramp=ramp, vx=vx, vy=vy, why=why, aspect=0.0))
        if dx > CLUSTER_DX:
            return eng.add_start(StormStart(x=float(x), y=float(y), t0=now, kind="nudge", radius=rh, strength=wt,
                                            zc=r.uniform(1000, 1500), depth=r.uniform(1000, 1400), duration=dur,
                                            ramp=ramp, vx=vx, vy=vy, why=why,
                                            **dict(self._shape(axis, noise=(0.3, 0.6)), tau=10.0)))
        lift = eng.add_start(StormStart(x=float(x), y=float(y), t0=now, kind="nudge", radius=rh, strength=LIFT_W * wt,
                                        zc=r.uniform(800, 1100), depth=r.uniform(700, 900), duration=dur,
                                        ramp=ramp, vx=vx, vy=vy, why=why, **dict(self._shape(axis), tau=LIFT_TAU)))
        if lift is None:
            return None
        if CORE_W > 0:
            core = StormStart(x=float(x), y=float(y), t0=now, kind="nudge", radius=CORE_R * rh,
                              strength=CORE_W * wt, zc=r.uniform(1000, 1500), depth=r.uniform(1000, 1400),
                              duration=dur, ramp=ramp, vx=vx, vy=vy, why="thermal", tau=NUDGE_TAU, aspect=1.0)
            core.id = eng._next_start_id
            eng._next_start_id += 1
            eng.starts.append(core)
        n = int(round(r.uniform(*CLUSTER_N) * rh / 10000.0))
        n = min(max(n, 4), 24)
        area = 0.8 * rh
        span = min(ramp + 0.6 * dur, 1500.0)
        for k in range(n):
            # the first two go up at once near the middle, the rest anywhere on the lift through its first ~25 min
            tk = now + (r.uniform(0.0, 120.0) if k < 2 else r.uniform(60.0, span))
            rr = math.sqrt(r.random()) * area * (0.4 if k < 2 else 1.0)
            th = r.uniform(0.0, 2.0 * math.pi)
            px = x + vx * (tk - now) + rr * math.cos(th)
            py = y + vy * (tk - now) + rr * math.sin(th)
            tr = max(r.uniform(*THERMAL_R), THERMAL_MIN_DX * dx)
            s0 = StormStart(x=float(px), y=float(py), t0=float(tk), kind="nudge", radius=tr,
                            strength=wt * r.uniform(*THERMAL_W), zc=r.uniform(700, 1300), depth=r.uniform(500, 900),
                            duration=r.uniform(240.0, 600.0), ramp=r.uniform(90, 240), vx=vx, vy=vy, why="thermal",
                            **dict(self._shape(), tau=THERMAL_TAU))
            s0.id = eng._next_start_id
            eng._next_start_id += 1
            eng.starts.append(s0)
            if not eng._ensure_covered(s0):         # (inside the lift's patch, so practically never)
                eng.starts.remove(s0)
        eng.note("start", f"storm start #{lift.id}: {n} thermals over {2 * area / 1000:.0f} km on a "
                          f"{LIFT_W * wt:.1f} m/s lift")
        return lift

    def _start_storm(self, x: float, y: float, why: str, strength: float = 10.0, axis: float | None = None,
                     motion=None, outbreak: bool = False):
        """A storm start at (x, y) now, each with its own character so no two grow alike: a main
        push of its own size (x0.6-1.4), strength (x0.8-1.25), height and timing, moving at about
        the Bunkers right-mover motion (+-20 %, +-15 deg).  In OutbreakLab mode it grows for 7-20 min
        and lasts 17-42 min, else 4-10 min and 10-20 min.  Then 0-5 smaller satellite towers that go
        up at different times and merge into it, scattered over an ellipse (round cluster .. line of
        towers; lined up along ``axis`` (rad) on a boundary), and 0-3 separate cells sent into it
        later on their own motion (_plan_mergers).  Returns the main start, or None if a fixed-size
        domain could not take it."""
        eng, r = self.engine, self._rng
        vx, vy = motion if motion is not None else self._motion("rm")
        jitter, ang = r.uniform(0.8, 1.2), math.radians(r.uniform(-15, 15))
        vx, vy = (jitter * (vx * math.cos(ang) - vy * math.sin(ang)),
                  jitter * (vx * math.sin(ang) + vy * math.cos(ang)))
        ao = self._auto()
        size, push = r.uniform(0.6, 1.4), r.uniform(0.8, 1.25)
        if outbreak:
            # synoptic initiation: a gentler, slower lift than OutbreakLab's (8 m/s at most) -- the trigger
            # only; whether a storm grows out of it is up to the simulated air
            rh = float(ao.get("radius_km", 12.0)) * 1000.0 * size
            wt = 0.8 * float(ao.get("push", 10.0)) * push
            ramp = r.uniform(600, 1200)
            dur = ramp + r.uniform(600, 1200)
        elif ao.get("on"):
            rh = float(ao.get("radius_km", 12.0)) * 1000.0 * size
            wt = float(ao.get("push", 10.0)) * push
            ramp = r.uniform(420, 1200)
            dur = ramp + r.uniform(600, 1300)
        else:
            rh, wt = 10000.0 * size, strength * push
            ramp = r.uniform(240, 600)
            dur = ramp + r.uniform(360, 600)
        wt, rh, dur = self._boosted(x, y, wt, rh, dur)       # strong shear: a stronger, wider, longer push
        main = self._thermal_cluster(x, y, rh, wt, ramp, dur, vx, vy, why, axis=axis)
        if main is None:
            return None
        # synoptic storms (outbreak): no merger cells or satellite towers -- those are OutbreakLab's forcing; storms
        # that grow out of the simulated weather meet and merge on their own
        if ao.get("mergers", True) and not outbreak:
            self._plan_mergers(main)
        if not ao.get("towers", True) or outbreak or (self.env.grid.dx <= CLUSTER_DX and not CLASSIC_STARTS):
            return main                             # (a thermal cluster has its own towers)
        zc, dp = ((1200, 1600), (1100, 1400)) if CLASSIC_STARTS else ((1000, 1500), (1000, 1300))
        n = r.choices(range(6), weights=[0.08, 0.17, 0.25, 0.25, 0.15, 0.10])[0]
        a = axis + r.uniform(-0.4, 0.4) if axis is not None else r.uniform(0.0, math.pi)
        aspect = r.uniform(1.8, 4.0) if axis is not None else r.uniform(1.0, 3.5)
        ca, sa = math.cos(a), math.sin(a)
        for _ in range(n):
            rr, th = math.sqrt(r.random()) * 0.7 * rh, r.uniform(0.0, 2.0 * math.pi)
            u, w = rr * math.cos(th), rr * math.sin(th) / aspect
            eng.add_start(StormStart(
                x=float(x + u * ca - w * sa), y=float(y + u * sa + w * ca), t0=eng.t + r.uniform(0, 600),
                kind="nudge", radius=rh * r.uniform(0.25, 0.5), zc=r.uniform(*zc),
                depth=r.uniform(*dp), strength=wt * r.uniform(0.7, 1.1), duration=r.uniform(360, 960),
                ramp=r.uniform(90, 300), vx=vx, vy=vy, why="tower", **self._shape(noise=(0.3, 0.6))))
        return main

    def _plan_mergers(self, main: StormStart) -> None:
        """0-3 separate cells (weights 30/40/20/10 %) sent into a new storm 15-50 min into its life,
        as nudgers (interactions.py): each moves on its own motion (the 0-6 km mean wind turned
        toward the left mover, +-15 deg) and meets the storm's updraft 12-25 min later, passing
        up to ~14 km ahead of or behind it -- under 10 km they usually merge, wider ones bridge or
        stay apart (Hastings & Richardson 2016), so each storm's later life goes its own way."""
        r = self._rng
        n = r.choices(range(4), weights=[0.3, 0.4, 0.2, 0.1])[0]
        mean, lm = np.asarray(self._motion("mean")), np.asarray(self._motion("lm"))
        for _ in range(n):
            mv = mean + r.uniform(0.0, 0.5) * (lm - mean)
            a = math.radians(r.uniform(-15, 15))
            mv = (float(mv[0] * math.cos(a) - mv[1] * math.sin(a)), float(mv[0] * math.sin(a) + mv[1] * math.cos(a)))
            self._mergers.append({"at": main.t0 + r.uniform(900, 3000), "main": main, "motion": mv,
                                  "lead": r.uniform(720, 1500), "aim": max(-14e3, min(14e3, r.gauss(0.0, 6000.0))),
                                  "strength": r.uniform(7.0, 11.0), "radius": r.uniform(4500, 8000)})
        self._mergers.sort(key=lambda m: m["at"])

    def _mergers_step(self) -> None:
        """Send the merger cells that are due at their storm's updraft; a storm without a deep
        updraft yet is tried again every 2 min for 16 min, then that cell is dropped."""
        eng = self.engine
        while self._mergers and eng.t >= self._mergers[0]["at"]:
            m = self._mergers.pop(0)
            m.setdefault("first", m["at"])
            try:
                eng.send_nudger(m["main"].position(eng.t), m["motion"], m["lead"], aim=m["aim"],
                                strength=m["strength"], radius=m["radius"])
            except ValueError as exc:
                if eng.t - m["first"] < 960.0:
                    m["at"] = eng.t + 120.0
                    self._mergers.append(m)
                    self._mergers.sort(key=lambda q: q["at"])
                else:
                    eng.note("nudger", f"merger cell for storm start #{m['main'].id} dropped: {exc}")
                break

    def _storm_near(self, x: float, y: float, r: float, echo: bool = True) -> bool:
        """A storm started within r (m) of (x, y) in the last 90 min, or (echo)
        precipitation below 9 km within r of it in a patch."""
        eng = self.engine
        if any(math.hypot(*np.subtract(s.position(eng.t), (x, y))) < r
               for s in eng.starts if s.t0 > eng.t - 5400.0 and s.why not in ("tower", "latch")):
            return True
        if not echo:
            return False
        dx = self.env.grid.dx
        k1 = max(int(np.searchsorted(self.env.z_h, 9000.0)), 1)
        n = r / dx
        for p in eng.patches:
            fi, fj = (x - p.world_x0()) / dx - 0.5, (y - p.world_y0()) / dx - 0.5
            i0, i1 = max(0, math.floor(fi - n)), min(p.rect.nx, math.ceil(fi + n) + 1)
            j0, j1 = max(0, math.floor(fj - n)), min(p.rect.ny, math.ceil(fj + n) + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            # every scheme's precipitation (forcing.precipitation): P3 keeps all its ice in qi
            p3 = getattr(p.state, "qir", None) is not None
            for name in ("qr", "qs", "qg", "qh") + (("qi",) if p3 else ()):
                a = getattr(p.state, name, None)
                if a is not None and float(a[:k1, j0:j1, i0:i1].max()) > 3e-4:
                    return True
        return False

    def _warm_sector_step(self):
        """OutbreakLab mode: storms start at random in the open warm sector of the
        area -- not inside a front's air mass, not within 35 km of a storm."""
        ao, eng = self._auto(), self.engine
        if not ao.get("on") or not (float(ao.get("start_min", 15.0)) * 60.0 <= eng.t
                                    <= float(ao.get("end_min", 270.0)) * 60.0):
            return
        half = 500.0 * float(ao.get("area_km", 400.0))
        x0, y0, x1, y1 = _within((-half, -half, half, half), self._start_box())
        if x1 <= x0 or y1 <= y0:
            return
        lam = float(ao.get("rate", 1.2)) * (x1 - x0) / 1000.0 * (y1 - y0) / 1000.0 / 1e5 * eng.dt / 3600.0
        if self._rng.random() >= lam:
            return
        am = self.env.airmass
        for _ in range(12):
            x, y = self._rng.uniform(x0, x1), self._rng.uniform(y0, y1)
            if am and float(am.weight(np.array([x]), np.array([y]), eng.t)[0]) > 0.25:
                continue
            # mostly where the warm sector's texture is warmer and moister
            if self.texture is not None and self._rng.random() > 0.25 + 0.75 * float(
                    self.texture.favour(np.array([x]), np.array([y]), eng.t)[0]):
                continue
            if self._storm_near(x, y, 35000.0):
                continue
            if self._start_storm(x, y, "warm sector") is not None:
                eng.note("storm", f"storm fired in the warm sector at ({x / 1000:.0f}, {y / 1000:.0f}) km")
            break

    # ------------------------------------------------------------ synoptic mode
    def _syn_cfg(self) -> dict:
        raw = self.settings.get("synoptic") or {}
        cfg = {**syn.DEFAULTS, **raw}
        if raw and "area_v" not in raw:
            # settings from before storms could form anywhere carried the old 700 km box: anywhere now
            cfg["area_km"] = 0.0
        cfg["area_v"] = 2
        return cfg

    def _build_synoptic(self, start: dt.datetime) -> None:
        """The simulated weather for this run (the one the page previewed, if it did), and the run's base
        sounding from it: the column at the map centre at the start."""
        cfg = self._syn_cfg()
        seed = int(cfg.get("seed") or self.settings["seed"])
        cfg["seed"] = seed
        self.settings["synoptic"] = cfg          # the run's settings.json repeats this weather
        self.status = "building the weather"

        def progress(f, msg):
            self.syn_progress = (float(f), msg)

        self.synoptic = syn.world_for(self.proj, start, cfg, seed, progress=progress)
        self.synoptic.keep_hourly = False
        self.syn_progress = None
        s = self.synoptic.sounding(0.0, 0.0, 0.0)
        if (self.settings.get("sounding") or {}).get("ice_cap", True):
            s = s.ice_capped()
        self.sounding = s
        self.diag = snd.diagnostics(s)

    def _synoptic_prepare(self) -> None:
        """Advance the weather ahead of the storms and work out what the next steps will read (fronts,
        initiation fields), outside the run's lock: a new 15-min time takes a second or two of CPU."""
        w, eng = self.synoptic, self.engine
        if w is None or eng is None:
            return
        w.advance_to(eng.t + syn.AHEAD)
        if eng.t >= self._syn_next_fronts - 60.0 or eng.t >= self._syn_next_ci - 60.0:
            w.diagnostics(max(eng.t, self._syn_next_ci))

    def _synoptic_step(self) -> None:
        w, eng = self.synoptic, self.engine
        if w is None:
            return
        if eng.t >= self._syn_next_fronts:
            self._syn_next_fronts = eng.t + SYN_FRONTS_EVERY
            an = w.analysis(eng.t)
            self.syn_boundaries = self._syn_fronts_as_boundaries(an, eng.t)
            self._syn_summary = w.summary(eng.t)
        if eng.t >= self._syn_next_ci:
            self._syn_next_ci = eng.t + SYN_CI_EVERY
            self._synoptic_feedback()
            self._synoptic_initiate()

    def _synoptic_feedback(self) -> None:
        """Two-way: the storm patches' lowest-level air and rain -- what their real microphysics made of it:
        cold pools, outflow, rain-cooled and moistened air -- block-averaged on the weather's 25 km grid and
        handed back to it (SynopticWorld.ingest_storms).  The weather then carries the outflow on, other patches'
        edges see it, later storms fire on its boundaries, and its own convection scheme stays off there."""
        from .forcing import precipitation
        eng, w, env = self.engine, self.synoptic, self.env
        if not eng.patches:
            return
        host = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
        dx = env.grid.dx
        b = max(1, int(round(syn.DX / dx)))
        z0 = float(env.z_h[0])
        k1 = max(int(np.searchsorted(env.z_h, 1000.0)), 1)
        thb0 = float(np.asarray(env.base.thb)[0])
        X, Y, TH, Q, R = [], [], [], [], []
        for p in eng.patches:
            st = p.state
            ny, nx = p.rect.ny // b * b, p.rect.nx // b * b
            if ny == 0 or nx == 0:
                continue

            def blocks(a):
                return host(a[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))).astype(np.float64)

            pr = precipitation(st, k1)
            # surface rain (mm/h): the lowest kilometre's precipitation at ~6 m/s fall speed
            rain = blocks(pr.mean(axis=0)) * 1.2 * 6.0 * 3600.0 if pr is not None else np.zeros((ny // b, nx // b))
            jj, ii = np.meshgrid(np.arange(ny // b), np.arange(nx // b), indexing="ij")
            X.append(p.world_x0() + (ii * b + 0.5 * b) * dx)
            Y.append(p.world_y0() + (jj * b + 0.5 * b) * dx)
            TH.append(blocks(st.thp[0]) + thb0)
            Q.append(blocks(st.qv[0]))
            R.append(rain)
        if X:
            cat = lambda L: np.concatenate([a.ravel() for a in L])
            w.ingest_storms(cat(X), cat(Y), cat(TH), cat(Q), cat(R), z0, eng.t)

    @staticmethod
    def _syn_fronts_as_boundaries(an: dict, t: float) -> list:
        """The simulated fronts and drylines as boundaries.Boundary (ground metres, air mass on the cold / dry
        side, moving the way the cold air pushes), for latching storms and firing squall lines."""
        kinds = {"cold": "cold", "warm": "warm", "stationary": "stationary", "dry": "dry"}
        out = []
        for k, fr in enumerate(an.get("fronts", [])):
            P = np.asarray(fr["points"], dtype=np.float64) * 1000.0
            if len(P) < 2:
                continue
            d = P[-1] - P[0]
            left = np.array([-d[1], d[0]]) / max(float(np.hypot(*d)), 1.0)
            warm = left * fr["side"]                     # toward the warm / moist air
            toward = warm if fr["kind"] in ("cold", "dry") else -warm
            b = Boundary.new(kinds[fr["kind"]], P.tolist(), side=-int(fr["side"]),
                             heading=float(math.degrees(math.atan2(toward[0], toward[1])) % 360.0),
                             init_rate=0.0)
            vx, vy = b.velocity()                        # stored as where it was at t = 0, like every boundary
            b.points = [[px - vx * t, py - vy * t] for px, py in b.points]
            b.id = 1000 + k
            b.seed = 1000 + k
            out.append(b)
        return out

    def _syn_why(self, x: float, y: float) -> tuple:
        """What lifts the air at (x, y): the nearest simulated boundary within 90 km on its warm side, or the
        open warm sector.  Returns (label, boundary or None)."""
        best = None
        for b in self.syn_boundaries:
            P = b.points_at(self.engine.t)
            d, over, _ = signed_distance(P, np.array([x]), np.array([y]))
            ahead = -float(d[0]) * b.side
            if over[0] <= 0 and -15000.0 <= ahead <= 90000.0 and (best is None or ahead < best[0]):
                best = (ahead, b)
        if best is None:
            return "warm sector", None
        names = {"cold": "cold front", "warm": "warm front", "stationary": "stationary front", "dry": "dryline"}
        return names.get(best[1].kind, best[1].kind), best[1]

    def _synoptic_initiate(self) -> None:
        """Storms form where the simulated air lets them: drawn every SYN_CI_EVERY s from the initiation-rate
        field (CAPE, CIN and lift from fronts, drylines, convergence and the trough's ascent) inside the
        storm area, at most max_storms at a time, each moving as the column it forms in says (Bunkers right
        mover); on a cold front whose shear runs along it, a squall line instead."""
        cfg, eng, w = self._syn_cfg(), self.engine, self.synoptic
        d = w.diagnostics(eng.t)
        rate = np.asarray(d["ci"], dtype=np.float64) * max(float(cfg.get("rate", 1.0)), 0.0)
        Xd, Yd = w.X[w.dsl], w.Y[w.dsl]
        from scipy.ndimage import maximum_filter
        cape = np.maximum(np.asarray(d["mlcape"], np.float64), np.asarray(d.get("mucape", 0.0), np.float64))
        self._syn_cape = (np.asarray(Xd, np.float64).ravel(), np.asarray(Yd, np.float64).ravel(),
                          maximum_filter(cape, size=3, mode="nearest"),     # storm_scale's inflow CAPE
                          np.asarray(d["shear6"], np.float64))               # start_boost's shear
        # anywhere the weather allows (area_km 0, the default), or a square of area_km around the map centre;
        # max_storms keeps the GPU's patches in check either way
        area = float(cfg.get("area_km") or 0.0)
        half = 500.0 * area if area > 0 else 1e12
        x0, y0, x1, y1 = _within((-half, -half, half, half), self._start_box())
        if x1 <= x0 or y1 <= y0:
            return
        rate = np.where((Xd >= x0) & (Xd <= x1) & (Yd >= y0) & (Yd <= y1), rate, 0.0)
        spacing = 1000.0 * max(float(cfg.get("spacing_km") or 45.0), 10.0)
        cell_km2 = (2.0 * syn.DX / 1000.0) ** 2
        lam = float(rate.sum()) * cell_km2 / 1e5 * SYN_CI_EVERY / 3600.0
        if lam <= 0.0:
            return
        n, p = 0, math.exp(-lam)                # Poisson draw (Knuth)
        acc = self._rng.random()
        while acc > p and n < 20:
            n += 1
            acc *= self._rng.random()
        live = sum(1 for s in eng.starts if (s.why.startswith("synoptic") or s.why.endswith(" line")) and s.t0 > eng.t - 5400.0)
        n = min(n, int(cfg.get("max_storms", 10)) - live)
        if n <= 0:
            return
        cum = np.cumsum(rate.ravel())
        for _ in range(n):
            for _try in range(12):
                c = int(np.searchsorted(cum, self._rng.random() * cum[-1]))
                c = min(c, cum.size - 1)
                x = float(Xd.ravel()[c] + self._rng.uniform(-syn.DX, syn.DX))
                y = float(Yd.ravel()[c] + self._rng.uniform(-syn.DX, syn.DX))
                # discrete storms keep their distance, as on a real dryline (~30-60 km apart)
                if not (x0 <= x <= x1 and y0 <= y <= y1) or self._storm_near(x, y, spacing * self._rng.uniform(0.8, 1.35)):
                    continue
                label, b = self._syn_why(x, y)
                if b is not None and b.kind == "cold" and self._rng.random() < self._line_chance(b, b.points_at(eng.t), eng.t):
                    if eng.t - self._line_last.get(b.id, -1e9) >= LINE_EVERY:
                        P = b.points_at(eng.t)
                        L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
                        if self._fire_line(b, P, L, (x0, y0, x1, y1), None, spacing=SYN_LINE_SPACING):
                            self._line_last[b.id] = eng.t
                            break
                motion = w.storm_motion(x, y, eng.t)
                if self._start_storm(x, y, f"synoptic: {label}", motion=motion, outbreak=True) is not None:
                    eng.note("storm", f"a storm formed on the {label} at ({x / 1000:.0f}, {y / 1000:.0f}) km "
                                      f"(simulated weather)")
                break

    # ------------------------------------------------------------ zones
    def _start_box(self):
        """Where new storms may fire: the firing area the user dragged out (settings "fire_area", ground
        metres x0, y0, x1, y1), within a fixed-size domain's start area; None = anywhere.  Storms placed
        by hand and nudgers are not limited by it."""
        fa = self.settings.get("fire_area")
        box = None
        if fa and len(fa) == 4:
            box = (min(fa[0], fa[2]), min(fa[1], fa[3]), max(fa[0], fa[2]), max(fa[1], fa[3]))
        return _intersect(box, self.engine.start_area())

    def _zones_step(self):
        eng = self.engine
        area = self._start_box()
        for z in self.zones:
            if not (z.t_start <= eng.t <= z.t_end):
                continue
            zone = (min(z.x0, z.x1), min(z.y0, z.y1), max(z.x0, z.x1), max(z.y0, z.y1))
            x0, y0, x1, y1 = _within(zone, area)
            if x1 <= x0 or y1 <= y0:
                continue                        # wholly outside the fixed-size domain
            # the zone's rate is for all of it; a fixed-size domain gets its share
            share = (x1 - x0) * (y1 - y0) / max((zone[2] - zone[0]) * (zone[3] - zone[1]), 1.0)
            lam = z.rate * share * eng.dt / 3600.0
            if self._rng.random() >= lam:
                continue
            for _ in range(20):
                x = self._rng.uniform(x0, x1)
                y = self._rng.uniform(y0, y1)
                if not self._storm_near(x, y, z.spacing, echo=bool(self._auto().get("on"))):
                    self._start_storm(x, y, f"initiation area {z.id}", strength=z.strength)
                    break

    # ------------------------------------------------------------ radars
    def _cut_offsets(self, r: rd.Radar):
        offs, acc = [], 0.0
        for c in r.cuts:
            offs.append(acc)
            acc += c.seconds
        return offs

    def _radar_due(self) -> bool:
        t = self.engine.t
        for r in self.radars:
            if r.vol_start is None:
                r.vol_start = t
            if r.next_cut < len(r.cuts) and t >= r.vol_start + self._cut_offsets(r)[r.next_cut]:
                return True
        return False

    def _radar_step(self, fields):
        t = self.engine.t
        for r in self.radars:
            offs = self._cut_offsets(r)
            while r.next_cut < len(r.cuts) and t >= r.vol_start + offs[r.next_cut]:
                cut = r.cuts[r.next_cut]
                data = rd.sample_cut(r, cut, self.proj, fields,
                                     wind=(self.env.z_h, self.env.u_ground, self.env.v_ground),
                                     terrain=self.env.terrain)
                start_ms = self.start_ms + int(round((r.vol_start + offs[r.next_cut]) * 1000))
                if r.noise:      # the same sweep scatters the same way every time (map, polling, Level II)
                    rd.add_noise(r, cut, data, (zlib.crc32(r.icao.encode()) ^ start_ms) & 0x7FFFFFFF)
                # RadarPainter's clutter / clear-air layer, drifting with the 0-1 km wind
                rd.add_contamination(r, cut, data, (start_ms - self.start_ms) / 1000.0, self._low_wind(),
                                     when=level2.utc(start_ms),
                                     towns=self._radar_towns(r))
                rd.add_terrain_clutter(r, cut, data, (start_ms - self.start_ms) / 1000.0)    # the hills it meets
                rd.range_fold(r, cut, data)             # distant echoes overlaying a Doppler cut: RF
                enc = rd.encode(r, cut, data)
                r.vol_sweeps.append({"cut": cut, "nrad": cut.nrad, "start_ms": start_ms,
                                     "first_gate": rd.LEGACY_DOP_FIRST if r.kind == "legacy" else cut.first,
                                     "gate": cut.gate,
                                     "ref_first": rd.LEGACY_REF_FIRST if r.kind == "legacy" else cut.first,
                                     "ref_gate": cut.gate * (rd.LEGACY_REF_GATES if r.kind == "legacy" else 1),
                                     "dur_ms": cut.seconds * 1000.0, "az0": 0.0, "moments": enc,
                                     "nyquist": r.nyquist(cut)})
                # display: the lowest tilt of each kind
                if abs(cut.elev - r.cuts[0].elev) < 0.01:
                    self._store_display(r, cut, data, start_ms)
                    self._sample_tornadoes(r, cut, data)
                r.next_cut += 1
            if r.next_cut >= len(r.cuts):
                sweeps, vol_ms = r.vol_sweeps, self.start_ms + int(round(r.vol_start * 1000))
                r.vol_sweeps = []
                r.next_cut = 0
                r.vol_start += r.volume_seconds
                if r.export and sweeps:
                    self._l2_pending += 1
                    self._l2_pool.submit(self._write_volume, r.icao, r.lat, r.lon, r.elev_m, r.vcp,
                                         vol_ms, sweeps, r.kind == "legacy")
                r.volumes_written += 1

    def _radar_towns(self, r):
        """(n, 3): the towns within 70 km of radar r, km east and north of it and population (cached)."""
        cache = self.__dict__.setdefault("_rtowns", {})
        if r.icao not in cache:
            cs = geo.cities() or []
            lat = np.array([c["lat"] for c in cs], np.float64)
            lon = np.array([c["lon"] for c in cs], np.float64)
            pop = np.array([c["pop"] for c in cs], np.float64)
            ex = (lon - r.lon) * 111.32 * math.cos(math.radians(r.lat))
            ny = (lat - r.lat) * 110.57
            keep = np.hypot(ex, ny) < 70.0
            cache[r.icao] = np.stack([ex[keep], ny[keep], pop[keep]], axis=1) if keep.any() else np.zeros((0, 3))
        return cache[r.icao]

    def _low_wind(self):
        low = np.asarray(self.env.z_h) <= 1000.0
        return float(np.mean(self.env.u_ground[low])), float(np.mean(self.env.v_ground[low]))

    TRACK_EVERY = 20.0             # s between a tornado track's points

    def _record_tornadoes(self, t: float) -> None:
        """A point on each confirmed tornado's track: where it is, the model's peak wind and Vrot, the radar's
        latest Vrot and CC at it and the EF estimate (radar.ef_estimate).  From the model's own tornado
        tracker, so the radar's contamination and noise never move the track."""
        for c in self.tornadoes.items:
            if not c.found or c.seen < rd.TORNADO_CONFIRM:
                continue
            pts = self.tor_paths.setdefault(c.id, [])
            if pts and t - pts[-1][0] < self.TRACK_EVERY:
                continue
            wind = c.ve if c.ve is not None else rd.DEBRIS_V0 + c.s * (rd.DEBRIS_V1 - rd.DEBRIS_V0)
            rad = self._tor_radar.get(c.id, {})
            fresh = t - rad.get("t", -1e9) < 300.0
            vr = rad.get("vrot") if fresh else None
            cc = rad.get("cc") if fresh else None
            ef = rd.ef_estimate(wind, vr if vr is not None else c.vrot, cc)
            pts.append((t, float(c.x), float(c.y), round(float(wind), 1), round(float(c.vrot), 1),
                        None if vr is None else round(vr, 1), None if cc is None else round(cc, 2), ef))

    def _sample_tornadoes(self, r, cut, data) -> None:
        """The radar's view of each confirmed tornado on its lowest tilt: Vrot from the strongest inbound and
        outbound within 2.5 km (Doppler cuts), the lowest CC within 1.5 km where there is echo (> 20 dBZ)."""
        import cupy as cp
        tors = [c for c in self.tornadoes.items if c.found and c.seen >= rd.TORNADO_CONFIRM]
        if not tors:
            return
        nrad, ng = data["REF"].shape
        daz = 360.0 / nrad
        for c in tors:
            lat, lon = self.proj.inv(np.array([c.x]), np.array([c.y]))
            lat, lon = float(np.asarray(lat).ravel()[0]), float(np.asarray(lon).ravel()[0])
            ex = (lon - r.lon) * 111320.0 * math.cos(math.radians(r.lat))
            ny = (lat - r.lat) * 110570.0
            rng = math.hypot(ex, ny)
            if rng < 5000.0 or rng > 230000.0:
                continue
            az = math.degrees(math.atan2(ex, ny)) % 360.0
            g0 = int((rng - cut.first) / cut.gate)
            out = self._tor_radar.setdefault(c.id, {})
            for name, reach in (("VEL", 2500.0), ("RHO", 1500.0)):
                if name == "VEL" and cut.kind not in ("dop", "both"):
                    continue
                if name == "RHO" and cut.kind not in ("surv", "both"):
                    continue
                ng_m = min(ng, cut.n_dop if name == "VEL" else rd.DOP_NGATES)
                dg = int(reach / cut.gate) + 1
                da = int(math.degrees(reach / rng) / daz) + 1
                g_lo, g_hi = max(g0 - dg, 0), min(g0 + dg + 1, ng_m)
                if g_lo >= g_hi:
                    continue
                rays = [(int(az / daz) + k) % nrad for k in range(-da, da + 1)]
                has = cp.asnumpy(data["_has"][rays, g_lo:g_hi])
                vals = cp.asnumpy(data[name][rays, g_lo:g_hi])
                if name == "VEL":
                    v = vals[has]
                    if v.size >= 4:
                        out["vrot"] = float(0.5 * (np.percentile(v, 98) - np.percentile(v, 2)))
                else:
                    ref = cp.asnumpy(data["REF"][rays, g_lo:g_hi])
                    v = vals[has & (ref > 20.0)]
                    if v.size >= 4:
                        out["cc"] = float(np.percentile(v, 5))
            out["t"] = self.engine.t

    def tornado_tracks(self) -> list:
        """The run's tornado tracks for the map: each one's points smoothed (a 5-point running mean of the
        positions), with the EF estimate along it, its peak and whether it is on the ground now."""
        live = {c.id for c in self.tornadoes.items if c.found and c.seen >= rd.TORNADO_CONFIRM}
        out = []
        for tid, pts in self.tor_paths.items():
            if len(pts) < 2:
                continue
            xy = np.array([[p[1], p[2]] for p in pts], np.float64)
            k = min(2, (len(xy) - 1) // 2)
            if k > 0:
                pad = np.concatenate([np.repeat(xy[:1], k, 0), xy, np.repeat(xy[-1:], k, 0)])
                xy = np.mean([pad[i:i + len(xy)] for i in range(2 * k + 1)], axis=0)
                xy[0], xy[-1] = [pts[0][1], pts[0][2]], [pts[-1][1], pts[-1][2]]
            out.append({"id": tid, "active": tid in live,
                        # t, x, y, EF, the model's peak wind (m/s), the Vrot rated on (m/s), CC (None: no fresh sweep)
                        "pts": [[round(p[0]), round(float(a), 1), round(float(b), 1), max(p[7], 0), round(float(p[3]), 1),
                                 round(float(p[5] if p[5] is not None else p[4]), 1), p[6]]
                                for p, (a, b) in zip(pts, xy)],
                        "ef": max(max(p[7] for p in pts), 0), "wind": max(p[3] for p in pts),
                        "vrot": max((p[5] or p[4]) for p in pts),
                        "cc": min((p[6] for p in pts if p[6] is not None), default=None),
                        "t0": pts[0][0], "t1": pts[-1][0]})
        return out

    def _store_display(self, r, cut, data, start_ms):
        import cupy as cp
        has = data["_has"] & (data["REF"] >= r.floor_dbz)
        if "_contam" in data:
            has = has | (data["_has"] & data["_contam"])
        ng = data["_ngate"]
        disp = r.latest
        meta = {"icao": r.icao, "lat": r.lat, "lon": r.lon, "x": r.x, "y": r.y, "elev": cut.elev,
                "nrad": cut.nrad, "first": cut.first, "gate": cut.gate, "t_ms": start_ms, "kind": r.kind,
                "conv": float(self.proj.convergence(r.lon))}
        if cut.kind in ("surv", "both"):
            ref, rh, rmeta = data["REF"], has, dict(meta, ngate=ng)
            if r.kind == "legacy":                        # a 1990s radar: 1 km reflectivity gates
                ref, rh = rd.legacy_ref(ref[:, :ng], has[:, :ng])
                rmeta = dict(meta, ngate=ref.shape[1], first=rd.LEGACY_REF_FIRST, gate=cut.gate * rd.LEGACY_REF_GATES)
            code = cp.where(rh, cp.clip(cp.rint(ref * 2 + 66), 2, 255), 0).astype(cp.uint8)
            disp["REF"] = (dict(rmeta, scale=2.0, offset=66.0), cp.asnumpy(code))
            for m in (("ZDR", "RHO") if r.dualpol else ()):
                sc, off, bits = rd.MOMENTS[m]
                nd = min(ng, rd.DOP_NGATES)
                c = cp.where(has[:, :nd], cp.clip(cp.rint(data[m][:, :nd] * sc + off), 2, 255), 0)
                disp[m] = (dict(meta, ngate=nd, scale=sc, offset=off), cp.asnumpy(c.astype(cp.uint8)))
        if cut.kind in ("dop", "both"):
            nd = min(ng, cut.n_dop)
            v = data["VEL"][:, :nd]
            if r.alias:                                   # as Level II carries it: folded at the Nyquist
                v = cp.mod(v + cut.nyquist, 2.0 * cut.nyquist) - cut.nyquist
            h = has[:, :nd]
            rf = data["_rf"][:, :nd] if "_rf" in data else cp.zeros(h.shape, dtype=bool)
            vsc = 2.0 if r.alias else rd.VEL_DEALIASED_SCALE      # dealiased: 1 m/s steps, past 142 mph
            code = cp.where(h, cp.clip(cp.rint(v * vsc + 129), 2, 255), 0)
            code = cp.where(rf, 1, code).astype(cp.uint8)             # range folded: purple
            disp["VEL"] = (dict(meta, ngate=nd, scale=vsc, offset=129.0), cp.asnumpy(code))
            # storm-relative: subtract the Bunkers right-mover along each radial (on the grid, as az is)
            um, vm = self.env.storm_motion
            az = cp.radians((cp.arange(cut.nrad) + 0.5) * 360.0 / cut.nrad - meta["conv"])
            vs = (um * cp.sin(az) + vm * cp.cos(az))[:, None] * math.cos(math.radians(cut.elev))
            srv = v - vs
            code = cp.where(rf, 1, cp.where(h, cp.clip(cp.rint(srv * vsc + 129), 2, 255), 0)).astype(cp.uint8)
            disp["SRV"] = (dict(meta, ngate=nd, scale=vsc, offset=129.0, motion=[um, vm]), cp.asnumpy(code))
            sw = data["SW"][:, :nd]
            code = cp.where(rf, 1, cp.where(h, cp.clip(cp.rint(sw * 2 + 129), 2, 255), 0)).astype(cp.uint8)
            disp["SW"] = (dict(meta, ngate=nd, scale=2.0, offset=129.0), cp.asnumpy(code))

    def _write_volume(self, icao, lat, lon, elev, vcp, vol_ms, sweeps, legacy=False):
        try:
            if legacy:                    # a 1990s WSR-88D: message type 1 radials, as archived then
                data = level2.encode_volume_legacy(icao, lat, lon, elev, vcp, vol_ms, sweeps)
            else:
                data = level2.encode_volume(icao, lat, lon, elev, vcp, vol_ms, sweeps)
            name = level2.archive_name(icao, vol_ms, legacy)
            d = self.run_dir / icao
            d.mkdir(parents=True, exist_ok=True)
            part = d / f".{name}.part"          # whole volumes only: polling lists names starting with the site
            part.write_bytes(data)
            os.replace(part, d / name)
            with self.lock:
                self.l2_index.setdefault(icao, []).append((name, len(data)))
            log.info("Level II %s (%d kB)", name, len(data) // 1024)
        except Exception:
            log.exception("Level II write failed")
        finally:
            with self.lock:
                self._l2_pending -= 1

    # ------------------------------------------------------------ frames
    def _vol_history(self, fields_cache, t):
        """The 3-D view's replay (volume.snapshot): the patches' volumes now, every volume.HISTORY_EVERY s of model
        time, written to run_dir/vol3d by a thread of its own (skipped while the last one is still being written)."""
        from . import volume
        if not self.settings.get("vol_history", True) or not fields_cache:
            return
        # past volume.HISTORY_MAX_MB on disk the snapshots thin out (twice the spacing, then four times, ...)
        every = volume.HISTORY_EVERY * 2 ** int(getattr(self, "_vol_hist_bytes", 0) / (volume.HISTORY_MAX_MB * 1e6))
        if t - getattr(self, "_vol_hist_t", -1.0e9) < every - 1.0:
            return
        th = getattr(self, "_vol_hist_thread", None)
        if th is not None and th.is_alive():
            return
        try:
            snap = volume.snapshot(fields_cache, {p.id: p for p in self.engine.patches + self.engine.nest_patches()})
        except Exception as exc:              # the replay must never stop the run (out of memory included)
            if not getattr(self, "_vol_hist_failed", False):
                self._vol_hist_failed = True
                log.warning("3-D replay snapshot failed: %s", exc)
            return
        self._vol_hist_t = t
        folder = self.run_dir / "vol3d"

        def write():
            try:
                self._vol_hist_bytes = getattr(self, "_vol_hist_bytes", 0) + volume.save_snapshot(folder, t, snap)
            except Exception as exc:
                log.warning("3-D replay snapshot not written: %s", exc)
        self._vol_hist_thread = threading.Thread(target=write, name="vol3d", daemon=True)
        self._vol_hist_thread.start()

    def _make_frame(self, fields_cache):
        eng = self.engine
        self._frame_now, self._frame_wall = False, time.time()
        per = {}
        metas = []
        dx = self.env.grid.dx
        made = {}                                   # patch id -> its products, for the nests' parents
        for p in eng.patches:
            prods = made[p.id] = products.patch_products(p, fields_cache.get(p.id))
            meta = p.to_json()
            metas.append(meta)
            for name, arr in prods.items():
                per.setdefault(name, []).append((meta, arr))
        self._frame_nests(fields_cache, made, per)
        epoch = self.start_ms + int(round(eng.t * 1000))
        fr = Frame(t=eng.t, epoch_ms=epoch, patches=metas, products={})
        for name, lst in per.items():
            fr.products[name] = products.compress(products.pack_frame_product(eng.t, name, lst))
        # storm-track swaths live on world tiles, so they stay after a patch moves on (_update_swaths)
        for dst in ("uhtrack", "windtrack"):
            blob = self.tracks.pack(eng.t, dst, dx)
            if blob is not None:
                fr.products[dst] = products.compress(blob)
        # the EF swath: packed again only when it has changed (frames share the bytes)
        ef = getattr(self, "_ef_blob", None)
        if ef is None or ef[0] != self.efswath.version:
            blob = self.efswath.pack(eng.t)
            ef = self._ef_blob = (self.efswath.version, None if blob is None else products.compress(blob))
        if ef[1] is not None:
            fr.products["efswath"] = ef[1]
        for r in self.radars:
            fr.radars[r.icao] = {m: products.compress(self._pack_polar(meta, arr))
                                 for m, (meta, arr) in r.latest.items()}
        fr.size = sum(len(b) for b in fr.products.values()) + sum(
            len(b) for d in fr.radars.values() for b in d.values())
        self.frames.append(fr)
        self.frame_bytes += fr.size
        if self.frame_bytes > self.max_frame_bytes and len(self.frames) > 10:
            # thin the older half: drop every other frame
            keep = []
            half = len(self.frames) // 2
            for i, f in enumerate(self.frames):
                if i < half and i % 2 == 1:
                    self.frame_bytes -= f.size
                    continue
                keep.append(f)
            self.frames = keep

    def _frame_nests(self, fields_cache, made, per):
        """The refinement nests' products on their own (finer) cells, after their parents' so the page draws them
        on top: each nest without its boundary frame (driven by the parent), trimmed to whole parent cells, and
        those parent cells blanked so the coarse grid doesn't show through where the nest has no echo.  Outer
        levels first, so an inner nest blanks its outer nest's codes before they are packed."""
        from .nest import WIDTH
        lv = self.engine.nests
        while lv is not None:
            for n in lv.items:
                pr, r = n.parent_rect(), n.ratio
                par = next((p for p in lv._parents() if p.rect.contains(pr) and p.id in made), None)
                if par is None:
                    continue
                m = -(-(WIDTH + 1) // r)                     # parent cells the boundary frame takes
                if pr.nx <= 2 * m or pr.ny <= 2 * m:
                    continue
                try:
                    prods = products.patch_products(n.patch, fields_cache.get(n.patch.id))
                except Exception:
                    log.exception("nest %d map products", n.id)
                    continue
                made[n.patch.id] = prods
                dxn = n.patch.env.grid.dx
                meta = n.patch.to_json()
                f = m * r
                meta.update(x0=meta["x0"] + f * dxn, y0=meta["y0"] + f * dxn, x1=meta["x1"] - f * dxn,
                            y1=meta["y1"] - f * dxn, nx=meta["nx"] - 2 * f, ny=meta["ny"] - 2 * f,
                            nest=getattr(n.patch, "nest_level", 1), dx=dxn)
                a0, b0 = pr.i0 + m - par.rect.i0, pr.j0 + m - par.rect.j0
                pa = made[par.id]
                for name, arr in prods.items():
                    if name not in pa:
                        continue
                    pa[name][b0:b0 + pr.ny - 2 * m, a0:a0 + pr.nx - 2 * m] = 0
                    per.setdefault(name, []).append((meta, arr[f:arr.shape[0] - f, f:arr.shape[1] - f]))
            lv = lv.inner

    def _ef_step(self) -> None:
        """The EF swath (efswath.py) after each model step: the fine grids' wind round the tornadoes followed,
        each moved on at its motion since the last radar refresh."""
        eng = self.engine
        grids = [p for p in eng.patches + eng.nest_patches() if p.env.grid.dx <= rd.VORTEX_DX]
        if not grids and not self.efswath.bufs:
            return
        tk = self.tornadoes
        age = max(0.0, eng.t - tk.t) if tk.t is not None else 0.0
        tors = [(c.x + c.vx * age, c.y + c.vy * age) for c in tk.items if c.found]
        try:
            self.efswath.step(grids, tors)
        except Exception as exc:              # (out of memory included) never the run's end
            if not getattr(self, "_ef_failed", False):
                self._ef_failed = True
                log.warning("EF swath: %s", exc)

    def _update_swaths(self):
        """The run-max swaths, every SWATH_EVERY s of model time: storm tracks (2-5 km UH) and wind
        swaths on world tiles that outlive the patches, and the run's strongest UH (when and where)
        for the page, the placefile and the Log."""
        import cupy as cp
        eng, dx = self.engine, self.env.grid.dx
        t = eng.t
        while self.uh_peaks and self.uh_peaks[-1]["t"] > t:      # from a run a rollback discarded
            self.uh_peaks.pop()
        ox = int(round(self.env.offset[0] / dx))
        oy = int(round(self.env.offset[1] / dx))
        best = None
        for p in eng.patches:
            r = p.rect
            codes, uh = products.swath_codes(p)
            for name, arr in codes.items():
                self.tracks.update(name, Rect(r.i0 + ox, r.j0 + oy, r.nx, r.ny), arr)
            k = int(cp.argmax(uh))
            v = float(uh.ravel()[k])
            if math.isfinite(v) and (best is None or v > best[0]):
                j, i = divmod(k, r.nx)
                best = (v, p.world_x0() + (i + 0.5) * dx, p.world_y0() + (j + 0.5) * dx)
        try:
            self.efswath.flush()
        except Exception:                     # the EF swath must never stop the run
            log.exception("EF swath flush failed")
        self.swath_version += 1
        self._next_swath = t + SWATH_EVERY
        prev = self.uh_peaks[-1]["value"] if self.uh_peaks else 0.0
        if best is None or best[0] <= prev:
            return
        v, x, y = best
        self.uh_peaks.append({"value": v, "t": t, "x": x, "y": y})
        if v >= 500.0 and v // 500 > prev // 500:
            eng.note("storm", f"run-max 2-5 km UH {v:.0f} m2/s2, at ({x / 1000:.0f}, {y / 1000:.0f}) km")

    @staticmethod
    def _pack_polar(meta, arr) -> bytes:
        hj = json.dumps(meta).encode()
        pad = (-(4 + len(hj))) % 4
        return len(hj + b" " * pad).to_bytes(4, "little") + hj + b" " * pad + np.ascontiguousarray(arr).tobytes()

    # ------------------------------------------------------------ status
    PACE_WINDOW = 20.0      # s of wall time the page's "x real time" is measured over

    def _pace(self, t: float) -> None:
        """(wall, model time) now, the last PACE_WINDOW s of them kept: the run's real speed, everything
        between the model steps included (radar sweeps, frames, re-fitted patches, warnings)."""
        h = self.__dict__.setdefault("_pace_hist", collections.deque())
        now = time.perf_counter()
        if h and t < h[-1][1]:                   # a rollback took the clock back
            h.clear()
        h.append((now, t))
        while len(h) > 2 and now - h[1][0] >= self.PACE_WINDOW:
            h.popleft()

    def real_speed(self) -> float:
        h = getattr(self, "_pace_hist", None)
        if not h or len(h) < 2 or self.status != "running":
            return 0.0
        (w0, t0), (w1, t1) = h[0], h[-1]
        # up to now, not the last step: a run stuck between steps shows it
        return (t1 - t0) / max(time.perf_counter() - w0, 1e-6)

    def info(self) -> dict:
        eng = self.engine
        d = {
            "run_id": self.run_id, "status": self.status, "error": self.error,
            "settings": self.settings, "diag": self.diag, "sounding": self.sounding.to_json(),
            "proj": self.proj.to_json(), "speed": self.speed,
            "frames": len(self.frames), "display_radar": self.display_radar,
            "radars": [r.to_json() for r in self.radars],
            "zones": [z.to_json() for z in self.zones],
            "nudger_plans": [dict(n, sent=n["id"] in self._plans_done) for n in self.nudger_plans],
            "boundaries": [b.to_json() for b in self.boundaries],
            "lows": [lo.to_json(self.boundaries) for lo in self.lows],
            "uh_peak": self.uh_peaks[-1] if self.uh_peaks else None,
        }
        if self.syn_progress is not None:
            d["synoptic_progress"] = {"fraction": self.syn_progress[0], "message": self.syn_progress[1]}
        if self.synoptic is not None:
            d["synoptic"] = self._syn_summary or {"scenario": self.synoptic.scenario_note, "fronts": [],
                                                  "centres": []}
        if eng is not None:
            d.update(eng.status())
            # the engine's "speed" is its last model step's alone; the page shows the run's real one
            d["step_speed"] = d.get("speed", 0.0)
            d["speed"] = self.real_speed()
            d["starts"] = [s.to_json() for s in eng.starts]
            d["events"] = eng.events[-80:]
            d["start_ms"] = self.start_ms
            d["model_ms"] = self.start_ms + int(eng.t * 1000)
            d["dx"] = self.env.grid.dx
            d["grid"] = self.env.describe()
            d["gpu_bytes"] = gpu_held()
            d["ground_fixed"] = getattr(self, "ground_fixed", False)
            dm = getattr(self.engine, "domain_motion", None) if self.engine is not None else None
            d["domain_motion"] = list(dm) if dm is not None else None
            d["tornado_tracks"] = self.tornado_tracks()
            d["cyclones"] = [self._cyclone_json(c) for c in self.cyclones]
            d["warnings"] = self.warnings.to_json()
            d["metars_version"] = self.metars.version
        return d
