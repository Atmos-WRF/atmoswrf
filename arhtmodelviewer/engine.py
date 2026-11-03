"""The model engine: ArWen patches on an unlimited world, stepped together.

Per step:
  1. adaptive dt (WRF targets: vertical CFL ~1.2, horizontal ~0.84; we use
     a little less), the same for every patch;
  2. ``gpuwm.core.dycore.step`` on every patch (ArWen's full WRF ARW step:
     RK3 + acoustic substeps, open lateral boundaries, Morrison, TKE...);
  3. the same step on a small periodic reference column of pure environment;
     its change over the step is the environment's own drift (surface drag,
     mixing, Coriolis on the mean wind) and is removed from every patch and
     from the reference itself -- a horizontally uniform large-scale forcing
     that keeps the storms' inflow environment exactly as configured;
  4. storm-start nudging/bubbles and the relaxation rim at patch edges.

Every ``manage_interval`` seconds the patches are re-fitted to where the
atmosphere is disturbed (grow ahead of storms, shift, shrink, merge, retire).
With a fixed-size domain (``fixed_km``) there is one patch of that size,
which only shifts to keep its storms away from its edges: it never grows, so
the GPU memory a run takes is known from the start.
A watchdog checks for non-finite or runaway values and rolls back to the
last checkpoint with a shorter time step instead of crashing.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import forcing, interactions
from .patch import HostCopy, Patch, Rect, release_freed

log = logging.getLogger(__name__)


@dataclass
class EngineOptions:
    margin_km: float | None = None        # space kept between activity and patch edges
    trigger_km: float | None = None       # re-fit when activity gets this close to an edge
    min_side_km: float | None = None
    block: int = 16                       # patch sizes are multiples of this (kernel reuse)
    manage_interval: float = 60.0         # s
    retire_after: float = 1200.0          # s of quiet before a patch is removed
    lookahead: float = 900.0              # s of storm motion added on the leading side
    shrink_after: float = 600.0           # s a patch must stay oversized before it is trimmed
    rim_km: float | None = None           # relaxation rim width
    rim_tau: float = 300.0                # s at the edge itself
    drift_correction: bool = True
    manage: bool = True                   # re-fit patches to the storms (unlimited domain)
    fixed_km: float | None = None         # one domain this wide (km a side): shifts with its storms, never grows
    fixed_km_y: float | None = None       # ... and this tall (km); None = square
    max_km: float | None = None           # unlimited domain: no patch wider than this (km a side); one that
                                          # reaches it shifts with its storms instead, like the fixed domain
    fixed_settle: float = 600.0           # s after a shift before the fixed domain shifts again (its strongest storm still inside)
    rim: bool = True                      # relaxation rim at patch edges
    persistence: float = 0.0              # 0 = pure physics (the default since 2026-10-28); 1 = strongest low-level lift under deep updrafts
    persistence_wmax: float = 8.0         # m/s of lift at persistence 1
    persistence_zpeak: float = 1200.0     # height of strongest lift (m)
    persistence_zhalf: float = 1100.0     # half depth of the lifted layer (m)
    persistence_memory: float = 0.0       # s: e-folding time of a deep updraft's remembered strength (0 = strength now; no gain measured)
    persistence_root: float = 2.0         # x persistence: lift the cold rising air of updraft roots (0.5-2.5 km) by this x its negative buoyancy (OUN 1999: 160 -> 190 min)
    cap_easing: float = 0.0               # x persistence: ease the warm cap ahead of updrafts (off: it weakened OUN storms)
    cfl_v_target: float = 1.1
    cfl_h_target: float = 0.75
    checkpoint_interval: float = 600.0    # s of model time
    watchdog_every: int = 10              # steps
    max_cells: int | None = None          # memory cap for all patches together
    merge_waste: float = 1.5              # merge only if the union is at most this x the patches it replaces
    follow_storms: bool = False           # re-centre the grid motion on the storms' measured motion
    follow_interval: float = 600.0        # s between grid-motion updates
    follow_max_step: float = 3.0          # m/s: largest change of grid motion per update
    airmass_refresh: float = 60.0         # s between updates of the air-mass maps of a patch
    airmass_keep_tau: float = 1200.0      # s: "keep the air mass" relaxation deep inside air masses (x2 near the boundary)
    lbc_renew_margin: float = 300.0       # s before moving-air-mass edge tables run out to renew them
    refine_ratio: int = 0                 # storm-following refinement nests (nest.py): this x finer (0 = none)
    refine_km: float = 40.0               # km a side
    refine_max: int = 2                   # at most this many at once
    refine_inner_ratio: int = 0           # a second level inside each nest, on its low-level mesocyclone (0 = none)
    refine_inner_km: float = 16.0
    ls_w_cms: float = 0.0                 # large-scale ascent (largescale.py): peak cm/s, 0 = none
    ls_top_km: float = 10.0
    follow_environment: bool = False      # physics-suite runs without the steady environment: new air takes on the
                                          # single-column suite reference's day (Engine._follow_reference)


def _on_levels(prof, env, to):
    """A run-level column profile (nz or nz + 1 levels, (k, 1, 1) on the device) on the levels of grid ``to``: as it
    is when they are the run's, interpolated in height for a nest on its own levels (the tornado zoom's)."""
    import cupy as cp
    if to is env or to.grid.nz == env.grid.nz:
        return prof
    src = env.z_h if prof.shape[0] == len(env.z_h) else env.z_w
    dst = to.z_h if prof.shape[0] == len(env.z_h) else to.z_w
    host = np.interp(np.asarray(dst, np.float64), np.asarray(src, np.float64), cp.asnumpy(prof).reshape(-1))
    return cp.asarray(host, dtype=prof.dtype).reshape((-1,) + prof.shape[1:])


def _clamp_vapour(st) -> None:
    """qv = max(0, qv), in place.  The next step's radiation reads this state:
    WRF's radiation driver clamps every water species it reads, gpuwm's RRTMGP
    refuses a negative qv instead ("qv must be finite and non-negative").  The
    drift correction's uniform subtraction and the unlimited mixing / boundary
    ring leave rounding-level negatives (-4e-9 seen with a radiation suite)."""
    import cupy as cp
    qv = getattr(st, "qv", None)
    if qv is not None:
        cp.maximum(qv, 0.0, out=qv)


CELL_BYTES = 600.0      # GPU bytes per patch cell: Morrison + TKE + scratch + one radar's fields, measured
                        # 555 in big 1 km patches, more in small ones (2026-10-22; was 450, the model alone)


def _gpu_free_bytes() -> int:
    """Free device memory plus what CuPy's pool holds unused (gpuwm's own measure)."""
    try:
        from gpuwm.core.resident_admission import device_free_bytes
        return int(device_free_bytes() or 0)
    except Exception:
        return 0


VRAM_SHARE = 0.72       # of the card's own memory this run may hold in all (its arrays, CUDA's own, the step's
                        # scratch).  Windows starts moving a program into shared RAM before the card is full: at
                        # 0.80 an 8 GB 4060 held 6.35 GB with 1.57 GB spilled (desktop, browser ~1 GB) at 2.2x
CONTEXT_BYTES = 0.9 * 2**30   # GPU memory outside CuPy's pool: CUDA's context, cuFFT / cuBLAS, the kernels
                              # (7.9 GB held by the process for 7.0 GB pooled, 2026-10-31)
MEASURE_CELLS = 3_000_000     # cells in patches before their measured bytes per cell replace CELL_BYTES: below
                              # it the run's other arrays (radar fields, nests, swaths) weigh too much per cell,
                              # and a run that small is far from filling any card


def _vram_total() -> int:
    try:
        import cupy as cp
        return int(cp.cuda.runtime.memGetInfo()[1])
    except Exception:
        return 0


def _pool_bytes() -> int:
    """What CuPy's pool holds (its free blocks too: the step's scratch comes back every step)."""
    try:
        import cupy as cp
        return int(cp.get_default_memory_pool().total_bytes())
    except Exception:
        return 0


def _vram_cells(cell_bytes: float = CELL_BYTES, base: int = 0) -> int:
    """The most cells the card's physical memory holds (VRAM_SHARE of it, less CUDA's own and the
    arrays that are not patches').  Windows lets a program run past its GPU's memory into shared system
    RAM, so "free" device memory can read more than the card has; there the model runs ~100x slower (a
    5090 run fell from 10x to 0.1x real time at 35.7 GB of 32), and an 8 GB card 3-5x slower already
    with a fifth of the run spilled."""
    total = _vram_total()
    if not total:
        return 1 << 62
    return max(0, int((VRAM_SHARE * total - CONTEXT_BYTES - base) / cell_bytes))


_REFUSED = object()      # Engine._place: patches at their size limit cannot take a storm start

FOLLOW_LEAD = 450.0      # s of storm motion a patch at its size limit sits ahead of its strongest storm
FOLLOW_RECENTRE = 0.12   # x its side: how far off that place it may drift before it shifts
FOLLOW_EDGE = 0.15       # x its side: a strong storm this close to an edge shifts it whatever its centre
FOLLOW_STRONG = 0.6      # x the highest ranked storm (nest.Rotation.score): the storms a patch at its size limit keeps
HAND_HOLD = 900.0        # s a patch moved by hand stays where it was put before following its storms again
SPIN_UP_MIN = 20.0       # min the environment's wind is mixed before a single-sounding run (Engine.spin_up)
SPIN_UP_GROUND_MIN = 30.0   # min of it, with the ground's friction, without the steady environment
EDGE_TKE_CELLS = 5       # cells at a patch edge given the reference column's TKE, then as many fading out
FOLLOW_REF_EVERY = 300.0  # s between updates of a suite run's environment from its reference column (_follow_reference)


_DEBUG_NAN = bool(__import__("os").environ.get("ARHTMODELVIEWER_DEBUG_NAN"))
_NAN_FIELDS = ("u", "v", "w", "thp", "php", "mup", "qv", "qc", "qr", "qg", "tke")


def _debug_nan(p, where: str) -> None:
    """ARHTMODELVIEWER_DEBUG_NAN: log the first non-finite field of a nest after ``where`` (once per nest)."""
    if not _DEBUG_NAN or getattr(p, "_nan_logged", False):
        return
    import cupy as cp
    for name in _NAN_FIELDS:
        a = getattr(p.state, name, None)
        if a is None:
            continue
        bad = ~cp.isfinite(a)
        if bool(bad.any()):
            idx = cp.argwhere(bad)
            ks = cp.unique(idx[:, 0]) if a.ndim == 3 else cp.zeros(1)
            log.warning("NaN debug: nest %d (%.3g m, nz %d) %s non-finite after %s: %d cells, levels %s",
                        p.id, p.env.grid.dx, len(p.env.z_h), name, where, int(bad.sum()),
                        cp.asnumpy(ks[:8]).tolist())
            p._nan_logged = True
            return


def _out_of_memory(exc: Exception) -> bool:
    """gpuwm's admission refusal or a CUDA allocation failure."""
    name = type(exc).__name__
    return (name in ("ResidentMemoryRefused", "OutOfMemoryError", "CUDARuntimeError")
            and ("memory" in str(exc).lower() or name != "CUDARuntimeError"))


# (dx m: margin, trigger, min side, rim km) of the grid presets; spacings between them interpolate
_PATCH_DEFAULTS = ((250.0, (15.0, 8.0, 36.0, 5.0)), (500.0, (25.0, 12.0, 56.0, 8.0)),
                   (750.0, (35.0, 15.0, 72.0, 12.0)), (1000.0, (45.0, 20.0, 96.0, 16.0)),
                   (1500.0, (60.0, 25.0, 120.0, 24.0)), (2000.0, (60.0, 25.0, 120.0, 24.0)),
                   (3000.0, (75.0, 30.0, 150.0, 30.0)))


def _defaults_for(dx: float):
    """(margin, trigger, min side, rim) in km for a grid spacing."""
    xs = [x for x, _ in _PATCH_DEFAULTS]
    return tuple(float(np.interp(dx, xs, [v[k] for _, v in _PATCH_DEFAULTS])) for k in range(4))


class Engine:
    def __init__(self, env, options: EngineOptions | None = None):
        import cupy as cp

        self.env = env
        self.opt = options or EngineOptions()
        dx = env.grid.dx
        m, tr, ms, rim = _defaults_for(dx)
        self.margin = int(round((self.opt.margin_km or m) * 1000.0 / dx))
        self.trigger = int(round((self.opt.trigger_km or tr) * 1000.0 / dx))
        self.min_side = int(round((self.opt.min_side_km or ms) * 1000.0 / dx))
        self.rim = max(6.0, (self.opt.rim_km or rim) * 1000.0 / dx)
        # fixed-size domain: its side in cells (a multiple of the block, like every patch); it shifts
        # when storms come within the trigger distance of an edge, which a small domain shortens
        self.fixed = self.fixed_ny = 0           # a fixed domain's width and height in cells (0: unlimited)
        if self.opt.fixed_km:
            b = self.opt.block
            self.fixed = max(2 * b, -(-int(round(self.opt.fixed_km * 1000.0 / dx)) // b) * b)
            self.fixed_ny = max(2 * b, -(-int(round((self.opt.fixed_km_y or self.opt.fixed_km) * 1000.0 / dx)) // b) * b)
            self.trigger = min(self.trigger, min(self.fixed, self.fixed_ny) // 6)
        # unlimited domain with a size limit: no patch wider than this many cells (a block multiple);
        # patches start no bigger than it either
        self.cap = 0
        if self.opt.max_km and not self.fixed:
            b = self.opt.block
            self.cap = max(2 * b, -(-int(round(self.opt.max_km * 1000.0 / dx)) // b) * b)
            self.trigger = min(self.trigger, self.cap // 6)
            self.min_side = min(self.min_side, self.cap)
        self.t = 0.0
        self.dt = env.grid.dt
        self.dt_cap = env.grid.dt
        self.dt_cap_until = 0.0
        self.nstep = 0
        self.patches: list[Patch] = []
        self.starts: list[forcing.StormStart] = []
        self._next_start_id = 1
        self.tracks = interactions.Tracks()    # every deep updraft, minute by minute (manage)
        self.track_w = None                    # (x, y) -> m/s a deep updraft there needs to be followed (12 if None)
        self.nudgers: list[interactions.Nudger] = []
        self.events: list[dict] = []           # log for the UI
        self._next_manage = 0.0
        self._next_follow = self.opt.follow_interval
        self._measured = None                 # the storms' measured motion at the last follow check
        self._last_manage = None
        self._next_checkpoint = 0.0
        self._checkpoint = None
        self._mem_full_until = -1.0             # model time: no new patches before this after a refusal
        # ground motion (m/s) a patch at its size limit drifts at instead of chasing its storms: a grid fixed to
        # the ground (a physics suite) whose run asked for the grid to move with, say, Bunkers right mover
        self.domain_motion = None
        self._ref = None
        self._ref_profile = None
        self._edge_w = {}
        self.hold_point = None                 # hurricane runs: (ground x, y, radius m) of the cyclone the domain keeps
        self._cores = {}
        self._roots = {}
        self._memory = forcing.CoreMemory(self.opt.persistence_memory) if self.opt.persistence_memory > 0 else None
        self.step_seconds = 0.0               # wall time of the last step
        dz = np.diff(env.z_w)
        self._rdzw = cp.asarray(1.0 / dz, dtype=cp.float32)          # per half level
        dzf = np.diff(env.z_h)
        rdzf = np.zeros(env.grid.nz + 1, dtype=np.float32)
        rdzf[1:-1] = 1.0 / dzf
        if env.terrain is not None:            # the sigma levels squeeze over high ground (4.5 km at most)
            rdzf /= 1.0 - min(4500.0, env.terrain.top) / env.grid.ztop
        self._rdz_face = cp.asarray(rdzf)[:, None, None]
        # coldest / warmest th' the watchdog accepts at each level (80 K at least): air carried up
        # from the column's lowest theta, or brought down from up to 3 km higher, plus 20 K.
        # Overshooting tops into a very stable stratosphere (30 K/km above a 13 km tropopause) are
        # legitimately 80-120 K colder than the air around them, the stratospheric air pushed down
        # beside them ~50 K warmer
        thb = np.asarray(env.base.thb, dtype=np.float64)
        zh = np.asarray(env.z_h, dtype=np.float64)
        above = np.array([thb[(zh >= z) & (zh <= z + 3000.0)].max() for z in zh])
        self._cold_limit = cp.asarray(np.maximum(80.0, thb - thb.min() + 20.0), dtype=cp.float32)[:, None, None]
        self._warm_limit = cp.asarray(np.maximum(80.0, above - thb + 20.0), dtype=cp.float32)[:, None, None]
        self._auto_budget = self.opt.max_cells is None     # follows the free memory (_refresh_budget)
        if self.opt.max_cells is None:
            # keep a quarter of the free memory as head-room (other programs, CUDA's own, the step's temporaries)
            self.opt.max_cells = int(0.75 * _gpu_free_bytes() / CELL_BYTES)
        self._cell_bytes = CELL_BYTES          # measured once the patches are big enough (_measure_memory)
        self._nest_mem = 1.3                   # x _cell_bytes a nest cell is counted at (edge tables, LES); up after an OOM
        self._pool_base = _pool_bytes()        # arrays that are not patches' (a previous run's are freed by now)
        self._vram_cells = _vram_cells(self._cell_bytes, self._pool_base)
        self.opt.max_cells = min(self.opt.max_cells, self._vram_cells)
        self._max_cells0 = self.opt.max_cells
        self._make_reference()                 # also with steady environment off: its TKE, and the METARs' environment
        self.ls = None
        if self.opt.ls_w_cms and self.opt.ls_w_cms > 0.0:
            from .largescale import LargeScaleAscent
            self.ls = LargeScaleAscent(env, self.opt.ls_w_cms, self.opt.ls_top_km)
        self.nests = None
        if self.opt.refine_ratio and self.opt.refine_ratio >= 2:
            from .nest import Nests
            self.nests = Nests(self, self.opt.refine_ratio, self.opt.refine_km, self.opt.refine_max)
            if self.opt.refine_inner_ratio and self.opt.refine_inner_ratio >= 2:
                self.nests.inner = Nests(self, self.opt.refine_inner_ratio, self.opt.refine_inner_km,
                                         self.opt.refine_max, parent=self.nests)
        self.tornado = None                   # the tornado zoom (tornado.py): a box the user put on a vortex

    def tornado_box(self, x: float, y: float, size_km: float, detail: str = "high") -> None:
        """The tornado zoom on ground point (x, y) m, a box of size_km (a new one replaces the old); ``detail``
        "high" (~110 m), "ultra" (~55 m), "extreme" (~37 m), "pc10" (~10 m) or "pc1" (~1 m), the finest on its own
        levels (tornado.DETAIL)."""
        from .tornado import TornadoZoom
        self.tornado_clear()
        self.tornado = TornadoZoom(self, x, y, size_km, detail)

    def tornado_clear(self) -> None:
        if self.tornado is not None:
            self.tornado.clear()
            self.tornado = None

    def nest_patches(self) -> list:
        """The refinement nests' patches, every level (none without nests)."""
        return [n.patch for n in self.nests.all_items()] if self.nests else []

    # ---------------------------------------------------------- by hand
    def move_patch(self, pid: int, x: float, y: float, lock: bool = False) -> bool:
        """Patch ``pid`` moved by hand, its centre to ground point (x, y) m, its size kept: what it overlaps
        of its old place comes along, the rest starts from the environment.  It then stays there HAND_HOLD
        (``lock``: until released) before following its storms again.  Onto another patch: the two merge,
        if one patch holding both fits the size limit and the GPU budget."""
        p = next((q for q in self.patches if q.id == pid), None)
        if p is None:
            return False
        dx, b = self.env.grid.dx, self.opt.block
        ox, oy = self.env.offset
        rect = Rect(int(round(((x - ox) / dx - 0.5 * p.rect.nx) / b)) * b,
                    int(round(((y - oy) / dx - 0.5 * p.rect.ny) / b)) * b, p.rect.nx, p.rect.ny)
        members = [p]
        union, touching = self._union_touching(rect, exclude=(p,))
        if touching:
            if self.fixed or self._over_cap(union) or not self._fits_budget(union, [p] + touching):
                self.note("patch", f"patch {pid} cannot go there: it would overlap patch "
                                   f"{', '.join(str(q.id) for q in touching)} and one patch holding both is too big")
                return False
            rect, members = union, [p] + touching
        got = self._commit(rect, members, "moved by hand" + (", locked there" if lock else ""))
        if got is None:
            return False
        got.hand_until = math.inf if lock else self.t + HAND_HOLD
        got.storm_center = None              # its storms did not jump: no motion from the move
        got.drift_anchor = None              # moving with domain_motion starts from here, not from the old place
        return True

    def release_patch(self, pid: int) -> bool:
        """A patch moved by hand follows its storms again now."""
        p = next((q for q in self.patches if q.id == pid), None)
        if p is None:
            return False
        p.hand_until = -math.inf
        p.drift_anchor = None
        self.note("patch", f"patch {pid} follows its storms again")
        return True

    def move_nest(self, nid: int, x: float, y: float) -> bool:
        return bool(self.nests) and self.nests.move(nid, x, y)

    def add_nest(self, x: float, y: float) -> bool:
        if not self.nests:
            self.note("nest", "this run has no refinement nests (Grid: storm nests) to add one")
            return False
        return self.nests.add_on(x, y)

    # ---------------------------------------------------------- logging
    def note(self, kind: str, text: str) -> None:
        self.events.append({"t": self.t, "kind": kind, "text": text})
        if len(self.events) > 500:
            del self.events[:100]
        log.info("[t=%7.0f s] %s: %s", self.t, kind, text)

    # ---------------------------------------------------------- reference
    def _make_reference(self) -> None:
        cfg = self.env.run_config(16, 16, self.dt, bc="periodic")
        self._ref_cfg = cfg
        self._ref = self.env.new_state(cfg)
        st = self._ref
        self._ref_profile = {n: getattr(st, n).mean(axis=(1, 2)).copy() for n in ("u", "v", "thp", "qv")}

    def spin_up(self, minutes: float = SPIN_UP_MIN, ground: bool = False) -> None:
        """Before the first patch: the environment's wind as the model's own mixing leaves it.  A sounding
        whose low-level jet has a sharp kink (Richardson number < 0.25: DDC 2019-04-10, 45 kt of 0-1 km
        shear over a near-neutral layer) is torn into turbulence by the TKE scheme within minutes.  The
        steady environment kept re-sharpening that kink everywhere, so the air in the domain at the start
        (mixed in step with the reference column) and the air brought in later through the inflow edge
        (unmixed, yet corrected the same) met in a seam that the unstable layer grew into a straight line
        of updrafts and light rain, 100 km long, drifting in from the inflow edge.  A throwaway column runs
        ``minutes`` with turbulence but no drag (the surface wind stays the sounding's); its mean wind
        becomes the environment's and its TKE every new state's.  Theta and moisture stay the sounding's
        (mixing the inversion would change CAPE and CIN): 0-1 km SRH 742 -> 738 on that sounding.

        ``ground`` (runs without the steady environment): the column also feels the ground's friction and keeps
        what its turbulence does to theta and moisture (env.ls_dth / ls_dqv, which new states, edge tables and
        rim targets read).  Inside a patch, friction and shear build a turbulent surface layer within ~30 min
        (OUN 2025-05-22, 1 km: TKE ~1.5 m2/s2 below 200 m, +0.4 K, -0.6 g/kg at the ground) while air added
        later by growth and brought in at the edges arrived unmixed: every old patch edge left a seam, the
        faint rectangles of uncapped soundings.  Starting everything from the mixed column removes the
        contrast."""
        import cupy as cp
        import dataclasses
        from gpuwm.core.dycore import step
        if minutes <= 0 or self.patches:
            return
        env = self.env
        cfg = dataclasses.replace(env.run_config(16, 16, self.dt, bc="periodic"), dt=self.dt)
        st = env.new_state(cfg)
        if getattr(st, "tke", None) is None:
            return
        mean = lambda a: cp.asnumpy(a.mean(axis=(1, 2))).astype(np.float64)
        u0, v0, th0, q0 = mean(st.u), mean(st.v), mean(st.thp), mean(st.qv)
        drag = ground and env.own_drag
        for _ in range(int(round(minutes * 60.0 / self.dt))):
            step(st, cfg)
            if drag:
                self._surface_drag(st)
        du, dv = mean(st.u) - u0, mean(st.v) - v0
        env.u_ground = env.u_ground + du
        env.v_ground = env.v_ground + dv
        env._apply_motion(env.motion)
        env.tke_profile = mean(st.tke)
        what = "the wind profile"
        if ground:
            dth, dq = mean(st.thp) - th0, mean(st.qv) - q0
            env.ls_dth = (getattr(env, "ls_dth", None) if getattr(env, "ls_dth", None) is not None else 0.0) + dth
            env.ls_dqv = (getattr(env, "ls_dqv", None) if getattr(env, "ls_dqv", None) is not None else 0.0) + dq
            env._bsnap = None
            what = (f"the low levels over the ground: surface {dth[0]:+.1f} K, {1000 * dq[0]:+.1f} g/kg, TKE "
                    f"{float(env.tke_profile[:6].max()):.1f} m2/s2 below {env.z_h[5]:.0f} m; the wind")
        del st
        if self._ref is not None:
            self._make_reference()
        k = int(np.argmax(np.hypot(du, dv)))
        self.note("physics", f"environment spun up {minutes:.0f} min (turbulent mixing of {what}, "
                             f"largest change {math.hypot(du[k], dv[k]):.1f} m/s at {env.z_h[k]:.0f} m)")

    def _edge_tke(self, p: Patch) -> None:
        """Air coming in through a patch's edge arrives as turbulent as the reference column's: the edge
        tables carry no TKE, and air the drift correction treats as mixed but that is not yet mixing
        made the seam spin_up describes."""
        import cupy as cp
        T = getattr(p.state, "tke", None)
        if T is None or self._ref is None or getattr(self._ref, "tke", None) is None:
            return
        ny, nx = T.shape[1:]
        w = self._edge_w.get((ny, nx))
        if w is None:
            j = cp.arange(ny)[:, None]
            i = cp.arange(nx)[None, :]
            d = cp.minimum(cp.minimum(i, nx - 1 - i), cp.minimum(j, ny - 1 - j)).astype(T.dtype)
            w = self._edge_w[(ny, nx)] = cp.clip(1.0 - (d - (EDGE_TKE_CELLS - 1)) / (EDGE_TKE_CELLS + 1), 0.0, 1.0)[None]
        T += w * (self._ref.tke.mean(axis=(1, 2))[:, None, None] - T)

    def _drift_correction(self) -> None:
        if self._ref is None:
            return
        from .patch import step_on_time
        import dataclasses
        if abs(self._ref_cfg.dt - self.dt) > 1e-9:
            self._ref_cfg = dataclasses.replace(self._ref_cfg, dt=self.dt)
        step_on_time(self._ref, self._ref_cfg)
        oc = getattr(self.env, "ocean", None)
        if oc is not None:                         # open sea at the environment's SST: its fluxes are the drift
            self._sea_surface(self._ref, None)
        elif self.env.own_drag:
            self._surface_drag(self._ref)
        st = self._ref
        keep = self._ambient_keep() if oc is not None else None
        sun = self._suite_weights() if self.env.suite else None
        for n, prof in self._ref_profile.items():
            arr = getattr(st, n)
            delta = arr.mean(axis=(1, 2)) - prof
            d3 = delta[:, None, None]
            arr -= d3
            dp = d3 * keep if keep is not None and n in ("thp", "qv") else d3
            for p in self.patches + self.nest_patches():
                dq = _on_levels(dp, self.env, p.env)
                w = sun(p, n) if sun is not None and n in ("thp", "qv") else None
                if w is not None and w.shape[0] != dq.shape[0]:      # a nest on its own levels: the plain drift
                    w = None
                getattr(p.state, n)[...] -= dq if w is None else dq * w
        _clamp_vapour(st)

    def _suite_weights(self):
        """A physics suite's theta / moisture drift is the column's sunny day: its land surface heating and
        moistening a growing boundary layer.  The steady environment takes it off each patch cell in the share
        of that day the cell has had -- its own surface sensible (theta) or latent (qv) heat flux over the
        column's, 0..1.5, inside the column's boundary layer (fading out to 1.4 x its depth; above it, the
        whole drift).  Taken off in full everywhere, every cell the anvil or the cumulus shaded cooled as much as
        the sun warmed the column: cold pools without storms (2026-11-02, DVN 2008-06-10, 750 m).  Not taken
        off at all, the sun kept heating an afternoon sounding that already holds the day's heating and the
        whole domain fired (OUN 1999-05-03, 500 m: popcorn and lines everywhere, the storms starved)."""
        import cupy as cp
        from .patch import driver_lookup
        ref = self._ref
        hr, lr = driver_lookup(ref, "fields/hfx"), driver_lookup(ref, "fields/lh")
        pb = driver_lookup(ref, "fields/pblh")
        if hr is None or pb is None:
            return None
        hr, lr, pb = float(hr.mean()), float(lr.mean()) if lr is not None else 0.0, float(pb.mean())
        z = np.asarray(self.env.z_h, dtype=np.float64)
        bl = cp.asarray(np.clip((1.4 * pb - z) / max(0.4 * pb, 1.0), 0.0, 1.0), dtype=cp.float32)[:, None, None]

        def weights(p, n):
            ref_flux, key = (hr, "fields/hfx") if n == "thp" else (lr, "fields/lh")
            f = driver_lookup(p.state, key)
            if f is None or ref_flux < 20.0:             # little sun (morning, evening, night): the plain drift
                return None
            r = cp.clip(f / ref_flux, 0.0, 1.5)[None]
            return 1.0 + bl * (r - 1.0)
        return weights

    def _follow_reference(self) -> None:
        """Physics-suite runs (with the steady environment it holds their wind only): the reference column runs the suite (land surface,
        surface layer, PBL, radiation) as a single column, and every FOLLOW_REF_EVERY s the air new cells, edge
        tables and the rim are built from takes on its day: its theta / qv drift (env.ls_dth / ls_dqv), its wind's
        (env.u_ground / v_ground) and its land surface (env.driver_template, the physics driver's arrays).  Held
        at the sounding, the new air met the patch's own heated, mixed boundary layer in a weak boundary at every
        place a patch edge had been before it grew: straight lines of shallow cloud and echo (2026-11-02, 750 m,
        Thompson + YSU + MM5 + Noah, six growth steps).  The column is put back to its horizontal mean at each
        update, so it can never overturn into a storm of its own: the 10-31 following environment of the
        built-in core did (the 16 x 16 column's 1-2 km layer overturned 1.9 K an hour) and every patch
        convected with it."""
        import cupy as cp
        import dataclasses
        from .patch import _driver_arrays, step_on_time
        if self._ref is None:
            return
        if not self.opt.drift_correction:           # (with the steady environment it has stepped the column)
            if abs(self._ref_cfg.dt - self.dt) > 1e-9:
                self._ref_cfg = dataclasses.replace(self._ref_cfg, dt=self.dt)
            step_on_time(self._ref, self._ref_cfg)
            _clamp_vapour(self._ref)
        if self.t < getattr(self, "_next_follow_ref", FOLLOW_REF_EVERY):
            return
        self._next_follow_ref = self.t + FOLLOW_REF_EVERY
        st, env = self._ref, self.env
        for name, val in list(vars(st).items()):
            if isinstance(val, cp.ndarray) and val.ndim >= 2 and val.dtype.kind == "f" and val.shape[-2:] in (
                    (16, 16), (16, 17), (17, 16)):
                val[...] = val.mean(axis=(-2, -1), keepdims=True)
        nz = len(env.z_h)
        if getattr(env, "ls_dth", None) is None:
            env.ls_dth, env.ls_dqv = np.zeros(nz), np.zeros(nz)
        d = {n: cp.asnumpy(getattr(st, n).mean(axis=(1, 2)) - self._ref_profile[n]).astype(np.float64)
             for n in ("thp", "qv", "u", "v")}
        qv0 = np.asarray(getattr(env, "qv_env_h", env.qv_f(env.z_h)), dtype=np.float64)
        d["qv"] = np.maximum(d["qv"], -(qv0 + env.ls_dqv))
        env.ls_dth = env.ls_dth + d["thp"]
        env.ls_dqv = env.ls_dqv + d["qv"]
        env.u_ground = env.u_ground + d["u"]
        env.v_ground = env.v_ground + d["v"]
        env._apply_motion(env.motion)
        for n in ("thp", "qv", "u", "v"):
            self._ref_profile[n] = self._ref_profile[n] + cp.asarray(d[n], dtype=self._ref_profile[n].dtype)
        env.driver_template = {path: a[..., :1, :1].copy() for path, a in _driver_arrays(st)
                               if a.ndim >= 2 and a.shape[-2:] == (16, 16)
                               and path.rsplit("/", 1)[-1] not in ("ivgtyp", "isltyp")}
        env._bsnap = None
        cache = env.__dict__.get("_dev_cache") or {}
        cache.pop("ls_thp", None)
        cache.pop("ls_qv", None)
        if self.ls is not None:
            self.ls.reset_balance()
        for p in self.patches:
            p.renew_boundaries(self.t)
        k = int(np.argmax(np.abs(d["thp"][:20])))
        if not hasattr(self, "_follow_noted"):
            self._follow_noted = True
            self.note("physics", "the environment follows the suite's single-column day (new cells, edges, rim)")
        log.info("follow reference: theta %+.2f K at %.0f m, surface %+.2f K, %+.2f g/kg", d["thp"][k],
                 env.z_h[k], env.ls_dth[0], 1000 * env.ls_dqv[0])

    # ------------------------------------------------------- moving grid
    def _set_offset(self, t: float) -> None:
        """Ground position of the grid origin at time t (the grid moves with env.motion)."""
        self.env.offset[0], self.env.offset[1] = self.env.offset_at(t)
        self.env.clock = t

    # ------------------------------------------------- following storms
    def _track_updrafts(self) -> None:
        """Ground position of each patch's strongest mid-level updraft, for the
        storms' measured motion.  A jump (a new cell taking over) restarts the track."""
        import cupy as cp
        env = self.env
        k3 = int(np.searchsorted(env.z_w, 3000.0))
        k7 = max(int(np.searchsorted(env.z_w, 7000.0)), k3 + 1)
        dx = env.grid.dx
        for p in self.patches:
            wm = p.state.w[k3:k7].max(axis=0)
            w = float(wm.max())
            track = p.__dict__.setdefault("updraft_track", [])
            while track and track[-1][0] >= self.t:     # points from a run a rollback discarded
                track.pop()
            if w < 15.0:
                track.clear()
                continue
            j, i = np.unravel_index(int(cp.argmax(wm)), wm.shape)
            x = p.world_x0() + (i + 0.5) * dx
            y = p.world_y0() + (j + 0.5) * dx
            if track:
                t0, x0, y0, _ = track[-1]
                if math.hypot(x - x0, y - y0) > 12000.0 + 30.0 * (self.t - t0):
                    track.clear()
            track.append((self.t, x, y, w))
            while track and self.t - track[0][0] > 1800.0:
                track.pop(0)

    def storm_motion_measured(self):
        """Ground-relative motion (m/s) of the tracked updrafts: least squares over
        each track of at least 15 min, weighted by updraft strength.  None if no track."""
        vx = vy = wsum = 0.0
        for p in self.patches:
            track = getattr(p, "updraft_track", None) or []
            if len(track) < 6 or track[-1][0] - track[0][0] < 900.0:
                continue
            a = np.asarray(track)
            t = a[:, 0] - a[:, 0].mean()
            den = float((t * t).sum())
            if den <= 0:
                continue
            w = float(a[:, 3].mean())
            vx += w * float((t * (a[:, 1] - a[:, 1].mean())).sum()) / den
            vy += w * float((t * (a[:, 2] - a[:, 2].mean())).sum()) / den
            wsum += w
        if wsum <= 0:
            return None
        return vx / wsum, vy / wsum

    def _follow_storms(self) -> None:
        m = self._measured = self.storm_motion_measured()
        if m is None:
            return
        mx, my = self.env.motion
        dvx, dvy = m[0] - mx, m[1] - my
        mag = math.hypot(dvx, dvy)
        if mag < 1.0:
            return
        f = min(1.0, self.opt.follow_max_step / mag) * 0.8
        self.set_grid_motion((mx + f * dvx, my + f * dvy), reason="following the storms")

    def _follow_step(self) -> None:
        """At each re-fit: track the storms' deep updrafts (when not following too, so that following
        switched on mid-run has them) and every follow_interval turn the grid toward their motion."""
        self._track_updrafts()
        if self.opt.follow_storms and self.t >= self._next_follow:
            self._follow_storms()
            self._next_follow = self.t + self.opt.follow_interval

    def set_follow(self, on: bool, interval: float | None = None) -> None:
        """Follow the storms' measured motion from now on (checked every ``interval`` s, the
        first time at the next re-fit), or stop following and keep the grid's motion."""
        if interval:
            self.opt.follow_interval = max(60.0, float(interval))
        if on and not self.opt.follow_storms:
            self._next_follow = self.t
        elif on:
            self._next_follow = min(self._next_follow, self.t + self.opt.follow_interval)
        self.opt.follow_storms = bool(on)

    def set_grid_motion(self, motion, reason: str = "") -> None:
        """Move the grid with a new motion from now on: every grid-relative wind
        (patches, the reference column, boundary tables) gets the Galilean shift."""
        from gpuwm.core.state import DTYPE
        self._set_offset(self.t)
        dx_, dy_ = self.env.set_motion(self.t, motion)
        if not (dx_ or dy_):
            return
        self._shift_winds(dx_, dy_)
        m = self.env.motion
        self.note("grid", f"grid now moves at {m[0]:.1f}, {m[1]:.1f} m/s ({reason})")

    def _shift_winds(self, dx_: float, dy_: float) -> None:
        from gpuwm.core.state import DTYPE
        states = [p.state for p in self.patches + self.nest_patches()] + ([self._ref] if self._ref is not None else [])
        for st in states:
            st.u -= DTYPE(dx_)
            st.v -= DTYPE(dy_)
        for n in (self.nests.all_items() if self.nests else []):
            n.sides = None                        # its last edge strips were in the old frame
        if self._ref_profile is not None:
            self._ref_profile["u"] = self._ref_profile["u"] - dx_
            self._ref_profile["v"] = self._ref_profile["v"] - dy_
        for p in self.patches:
            p.storm_velocity = (p.storm_velocity[0] - dx_, p.storm_velocity[1] - dy_)
            if p.storm_center is not None:
                p.storm_center = None             # positions in the grid frame restart
            p.renew_boundaries(self.t)

    # -------------------------------------------------------- air masses
    def set_airmass(self, airmass) -> None:
        """Use a new set of boundaries / air masses from now on.  Running patches
        get the change as a balanced increment (theta', qv, mu', phi', winds),
        new edge tables and new maps."""
        import cupy as cp
        from gpuwm.core.diagnostics import update_diagnostics
        from gpuwm.core.state import DTYPE
        from .boundaries import AirMasses
        env = self.env
        old = env.airmass
        self._set_offset(self.t)
        none = AirMasses([], env)
        for p in self.patches:
            st = p.state
            X, Y = env.ground_xy(p.rect, t=self.t)
            cols = []
            for am in (old or none, airmass or none):
                env.airmass = am
                if am:
                    cols.append(env.airmass_columns(X, Y, self.t, xp=cp))
                else:
                    z = cp.zeros((env.grid.nz,) + X.shape)
                    qv = cp.asarray(env.qv_f(env.z_h))[:, None, None] * cp.ones(X.shape)[None]
                    cols.append((z, qv) + env.balance(z, qv))
            (t0, q0, m0, f0), (t1, q1, m1, f1) = cols
            st.thp += (t1 - t0).astype(DTYPE)
            st.qv[...] = cp.maximum(st.qv + (q1 - q0).astype(DTYPE), 0.0)
            st.mup += (m1 - m0).astype(DTYPE)
            st.php += (f1 - f0).astype(DTYPE)
            del cols, t0, q0, f0, t1, q1, f1
            for name, sx, sy in (("u", 1, 0), ("v", 0, 1)):
                Xf, Yf = env.ground_xy(p.rect, sx, sy, t=self.t)
                d = 0.0
                for sign, am in ((-1.0, old), (1.0, airmass)):
                    if am:
                        env.airmass = am
                        d = d + sign * env.airmass_wind(Xf, Yf, self.t, name, xp=cp)
                if not isinstance(d, float):
                    getattr(st, name)[...] += d.astype(DTYPE)
            update_diagnostics(st)
        env.airmass = airmass if airmass else None
        for p in self.patches:
            p.renew_boundaries(self.t)
            p.am_maps, p.am_time = None, -math.inf
        n = len(airmass.boundaries) if airmass else 0
        self.note("boundaries", f"{n} boundar{'y' if n == 1 else 'ies'} in the run")

    def _update_airmass(self) -> None:
        """Renew moving edge tables before they run out; refresh the patch maps."""
        if not self.env.airmass:
            if self.env.terrain is not None:          # the terrain's rim targets, once per patch
                for p in self.patches:
                    if p.am_maps is None:
                        p.refresh_airmass(self.t)
            return
        for p in self.patches:
            if math.isfinite(p.lbc_end) and p.lbc_end - self.t < self.opt.lbc_renew_margin + self.dt:
                p.renew_boundaries(self.t)
            if p.am_maps is None or self.t - p.am_time >= self.opt.airmass_refresh:
                p.refresh_airmass(self.t)

    def _surface_drag(self, st, cd=None, dt: float | None = None, env=None) -> None:
        """Bulk drag on the lowest level from the ground-relative wind, implicit:
        V_g <- V_g / (1 + dt Cd |V_g| / dz1).  ``cd``: each column's coefficient (ny, nx), from the
        ground under it (_cd_map), or None for the environment's one.  ``env``: the state's grid (a nest
        on its own levels has a thinner lowest layer), the run's by default."""
        import cupy as cp
        env = env or self.env
        mx, my = self.env.motion
        dt = self.dt if dt is None else dt
        dz1 = float(env.z_w[1] - env.z_w[0])
        u, v = st.u[0], st.v[0]                       # (ny, nx+1), (ny+1, nx)
        if cd is None:
            au = av = dt * env.cd / dz1
        else:                                         # at the u and v faces: the two columns' mean
            c = cd * (dt / dz1)
            au = cp.empty_like(u)
            au[:, 1:-1] = 0.5 * (c[:, 1:] + c[:, :-1])
            au[:, 0], au[:, -1] = c[:, 0], c[:, -1]
            av = cp.empty_like(v)
            av[1:-1, :] = 0.5 * (c[1:, :] + c[:-1, :])
            av[0, :], av[-1, :] = c[0, :], c[-1, :]
        ug = 0.5 * (u[:, 1:] + u[:, :-1]) + mx
        vg = 0.5 * (v[1:, :] + v[:-1, :]) + my
        spd = cp.sqrt(ug * ug + vg * vg)
        su = cp.empty_like(u)
        su[:, 1:-1] = 0.5 * (spd[:, 1:] + spd[:, :-1])
        su[:, 0], su[:, -1] = spd[:, 0], spd[:, -1]
        sv = cp.empty_like(v)
        sv[1:-1, :] = 0.5 * (spd[1:, :] + spd[:-1, :])
        sv[0, :], sv[-1, :] = spd[0, :], spd[-1, :]
        u[...] = (u + mx) / (1.0 + au * su) - mx
        v[...] = (v + my) / (1.0 + av * sv) - my

    def _ambient_keep(self):
        """Hurricane runs: the share of the open sea's heat and moisture the steady environment takes away again
        (1 - ocean.ambient below 1.5 km, fading to all of it by 2.5 km): the rest stays, so the open ocean
        around a storm builds its scattered convection and outer rainbands (with all of it removed, nothing but
        the core ever convected)."""
        import cupy as cp
        a = float(getattr(self.env.ocean, "ambient", 0.0) or 0.0)
        if a <= 0.0:
            return None
        if getattr(self, "_keep", None) is None or self._keep[0] != a:
            z = np.asarray(self.env.z_h)
            w = np.clip((2500.0 - z) / 1000.0, 0.0, 1.0)
            self._keep = (a, cp.asarray(1.0 - a * w, dtype=cp.float32)[:, None, None])
        return self._keep[1]

    def _sea_surface(self, st, p, dt: float | None = None) -> None:
        """Hurricane runs (env.ocean): the sea's drag and heat and moisture fluxes over water, the land cover's
        drag over land (hurricane.sea_surface); ``p`` None = the reference column, open sea at the
        environment's SST."""
        import cupy as cp
        from .hurricane import sea_surface
        oc = self.env.ocean
        dt = self.dt if dt is None else dt
        if p is None:
            ny, nx = st.thp.shape[1:]
            sst = cp.full((ny, nx), oc.ref_sst_k, dtype=cp.float32)
            water = cp.ones((ny, nx), dtype=bool)
            land = 0.0
        else:
            sst, water = oc.maps(p, self.t)
            land = self._cd_map(p)
            if land is None:
                land = float(self.env.cd)
        cd, u10 = sea_surface(self.env, st, dt, sst, water, land)
        if p is not None:
            oc.u10[p.id] = u10
        self._surface_drag(st, cd, dt=dt)

    CD_REFRESH = 60.0                                 # s between updates of a patch's drag map

    def _cd_map(self, p):
        """The drag coefficient of each column of patch p from the ground under it now (land cover,
        env.roughness), cached and renewed every CD_REFRESH s as the grid moves; None without one."""
        import cupy as cp
        rough = getattr(self.env, "roughness", None)
        if rough is None:
            return None
        key = (p.rect.i0, p.rect.j0, p.rect.nx, p.rect.ny)
        cache = getattr(p, "_cd", None)
        if cache is None or cache[0] != key or self.t - cache[1] >= self.CD_REFRESH or self.t < cache[1]:
            X, Y = p.env.ground_xy(p.rect, t=self.t)
            cd = p.env.drag_coefficient(np.asarray(rough(X, Y), dtype=np.float64))    # over its lowest layer
            p._cd = (key, self.t, cp.asarray(cd, dtype=cp.float32))
        return p._cd[2]

    # ------------------------------------------------------- storm starts
    def add_start(self, s: forcing.StormStart) -> forcing.StormStart | None:
        """Start a storm; None if a fixed-size domain cannot take it."""
        s.id = self._next_start_id
        self._next_start_id += 1
        self.starts.append(s)
        if not self._ensure_covered(s):
            self.starts.remove(s)
            return None
        self.note("start", f"storm start #{s.id} ({s.kind}) at "
                           f"({s.x / 1000:.0f}, {s.y / 1000:.0f}) km, t0={s.t0:.0f} s")
        return s

    def send_nudger(self, target, motion, lead: float, aim: float = 0.0, strength: float = 8.0,
                    radius: float = 6000.0) -> interactions.Nudger:
        """Send a nudger cell (interactions.py) at a storm: ``target`` "strongest" or a ground
        point (x, y) near its updraft; ``motion`` the cell's ground motion (m/s).  It meets the
        storm's updraft after ``lead`` s (10 min at least), passing ``aim`` m ahead of it (+,
        its forward flank) or behind (-, its rear flank).  It must start in clear air: if that
        start point is on an updraft, in rain or in cold air, the lead grows in 2 min steps (up
        to 20 min more) until it is not.  ValueError says why it cannot."""
        tr = self.tracks.strongest(self.t) if target == "strongest" else self.tracks.nearest(*target, self.t)
        if tr is None:
            raise ValueError("there is no storm with a deep updraft (12 m/s at 3-7 km) "
                             + ("yet" if target == "strongest" else "within 25 km of that point"))
        asked = max(600.0, float(lead))
        tv = tr.velocity(self.env.storm_motion)
        lt, x, y, _ = tr.last
        now = (x + tv[0] * (self.t - lt), y + tv[1] * (self.t - lt))
        why = ""
        for lead in np.arange(asked, asked + 1201.0, 120.0):
            got = interactions.plan(now, tv, motion, lead, aim)
            if got is None:
                raise ValueError("that cell would move almost like the storm and never reach it: pick another motion")
            why = self._blocked_at(*got[0])
            if not why:
                break
        else:
            raise ValueError(f"with {asked / 60:.0f}-{asked / 60 + 20:.0f} min to go its start point is always {why}: "
                             "a separate cell cannot start there (try another motion or aim)")
        lead = float(lead)
        start, meet = got
        if lead > asked:
            self.note("nudger", f"a {asked / 60:.0f} min lead would start the nudger {self._blocked_at(*interactions.plan(now, tv, motion, asked, aim)[0])}; "
                                f"using {lead / 60:.0f} min")
        push = min(900.0, lead - 300.0)                # over at least 5 min before contact
        s = forcing.StormStart(x=start[0], y=start[1], t0=self.t, kind="nudge", radius=radius, strength=strength,
                               duration=push, ramp=min(300.0, push / 3.0), vx=motion[0], vy=motion[1],
                               why="nudger")
        if self.add_start(s) is None:
            raise ValueError("its start point is too far from the storms in the fixed-size domain")
        n = interactions.Nudger(len(self.nudgers) + 1, self.t, self.t + lead, tr.id, float(aim), tuple(motion),
                                start, meet, start_id=s.id, push_end=self.t + push)
        self.nudgers.append(n)
        side = "into" if abs(aim) < 500.0 else f"{abs(aim) / 1000:.0f} km {'ahead of' if aim > 0 else 'behind'}"
        self.note("nudger", f"nudger #{n.id} sent at storm {tr.id} ({tr.strength():.0f} m/s): starts "
                            f"{math.hypot(start[0] - now[0], start[1] - now[1]) / 1000:.0f} km from its updraft, "
                            f"meets it at +{n.meet / 60:.0f} min passing {side} the updraft, push over at "
                            f"+{n.push_end / 60:.0f} min")
        return n

    def _blocked_at(self, x: float, y: float) -> str:
        """Why a new cell cannot start at the ground point ("" if it can): within 8 km of a
        deep updraft, or within 3 km of rain below 2 km or of cold surface air."""
        import cupy as cp
        for tr in self.tracks.live:
            if self.t - tr.last[0] <= 120.0 and math.hypot(tr.last[1] - x, tr.last[2] - y) < 8000.0:
                return f"on the updraft of track {tr.id}"
        env = self.env
        dx = env.grid.dx
        r = max(1, int(round(3000.0 / dx)))
        kq = max(int(np.searchsorted(env.z_h, 2000.0)), 1)
        for p in self.patches:
            i, j = int((x - p.world_x0()) // dx), int((y - p.world_y0()) // dx)
            if not (0 <= i < p.rect.nx and 0 <= j < p.rect.ny):
                continue
            js, is_ = slice(max(j - r, 0), j + r + 1), slice(max(i - r, 0), i + r + 1)
            th = float(p.state.thp[0, js, is_].min())
            if th < -1.0:
                return f"in cold air (surface θ′ {th:.1f} K)"
            if float(forcing.precipitation(p.state, kq)[:, js, is_].max()) > 3.0e-4:
                return "in rain"
            return ""
        return ""

    def _update_tracks(self) -> None:
        peaks = [pk for p in self.patches for pk in interactions.updraft_peaks(p, w_min=self._w_min(p))]
        self.tracks.update(self.t, peaks, self.env.storm_motion)
        for n in self.nudgers:
            msg = n.watch(self.tracks, self.t)
            if msg:
                self.note("nudger", msg)
            self._end_push_near(n)

    def _w_min(self, p: Patch) -> float:
        """The 3-7 km w a deep updraft in this patch needs to count as a storm (track_w at its centre)."""
        if self.track_w is None:
            return 12.0
        dx = self.env.grid.dx
        ox, oy = self.env.offset
        return float(self.track_w(ox + (p.rect.i0 + 0.5 * p.rect.nx) * dx, oy + (p.rect.j0 + 0.5 * p.rect.ny) * dx))

    def _end_push_near(self, n: interactions.Nudger) -> None:
        """A nudger whose cell gets within 10 km of the storm's updraft while still pushed (the storm
        did not move as measured) loses its push now: the merger stays the model's own."""
        if n.done or self.t >= n.push_end - 60.0:
            return
        tgt = self.tracks.resolve(n.target)
        cell = self.tracks.get(n.cell)
        if tgt is None:
            return
        cx, cy = (cell.last[1], cell.last[2]) if cell is not None else n.expected(self.t)
        d = math.hypot(cx - tgt.last[1], cy - tgt.last[2])
        if d > 10000.0:
            return
        for s in self.starts:
            if s.id == n.start_id:
                s.duration = self.t + 60.0 - s.t0          # its fade-out (the last fifth) starts at once
        n.push_end = self.t + 60.0
        self.note("nudger", f"nudger #{n.id}: {d / 1000:.0f} km from the storm's updraft already, push ended")

    def _start_rect(self, s: forcing.StormStart) -> Rect:
        """Grid cells a storm start touches over its life (in the moving grid's frame)."""
        dx = self.env.grid.dx
        dur = s.duration if s.kind == "nudge" else 0.0
        t0, t1 = s.t0, s.t0 + dur
        o0, o1 = self.env.offset_at(t0), self.env.offset_at(t1)
        gx0, gy0 = s.x - o0[0], s.y - o0[1]
        gx1, gy1 = s.x + s.vx * dur - o1[0], s.y + s.vy * dur - o1[1]
        rad = 3.0 * s.rmw if s.kind == "vortex" else s.radius      # a seed vortex: the domain centres on its core
        x0, x1 = min(gx0, gx1) - rad, max(gx0, gx1) + rad
        y0, y1 = min(gy0, gy1) - rad, max(gy0, gy1) + rad
        i0, j0 = int(math.floor(x0 / dx)), int(math.floor(y0 / dx))
        i1, j1 = int(math.ceil(x1 / dx)), int(math.ceil(y1 / dx))
        return Rect(i0, j0, i1 - i0, j1 - j0)

    def _ensure_covered(self, s: forcing.StormStart) -> bool:
        """A patch for a storm start.  False: refused (a fixed-size domain, or patches at their
        size limit, cannot hold it with their storms).  A start the GPU memory cannot take now
        waits (``s.waiting``) and is tried again as memory frees up (_retry_waiting)."""
        if self.fixed:
            return self._cover_fixed(s)
        core = self._start_rect(s).expanded(self.trigger)
        for p in self.patches:
            if p.rect.contains(core):
                return True
        want = self._start_rect(s).expanded(self.margin).snapped(self.opt.block, self.min_side)
        got = self._place(want, reason=f"storm start #{s.id}", core=core)
        if got is _REFUSED:
            self.note("domain", f"storm start #{s.id} at ({s.x / 1000:.0f}, {s.y / 1000:.0f}) km is too close to "
                                f"storms whose patch is at its size limit ({self.cap * self.env.grid.dx / 1000:.0f} km) "
                                "for one patch to hold them all: not started")
            return False
        if got is None:
            self._wait_for_memory(s)
        return True

    def _wait_for_memory(self, s: forcing.StormStart) -> None:
        if s.retry_at == 0.0:
            self.note("memory", f"storm start #{s.id} waits for GPU memory (tried again every 5 min while its "
                                "push lasts)")
        s.waiting = True
        s.retry_at = self.t + 300.0

    def _retry_waiting(self) -> None:
        """Storm starts that found no GPU memory get it once memory frees up, while their push lasts."""
        for s in list(self.starts):
            if not s.waiting or self.t < max(s.retry_at, self._mem_full_until):
                continue
            s.waiting = False
            if self.t > s.t1 - 120.0:
                self.note("memory", f"storm start #{s.id} never got GPU memory before its push ended: not started")
                self.starts.remove(s)
            elif not self._ensure_covered(s):
                self.starts.remove(s)
            elif not s.waiting:
                self.note("start", f"storm start #{s.id} started: GPU memory came free")

    def _cover_fixed(self, s: forcing.StormStart) -> bool:
        """A fixed-size domain takes a storm start inside it, or shifts to hold it along
        with the storms it has; a start it cannot hold is refused (False).  The first
        start places the domain."""
        core = self._start_rect(s).expanded(self.trigger)
        if not self.patches:
            if self._commit(self._fixed_rect(core, core), [], f"storm start #{s.id}") is None:
                self._wait_for_memory(s)
            return True
        p = self.patches[0]
        if p.rect.contains(core):
            return True
        box = self._activity_box(p)
        hold = core if box is None else core.union(box.expanded(self.trigger))
        b = self.opt.block                           # a block to spare, as in _follow_fixed
        if hold.nx <= self.fixed - b and hold.ny <= self.fixed_ny - b:
            if self._commit(self._fixed_rect(hold, hold), [p], f"storm start #{s.id}") is None:
                self._wait_for_memory(s)
            return True
        self.note("domain", f"storm start #{s.id} at ({s.x / 1000:.0f}, {s.y / 1000:.0f}) km is too far from "
                            f"the storms in the fixed {self.fixed * self.env.grid.dx / 1000:.0f} x "
                            f"{self.fixed_ny * self.env.grid.dx / 1000:.0f} km domain: "
                            "not started")
        return False

    def _fit_rect(self, around: Rect, hold: Rect, nx: int, ny: int, lead=(0.0, 0.0)) -> Rect:
        """An nx x ny rectangle centred on ``around`` (moved ``lead`` cells ahead of the storms),
        shifted as little as needed to hold ``hold``; block-aligned like every patch."""
        b = self.opt.block

        def edge(centre, lo, hi, n):
            first, last = math.ceil((hi - n) / b), math.floor(lo / b)   # blocks where [a, a + n) holds [lo, hi)
            k = round((centre - 0.5 * n) / b)
            if first <= last:
                return min(max(k, first), last) * b
            return round((0.5 * (lo + hi) - 0.5 * n) / b) * b          # too big to hold: centred on it

        return Rect(edge(around.i0 + 0.5 * around.nx + lead[0], hold.i0, hold.i1, nx),
                    edge(around.j0 + 0.5 * around.ny + lead[1], hold.j0, hold.j1, ny), nx, ny)

    def _fixed_rect(self, around: Rect, hold: Rect, lead=(0.0, 0.0)) -> Rect:
        """The fixed-size domain centred on ``around``, shifted as little as needed to hold ``hold``."""
        return self._fit_rect(around, hold, self.fixed, self.fixed_ny, lead)

    # ------------------------------------------------ patch size limit
    def _over_cap(self, r: Rect) -> bool:
        return bool(self.cap) and (r.nx > self.cap or r.ny > self.cap)

    def _capped(self, rect: Rect, hold: Rect) -> Rect:
        """``rect`` cut to the size limit where it is wider, holding ``hold`` (what must stay in)."""
        if not self._over_cap(rect):
            return rect
        return self._fit_rect(rect, hold, min(rect.nx, self.cap), min(rect.ny, self.cap))

    def _hold(self, p: Patch, box: Rect) -> Rect | None:
        """What a patch at its size limit keeps inside: its storms and the trigger distance
        around them, or, once they spread wider than the limit leaves room for, its strongest
        storm (as the fixed-size domain does).  None: too spread out, no deep updraft to follow."""
        hold = box.expanded(self.trigger)
        room = self.cap - self.opt.block            # a block to spare, as in _follow_fixed
        if hold.nx <= room and hold.ny <= room:
            return hold
        core = self._main_updraft(p)
        return None if core is None else core.expanded(self.trigger)

    def _strongest_hold(self, patches: list) -> Rect | None:
        """The strongest deep updraft of these patches with the trigger distance around it."""
        best = None
        for q in patches:
            got = self._updraft_core(q)
            if got is not None and (best is None or got[0] > best[0]):
                best = got
        return None if best is None else best[1].expanded(self.trigger)

    def start_area(self):
        """Ground rectangle (x0, y0, x1, y1, m) new storms must start in, or None for
        anywhere: a fixed-size domain takes them only well inside its edges."""
        if not self.fixed or not self.patches:
            return None
        p = self.patches[0]
        dx = self.env.grid.dx
        inset = self.trigger * dx + 15000.0          # its edge zone and a storm start's radius
        x0, y0 = p.world_x0(), p.world_y0()
        return (x0 + inset, y0 + inset, x0 + p.rect.nx * dx - inset, y0 + p.rect.ny * dx - inset)

    # ---------------------------------------------------------- patches
    def total_cells(self) -> int:
        return sum(p.rect.cells for p in self.patches) * self.env.grid.nz

    def _union_touching(self, rect: Rect, exclude=()) -> tuple[Rect, list]:
        """Grow ``rect`` over every patch it touches (until stable)."""
        touching = []
        changed = True
        while changed:
            changed = False
            for q in self.patches:
                if q in exclude or q in touching:
                    continue
                if q.rect.intersects(rect):
                    touching.append(q)
                    rect = rect.union(q.rect)
                    changed = True
        return rect, touching

    def _fits_budget(self, rect: Rect, replacing: list) -> bool:
        others = sum(q.rect.cells for q in self.patches if q not in replacing)
        return (others + rect.cells) * self.env.grid.nz <= self.opt.max_cells

    def _commit(self, rect: Rect, members: list, reason: str) -> Patch | None:
        """Put one patch on ``rect`` holding the contents of ``members``
        (patches that do not overlap one another)."""
        # a fixed-size domain is block-aligned already and keeps its size, even below the minimum side
        rect = rect.snapped(self.opt.block, 0 if self.fixed else self.min_side)
        if self._over_cap(rect):                 # snapping to the minimum side can add a block
            rect = self._capped(rect, rect)
        if not self._fits_budget(rect, members):
            self.note("memory", f"{reason}: a {rect.nx}x{rect.ny} patch would exceed the GPU "
                                f"budget ({self.opt.max_cells / 1e6:.1f} M cells, {self.total_cells() / 1e6:.1f} M "
                                "in use); keeping the current layout")
            return None
        try:
            return self._commit_alloc(rect, members, reason)
        except Exception as exc:
            if not _out_of_memory(exc):
                raise
            import gc
            import cupy as cp
            gc.collect()
            pool = cp.get_default_memory_pool()
            pool.free_all_blocks()
            free = _gpu_free_bytes()
            # the card is fuller than the budget thought (other programs, or arrays not yet
            # released): no new patches for 5 model minutes, then the budget follows what is
            # actually free again (_refresh_budget) -- it used to stay at the cells in use for
            # good, 0 when the refusal came with no patches, and no storm could start again
            self.opt.max_cells = min(self.opt.max_cells, self.total_cells() + int(0.75 * free / self._cell_bytes))
            self._mem_full_until = self.t + 300.0
            self.note("memory", f"{reason}: a {rect.nx}x{rect.ny} patch does not fit in the GPU memory left "
                                f"({free / 2**30:.1f} GB free, this run holds {pool.used_bytes() / 2**30:.1f} GB); "
                                "keeping the current layout, trying again in 5 min")
            # a patch whose arrays could not even be put back (Patch.replace / merged) is gone
            for q in members:
                if q.state is None and q in self.patches:
                    self.patches.remove(q)
                    self.note("memory", f"patch {q.id} lost: no GPU memory to put it back")
            return None

    def _commit_alloc(self, rect: Rect, members: list, reason: str) -> Patch | None:
        if len(members) == 1:
            p = members[0]
            if p.rect == rect:
                return p
            old = p.rect
            p.replace(rect)
            p.oversize_seconds = 0.0
            p.moved_at = self.t
            self.note("patch", f"patch {p.id} {old.nx}x{old.ny} at ({old.i0},{old.j0}) -> "
                               f"{rect.nx}x{rect.ny} at ({rect.i0},{rect.j0}) ({reason})")
            return p
        if members:
            newp = Patch.merged(self.env, members, rect, self.dt)
            vx = sum(q.storm_velocity[0] for q in members) / len(members)
            vy = sum(q.storm_velocity[1] for q in members) / len(members)
            newp.storm_velocity = (vx, vy)
            newp.hand_until = max(getattr(q, "hand_until", -math.inf) for q in members)
            for q in members:
                self.patches.remove(q)
                q.free()
            self.note("patch", f"merged {len(members)} patches into {rect.nx}x{rect.ny} ({reason})")
        else:
            newp = Patch(self.env, rect, self.dt)
            newp.storm_velocity = tuple(self.default_motion)
            self.note("patch", f"new patch {newp.id} {rect.nx}x{rect.ny} ({reason})")
        newp.created_at = self.t
        newp.state.elapsed_seconds = self.t
        self.patches.append(newp)
        return newp

    def _place(self, want: Rect, reason: str, core: Rect | None = None):
        """Cover ``want``: extend/merge the patches it touches, or start a new one
        beside them when merging would mostly add empty air (or, with a size limit,
        make a patch wider than the limit).  Returns the patch, None when the GPU
        memory cannot take it, or _REFUSED when patches at their size limit cannot."""
        if core is not None:
            want = self._capped(want, core)
        rect, touching = self._union_touching(want)
        if touching and core is not None and (self._over_cap(rect) or
                                              self._wasteful(rect, [want] + [q.rect for q in touching])):
            trimmed = self._trim_away(want.snapped(self.opt.block, self.min_side), touching, core)
            if trimmed is not None:
                return self._commit(trimmed, [], reason + ", beside its neighbours")
        if self._over_cap(rect):
            # one patch at its size limit beside it: shifted to hold both, if they fit (else refused)
            if len(touching) != 1 or core is None:
                return _REFUSED
            q = touching[0]
            box = self._activity_box(q)
            hold = core if box is None else core.union(box.expanded(self.trigger))
            if hold.nx > self.cap - self.opt.block or hold.ny > self.cap - self.opt.block:
                return _REFUSED
            rect = self._capped(rect, hold)
        return self._commit(rect, touching, reason)

    def _wasteful(self, union: Rect, parts: list) -> bool:
        return union.snapped(self.opt.block, self.min_side).cells > self.opt.merge_waste * sum(r.cells for r in parts)

    def _trim_away(self, want: Rect, others: list, core: Rect, ignore=()) -> Rect | None:
        """``want`` cut back (keeping ``core``) so it does not overlap ``others``:
        patches far apart sit side by side instead of becoming one rectangle that
        is mostly empty.  None if that cannot keep the core or the minimum size."""
        r = want
        for q in others:
            o = q.rect
            if not r.intersects(o):
                continue
            cands = []
            if o.i0 >= core.i1:
                cands.append(Rect(r.i0, r.j0, o.i0 - r.i0, r.ny))
            if o.i1 <= core.i0:
                cands.append(Rect(o.i1, r.j0, r.i1 - o.i1, r.ny))
            if o.j0 >= core.j1:
                cands.append(Rect(r.i0, r.j0, r.nx, o.j0 - r.j0))
            if o.j1 <= core.j0:
                cands.append(Rect(r.i0, o.j1, r.nx, r.j1 - o.j1))
            cands = [c for c in cands if c.nx >= self.min_side and c.ny >= self.min_side and c.contains(core)]
            if not cands:
                return None
            r = max(cands, key=lambda c: c.cells)
        if any(r.intersects(q.rect) for q in self.patches if q not in ignore):
            return None
        return r

    @property
    def default_motion(self):
        """Expected storm motion relative to the (possibly moving) grid."""
        rm = getattr(self.env, "storm_motion", None) or (0.0, 0.0)
        mx, my = self.env.motion
        return (rm[0] - mx, rm[1] - my)

    def _activity_box(self, p: Patch) -> Rect | None:
        import cupy as cp
        b = 4
        act = forcing.activity_map(p)
        ny, nx = act.shape
        pad_y, pad_x = (-ny) % b, (-nx) % b
        if pad_y or pad_x:
            act = cp.pad(act, ((0, pad_y), (0, pad_x)))
        coarse = cp.asnumpy(act.reshape(act.shape[0] // b, b, act.shape[1] // b, b).any(axis=(1, 3)))
        box = None
        if coarse.any():
            jj, ii = np.nonzero(coarse)
            box = Rect(p.rect.i0 + int(ii.min()) * b, p.rect.j0 + int(jj.min()) * b,
                       (int(ii.max()) - int(ii.min()) + 1) * b, (int(jj.max()) - int(jj.min()) + 1) * b)
        for s in self.starts:
            if s.t1 + 600.0 >= self.t and p.rect.intersects(self._start_rect(s)):
                r = self._start_rect(s)
                box = r if box is None else box.union(r)
        r = self._hold_rect()
        if r is not None and p.rect.intersects(r):
            box = r if box is None else box.union(r)
        return box

    def _hold_rect(self) -> Rect | None:
        """Grid cells around the tracked cyclone (hold_point, ground x, y, radius m), or None."""
        hp = self.hold_point
        if hp is None:
            return None
        dx = self.env.grid.dx
        ox, oy = self.env.offset
        i, j, n = (hp[0] - ox) / dx, (hp[1] - oy) / dx, hp[2] / dx
        return Rect(int(math.floor(i - n)), int(math.floor(j - n)), int(2 * n) + 1, int(2 * n) + 1)

    def _want(self, p: Patch, box: Rect) -> Rect:
        """Activity box + margin, with extra room ahead of the storms."""
        dx = self.env.grid.dx
        vx, vy = p.storm_velocity
        li = vx * self.opt.lookahead / dx
        lj = vy * self.opt.lookahead / dx
        r = box.expanded(self.margin)
        i0 = r.i0 + min(0, int(math.floor(li)))
        i1 = r.i1 + max(0, int(math.ceil(li)))
        j0 = r.j0 + min(0, int(math.floor(lj)))
        j1 = r.j1 + max(0, int(math.ceil(lj)))
        return Rect(i0, j0, i1 - i0, j1 - j0).snapped(self.opt.block, self.min_side)

    def _refresh_budget(self) -> None:
        """Let the budget follow the memory actually free.  After a refusal it comes back once
        the card has room; an automatic budget also grows past its first value when memory
        frees up later (the run may have measured the card while a previous run's arrays, or
        another program, still held it).  A budget set by hand never grows past itself."""
        self._measure_memory()
        if self.t < self._mem_full_until:
            return
        if not self._auto_budget and self.opt.max_cells >= self._max_cells0:
            return
        room = self.total_cells() + int(0.75 * _gpu_free_bytes() / self._cell_bytes)
        ceiling = room if self._auto_budget else min(self._max_cells0, room)
        ceiling = min(ceiling, getattr(self, "_vram_cells", ceiling))     # never past the card itself
        if ceiling > 1.05 * self.opt.max_cells:
            self.opt.max_cells = ceiling
            self.note("memory", f"GPU memory free again: budget {self.opt.max_cells / 1e6:.1f} M cells")

    def nest_cell_bytes(self) -> float:
        """GPU bytes a nest cell is counted at (nest.Nests._room_for)."""
        return self._cell_bytes * self._nest_mem

    def gpu_room(self, moving: float = 0.0) -> float:
        """GPU bytes a new grid (a nest) may still take: within VRAM_SHARE of the card in all -- the arrays in
        use with the step's scratch over them -- and within most of what the device reports free.  ``moving``:
        bytes of a grid the new one replaces (freed right after), counted back toward the share."""
        try:
            import cupy as cp
            used = int(cp.get_default_memory_pool().used_bytes())
        except Exception:
            return math.inf
        total = _vram_total()
        if not total:
            return math.inf
        room = VRAM_SHARE * total - CONTEXT_BYTES - used * getattr(self, "_scratch", 1.15) + moving
        free = _gpu_free_bytes()
        if free > 0:
            room = min(room, (0.9 if moving else 0.8) * free)
        return max(0.0, room)

    def recover_memory(self, mid_step: bool) -> str:
        """After the card ran out of memory (cudaErrorMemoryAllocation), instead of the run ending: what the
        failed step held goes back to the card, the biggest nest goes (with the nests inside it), every nest
        level waits nest.MEM_WAIT and counts a nest cell bigger from now on, and a step cut off part way rolls
        back to the checkpoint (its arrays are half updated).  Returns what was done."""
        import gc
        from .nest import MEM_WAIT
        gc.collect()
        release_freed()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        self._nest_mem = min(4.0, self._nest_mem * 1.5)
        done = []
        levels, lv = [], self.nests
        while lv is not None:
            levels.append(lv)
            lv = lv.inner
        found = [(lv, n) for lv in levels for n in lv.items]
        if found:
            lv, n = max(found, key=lambda q: q[1].patch.rect.cells * len(q[0].env.z_h))
            lv.drop(n, "the GPU ran out of memory")
            done.append(f"nest {n.id} removed")
        for lv in levels:
            lv._mem_wait = self.t + MEM_WAIT
        if mid_step and self._checkpoint is not None:
            t = self.t
            self._restore_checkpoint()
            done.append(f"rolled back {t - self.t:.0f} s to the last checkpoint")
        # no new or bigger patches for a while either, the budget down to what is in use
        self.opt.max_cells = min(self.opt.max_cells, self.total_cells())
        self._mem_full_until = self.t + 300.0
        release_freed()
        return ", ".join(done) or "memory given back"

    def _measure_memory(self) -> None:
        """Bytes per cell as this run really uses them (microphysics species, radars, the suite's scratch
        differ from CELL_BYTES: 760 measured with aerosol Thompson at 1 km), and the card's cap from them.
        It goes up at once and down slowly, and never below CELL_BYTES.  A budget over the new cap comes
        down to it: the patches stop growing and new storms wait instead of the run spilling into shared
        RAM (patches already bigger keep their cells until they shrink or retire).

        Between steps the pool holds the patches' arrays (in use) and the step's scratch (free blocks it
        takes back every step), but also the blocks of patches that since grew, shrank or went.  So: the
        arrays in use x the scratch ratio, that ratio learnt while the patch sizes stay the same, and the
        stale blocks handed back to the card when they change."""
        cells = self.total_cells()
        if cells < MEASURE_CELLS:
            return
        try:
            import cupy as cp
            pool = cp.get_default_memory_pool()
            used, held = int(pool.used_bytes()), int(pool.total_bytes())
        except Exception:
            return
        base = self._pool_base
        layout = sorted((p.rect.nx, p.rect.ny) for p in self.patches)
        if layout == getattr(self, "_measured_layout", None):
            self._scratch = max(1.0, (held - base) / max(1, used - base))
        else:
            self._measured_layout = layout
            if held > 1.1 * used * getattr(self, "_scratch", 1.15):
                pool.free_all_blocks()
        now = (used - base) / cells * getattr(self, "_scratch", 1.15)
        if now <= 0:
            return
        old = self._cell_bytes
        self._cell_bytes = max(CELL_BYTES, now if now > old else 0.9 * old + 0.1 * now)
        self._vram_cells = _vram_cells(self._cell_bytes, base)
        if self.opt.max_cells > self._vram_cells:
            lowered = self.opt.max_cells > 1.05 * self._vram_cells
            self.opt.max_cells = self._vram_cells
            if lowered:
                self.note("memory", f"this run uses {self._cell_bytes:.0f} bytes per cell: budget "
                                    f"{self.opt.max_cells / 1e6:.1f} M cells, so it stays in the GPU's own memory")

    def remove_patch(self, pid: int) -> bool:
        """Delete a patch by hand: its storms, the starts still to come inside it (they would only build
        it again) and its copy in the checkpoint go too, and its memory is free for the others at once."""
        p = next((q for q in self.patches if q.id == pid), None)
        if p is None:
            return False
        dx = self.env.grid.dx
        ox, oy = self.env.offset                 # starts and tracks are on the ground, the patch moves with the grid
        x0, y0 = p.rect.i0 * dx + ox, p.rect.j0 * dx + oy
        x1, y1 = p.rect.i1 * dx + ox, p.rect.j1 * dx + oy

        def inside(x, y):
            return x0 <= x <= x1 and y0 <= y <= y1

        self.patches.remove(p)
        p.free()
        release_freed()
        self.starts[:] = [s for s in self.starts if s.t1 <= self.t or not inside(*s.position(self.t))]
        for tr in list(self.tracks.live):
            if inside(tr.last[1], tr.last[2]):
                tr.ended = self.t
                self.tracks.live.remove(tr)
                self.tracks.done.append(tr)
        if self._checkpoint is not None:
            t, nstep, snap, applied = self._checkpoint
            self._checkpoint = (t, nstep, [c for c in snap if c[0] != pid], applied)
        self._mem_full_until = -1.0
        self._refresh_budget()
        self.note("patch", f"patch {pid} removed by hand ({p.rect.nx}x{p.rect.ny} cells freed)")
        return True

    def memory_full(self) -> bool:
        """New storms cannot get a patch now (the UI says so)."""
        if self.fixed and self.patches:
            return False                          # they start inside the fixed domain
        need = self.fixed * self.fixed_ny if self.fixed else self.min_side * self.min_side
        return self.t < self._mem_full_until or self.opt.max_cells - self.total_cells() < need * self.env.grid.nz

    def manage(self) -> None:
        """Re-fit every patch to its storms: follow them, grow ahead of them,
        trim what they left behind, merge patches that meet, retire quiet ones."""
        self._refresh_budget()
        self._update_tracks()
        dtm = self.opt.manage_interval if self._last_manage is None else max(self.t - self._last_manage, 1.0)
        self._last_manage = self.t
        dx = self.env.grid.dx
        for p in list(self.patches):
            if p not in self.patches:
                continue
            box = self._activity_box(p)
            held = self.t < getattr(p, "hand_until", -math.inf)      # moved by hand: it stays put
            if box is None:
                if held:
                    continue
                p.quiet_seconds += dtm
                p.storm_center = None
                if p.quiet_seconds >= self.opt.retire_after:
                    self.patches.remove(p)
                    p.free()
                    release_freed()
                    self.note("patch", f"patch {p.id} retired (quiet for {self.opt.retire_after / 60:.0f} min)")
                continue
            p.quiet_seconds = 0.0
            cx = (box.i0 + 0.5 * box.nx) * dx
            cy = (box.j0 + 0.5 * box.ny) * dx
            if p.storm_center is not None:
                vx = (cx - p.storm_center[0]) / dtm
                vy = (cy - p.storm_center[1]) / dtm
                sp = math.hypot(vx, vy)
                if sp > 45.0:                       # a new cell appeared, not motion
                    vx, vy = vx * 45.0 / sp, vy * 45.0 / sp
                a = 0.25
                p.storm_velocity = ((1 - a) * p.storm_velocity[0] + a * vx,
                                    (1 - a) * p.storm_velocity[1] + a * vy)
            p.storm_center = (cx, cy)
            if held:
                continue
            if self.fixed:
                self._follow_fixed(p, box)
                continue
            need = box.expanded(self.trigger)
            want = self._want(p, box)
            why = "following the storms"
            recentre = False
            if self._over_cap(want):
                # at its size limit: it shifts instead of growing, holding its storms or, once they
                # spread wider than that leaves room for, its strongest storm
                need = self._hold(p, box)
                if need is None:
                    continue                          # no deep updraft left to follow: stay
                ride = self._ride(p)
                if ride is not None:
                    if ride != p.rect:
                        self._commit(ride, [p], "moving with the grid motion asked for")
                    continue
                core = self._storms_core(p, self.cap)
                if core is not None:
                    # on its strong storms with room ahead of them, before they reach an edge
                    # all its storms when they fit, else what _storms_core chose (not _hold's one storm
                    # nearest the centre, which pulled the window off the storms ranked first)
                    win = self._storm_window(p, core, self.cap, need if need == box.expanded(self.trigger) else None)
                    if win is None:
                        continue
                    want, recentre = win, True
                else:
                    if not p.rect.contains(need) and self.t - p.moved_at < self.opt.fixed_settle:
                        continue
                    want = self._capped(want, need)
                why = "following the storms at its size limit"
            if not p.rect.contains(need) or recentre:
                rect, touching = self._union_touching(want, exclude=(p,))
                if touching and (self._over_cap(rect) or
                                 self._wasteful(rect, [p.rect, want] + [q.rect for q in touching])):
                    others = [q for q in self.patches if q is not p]
                    trimmed = self._trim_away(want, others, need, ignore=(p,))
                    if trimmed is None and self._over_cap(rect):
                        # a neighbour is in the way and one patch for both would be too wide: hold only
                        # its strongest storm, so the patches merge only once the storms really meet
                        core = self._main_updraft(p)
                        if core is not None:
                            core = core.expanded(self.trigger)
                            if p.rect.contains(core):
                                continue
                            trimmed = self._trim_away(self._capped(want, core), others, core, ignore=(p,))
                    if trimmed is not None:
                        self._commit(trimmed, [p], why + ", beside its neighbours")
                        continue
                    if self._over_cap(rect):
                        # its storms meet another patch's and one patch holding them all would be wider
                        # than the limit: one patch of the largest size on the strongest storm
                        rect = self._capped(rect, self._strongest_hold([p] + touching) or need)
                        why = "storms meeting, at the size limit: on the strongest storm"
                self._commit(rect, [p] + touching, why)
            elif p.rect.cells > 2.5 * want.cells:
                p.oversize_seconds += dtm
                if p.oversize_seconds >= self.opt.shrink_after:
                    rect, touching = self._union_touching(want, exclude=(p,))
                    if not touching:
                        self._commit(rect, [p], "trimmed to the storms")
            else:
                p.oversize_seconds = 0.0
        # patches that grew into each other become one (with a size limit: on the strongest storm)
        merged = not self.fixed
        while merged:
            merged = False
            for a in list(self.patches):
                rect, touching = self._union_touching(a.rect, exclude=(a,))
                if touching:
                    if self._over_cap(rect):
                        rect = self._capped(rect, self._strongest_hold([a] + touching) or a.rect)
                    if self._commit(rect, [a] + touching, "storms meeting") is not None:
                        merged = True
                    break
        self._retry_waiting()

    def _follow_fixed(self, p: Patch, box: Rect) -> None:
        """A fixed-size domain shifts (it never grows) to keep its storms away from its
        edges, or, once they no longer fit in it, its strongest storm."""
        n, b = min(self.fixed, self.fixed_ny), self.opt.block
        hr = self._hold_rect()
        if hr is not None:
            # a hurricane run: the domain stays centred on its cyclone (rainbands and outer cells must not
            # drag it off), re-centring once the centre drifts FOLLOW_RECENTRE of a side away
            ci, cj = hr.i0 + 0.5 * hr.nx, hr.j0 + 0.5 * hr.ny
            if (max(abs(ci - (p.rect.i0 + 0.5 * p.rect.nx)), abs(cj - (p.rect.j0 + 0.5 * p.rect.ny)))
                    > FOLLOW_RECENTRE * n and self.t - p.moved_at >= 120.0):
                rect = self._fixed_rect(hr, hr)
                if rect != p.rect:
                    self._commit(rect, [p], "following the cyclone")
            return
        hold, why = box.expanded(self.trigger), "following the storms"
        # a block to spare: a box any bigger may fit no block-aligned place, and the domain would
        # hop between two places every minute
        if hold.nx > self.fixed - b or hold.ny > self.fixed_ny - b:
            core = self._main_updraft(p)
            if core is None:
                return                            # no deep updraft left to follow: stay
            hold, why = core.expanded(self.trigger), "following its strongest storm"
        ride = self._ride(p)
        if ride is not None:
            if ride != p.rect:
                self._commit(ride, [p], "moving with the grid motion asked for")
            return
        core = self._storms_core(p, n)
        if core is not None:
            rect = self._storm_window(p, core, self.fixed, hold if hold == box.expanded(self.trigger) else None,
                                      ny=self.fixed_ny)
            if rect is not None:
                self._commit(rect, [p], why)
            return
        if p.rect.contains(hold) or self.t - p.moved_at < self.opt.fixed_settle:
            return
        dx = self.env.grid.dx
        vx, vy = p.storm_velocity
        lead = (0.5 * vx * self.opt.lookahead / dx, 0.5 * vy * self.opt.lookahead / dx)
        rect = self._fixed_rect(box, hold, lead)
        if rect != p.rect:
            self._commit(rect, [p], why)

    def _patch_motion(self, p: Patch):
        """Grid-relative motion (m/s) of the patch's strongest updraft: least squares over its track
        when that holds 5 points over 8 min, else the expected storm motion."""
        track = getattr(p, "updraft_track", None) or []
        if len(track) >= 5 and track[-1][0] - track[0][0] >= 480.0:
            a = np.asarray(track[-15:])
            t = a[:, 0] - a[:, 0].mean()
            den = float((t * t).sum())
            if den > 0:
                mx, my = self.env.motion
                return (float((t * (a[:, 1] - a[:, 1].mean())).sum()) / den - mx,
                        float((t * (a[:, 2] - a[:, 2].mean())).sum()) / den - my)
        return self.default_motion

    def _ride(self, p: Patch) -> Rect | None:
        """Where a patch at its size limit (a fixed domain) goes by riding domain_motion: its drifted window (or
        its own rectangle), or None when it should follow its storms instead -- no motion was asked for (the grid
        "Fixed to the ground" with the built-in core: domain_motion (0, 0) left the domain standing while its storm
        drove out of the edge, 2026-11-03), or riding it would leave the main storm within the trigger distance
        of an edge (a storm moving off the set motion)."""
        dm = self.domain_motion
        if dm is None or (abs(dm[0]) < 0.05 and abs(dm[1]) < 0.05):
            return None
        win = self._drift_window(p) or p.rect
        core = self._main_updraft(p)
        if core is not None and not win.contains(core.expanded(self.trigger)):
            return None
        return win

    def _drift_window(self, p: Patch) -> Rect | None:
        """The patch moved on at domain_motion since it reached this size (whole blocks, so it steps every
        block's worth of travel: 8 km at 500 m), or None while it has not travelled a half block yet."""
        dx, b = self.env.grid.dx, self.opt.block
        a = getattr(p, "drift_anchor", None)
        if a is None or a[3:] != (p.rect.nx, p.rect.ny):
            a = p.drift_anchor = (self.t, float(p.rect.i0), float(p.rect.j0), p.rect.nx, p.rect.ny)
        mx, my = self.env.motion
        vx, vy = self.domain_motion[0] - mx, self.domain_motion[1] - my      # relative to the grid
        ti = a[1] + vx * (self.t - a[0]) / dx
        tj = a[2] + vy * (self.t - a[0]) / dx
        di = int(round((ti - p.rect.i0) / b)) * b
        dj = int(round((tj - p.rect.j0) / b)) * b
        if di == 0 and dj == 0:
            return None
        return Rect(p.rect.i0 + di, p.rect.j0 + dj, p.rect.nx, p.rect.ny)

    def _storm_window(self, p: Patch, core: Rect, n: int, need: Rect | None = None, ny: int = 0) -> Rect | None:
        """An n x n window on the strongest storm, moved FOLLOW_LEAD s of its motion ahead (its inflow side
        has room, its outflow trails), or None when the patch is near enough to that already (within
        FOLLOW_RECENTRE of its side, the storm well inside) or has only just moved.  With a grid fixed to
        the ground (a physics suite) the storms cross the patches, which only shifted once the storm was a
        trigger distance from an edge: the storm rode the leading edge and the domain seemed to stand still."""
        need = need or core.expanded(self.trigger)       # what must stay inside: all its storms when they fit
        if self.t - p.moved_at < self.opt.fixed_settle and p.rect.contains(need):
            return None
        vx, vy = self._patch_motion(p)
        dx = self.env.grid.dx
        rect = self._fit_rect(core, need, n, ny or n, (vx * FOLLOW_LEAD / dx, vy * FOLLOW_LEAD / dx))
        off = math.hypot(rect.i0 + 0.5 * rect.nx - (p.rect.i0 + 0.5 * p.rect.nx),
                         rect.j0 + 0.5 * rect.ny - (p.rect.j0 + 0.5 * p.rect.ny))
        # settled: near its place and every strong storm FOLLOW_EDGE of the side clear of the edges (storms
        # spread either side of the centre kept it from shifting while the outer ones reached an edge)
        e = max(self.trigger, int(FOLLOW_EDGE * n))
        inner = Rect(p.rect.i0 + e, p.rect.j0 + e, max(p.rect.nx - 2 * e, 1), max(p.rect.ny - 2 * e, 1))
        settled = off < FOLLOW_RECENTRE * n and inner.contains(core)
        if p.rect.contains(need) and (settled or rect == p.rect):
            return None
        return rect

    def _storms_core(self, p: Patch, side: int, radius_km: float = 10.0) -> Rect | None:
        """What a patch at its size limit centres on: its nests (the tornado zoom first: a patch that leaves
        a nest drops it), then its storms best ranked first (nest.Rotation.score: a rotating storm over a
        stronger plain updraft), each taken if the box round all taken so far still fits with room to spare.
        Storms ranked under FOLLOW_STRONG of the best are left out.  It used to take every strong updraft
        if they all fit, else only the strongest one nearest the patch centre: with storms spread wider than
        the patch, a supercell or tornadic storm away from the centre rode an edge, or out of the patch."""
        from .nest import PARENT_MARGIN, Rotation
        peaks = interactions.updraft_peaks(p, w_min=self._w_min(p))
        dx = self.env.grid.dx
        ox, oy = self.env.offset
        r = int(round(radius_km * 1000.0 / dx))
        room = side - 2 * self.trigger - self.opt.block
        box = None

        def take(cand: Rect) -> bool:
            nonlocal box
            u = cand if box is None else box.union(cand)
            if u.nx <= room and u.ny <= room:
                box = u
                return True
            return False

        if self.nests:
            mine = [n for n in self.nests.items if n.parent_rect().intersects(p.rect)]
            for n in sorted(mine, key=lambda n: (n.pin is None, not n.hand, -n.rank)):
                take(n.parent_rect().expanded(PARENT_MARGIN))
        if peaks:
            rot = Rotation([p])
            scored = sorted(((rot.score(w, x, y), x, y) for x, y, w in peaks), reverse=True)
            for s, x, y in scored:
                if s < FOLLOW_STRONG * scored[0][0]:
                    break
                i, j = int((x - ox) / dx), int((y - oy) / dx)
                take(Rect(i - r, j - r, 2 * r + 1, 2 * r + 1))
        return box if box is not None else self._main_updraft(p, radius_km)

    def _main_updraft(self, p: Patch, radius_km: float = 10.0) -> Rect | None:
        """World cells around the patch's strongest 3-7 km updraft (of those within 25 % of
        it, the one nearest the patch centre, so the domain does not jump between storms
        of about the same strength).  None without a deep updraft."""
        got = self._updraft_core(p, radius_km)
        return None if got is None else got[1]

    def _updraft_core(self, p: Patch, radius_km: float = 10.0):
        """(peak w, world cells around it) of _main_updraft, or None."""
        import cupy as cp
        env = self.env
        k3 = int(np.searchsorted(env.z_w, 3000.0))
        k7 = max(int(np.searchsorted(env.z_w, 7000.0)), k3 + 1)
        wm = p.state.w[k3:k7].max(axis=0)
        peak = float(wm.max())
        if peak < 10.0:
            return None
        jj, ii = (cp.asnumpy(a) for a in cp.nonzero(wm >= 0.75 * peak))
        m = int(np.argmin((jj - 0.5 * p.rect.ny) ** 2 + (ii - 0.5 * p.rect.nx) ** 2))
        r = int(round(radius_km * 1000.0 / env.grid.dx))
        side = min(self.fixed, self.fixed_ny) or self.cap
        if side:                                   # small enough to fit with the trigger distance around it
            r = min(r, side // 2 - self.trigger - 1)
        r = max(1, r)
        return peak, Rect(p.rect.i0 + int(ii[m]) - r, p.rect.j0 + int(jj[m]) - r, 2 * r + 1, 2 * r + 1)

    # ------------------------------------------------------- time step
    def _cfl(self):
        import cupy as cp
        dx = self.env.grid.dx
        ch, cv = 0.0, 0.0
        for p in self.patches + self.nest_patches():
            st = p.state
            umax = float(cp.abs(st.u).max())
            vmax = float(cp.abs(st.v).max())
            f = p.env.grid.dx / dx                        # a nest steps dt / r over dx / r
            ch = max(ch, max(umax, vmax) * self.dt / dx)
            if st.w.shape[0] != self._rdz_face.shape[0]:
                continue        # a nest on its own levels (tornado zoom) takes extra sub-steps for its own w
            wv = float((cp.abs(st.w) * self._rdz_face).max())
            cv = max(cv, wv * self.dt * f)
        return ch, cv

    def _adapt_dt(self) -> None:
        if not self.patches:
            return
        ch, cv = self._cfl()
        cap = self.dt_cap if self.t < self.dt_cap_until else self.env.grid.dt
        if self.t >= self.dt_cap_until:
            self._rollbacks = 0
        new = self.dt
        if cv > self.opt.cfl_v_target or ch > self.opt.cfl_h_target:
            new = self.dt * 0.8
        elif cv < 0.7 * self.opt.cfl_v_target and ch < 0.7 * self.opt.cfl_h_target:
            new = self.dt * 1.05
        quantum = self.env.grid.dt / 24.0
        new = max(quantum * 4, min(cap, round(new / quantum) * quantum))
        if abs(new - self.dt) > 1e-9:
            self.dt = new
        self.cfl = (ch, cv)

    # ------------------------------------------------------- watchdog
    def _healthy(self) -> tuple[bool, str]:
        import cupy as cp
        for p in self.patches:
            st = p.state
            wmax = float(cp.abs(st.w).max())
            warm = float((st.thp - self._warm_limit).max())      # > 0: warmer than any air can be
            cold = float((-st.thp - self._cold_limit).max())     # > 0: colder than any parcel can be
            if not (math.isfinite(wmax) and math.isfinite(warm) and math.isfinite(cold)):
                return False, f"non-finite values in patch {p.id}"
            if wmax > 160.0 or warm > 0.0 or cold > 0.0:
                tmax = float(cp.abs(st.thp).max())
                return False, f"runaway values in patch {p.id} (|w| {wmax:.0f} m/s, |th'| {tmax:.0f} K)"
        return True, ""

    # what a patch knows about its storms besides its arrays (a rollback restores it too: a patch rebuilt
    # without it forgot its storms' motion and, as a fixed-size domain, when it last moved)
    _PATCH_BOOKKEEPING = ("quiet_seconds", "storm_velocity", "storm_center", "moved_at", "created_at",
                          "oversize_seconds")

    def _save_checkpoint(self) -> None:
        snap = [(p.id, HostCopy(p), {k: getattr(p, k) for k in self._PATCH_BOOKKEEPING},
                 list(getattr(p, "updraft_track", []))) for p in self.patches]
        applied = {s.id: s.applied for s in self.starts}
        self._checkpoint = (self.t, self.nstep, snap, applied)
        if getattr(self.env, "ls_dth", None) is not None:      # the lifted / followed environment goes back too
            self._ls_checkpoint = (self.env.ls_dth.copy(), self.env.ls_dqv.copy(),
                                   self.ls.lift_m if self.ls is not None else 0.0)

    def _restore_checkpoint(self) -> bool:
        if self._checkpoint is None:
            return False
        t, nstep, snap, applied = self._checkpoint
        for p in self.patches:
            p.free()
        self.patches = []
        release_freed()                          # before the restored patches take their place
        self.env.truncate_motion(t)              # the saved winds are relative to the grid motion then
        if getattr(self, "_ls_checkpoint", None) is not None:
            self.env.ls_dth, self.env.ls_dqv = self._ls_checkpoint[0].copy(), self._ls_checkpoint[1].copy()
            self.env._bsnap = None
            if self.ls is not None:
                self.ls.lift_m = self._ls_checkpoint[2]
                self.ls.reset_balance()
            cache = self.env.__dict__.get("_dev_cache") or {}
            cache.pop("ls_thp", None)
            cache.pop("ls_qv", None)
        self._set_offset(t)
        for pid, held, book, track in snap:
            p = Patch(self.env, held.rect, self.dt)
            p.id = pid
            held.put(p.state, held.rect, whole=True)
            for k, v in book.items():
                setattr(p, k, v)
            if track:
                p.updraft_track = list(track)
            self.patches.append(p)
        self.t, self.nstep = t, nstep
        if self.nests:
            self.nests.clear("rolled back with the run")
        for s in self.starts:
            s.applied = applied.get(s.id, s.applied)
        for tr in self.tracks.live:                  # the updraft tracks go back with the model
            tr.pts = [p for p in tr.pts if p[0] <= t]
        self.tracks.live = [tr for tr in self.tracks.live if tr.pts]
        if self._ref is not None:
            self._make_reference()
        return True

    # ------------------------------------------------------ persistence
    def _find_cores(self) -> None:
        """Roots of deep updrafts for the persistence lift, with their remembered
        strength when persistence_memory is on, and their surroundings for the root lift."""
        tau = self.opt.persistence_memory
        if tau > 0:
            if self._memory is None or self._memory.tau != tau:
                self._memory = forcing.CoreMemory(tau)
            found = {p.id: forcing.updraft_cores(p, any_root=True) for p in self.patches + self.nest_patches()}
            self._cores = self._memory.update(found, self.t, self.env.offset)
        else:
            self._cores = {p.id: forcing.updraft_cores(p) for p in self.patches + self.nest_patches()}
        self._roots = ({p.id: forcing.root_surroundings(p, self._cores.get(p.id))
                        for p in self.patches + self.nest_patches()}
                       if self.opt.persistence_root > 0 else {})

    # ------------------------------------------------------------ step
    def step(self) -> None:
        wall0 = time.perf_counter()
        if self.opt.manage and self.t >= self._next_manage:
            self.manage()
            self._next_manage = self.t + self.opt.manage_interval
            self._follow_step()
            if self.nests:
                self.nests.manage()
        if self.t >= self._next_checkpoint and self.patches:
            if self._healthy()[0]:              # never checkpoint a state that is already going bad
                self._save_checkpoint()
                self._rollbacks = 0
            self._next_checkpoint = self.t + self.opt.checkpoint_interval
        if self.nstep % 3 == 0:
            self._adapt_dt()
        self._set_offset(self.t)
        self._update_airmass()
        # bubbles due now
        for s in self.starts:
            if s.kind == "bubble" and not s.applied and s.t0 <= self.t:
                for p in self.patches:
                    if p.rect.intersects(self._start_rect(s)):
                        forcing.apply_bubble(p, s)
                s.applied = True
            elif s.kind == "vortex" and not s.applied and s.t0 <= self.t:
                from .hurricane import apply_vortex
                drop = 0.0
                for p in self.patches:
                    drop = max(drop, apply_vortex(p, s.x, s.y, s.strength, s.rmw, s.radius))
                s.applied = True
                self.note("cyclone", f"seed vortex #{s.id}: {s.strength / 0.514444:.0f} kt at {s.rmw / 1000:.0f} km, "
                                     f"{s.radius / 1000:.0f} km across its circulation, central pressure "
                                     f"{drop:.1f} hPa below the environment")
        self._set_offset(self.t)
        # a physics suite calls radiation every so many STEPS: after the step shortens, the steps counted
        # at the old length no longer reach its interval in model time and its surface scheme refuses the
        # older sky (gpuwm's carrier check).  Radiation is called anew on the first shorter step.
        refresh = self.dt < getattr(self, "_rad_dt", self.dt) - 1e-9
        self._rad_dt = self.dt
        states = ([p.state for p in self.patches + self.nest_patches()]
                  + ([self._ref] if self._ref is not None else []))
        if refresh:
            for st in states:
                ph = getattr(st, "physics", None)
                if ph is not None and hasattr(ph, "radiation_due_override"):
                    ph.radiation_due_override = True
        for p in self.patches:
            p.set_dt(self.dt)
            p.step()
            if getattr(self.env, "ocean", None) is not None:
                self._sea_surface(p.state, p)
            elif self.env.own_drag:
                self._surface_drag(p.state, self._cd_map(p))
        if getattr(self.env, "ocean", None) is not None:
            self.env.ocean.stir(self.patches, self.t)        # the cold wake under strong winds
        if self.opt.drift_correction and self.patches:      # nothing to correct without patches
            self._drift_correction()
        if self.opt.follow_environment and (self.patches or not self.opt.drift_correction):
            self._follow_reference()
        if refresh:
            for st in states:
                ph = getattr(st, "physics", None)
                if ph is not None and getattr(ph, "radiation_due_override", None) is not None:
                    ph.radiation_due_override = None
        self._set_offset(self.t + self.dt)
        if self.opt.persistence > 0 and self.nstep % 5 == 0:
            self._find_cores()
        for p in self.patches:
            forcing.apply_nudging(p, self.starts, self.t + self.dt, self.dt)
            if self.opt.persistence > 0:
                forcing.apply_keep_alive(p, self._cores.get(p.id), self.dt,
                                         self.opt.persistence * self.opt.persistence_wmax,
                                         z_peak=self.opt.persistence_zpeak,
                                         z_half=self.opt.persistence_zhalf)
                if self.opt.persistence_root > 0:
                    forcing.apply_root_lift(p, self._roots.get(p.id), self.dt,
                                            self.opt.persistence * self.opt.persistence_root)
                if self.opt.cap_easing > 0:
                    forcing.apply_ease_cap(p, self._cores.get(p.id), self.dt,
                                           min(1.0, self.opt.persistence * self.opt.cap_easing))
            if self.opt.rim:
                forcing.apply_rim(p, self.dt, self.rim, self.opt.rim_tau)
            if self.opt.drift_correction:
                # only with the steady environment (its seam): without it the reference column runs free, and on
                # an uncapped sounding its overturning boundary layer's TKE churned every patch edge into echo lines
                self._edge_tke(p)
            if p.am_maps is not None:
                forcing.apply_airmass(p, self.dt, self.rim, self.opt.rim_tau, self.opt.airmass_keep_tau,
                                      rim=self.opt.rim)
                forcing.apply_wind_hold(p, self.dt)
            _clamp_vapour(p.state)
        if self.ls is not None and not getattr(self.env.airmass, "synoptic", False):
            self._large_scale()
        if self.nests and self.nests.items:
            self.nests.step(self.t, self.dt, self._nest_substep)
        if self.tornado is not None:
            self.tornado.advect(self.dt)
        if refresh:
            for st in (p.state for p in self.nest_patches()):
                ph = getattr(st, "physics", None)
                if ph is not None and getattr(ph, "radiation_due_override", None) is not None:
                    ph.radiation_due_override = None
        self.t += self.dt
        self.nstep += 1
        if self.tornado is not None and self.t >= self.tornado.next_follow:
            self.tornado.follow()
        if self.nstep % self.opt.watchdog_every == 0:
            if self.nests:
                self.nests.check()
            ok, why = self._healthy()
            if not ok:
                self._rollbacks = getattr(self, "_rollbacks", 0) + 1
                # each failed try halves the time step again (down to 1/6 of the normal step, the
                # shortest _adapt_dt takes), and always below the step that failed: the model is
                # deterministic, so a retry with the same step (a cap still in force from an earlier
                # rollback, or the floor) fails the same way
                cap = max(self.env.grid.dt / 6.0, min(self.env.grid.dt * 0.5 ** self._rollbacks, 0.5 * self.dt))
                if self._rollbacks > 4 or self._checkpoint is None or cap > self.dt - 1e-9:
                    raise RuntimeError(f"{why}; the run could not be recovered by rolling back "
                                       f"({self._rollbacks - 1} tries with shorter time steps)")
                self.note("watchdog", f"{why}; rolling back to t={self._checkpoint[0]:.0f} s with dt {cap:.2f} s")
                self.dt_cap = cap
                self.dt_cap_until = self._checkpoint[0] + 1800.0
                self.dt = self.dt_cap
                self._restore_checkpoint()
        self.step_seconds = time.perf_counter() - wall0

    def _large_scale(self) -> None:
        """Large-scale ascent this step: the environment lifted (largescale.py), every patch, nest and the
        reference column with it (and the reference's target, so the drift correction keeps it), the
        patches' edge tables renewed from it every largescale.REFRESH s."""
        dth, dq, dmu, dphp = self.ls.advance(self.dt)
        for p in self.patches + self.nest_patches():
            p.state.thp += _on_levels(dth, self.env, p.env)
            p.state.qv += _on_levels(dq, self.env, p.env)
            p.state.mup += dmu
            p.state.php += _on_levels(dphp, self.env, p.env)
            _clamp_vapour(p.state)
        if self._ref is not None:
            self._ref.thp += dth
            self._ref.qv += dq
            self._ref.mup += dmu
            self._ref.php += dphp
            self._ref_profile["thp"] = self._ref_profile["thp"] + dth[:, 0, 0]
            self._ref_profile["qv"] = self._ref_profile["qv"] + dq[:, 0, 0]
        if self.ls.due_refresh(self.t):
            for p in self.patches:
                p.renew_boundaries(self.t)

    def _nest_substep(self, p: Patch, t: float, h: float) -> None:
        """One nest step of h s from t, with the forcing a patch step gets (drag, storm starts, persistence),
        but no rim or air-mass relaxation: its edges are its parent's air."""
        p.set_dt(h)
        p.step()
        _debug_nan(p, "dynamics + physics")
        if getattr(self.env, "ocean", None) is not None:
            self._sea_surface(p.state, p, dt=h)
        elif self.env.own_drag:
            self._surface_drag(p.state, self._cd_map(p), dt=h, env=p.env)
        _debug_nan(p, "surface drag")
        forcing.apply_nudging(p, self.starts, t + h, h)
        _debug_nan(p, "nudging")
        if self.opt.persistence > 0:
            forcing.apply_keep_alive(p, self._cores.get(p.id), h,
                                     self.opt.persistence * self.opt.persistence_wmax,
                                     z_peak=self.opt.persistence_zpeak, z_half=self.opt.persistence_zhalf)
            if self.opt.persistence_root > 0:
                forcing.apply_root_lift(p, self._roots.get(p.id), h,
                                        self.opt.persistence * self.opt.persistence_root)
        _debug_nan(p, "persistence forcing")
        _clamp_vapour(p.state)

    def release_gpu(self) -> None:
        """Drop the run's GPU arrays (patch states, the reference column) once it has
        stopped; its time, log and the places of its patches stay readable."""
        for p in self.patches + self.nest_patches():
            p.free()
        self._ref = None
        self._cores, self._roots, self._memory = {}, {}, None

    def run_until(self, t_end: float, callback=None) -> None:
        while self.t < t_end:
            self.step()
            if callback is not None:
                callback(self)

    # -------------------------------------------------------- summary
    def status(self) -> dict:
        return {
            "t": self.t, "dt": self.dt, "nstep": self.nstep,
            # held: s it stays where it was moved by hand (-1: locked there), 0 following its storms
            "patches": [dict(p.to_json(), held=(-1 if getattr(p, "hand_until", -math.inf) == math.inf else
                                                max(0.0, getattr(p, "hand_until", -math.inf) - self.t)))
                        for p in self.patches],
            "cells": self.total_cells(), "max_cells": self.opt.max_cells, "memory_full": self.memory_full(),
            "fixed_cells": self.fixed, "fixed_cells_y": self.fixed_ny, "max_side": self.cap, "start_area": self.start_area(),
            "tracks": [tr.to_json(self.env.storm_motion) for tr in self.tracks.live],
            "nudgers": [n.to_json() for n in self.nudgers],
            "nests": self.nests.to_json() if self.nests else [],
            "tornado": self.tornado.to_json() if self.tornado is not None else None,
            "large_scale": self.ls.describe() if self.ls is not None else None,
            "grid_motion": list(self.env.motion),
            "follow": ({"every": self.opt.follow_interval, "next": self._next_follow,
                        "measured": list(self._measured) if self._measured else None}
                       if self.opt.follow_storms else None),
            "step_seconds": self.step_seconds,
            "speed": (self.dt / self.step_seconds) if self.step_seconds > 0 else 0.0,
        }
