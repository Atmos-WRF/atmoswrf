"""Patches: ArWen (gpuwm) domains placed on an infinite world grid.

The model world is an unbounded plane of dx-sized cells.  Storms live in
rectangular patches; each is an ordinary gpuwm ``DomainState`` with open
lateral boundaries whose cell (0, 0) sits at world cell (i0, j0).  Everything
outside the patches is the undisturbed environment.

A patch can be re-placed on a new rectangle (grow, shrink, shift) or merged
with another patch.  The new state starts as the pure environment and every
array ArWen carries from one step to the next is copied in where the old and
new rectangles overlap.  "Carried" is decided by ArWen's own restart manifest
(``gpuwm.io.restart.classify_state_attr`` / ``classify_scratch_slot``), so a
re-placed patch continues exactly as a restarted ArWen run would.  The old
arrays wait on the host (``HostCopy``) while the new state is built, so the
card never holds the old and the new patch at once.
"""
from __future__ import annotations

import dataclasses
import math
import itertools
import logging
import traceback

import numpy as np

log = logging.getLogger(__name__)

_ids = itertools.count(1)

CARRIED_ATTR_CLASSES = ("serialize", "checkpoint_only")
CARRIED_SLOT_CLASSES = ("serialize", "carry")


@dataclasses.dataclass(frozen=True)
class Rect:
    """World cell rectangle [i0, i0+nx) x [j0, j0+ny)."""
    i0: int
    j0: int
    nx: int
    ny: int

    @property
    def i1(self) -> int:
        return self.i0 + self.nx

    @property
    def j1(self) -> int:
        return self.j0 + self.ny

    @property
    def cells(self) -> int:
        return self.nx * self.ny

    def intersects(self, o: "Rect", gap: int = 0) -> bool:
        return not (self.i1 + gap <= o.i0 or o.i1 + gap <= self.i0
                    or self.j1 + gap <= o.j0 or o.j1 + gap <= self.j0)

    def contains(self, o: "Rect") -> bool:
        return (o.i0 >= self.i0 and o.j0 >= self.j0
                and o.i1 <= self.i1 and o.j1 <= self.j1)

    def union(self, o: "Rect") -> "Rect":
        i0, j0 = min(self.i0, o.i0), min(self.j0, o.j0)
        i1, j1 = max(self.i1, o.i1), max(self.j1, o.j1)
        return Rect(i0, j0, i1 - i0, j1 - j0)

    def expanded(self, m: int) -> "Rect":
        return Rect(self.i0 - m, self.j0 - m, self.nx + 2 * m, self.ny + 2 * m)

    def snapped(self, block: int, min_cells: int = 0) -> "Rect":
        """Outward to multiples of ``block`` (so shapes repeat and ArWen's
        compiled kernels are reused), at least ``min_cells`` a side."""
        i0 = (self.i0 // block) * block
        j0 = (self.j0 // block) * block
        i1 = -((-self.i1) // block) * block
        j1 = -((-self.j1) // block) * block
        nx, ny = i1 - i0, j1 - j0
        if nx < min_cells:
            grow = -(-(min_cells - nx) // (2 * block)) * block
            i0 -= grow
            nx += 2 * grow
        if ny < min_cells:
            grow = -(-(min_cells - ny) // (2 * block)) * block
            j0 -= grow
            ny += 2 * grow
        return Rect(i0, j0, nx, ny)

    def to_json(self, dx: float) -> dict:
        return {"x0": self.i0 * dx, "y0": self.j0 * dx,
                "x1": self.i1 * dx, "y1": self.j1 * dx,
                "nx": self.nx, "ny": self.ny}


def _carried_arrays(state):
    """(kind, name, array) for every array ArWen carries across steps."""
    import cupy as cp
    from gpuwm.io.restart import (RestartManifestError, classify_scratch_slot,
                                  classify_state_attr)

    for name, val in vars(state).items():
        if not isinstance(val, cp.ndarray) or name.startswith("_"):
            continue
        try:
            cls = classify_state_attr(name)
        except RestartManifestError:
            log.warning("unclassified ArWen state array %s: carried", name)
            cls = "serialize"
        if cls in CARRIED_ATTR_CLASSES:
            yield "attr", name, val
    for slot, val in getattr(state, "_scratch", {}).items():
        if not isinstance(val, cp.ndarray):
            continue
        try:
            cls = classify_scratch_slot(slot)
        except RestartManifestError:
            cls = "rebuild"
        if cls in CARRIED_SLOT_CLASSES:
            yield "slot", slot, val
    for path, val in _driver_arrays(state):
        yield "driver", path, val


def _driver_arrays(state):
    """(path, array) for the arrays of the state's gpuwm physics driver (land
    surface, surface layer, PBL and radiation fields and held tendencies)."""
    import cupy as cp
    drv = getattr(state, "physics", None)
    if drv is None:
        return
    for key, val in (getattr(drv, "fields", None) or {}).items():
        if isinstance(val, cp.ndarray):
            yield f"fields/{key}", val
    for name, val in vars(drv).items():
        if isinstance(val, cp.ndarray):
            yield f"attr/{name}", val
        elif isinstance(val, dict) and name != "fields":
            for k, v in val.items():
                if isinstance(v, cp.ndarray):
                    yield f"dict/{name}/{k}", v


def driver_lookup(state, path: str):
    """The array at ``path`` (see _driver_arrays) in the state's driver, or None."""
    import cupy as cp
    drv = getattr(state, "physics", None)
    if drv is None:
        return None
    kind, _, rest = path.partition("/")
    if kind == "fields":
        return (getattr(drv, "fields", None) or {}).get(rest)
    if kind == "attr":
        v = getattr(drv, rest, None)
        return v if isinstance(v, cp.ndarray) else None
    if kind == "dict":
        name, _, key = rest.partition("/")
        d = getattr(drv, name, None)
        return d.get(key) if isinstance(d, dict) else None
    return None


def _horizontal_kind(shape, nx, ny):
    """(x_stagger, y_stagger) if the last two axes are this patch's
    horizontal axes, else None."""
    if len(shape) < 2:
        return None
    a, b = shape[-2], shape[-1]
    if a not in (ny, ny + 1) or b not in (nx, nx + 1):
        return None
    return b == nx + 1, a == ny + 1


def _overlap(src: Rect, dst: Rect, xs: bool, ys: bool):
    """Index slices (src, dst) of the overlap; faces include the far edge."""
    i0 = max(src.i0, dst.i0)
    j0 = max(src.j0, dst.j0)
    i1 = min(src.i1, dst.i1) + (1 if xs else 0)
    j1 = min(src.j1, dst.j1) + (1 if ys else 0)
    if i1 <= i0 or j1 <= j0:
        return None
    s = (slice(j0 - src.j0, j1 - src.j0), slice(i0 - src.i0, i1 - src.i0))
    d = (slice(j0 - dst.j0, j1 - dst.j0), slice(i0 - dst.i0, i1 - dst.i0))
    return s, d


def _copy_carried(items, src: Rect, new_state, new_rect: Rect) -> int:
    """Copy carried arrays ``items`` ((kind, name, array) of a patch on ``src``,
    on the device or the host) into ``new_state`` (on ``new_rect``) where the
    rectangles overlap.  Returns the number of arrays copied."""
    import cupy as cp

    nxo, nyo = src.nx, src.ny
    nxn, nyn = new_rect.nx, new_rect.ny
    n = 0
    for kind, name, arr in items:
        hk = _horizontal_kind(arr.shape, nxo, nyo)
        if hk is None:
            continue
        xs, ys = hk
        if kind == "attr":
            dst = getattr(new_state, name, None)
            if dst is None:
                continue
        elif kind == "driver":
            dst = driver_lookup(new_state, name)
            if dst is None or dst.dtype != arr.dtype:
                continue
        else:
            shape = arr.shape[:-2] + (nyn + (1 if ys else 0), nxn + (1 if xs else 0))
            dst = new_state.scratch(shape, name, dtype=arr.dtype)
        if _horizontal_kind(dst.shape, nxn, nyn) != hk or dst.shape[:-2] != arr.shape[:-2]:
            log.warning("skipping %s: shapes %s -> %s", name, arr.shape, dst.shape)
            continue
        ov = _overlap(src, new_rect, xs, ys)
        if ov is None:
            continue
        (sj, si), (dj, di) = ov
        dst[..., dj, di] = cp.asarray(arr[..., sj, si])
        n += 1
    return n


def step_on_time(state, cfg) -> None:
    """One gpuwm step, a physics suite's radiation on MODEL TIME: due when this step would carry its last run
    past the radiation interval, and on a state's first step (a new patch, a re-placed one, a nest or tornado-zoom
    level: its driver is new).  gpuwm counts radiation in steps of the state's own dt, so a grid whose step
    changed -- the run's adaptive dt, a nest's extra sub-steps, a tornado zoom's levels -- let the land surface
    read a sky older than the interval + one step allows, and gpuwm stopped the run ("GLW (downward longwave at
    the surface) is stale").  An override the engine set itself (its first shorter step) is left to it."""
    from gpuwm.core.dycore import step
    ph = getattr(state, "physics", None)
    target = float(getattr(ph, "radt_seconds", 0.0) or 0.0) if ph is not None else 0.0
    mine = False
    if target > 0.0 and getattr(ph, "radiation_due_override", None) is None and hasattr(ph, "call_counts"):
        last = getattr(ph, "_arht_rad_last", None)
        now = float(getattr(state, "elapsed_seconds", 0.0) or 0.0)
        ph.radiation_due_override = last is None or now + float(cfg.dt) - last >= target
        mine = True
    n0 = ph.call_counts.get("radiation", 0) if ph is not None and hasattr(ph, "call_counts") else 0
    t0 = float(getattr(state, "elapsed_seconds", 0.0) or 0.0)
    try:
        step(state, cfg)
    finally:
        if mine:
            ph.radiation_due_override = None
    if ph is not None and hasattr(ph, "call_counts") and ph.call_counts.get("radiation", 0) != n0:
        ph._arht_rad_last = t0


def release_freed() -> None:
    """Give the card back the arrays nobody holds any more.  A released gpuwm DomainState
    can be reachable only through a reference cycle (gpuwm's own note), so its arrays
    are freed only once the collector has run."""
    import gc
    import cupy as cp
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


class HostCopy:
    """A patch's carried arrays on the host, with its rectangle and model time:
    a checkpoint, or what a re-placed patch carries over while its GPU state is
    freed and the new one is built."""

    def __init__(self, patch: "Patch"):
        import cupy as cp
        self.rect = patch.rect
        self.elapsed = patch.state.elapsed_seconds
        self.arrays = [(kind, name, cp.asnumpy(a)) for kind, name, a in _carried_arrays(patch.state)]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for _, _, a in self.arrays)

    def put(self, state, rect: Rect, whole: bool = False) -> None:
        """Into ``state`` (a new state on ``rect``) where the rectangles overlap, or with
        ``whole`` (a rollback, an undone re-placement: ``rect`` is this copy's own)
        every array of a matching shape."""
        import cupy as cp
        if whole:
            for kind, name, host in self.arrays:
                if kind == "slot":
                    state.scratch(host.shape, name, dtype=host.dtype)[...] = cp.asarray(host)
                    continue
                dst = getattr(state, name, None) if kind == "attr" else driver_lookup(state, name)
                if dst is not None and dst.shape == host.shape:
                    dst[...] = cp.asarray(host, dtype=dst.dtype)
        else:
            _copy_carried(self.arrays, self.rect, state, rect)
        state.elapsed_seconds = self.elapsed


class Patch:
    """One gpuwm domain on the world grid."""

    def __init__(self, env, rect: Rect, dt: float, state=None):
        self.id = next(_ids)
        self.env = env
        self.rect = rect
        self.cfg = env.run_config(rect.nx, rect.ny, dt)
        self.state = state if state is not None else env.new_state(self.cfg, rect)
        self.lbc_end = getattr(self.state, "atmo_lbc_end", math.inf)   # model time the edge tables run out
        self.am_maps = None                 # air-mass maps on the device (engine), and their time
        self.am_time = -math.inf
        self.quiet_seconds = 0.0            # how long the patch has had no activity
        self.created_at = 0.0
        self.oversize_seconds = 0.0         # how long it has been much bigger than needed
        self.moved_at = -math.inf           # model time it was last re-placed (a fixed-size domain settles)
        self.storm_center = None            # (x, y) world m of the activity box, last re-fit check
        self.storm_velocity = (0.0, 0.0)    # smoothed motion of that box (m/s)

    # --------------------------------------------------------------- time
    def set_dt(self, dt: float) -> None:
        if abs(self.cfg.dt - dt) > 1e-9:
            self.cfg = dataclasses.replace(self.cfg, dt=float(dt))

    def step(self) -> None:
        step_on_time(self.state, self.cfg)

    # ----------------------------------------------------------- geometry
    def world_x0(self) -> float:
        """Ground x (m) of the west edge of the patch now (the grid may be moving)."""
        return self.rect.i0 * self.env.grid.dx + self.env.offset[0]

    def world_y0(self) -> float:
        return self.rect.j0 * self.env.grid.dx + self.env.offset[1]

    # ------------------------------------------------------ re-placement
    def replace(self, new_rect: Rect) -> None:
        """Move this patch onto ``new_rect`` (grow / shrink / shift).  Its arrays wait
        on the host while the new state is built; if that fails, the patch is put back."""
        held = HostCopy(self)
        self.state = None
        release_freed()
        try:
            cfg = self.env.run_config(new_rect.nx, new_rect.ny, self.cfg.dt)
            new_state = self.env.new_state(cfg, new_rect)
            held.put(new_state, new_rect)
        except Exception as exc:
            new_state = None
            traceback.clear_frames(exc.__traceback__)       # the half-built state's arrays
            self.put_back(held)
            raise
        self.state, self.cfg, self.rect = new_state, cfg, new_rect
        self.lbc_end = getattr(new_state, "atmo_lbc_end", math.inf)
        self.am_maps, self.am_time = None, -math.inf

    def put_back(self, held: HostCopy) -> None:
        """This patch's state rebuilt from ``held`` (taken on its own rectangle)."""
        release_freed()                           # what the failed attempt had built
        self.state = self.env.new_state(self.cfg, held.rect)
        held.put(self.state, held.rect, whole=True)
        self.lbc_end = getattr(self.state, "atmo_lbc_end", math.inf)

    def renew_boundaries(self, t: float) -> None:
        """Fresh lateral boundary tables from model time t (moving air masses)."""
        self.lbc_end = self.env.attach_boundaries(self.state, self.cfg, self.rect, t0=t)

    def refresh_airmass(self, t: float) -> None:
        """Air-mass maps (boundaries.MAP_NAMES) of this patch at time t, on the device."""
        import cupy as cp
        am = self.env.airmass
        if self.env.terrain is not None:
            # each column over its own ground: 3-D rim / keep targets (forcing._apply_synoptic)
            self.am_maps = self.env.terrain_targets(self.rect, t)
            self.am_time = t
            return
        if not am:
            self.am_maps = None
            return
        if getattr(am, "synoptic", False):
            # the simulated atmosphere: full 3-D targets on a ~10 km grid (forcing.apply_airmass)
            self.am_maps = am.patch_targets(self.rect, t)
            self.am_time = t
            return
        X, Y = self.env.ground_xy(self.rect, t=t)
        self.am_maps = cp.asarray(am.stack(am.maps(X, Y, t)))
        self.am_time = t

    @classmethod
    def merged(cls, env, patches: list["Patch"], rect: Rect, dt: float) -> "Patch":
        """A new patch on ``rect`` holding the contents of ``patches`` (where they
        overlap, the later one's).  Their arrays wait on the host while it is built;
        if that fails, they are put back as they were."""
        held = [HostCopy(q) for q in patches]
        for q in patches:
            q.state = None
        release_freed()
        try:
            p = cls(env, rect, dt)
            for h in held:
                h.put(p.state, rect)
        except Exception as exc:
            p = None
            traceback.clear_frames(exc.__traceback__)
            for q, h in zip(patches, held):
                q.put_back(h)
            raise
        return p

    def free(self) -> None:
        self.state = None

    def to_json(self) -> dict:
        d = self.rect.to_json(self.env.grid.dx)
        ox, oy = self.env.offset
        d.update(x0=d["x0"] + ox, x1=d["x1"] + ox, y0=d["y0"] + oy, y1=d["y1"] + oy, id=self.id)
        return d
