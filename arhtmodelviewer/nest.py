"""Storm-following refinement nests: a finer grid (ratio 2-4 x the run's) over each of the strongest storms.

A nest is an ordinary patch on a grid ``ratio`` times finer, lined up with its parent patch's cells (each
parent cell is ratio x ratio nest cells).  Every parent step (WRF's nesting, two-way):

1. FORCE: the parent's air along the nest's edges, in WRF's coupled boundary units
   (gpuwm.ingest.lateral_bc), interpolated onto the nest's edge strips: the nest's lateral boundary tables
   for the parent step, value at its start and tendency to its end (specified + relaxation zone);
2. the nest takes ``ratio`` steps of dt / ratio;
3. FEEDBACK: the nest's air, block-averaged (mass points) or face-averaged (u, v), replaces the parent's
   inside the nest (all but a margin along the nest's edges), and the parent's pressure is re-diagnosed.

So the parent carries the storm as the nest resolves it (its tracks, its cold pool, the synoptic weather
it feeds), and the radar samples the nest where there is one (radar.build_fields' holes).

A nest is placed on a storm's updraft (Engine tracks), keeps it near its centre (re-placed when the storm
has moved REPLACE_FRAC of its side: the overlap carried over, the rest interpolated from the parent), and
is dropped when its storm ends or its parent patch no longer holds it.  A rollback drops every nest (they
come back from the parent at once).

Nests stack: an inner level (Nests with ``parent``) sits inside the outer nests, on each one's low-level
mesocyclone (the 0-3 km updraft helicity maximum, UH03), stepped ``ratio`` times inside every outer sub-step
and fed back into its outer nest before that one feeds the run's grid -- 1 km everywhere, 500 m on the
storm, 250 m on the mesocyclone.

Over terrain a nest's ground is its parent grid's, interpolated (GridTerrain: the terrain data is ~1 km, so a
finer grid has nothing more to resolve, and the nest's columns then match the parent's they feed back into),
and what passes between the two grids -- the fill, the feedback -- is the full theta, geopotential and dry
mass (BASE_OF): each grid's perturbations are about its own terrain base state.
"""
from __future__ import annotations

import dataclasses
import inspect
import logging
import math

import numpy as np

from .patch import Patch, Rect, _carried_arrays, _horizontal_kind, driver_lookup, release_freed

log = logging.getLogger(__name__)

WIDTH = 5                    # spec_bdy_width of the nest's tables (spec_zone 1 + relax_zone 4)
REPLACE_FRAC = 0.15          # re-placed when where its storm is heading is this far (x its side) off its centre
PLACE_EVERY = 60.0           # s of model time between placement checks
# a nest goes where its storm will be LEAD s later (the track's motion), not where it is: re-placed only every few
# minutes, a nest centred on the updraft had it on its downstream side most of the time, its forward flank and
# anvil cut by the edge, and its inflow, RFD and mesocyclone (behind and right of the updraft) crowded by it
LEAD = 240.0
ROT_UH = 150.0               # m2/s2 of 0-3 km UH: a storm rotating this hard ranks 2 x its updraft (Rotation.score)
SWAP_RANK = 1.6              # x: a storm this much higher ranked than an auto nest's (placed SWAP_AFTER ago) takes it
SWAP_AFTER = 900.0
HAND_HOLD = 600.0            # s a nest moved by hand stays where it was put before following its storm again
HAND_MAX = 3                 # nests added by hand (on top of the automatic ones)
MIN_AGE = 600.0              # s a storm has been followed before it gets a nest
INNER_MIN_AGE = 300.0        # s an outer nest has run (its own spin-up) before an inner one goes in it
INNER_UH03 = 40.0            # m2/s2 of 0-3 km updraft helicity: a low-level mesocyclone worth an inner nest
# A nest's sub-step is its parent's over the ratio, but its updrafts are stronger (a 250 m nest's 90-160 m/s
# core over the parent's 60) on the same levels: when |w| dt / dz would pass CFL_W the parent step is cut into
# more sub-steps (a 250 m nest of May 3, 1999 blew up at 165 m/s without this).
CFL_W = 0.9
# The specified boundary tables carry u, v, theta, phi, mu and qv, not w: gpuwm gives the edge row its inner
# neighbour's w, so an updraft reaching a nest's edge kept feeding itself there (83 m/s held in a 250 m nest's
# first row for minutes, then 165-195 m/s).  w is relaxed toward the parent's, interpolated, across the
# WIDTH-cell frame (WRF nests force w at their edges too), after every sub-step.
MEM_WAIT = 120.0             # s a level places no nest after one did not fit in the GPU's memory
LOST_AFTER = 600.0          # s a nest keeps its place once its storm is no longer followed
REPLACE_PIN = 0.12           # a pinned nest is re-placed when its pin is this far (x its side) off its centre
PARENT_MARGIN = 4            # parent cells between a nest and its parent patch's edges (rim, stencils)
# the prognostic fields fed back to the parent (diagnostics such as p, al are re-diagnosed from them)
FEEDBACK = ("u", "v", "w", "thp", "php", "mup", "tke", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
            "nc", "nr", "ni", "ns", "ng", "nh")
# the hydrometeors a nest's edge frame takes from its parent (_parent_hydro): the boundary tables carry only qv
HYDRO = ("qc", "qr", "qi", "qs", "qg", "qh", "nc", "nr", "ni", "ns", "ng", "nh")
BASE_OF = {"thp": "thb", "php": "phb", "mup": "mub2d"}   # perturbation -> its base state (terrain: per column)


class GridTerrain:
    """A parent grid's terrain seen from a finer grid: the parent's ground (terrain.height at its cell centres),
    bilinear between them.  Every other attribute is the terrain's."""

    def __init__(self, terrain, env):
        self._t, self._env = terrain, env

    def __getattr__(self, name):
        return getattr(self._t, name)

    def height(self, X, Y):
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        dx = self._env.grid.dx
        ox, oy = self._env.offset_at(self._env.clock)
        fi, fj = (X - ox) / dx - 0.5, (Y - oy) / dx - 0.5
        i0, j0 = np.floor(fi), np.floor(fj)
        wi, wj = fi - i0, fj - j0
        h = lambda i, j: self._t.height((i + 0.5) * dx + ox, (j + 0.5) * dx + oy)
        return ((1 - wj) * ((1 - wi) * h(i0, j0) + wi * h(i0 + 1, j0))
                + wj * ((1 - wi) * h(i0, j0 + 1) + wi * h(i0 + 1, j0 + 1)))


def _finite(state) -> bool:
    """Are a nest's winds, theta and pressure all finite (one reduction each, on the device)?"""
    import cupy as cp
    for name in ("u", "v", "w", "thp", "php", "mup"):
        a = getattr(state, name, None)
        if a is not None and not bool(cp.isfinite(a).all()):
            return False
    return True


def _terrain_base(state) -> bool:
    """Is the state's base state per column (terrain)?"""
    thb = getattr(state, "thb", None)
    return thb is not None and thb.ndim == 3 and thb.shape[-1] > 1


class FineEnv:
    """The run's Environment on a grid ``ratio`` times finer: every method of the environment, run against
    this grid; every other attribute (sounding, physics, motion, air masses, clock, towns...) read live from
    the run's environment, so a change there (the grid motion, the air masses) is the nest's too."""

    def __init__(self, parent, ratio: int):
        d = self.__dict__
        d["_parent"] = parent
        d["ratio"] = int(ratio)
        g = parent.grid
        # the run's own grids of 750 m and finer take 6 acoustic sub-steps (environment.GRID_PRESETS)
        sound = max(g.sound, 6) if g.dx / ratio <= 750.0 else g.sound
        d["grid"] = dataclasses.replace(g, dx=g.dx / ratio, dt=g.dt / ratio, sound=sound)
        d["_dev_cache"] = parent.__dict__.setdefault("_dev_cache", {})   # column arrays: no dx in them
        if getattr(parent, "terrain", None) is not None:
            d["terrain"] = GridTerrain(parent.terrain, parent)

    def __getattr__(self, name):
        return getattr(self.__dict__["_parent"], name)

    def _boundary_snapshot(self):
        return self._parent._boundary_snapshot()

    def lateral_boundaries(self, nx: int, ny: int, width: int = 5, rect=None, t0=None):
        # what a new nest starts with until its first FORCE: the environment's (constant) tables
        from .environment import Environment
        return Environment.lateral_boundaries(self, nx, ny, width, None, t0)


class VFineEnv(FineEnv):
    """A FineEnv with its own vertical levels: the tornado zoom's finest nest, CM1-style -- layers from ``dz_bot`` m
    at the ground (about a fifth of its spacing, so the lowest few hundred metres are close to isotropic) growing
    by at most 8 % a layer to the run's deepest, the same model top; and LES mixing (no PBL scheme, the 3-D TKE
    closure with one mixing length (dx dy dz)^(1/3)).  Its levels' arrays are its own (built by an Environment on
    them); the profiles the run changes as it goes (its spun-up wind and turbulence, the large-scale ascent) are
    read live from the run and put on these levels."""

    # the run's level profiles a VFineEnv reads on its own levels (interpolated in height)
    LEVEL_PROFILES = ("u_ground", "v_ground", "tke_profile", "qv_env_h", "ls_dth", "ls_dqv")

    def __init__(self, parent, ratio: int, nz: int, dz_bot: float):
        super().__init__(parent, ratio)
        from .environment import Environment
        d = self.__dict__
        g = dataclasses.replace(d["grid"], nz=int(nz), dz_bot=float(dz_bot))
        phys = dataclasses.replace(parent.physics, mix_isotropic=1, les=True)
        v = Environment(parent.sounding, g, phys, motion=parent.motion)
        d.update(grid=v.grid, coord=v.coord, base=v.base, z_w=v.z_w, z_h=v.z_h, physics=v.physics, suite=v.suite,
                 _dev_cache={}, _bsnap=None, _php_top_env=None, env_column=None)

    def __getattr__(self, name):
        parent = self.__dict__["_parent"]
        if name in VFineEnv.LEVEL_PROFILES:
            a = getattr(parent, name)
            arr = None if a is None else np.asarray(a, dtype=np.float64)
            if arr is None or arr.ndim != 1 or len(arr) != len(parent.z_h):
                return a
            return np.interp(self.z_h, np.asarray(parent.z_h, np.float64), arr)
        if name in ("u_env", "v_env"):
            m = parent.motion[0 if name == "u_env" else 1]
            return getattr(self, "u_ground" if name == "u_env" else "v_ground") - m
        if name == "cd":                         # its own lowest layer is thinner: a larger coefficient
            return float(self.drag_coefficient(self.physics.z0)) if parent.cd else 0.0
        return getattr(parent, name)

    def _boundary_snapshot(self):
        from .environment import Environment
        return Environment._boundary_snapshot(self)


def _interp_matrix(src, dst) -> np.ndarray:
    """(len(dst), len(src)) weights of linear interpolation from points ``src`` to ``dst`` (constant beyond)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    order = np.argsort(src)
    s = src[order]
    W = np.zeros((len(dst), len(src)))
    for i, x in enumerate(dst):
        k = int(np.searchsorted(s, x))
        if k <= 0:
            W[i, order[0]] = 1.0
        elif k >= len(s):
            W[i, order[-1]] = 1.0
        else:
            a = (x - s[k - 1]) / (s[k] - s[k - 1])
            W[i, order[k - 1]], W[i, order[k]] = 1.0 - a, a
    return W


def _layer_matrix(src_w, dst_w) -> np.ndarray:
    """(nd, ns) weights of layer means: each destination layer (between its w levels) the mean of the source
    layers over it, each by how much of it it spans (the levels as eta: a mass-weighted mean)."""
    src_w, dst_w = np.asarray(src_w, np.float64), np.asarray(dst_w, np.float64)
    W = np.zeros((len(dst_w) - 1, len(src_w) - 1))
    for k in range(len(dst_w) - 1):
        lo, hi = sorted((dst_w[k], dst_w[k + 1]))
        for j in range(len(src_w) - 1):
            a, b = sorted((src_w[j], src_w[j + 1]))
            W[k, j] = max(0.0, min(hi, b) - max(lo, a))
        s = W[k].sum()
        if s > 0:
            W[k] /= s
    return W


class VMaps:
    """Between a parent's levels and a VFineEnv nest's, in eta (the same surface pressure and top, so the same
    eta is the same place in the column): ``up`` interpolates parent fields onto the nest's levels, ``down``
    gives the parent's levels the nest's layer means (w levels: interpolated)."""

    def __init__(self, parent_coord, nest_coord):
        import cupy as cp
        pu, pw = np.asarray(parent_coord.znu), np.asarray(parent_coord.znw)
        nu, nw = np.asarray(nest_coord.znu), np.asarray(nest_coord.znw)
        self.nzp, self.nzn = len(pu), len(nu)
        f = lambda W: cp.asarray(W, dtype=cp.float32)
        self.up_h, self.up_w = f(_interp_matrix(pu, nu)), f(_interp_matrix(pw, nw))
        self.down_h, self.down_w = f(_layer_matrix(nw, pw)), f(_interp_matrix(nw, pw))

    @staticmethod
    def _apply(W, a):
        n = a.shape[0]
        out = W @ a.reshape(n, -1).astype(W.dtype, copy=False)
        return out.reshape((W.shape[0],) + a.shape[1:])

    def up(self, a):
        """A parent-levelled array (nzp or nzp + 1 leading) on the nest's levels; anything else as it is."""
        if a.ndim >= 2 and a.shape[0] == self.nzp:
            return self._apply(self.up_h, a)
        if a.ndim >= 2 and a.shape[0] == self.nzp + 1:
            return self._apply(self.up_w, a)
        return a

    def down(self, a):
        if a.ndim >= 2 and a.shape[0] == self.nzn:
            return self._apply(self.down_h, a)
        if a.ndim >= 2 and a.shape[0] == self.nzn + 1:
            return self._apply(self.down_w, a)
        return a

    def maps(self, n_src: int, n_dst: int, up: bool) -> bool:
        """Do arrays of n_src levels go to n_dst levels through this (up: parent -> nest)?"""
        a, b = (self.nzp, self.nzn) if up else (self.nzn, self.nzp)
        return (n_src, n_dst) in ((a, b), (a + 1, b + 1))


def _base_part(st, name, j0, j1, i0, i1):
    """The base-state part of perturbation ``name`` (BASE_OF) of a state over cells [j0:j1, i0:i1], broadcastable
    to the field there (flat: a column; terrain: per column), or None."""
    bn = BASE_OF.get(name)
    b = getattr(st, bn, None) if bn else None
    if b is None:
        return None
    if b.ndim == 1:
        return b[:, None, None]
    if b.ndim == 2:
        return b[j0:j1, i0:i1] if b.shape[-1] > 1 else b
    return b[..., j0:j1, i0:i1] if b.shape[-1] > 1 else b


def _bind_methods():
    from .environment import Environment
    skip = {"__init__", "__getattr__", "_boundary_snapshot", "lateral_boundaries"}
    for name, member in vars(Environment).items():
        if name in skip or (name.startswith("__") and name.endswith("__")):
            continue
        if inspect.isfunction(member) or isinstance(member, (property, staticmethod, classmethod)):
            setattr(FineEnv, name, member)


# ------------------------------------------------------------------ interpolation
def _axis(pos: np.ndarray, n: int):
    import cupy as cp
    p = np.clip(pos, 0.0, n - 1.0)
    i0 = np.minimum(np.floor(p).astype(np.int64), max(n - 2, 0))
    w = p - i0
    return cp.asarray(i0), cp.asarray(np.minimum(i0 + 1, n - 1)), cp.asarray(w, dtype=cp.float32)


def _interp(a, jpos: np.ndarray, ipos: np.ndarray):
    """``a`` (..., ny, nx) at fractional indices jpos x ipos (bilinear; nearest for integer arrays)."""
    import cupy as cp
    j0, j1, wj = _axis(jpos, a.shape[-2])
    i0, i1, wi = _axis(ipos, a.shape[-1])
    if not cp.issubdtype(a.dtype, cp.floating):
        jj = cp.where(wj >= 0.5, j1, j0)
        ii = cp.where(wi >= 0.5, i1, i0)
        return a[..., jj[:, None], ii[None, :]]
    lo, hi = a[..., j0, :], a[..., j1, :]
    rows = lo + (hi - lo) * wj[:, None]
    l2, h2 = rows[..., i0], rows[..., i1]
    return (l2 + (h2 - l2) * wi).astype(a.dtype)


class Nest:
    """One nest: its patch (a FineEnv grid) and where it sits in its parent."""

    def __init__(self, patch: Patch, ratio: int, track_id: int | None):
        self.patch = patch
        self.ratio = ratio
        self.track_id = track_id
        self.sides = None              # the edge strips at the end of the last parent step
        self.lost_since = None         # model time its storm stopped being followed
        self.placed_at = 0.0
        self.pin = None                # Nests.pins key: a nest kept on a point given from outside (tornado.py)
        self.hand = False              # added by hand (Nests.add_on): not counted in max_nests, never swapped
        self.hold_until = -math.inf    # moved by hand (Nests.move): kept where it was put until then
        self.rank = 0.0                # its storm's Rotation.score at the last placement check

    @property
    def id(self) -> int:
        return self.patch.id

    @property
    def rect(self) -> Rect:
        return self.patch.rect

    def parent_rect(self) -> Rect:
        """The parent cells it covers."""
        r, q = self.ratio, self.patch.rect
        return Rect(q.i0 // r, q.j0 // r, q.nx // r, q.ny // r)

    def to_json(self) -> dict:
        d = self.patch.to_json()
        d.update(ratio=self.ratio, dx=self.patch.env.grid.dx, track=self.track_id,
                 level=getattr(self.patch, "nest_level", 1), pinned=self.pin is not None, hand=self.hand)
        return d


class Nests:
    """The engine's nests: placement, the per-step FORCE / sub-steps / FEEDBACK, radar holes."""

    def __init__(self, engine, ratio: int = 3, size_km: float = 40.0, max_nests: int = 2, parent: "Nests | None" = None,
                 vertical: tuple | None = None):
        _bind_methods()
        self.engine = engine
        self.parent = parent                  # None: nests in the run's patches; else inside that level's nests
        self.level = 1 if parent is None else parent.level + 1
        self.ratio = int(ratio)
        self.size_km = float(size_km)
        self.max_nests = int(max_nests)
        self.base = engine.env if parent is None else parent.env      # the grid this level nests in
        # vertical = (nz, dz_bot): its own levels (VFineEnv), every exchange with the parent through VMaps
        self.env = FineEnv(self.base, self.ratio) if not vertical else VFineEnv(self.base, self.ratio, *vertical)
        self.vmaps = VMaps(self.base.coord, self.env.coord) if vertical else None
        self.items: list[Nest] = []
        self.inner: Nests | None = None       # the next level in, if any
        self._next_place = 0.0
        self._mem_wait = -math.inf            # model time: no new nest on this level before it (no GPU room)
        self.mem_note = None                  # why its last nest was refused (_room_for), for the page
        # points a nest of this level is kept centred on (key -> ground x, y m), besides (and not counted in)
        # the max_nests placed on storms: the tornado zoom's (tornado.py)
        self.pins: dict = {}

    def _parents(self) -> list:
        return self.engine.patches if self.parent is None else [n.patch for n in self.parent.items]

    def all_items(self) -> list:
        """This level's nests and every level's inside them."""
        return self.items + (self.inner.all_items() if self.inner else [])

    # ---------------------------------------------------------------- geometry
    def _parent_of(self, pr: Rect):
        """The patch that holds parent cells ``pr`` with PARENT_MARGIN to spare, or None."""
        want = pr.expanded(PARENT_MARGIN)
        for p in self._parents():
            if p.rect.contains(want):
                return p
        return None

    def _side_cells(self) -> int:
        """The nest's side in parent cells (a multiple of 4, so its own side is too)."""
        n = int(round(self.size_km * 1000.0 / self.base.grid.dx))
        return max(8, -(-n // 4) * 4)

    def _parent_rect_at(self, x: float, y: float) -> Rect:
        """The parent cells of a nest centred on ground point (x, y)."""
        dx = self.base.grid.dx
        ox, oy = self.engine.env.offset
        n = self._side_cells()
        return Rect(int(round((x - ox) / dx - n / 2)), int(round((y - oy) / dx - n / 2)), n, n)

    def _fit(self, pr: Rect) -> tuple[Rect, Patch] | None:
        """``pr`` moved (not resized) to fit inside the patch it most overlaps, with PARENT_MARGIN."""
        best = None
        for p in self._parents():
            q = p.rect
            if not q.intersects(pr):
                continue
            room = q.expanded(-PARENT_MARGIN)
            if room.nx < pr.nx or room.ny < pr.ny:
                continue
            i0 = min(max(pr.i0, room.i0), room.i1 - pr.nx)
            j0 = min(max(pr.j0, room.j0), room.j1 - pr.ny)
            cand = Rect(i0, j0, pr.nx, pr.ny)
            shift = abs(i0 - pr.i0) + abs(j0 - pr.j0)
            if best is None or shift < best[0]:
                best = (shift, cand, p)
        return None if best is None else (best[1], best[2])

    def holes(self) -> list:
        """What every level's nests cover, for the radar (radar.build_fields): ground (x0, y0, x1, y1), level,
        the nest's patch id, and its crossfade -- coarser grid only within ``d0`` m of the nest's edge (its
        boundary zone, driven by the parent), then over ``ramp`` m (3 parent cells) the nest taking over."""
        out = []
        for n in self.all_items():
            d = n.patch.to_json()
            dxn = n.patch.env.grid.dx
            out.append((d["x0"], d["y0"], d["x1"], d["y1"], getattr(n.patch, "nest_level", 1), n.patch.id,
                        (WIDTH + 1) * dxn, 3.0 * dxn * n.ratio))
        return out

    # ---------------------------------------------------------------- building
    def _positions(self, nest_rect: Rect, win: Rect, parent: Patch, xs: bool, ys: bool):
        """Fractional indices, in the parent window ``win`` (the parent's local cells), of a nest array's points."""
        r = self.ratio
        oi, oj = parent.rect.i0 + win.i0, parent.rect.j0 + win.j0      # the window's first world parent cell
        i = np.arange(nest_rect.nx + (1 if xs else 0), dtype=np.float64)
        j = np.arange(nest_rect.ny + (1 if ys else 0), dtype=np.float64)
        ipos = (nest_rect.i0 + i) / r - oi if xs else (nest_rect.i0 + i + 0.5) / r - 0.5 - oi
        jpos = (nest_rect.j0 + j) / r - oj if ys else (nest_rect.j0 + j + 0.5) / r - 0.5 - oj
        return jpos, ipos

    def _window(self, parent: Patch, pr: Rect, halo: int = 2) -> Rect:
        """``pr`` plus a halo, in the parent's local cells."""
        q = parent.rect
        return Rect(pr.i0 - q.i0 - halo, pr.j0 - q.j0 - halo, pr.nx + 2 * halo, pr.ny + 2 * halo)

    def _fill_from_parent(self, nest_patch: Patch, parent: Patch, pr: Rect) -> int:
        """Every array the nest carries, interpolated from the parent's (same name, same stagger)."""
        import cupy as cp
        win = self._window(parent, pr)
        pst, nst = parent.state, nest_patch.state
        pnx, pny = parent.rect.nx, parent.rect.ny
        cnx, cny = nest_patch.rect.nx, nest_patch.rect.ny
        src = {(k, n): a for k, n, a in _carried_arrays(pst)}
        terrain = _terrain_base(pst) and _terrain_base(nst)
        vm = self.vmaps
        done = 0
        for kind, name, dst in list(_carried_arrays(nst)):
            a = src.get((kind, name))
            if a is None:
                continue
            hk, pk = _horizontal_kind(dst.shape, cnx, cny), _horizontal_kind(a.shape, pnx, pny)
            if hk is None or hk != pk:
                continue
            levels = vm is not None and a.ndim == 3 and vm.maps(a.shape[0], dst.shape[0], up=True)
            if a.shape[:-2] != dst.shape[:-2] and not levels:
                continue
            xs, ys = hk
            sub = a[..., win.j0:win.j1 + (1 if ys else 0), win.i0:win.i1 + (1 if xs else 0)]
            jpos, ipos = self._positions(nest_patch.rect, win, parent, xs, ys)
            # the full value, less the nest's own base: over terrain (each grid's base per column) and between
            # levels (each grid's base on its own levels; the dry mass's is the same for both, over flat ground)
            full = kind == "attr" and name in BASE_OF and (terrain or (levels and name != "mup"))
            if full:
                sub = sub + _base_part(pst, name, win.j0, win.j1, win.i0, win.i1)
            v = _interp(sub, jpos, ipos)
            if levels:
                v = vm.up(v)
            if full:
                v = v - _base_part(nst, name, 0, cny, 0, cnx)
            dst[...] = v.astype(dst.dtype)
            done += 1
        cp.cuda.Stream.null.synchronize()
        return done

    def _new_patch(self, nest_rect: Rect, parent: Patch, pr: Rect, old: Nest | None) -> Patch:
        from gpuwm.core.diagnostics import update_diagnostics
        from .patch import _copy_carried
        eng = self.engine
        p = Patch(self.env, nest_rect, parent.cfg.dt / self.ratio)
        p.nest_level = self.level
        self._fill_from_parent(p, parent, pr)
        if old is not None:                               # what the nest already resolved stays
            _copy_carried(list(_carried_arrays(old.patch.state)), old.patch.rect, p.state, nest_rect)
        update_diagnostics(p.state)
        p.state.elapsed_seconds = eng.t
        p.created_at = eng.t
        from .engine import _debug_nan
        _debug_nan(p, "placement (filled from its parent)")
        return p

    def place(self, pr: Rect, track_id, old: Nest | None = None, why: str = "", pin=None) -> Nest | None:
        fit = self._fit(pr)
        if fit is None:
            return None
        pr, parent = fit
        r = self.ratio
        nest_rect = Rect(pr.i0 * r, pr.j0 * r, pr.nx * r, pr.ny * r)
        if old is not None and old.patch.rect == nest_rect:
            return old
        if not self._room_for(nest_rect, old):
            return None
        try:
            p = self._new_patch(nest_rect, parent, pr, old)
        except Exception as exc:                          # out of memory, a refused shape: no nest
            log.exception("nest not placed")
            self.engine.note("nest", f"could not place a nest ({exc})")
            return None
        if old is not None:
            p.id = old.patch.id
            old.patch.free()
            n = old
            n.patch, n.sides = p, None
        else:
            n = Nest(p, r, track_id)
            n.pin = pin
            self.items.append(n)
            on = (f"storm {track_id}" if self.parent is None else f"the mesocyclone in nest {track_id}") if pin is None \
                else "the tornado zoom"
            self.engine.note("nest", f"nest {p.id}: {self.env.grid.label} grid, {pr.nx * self.base.grid.dx / 1000:.0f} km "
                                     f"a side, on {on}{why}")
        n.placed_at = self.engine.t
        return n

    def _room_for(self, nest_rect: Rect, old: Nest | None = None) -> bool:
        """Is there GPU memory for a nest on ``nest_rect`` (its cells x its levels x what a cell takes,
        Engine.nest_cell_bytes)?  The patches' budget does not count the nests, and a nest that does not fit
        stopped the whole run with cudaErrorMemoryAllocation.  Moving ``old``: its memory comes back right after,
        so the two only have to fit side by side for a moment (in what the card has free, not the run's share of
        it -- a 10 m zoom on an 8 GB card was dropped the first time it followed its vortex).  Refused: none on
        this level for MEM_WAIT."""
        eng = self.engine
        if eng.t < self._mem_wait:
            return False
        per = len(self.env.z_h) * eng.nest_cell_bytes()
        need = nest_rect.cells * per
        room = eng.gpu_room(moving=old.patch.rect.cells * per if old is not None else 0.0)
        if need <= room:
            self.mem_note = None
            return True
        self._mem_wait = eng.t + MEM_WAIT
        self.mem_note = (f"waiting for GPU memory: its {self.env.grid.dx:.3g} m level needs {need / 2**30:.1f} GB, "
                         f"{room / 2**30:.1f} GB free -- drag a smaller box")
        eng.note("memory", f"no {self.env.grid.label} nest for now: it would take {need / 2**30:.1f} GB of GPU memory "
                           f"and {room / 2**30:.1f} GB is left; trying again in {MEM_WAIT / 60:.0f} min (a smaller "
                           "nest, tornado-zoom box or patch makes room)")
        return False

    def drop(self, n: Nest, why: str) -> None:
        if self.inner:                                    # the nests inside it go with it
            for m in [m for m in self.inner.items if n.patch.rect.intersects(m.parent_rect())]:
                self.inner.drop(m, why)
        n.patch.free()
        self.items.remove(n)
        release_freed()
        self.engine.note("nest", f"nest {n.id} removed ({why})")

    def clear(self, why: str) -> None:
        if self.inner:
            self.inner.clear(why)
        for n in list(self.items):
            self.drop(n, why)

    # ---------------------------------------------------------------- placement
    def manage(self) -> None:
        """Every PLACE_EVERY: nests on the strongest followed storms, each kept on its storm (an inner level:
        on its outer nests' low-level mesocyclones, _manage_inner).  Then the next level in."""
        if self.parent is None:
            self._manage_outer()
        else:
            self._manage_inner()
        self._pins_here()
        if self.inner:
            self.inner.manage()

    def _auto(self) -> list:
        """The nests placed on storms (not on pins, not added by hand): the ones max_nests counts."""
        return [n for n in self.items if n.pin is None and not n.hand]

    def _followers(self) -> list:
        """Every nest kept on a storm (automatic or added by hand)."""
        return [n for n in self.items if n.pin is None]

    # ---------------------------------------------------------------- by hand
    def find(self, nid: int):
        """(level, nest) of nest ``nid`` on this level or any inside it, or None."""
        for n in self.items:
            if n.id == nid:
                return self, n
        return self.inner.find(nid) if self.inner else None

    def _aim(self, tr) -> tuple[float, float]:
        """Where a nest on track ``tr`` is centred: LEAD s ahead of its updraft."""
        _, x, y, _ = tr.last
        vx, vy = tr.velocity(getattr(self.engine.env, "storm_motion", None) or (0.0, 0.0))
        return x + vx * LEAD, y + vy * LEAD

    def move(self, nid: int, x: float, y: float) -> bool:
        """Nest ``nid`` moved by hand to ground point (x, y).  An outer nest dropped on a storm takes it (it
        follows that one from now on, from where it is heading); elsewhere, or an inner nest, it is kept where
        it was put for HAND_HOLD first."""
        got = self.find(nid)
        if got is None:
            return False
        lv, n = got
        eng = self.engine
        tr = None
        if lv.parent is None and n.pin is None:
            tr = eng.tracks.nearest(x, y, eng.t, within=15000.0)
            if tr is not None:
                n.track_id, n.lost_since = tr.id, None
                x, y = lv._aim(tr)
        moved = lv.place(lv._parent_rect_at(x, y), n.track_id, old=n)
        if moved is None:
            eng.note("nest", f"nest {nid} cannot go there: it has to sit inside a patch (or its outer nest)")
            return False
        n.hold_until = -math.inf if tr is not None else eng.t + HAND_HOLD
        eng.note("nest", f"nest {nid} moved by hand to ({x / 1000:.0f}, {y / 1000:.0f}) km"
                         + (f", now on storm {n.track_id}" if lv.parent is None and n.track_id is not None else ""))
        return True

    def add_on(self, x: float, y: float) -> bool:
        """A nest added by hand on the storm nearest ground point (x, y) (or on the point, without one near)."""
        eng = self.engine
        if sum(1 for n in self.items if n.hand) >= HAND_MAX:
            eng.note("nest", f"already {HAND_MAX} nests added by hand: move one, or let one end, first")
            return False
        tr = eng.tracks.nearest(x, y, eng.t, within=15000.0)
        if tr is not None and any(n.track_id == tr.id for n in self._followers()):
            eng.note("nest", f"storm {tr.id} has a nest already")
            return False
        ax, ay = self._aim(tr) if tr is not None else (x, y)
        n = self.place(self._parent_rect_at(ax, ay), tr.id if tr is not None else None, why=" (added by hand)")
        if n is None:
            eng.note("nest", "no nest there: it has to sit inside a patch, with room for its whole box")
            return False
        n.hand = True
        if tr is None:
            n.hold_until = eng.t + HAND_HOLD
        return True

    def manage_pins(self) -> None:
        """Pinned nests onto their pins now, this level and every level inside it (tornado.py, between manages)."""
        self._pins_here()
        if self.inner:
            self.inner.manage_pins()

    def _pins_here(self) -> None:
        """A nest centred on every pin of this level (re-placed when the pin is REPLACE_PIN of its side off);
        a pinned nest whose pin is gone is dropped."""
        for n in [n for n in self.items if n.pin is not None and n.pin not in self.pins]:
            self.drop(n, "the tornado zoom was removed")
        side = self._side_cells() * self.base.grid.dx
        for key, (x, y) in list(self.pins.items()):
            n = next((m for m in self.items if m.pin == key), None)
            if n is None:
                self.place(self._parent_rect_at(x, y), None, pin=key)
                continue
            d = n.patch.to_json()
            if math.hypot(x - 0.5 * (d["x0"] + d["x1"]), y - 0.5 * (d["y0"] + d["y1"])) > REPLACE_PIN * side:
                self.place(self._parent_rect_at(x, y), None, old=n)

    def _manage_inner(self) -> None:
        """One nest in each outer nest that has run INNER_MIN_AGE, on its UH03 maximum when that is a mesocyclone
        (INNER_UH03), kept on it; dropped with its outer nest or when the mesocyclone is gone LOST_AFTER."""
        eng = self.engine
        for n in list(self.items):              # its outer nest moved: moved with it (kept inside), or dropped
            if self._parent_of(n.parent_rect()) is None:
                d = n.patch.to_json()
                got = self.place(self._parent_rect_at(0.5 * (d["x0"] + d["x1"]), 0.5 * (d["y0"] + d["y1"])),
                                 n.track_id, old=n)
                if got is None:
                    self.drop(n, "its outer nest moved or ended")
        if eng.t < self._next_place:
            return
        self._next_place = eng.t + PLACE_EVERY / 2.0
        side = self._side_cells() * self.base.grid.dx
        mine = {n.track_id: n for n in self._auto()}
        for outer in self.parent.items:
            if eng.t - outer.placed_at < INNER_MIN_AGE and outer.id not in mine:
                continue
            got = uh03_peak(outer.patch)
            n = mine.get(outer.id)
            if got is None or got[2] < INNER_UH03:
                if n is not None:
                    if n.lost_since is None:
                        n.lost_since = eng.t
                    if eng.t - n.lost_since >= LOST_AFTER:
                        self.drop(n, "its mesocyclone weakened")
                continue
            x, y, _ = got
            tr = eng.tracks.resolve(outer.track_id)
            if tr is not None:                     # half the outer nest's lead: re-placed twice as often
                vx, vy = tr.velocity(getattr(eng.env, "storm_motion", None) or (0.0, 0.0))
                x, y = x + 0.5 * vx * LEAD, y + 0.5 * vy * LEAD
            if n is None:
                if len(self._auto()) < self.max_nests:
                    self.place(self._parent_rect_at(x, y), outer.id)
                continue
            n.lost_since = None
            if eng.t < n.hold_until:
                continue                           # moved by hand: stays where it was put for now
            d = n.patch.to_json()
            if math.hypot(x - 0.5 * (d["x0"] + d["x1"]), y - 0.5 * (d["y0"] + d["y1"])) > REPLACE_FRAC * side:
                self.place(self._parent_rect_at(x, y), outer.id, old=n)

    def _manage_outer(self) -> None:
        eng = self.engine
        for n in list(self.items):                 # its parent patch moved, shrank or went: refit or drop
            if self._parent_of(n.parent_rect()) is None:
                d = n.patch.to_json()
                got = self.place(self._parent_rect_at(0.5 * (d["x0"] + d["x1"]), 0.5 * (d["y0"] + d["y1"])),
                                 n.track_id, old=n)
                if got is None:
                    self.drop(n, "its patch no longer holds it")
        if eng.t < self._next_place:
            return
        self._next_place = eng.t + PLACE_EVERY
        live = {}
        for tr in eng.tracks.live:
            if eng.t - tr.last[0] <= 180.0 and eng.t - tr.born >= MIN_AGE:
                live[tr.id] = tr
        rot = Rotation(eng.patches)
        rank = {tid: rot.score(tr.strength(), tr.last[1], tr.last[2]) for tid, tr in live.items()}
        side = self._side_cells() * eng.env.grid.dx

        def centre(m):
            d = m.patch.to_json()
            return 0.5 * (d["x0"] + d["x1"]), 0.5 * (d["y0"] + d["y1"])

        for n in self._followers():
            if eng.t < n.hold_until:
                continue                           # moved by hand: stays where it was put for now
            tr = eng.tracks.resolve(n.track_id)
            if tr is None or tr.id not in live:
                # its track ended or split: a storm still under the nest's centre is its storm
                cx, cy = centre(n)
                near = eng.tracks.nearest(cx, cy, eng.t, within=0.3 * side)
                tr = near if near is not None and near.id in live and \
                    not any(m.track_id == near.id for m in self._followers() if m is not n) else None
            if tr is not None:
                n.track_id, n.lost_since, n.rank = tr.id, None, rank.get(tr.id, 0.0)
                ax, ay = self._aim(tr)
                cx, cy = centre(n)
                if math.hypot(ax - cx, ay - cy) > REPLACE_FRAC * side:
                    self.place(self._parent_rect_at(ax, ay), tr.id, old=n)
                continue
            if n.lost_since is None:
                n.lost_since = eng.t
            if eng.t - n.lost_since >= LOST_AFTER:
                self.drop(n, "its storm ended")
        owned = {n.track_id for n in self.items}
        for tr in sorted(live.values(), key=lambda q: -rank[q.id]):
            if tr.id in owned:
                continue
            _, x, y, _ = tr.last
            if any(math.hypot(x - cx, y - cy) < 0.5 * side for cx, cy in map(centre, self.items)):
                continue                           # inside another nest already
            if len(self._auto()) >= self.max_nests:
                # full: a much higher ranked storm (a rotating one) takes the weakest automatic nest's place
                old = [n for n in self._auto() if eng.t - n.placed_at >= SWAP_AFTER and eng.t >= n.hold_until]
                weak = min(old, key=lambda n: n.rank, default=None)
                if weak is None or rank[tr.id] < SWAP_RANK * max(weak.rank, 1e-6):
                    break
                self.drop(weak, f"storm {tr.id} rotates harder (rank {rank[tr.id]:.0f} over {weak.rank:.0f})")
            ax, ay = self._aim(tr)
            n = self.place(self._parent_rect_at(ax, ay), tr.id)
            if n is not None:
                n.rank = rank[tr.id]
                owned.add(tr.id)

    # ---------------------------------------------------------------- the step
    def _coupled(self, parent: Patch, win: Rect, names) -> dict:
        """WRF's coupled boundary fields (lateral_bc._coupled_device_fields) of the parent's cells ``win``
        (its local cells), with the u faces / v faces on and between them."""
        import cupy as cp
        st = parent.state
        j0, j1, i0, i1 = win.j0, win.j1, win.i0, win.i1
        mu_all = st.total_mu()
        muh = mu_all[j0 - 1:j1 + 1, i0 - 1:i1 + 1]
        mu = muh[1:-1, 1:-1]
        mux = 0.5 * (muh[1:-1, :-1] + muh[1:-1, 1:])
        muy = 0.5 * (muh[:-1, 1:-1] + muh[1:, 1:-1])
        c1h, c2h = st.c1h[:, None, None], st.c2h[:, None, None]
        c1f, c2f = st.c1f[:, None, None], st.c2f[:, None, None]
        chm = c1h * mu[None] + c2h
        chf = c1f * mu[None] + c2f
        thb = st.thb if st.thb.ndim == 3 else st.thb[:, None, None]
        if thb.ndim == 3 and thb.shape[-1] > 1:
            thb = thb[:, j0:j1, i0:i1]
        out = {
            "u": (c1h * mux[None] + c2h) * st.u[:, j0:j1, i0:i1 + 1],
            "v": (c1h * muy[None] + c2h) * st.v[:, j0:j1 + 1, i0:i1],
            "theta": chm * (thb + st.thp[:, j0:j1, i0:i1] - 300.0),
            "phi": chf * st.php[:, j0:j1, i0:i1],
            "mu": st.mup[None, j0:j1, i0:i1],
        }
        for name in names:
            if name not in out:
                out[name] = chm * getattr(st, name)[:, j0:j1, i0:i1]
        return out

    def _sides(self, n: Nest, parent: Patch) -> dict:
        """The nest's W/E/S/N boundary strips (coupled, child grid) from the parent now."""
        from gpuwm.ingest.lateral_bc import extract_lateral_side
        if self.vmaps is not None:
            return self._sides_levels(n, parent)
        names = sorted(self.env._boundary_snapshot())
        win = self._window(parent, n.parent_rect())
        cp_ = self._coupled(parent, win, names)
        snap = {}
        for name, a in cp_.items():
            xs, ys = name == "u", name == "v"
            jpos, ipos = self._positions(n.patch.rect, win, parent, xs, ys)
            snap[name] = _interp(a, jpos, ipos)
        return {side: extract_lateral_side(snap, side, WIDTH) for side in ("west", "east", "south", "north")}

    def _mass_positions(self, nest_rect: Rect, win: Rect, parent: Patch, xs: bool, ys: bool):
        """Fractional indices, in the parent window's MASS points, of a nest array's points (its faces too): to
        put the parent's dry mass on the nest's u / v faces."""
        r = self.ratio
        oi, oj = parent.rect.i0 + win.i0, parent.rect.j0 + win.j0
        i = np.arange(nest_rect.nx + (1 if xs else 0), dtype=np.float64)
        j = np.arange(nest_rect.ny + (1 if ys else 0), dtype=np.float64)
        ipos = (nest_rect.i0 + i + (0.0 if xs else 0.5)) / r - 0.5 - oi
        jpos = (nest_rect.j0 + j + (0.0 if ys else 0.5)) / r - 0.5 - oj
        return jpos, ipos

    def _sides_levels(self, n: Nest, parent: Patch) -> dict:
        """_sides for a nest on its own levels: the parent's air uncoupled (full theta and geopotential, the
        winds, the scalars), onto the nest's points and levels (VMaps.up), coupled again with the nest's
        coordinate and dry mass (WRF's coupled boundary units, as lateral_bc._coupled_device_fields)."""
        import cupy as cp
        from gpuwm.ingest.lateral_bc import extract_lateral_side
        vm = self.vmaps
        names = sorted(self.env._boundary_snapshot())
        win = self._window(parent, n.parent_rect())
        pst, nst = parent.state, n.patch.state
        j0, j1, i0, i1 = win.j0, win.j1, win.i0, win.i1
        nr = n.patch.rect

        def onto(a, xs=False, ys=False):            # a parent window array onto the nest's points and levels
            jp, ip = self._positions(nr, win, parent, xs, ys)
            return vm.up(_interp(a, jp, ip))

        theta = onto(pst.thp[:, j0:j1, i0:i1] + _base_part(pst, "thp", j0, j1, i0, i1))
        phi = onto(pst.php[:, j0:j1, i0:i1] + _base_part(pst, "php", j0, j1, i0, i1))
        mup = _interp(pst.mup[j0:j1, i0:i1], *self._positions(nr, win, parent, False, False))
        mu_tot = pst.total_mu()[j0:j1, i0:i1]
        mu_at = lambda xs, ys: _interp(mu_tot, *self._mass_positions(nr, win, parent, xs, ys))
        co = self.env.coord
        col = lambda a: cp.asarray(np.asarray(a, np.float64), dtype=cp.float32)[:, None, None]
        c1h, c2h, c1f, c2f = col(co.c1h), col(co.c2h), col(co.c1f), col(co.c2f)
        mu = mu_at(False, False)[None]
        chm, chf = c1h * mu + c2h, c1f * mu + c2f
        nphb = _base_part(nst, "php", 0, nr.ny, 0, nr.nx)
        out = {
            "u": (c1h * mu_at(True, False)[None] + c2h) * onto(pst.u[:, j0:j1, i0:i1 + 1], xs=True),
            "v": (c1h * mu_at(False, True)[None] + c2h) * onto(pst.v[:, j0:j1 + 1, i0:i1], ys=True),
            "theta": chm * (theta - 300.0),
            "phi": chf * (phi - nphb),
            "mu": mup[None],
        }
        for name in names:
            if name not in out:
                out[name] = chm * onto(getattr(pst, name)[:, j0:j1, i0:i1])
        return {side: extract_lateral_side(out, side, WIDTH) for side in ("west", "east", "south", "north")}

    def _attach(self, n: Nest, before: dict, after: dict, t0: float, t1: float) -> None:
        from gpuwm.ingest.lateral_bc import (LateralBoundaries, attach_lateral_boundaries,
                                             build_lateral_interval_from_sides)
        # a little past the parent step's end, so the nest's own sum of sub-steps never runs off the table
        interval = build_lateral_interval_from_sides(before, after, start_seconds=t0, end_seconds=t1 + 1.0e-3)
        st = n.patch.state
        el = st.elapsed_seconds
        attach_lateral_boundaries(st, LateralBoundaries((interval,), WIDTH, 1, 4))
        st.elapsed_seconds = el
        n.patch.lbc_end = math.inf

    def step(self, t0: float, dt: float, sub_step, within: Patch | None = None) -> None:
        """After the parent patches stepped from t0 to t0 + dt: force each nest, step it ``ratio`` times
        (``sub_step(patch, t, dt)`` does the step and the engine's own forcing; the next level in steps inside
        each of those), feed it back.  ``within``: only the nests of that parent (an inner level)."""
        for n in list(self.items):
            parent = self._parent_of(n.parent_rect())
            if parent is None or (within is not None and parent is not within):
                continue
            try:
                after = self._sides(n, parent)
                before = n.sides if n.sides is not None else after
                n.patch.state.elapsed_seconds = t0
                self._attach(n, before, after, t0, t0 + dt)
                nsub = self.ratio * self._extra_substeps(n, dt / self.ratio)
                h = dt / nsub
                w_edge = self._parent_w(n, parent)
                hydro_edge = self._parent_hydro(n, parent)
                from .engine import _debug_nan
                _debug_nan(n.patch, "boundary tables attached")
                for k in range(nsub):
                    sub_step(n.patch, t0 + k * h, h)
                    self._relax_w(n, w_edge)
                    self._relax_hydro(n, hydro_edge)
                    _debug_nan(n.patch, "edge relaxation")
                    if self.inner and self.inner.items:
                        self.inner.step(t0 + k * h, h, sub_step, within=n.patch)
                n.patch.state.elapsed_seconds = t0 + dt
                if not _finite(n.patch.state):
                    # blown up: never fed back (a ~1 m zoom's NaNs reached the run's patch through every level and
                    # rolled the whole run back); the levels inside it went with it in their own step
                    self.drop(n, "it went numerically unstable (nothing of it was fed back)")
                    continue
                self._feedback(n, parent)
                n.sides = self._sides(n, parent)
            except Exception as exc:
                log.exception("nest step failed")
                self.drop(n, f"its step failed: {exc}")

    def _parent_w(self, n: Nest, parent: Patch):
        """The parent's w interpolated onto the nest (its frame is all that is used)."""
        win = self._window(parent, n.parent_rect())
        sub = parent.state.w[:, win.j0:win.j1, win.i0:win.i1]
        jpos, ipos = self._positions(n.patch.rect, win, parent, False, False)
        w = _interp(sub, jpos, ipos)
        return self.vmaps.up(w) if self.vmaps is not None else w

    def _relax_w(self, n: Nest, target) -> None:
        """w across the edge frame toward ``target``: all of it on the edge row, nothing WIDTH cells in."""
        import cupy as cp
        ny, nx = n.patch.rect.ny, n.patch.rect.nx
        cache = getattr(n, "_wframe", None)
        if cache is None or cache.shape != (ny, nx):
            j = np.minimum(np.arange(ny), np.arange(ny)[::-1])[:, None]
            i = np.minimum(np.arange(nx), np.arange(nx)[::-1])[None, :]
            d = np.minimum(j, i).astype(np.float32)
            n._wframe = cache = cp.asarray(np.clip(1.0 - d / WIDTH, 0.0, 1.0))
        w = n.patch.state.w
        w += cache[None] * (target.astype(w.dtype) - w)

    def _frame_cells(self, n: Nest):
        """(rows, cols, weights) of the nest's edge frame -- the cells within WIDTH of an edge -- and each one's
        _relax_w weight (1 on the edge row, toward 0 WIDTH cells in), on the device."""
        import cupy as cp
        ny, nx = n.patch.rect.ny, n.patch.rect.nx
        cache = getattr(n, "_hframe", None)
        if cache is None or cache[0] != (ny, nx):
            j = np.minimum(np.arange(ny), np.arange(ny)[::-1])[:, None]
            i = np.minimum(np.arange(nx), np.arange(nx)[::-1])[None, :]
            d = np.minimum(j, i)
            jj, ii = np.nonzero(d < WIDTH)
            wgt = np.clip(1.0 - d[jj, ii] / WIDTH, 0.0, 1.0).astype(np.float32)
            n._hframe = cache = ((ny, nx), cp.asarray(jj), cp.asarray(ii), cp.asarray(wgt))
        return cache[1:]

    def _parent_hydro(self, n: Nest, parent: Patch) -> dict:
        """The parent's hydrometeors (masses and numbers, HYDRO) on the nest's edge frame only: {name: (nz, cells)}.
        gpuwm's specified boundary carries water vapour alone and gives every other species WRF's flow-dependent
        edge -- nothing coming in on an inflow side -- so a small nest on a hook lost the rain and hail the storm
        blew into it (an 83 m and a 42 m tornado-zoom box were near echo-free at 1 km inside a 50 dBZ core).
        WRF's nests take them from the parent; so do these, by relaxing across the frame (_relax_hydro)."""
        win = self._window(parent, n.parent_rect())
        jpos, ipos = self._positions(n.patch.rect, win, parent, False, False)
        jj, ii, _ = self._frame_cells(n)
        out = {}
        for name in HYDRO:
            src, dst = getattr(parent.state, name, None), getattr(n.patch.state, name, None)
            if src is None or dst is None or src.ndim != 3:
                continue
            full = _interp(src[:, win.j0:win.j1, win.i0:win.i1], jpos, ipos)
            if self.vmaps is not None:
                full = self.vmaps.up(full)
            if full.shape != dst.shape:
                continue
            out[name] = full[:, jj, ii].astype(dst.dtype)
            del full
        return out

    def _relax_hydro(self, n: Nest, target: dict) -> None:
        """The nest's hydrometeors across its edge frame toward the parent's (_parent_hydro), as _relax_w."""
        jj, ii, wgt = self._frame_cells(n)
        st = n.patch.state
        for name, tgt in target.items():
            a = getattr(st, name)
            cur = a[:, jj, ii]
            a[:, jj, ii] = cur + wgt[None] * (tgt - cur)

    def _extra_substeps(self, n: Nest, h: float) -> int:
        """1, or more when the nest's vertical CFL number at sub-step h would pass CFL_W (CFL_W)."""
        import cupy as cp
        env = n.patch.env
        rdz = getattr(self, "_rdz", None)
        if rdz is None:
            zh = np.asarray(env.z_h, dtype=np.float64)
            r = np.zeros(len(zh) + 1, dtype=np.float32)
            r[1:-1] = 1.0 / np.diff(zh)
            if getattr(env, "terrain", None) is not None:      # squeezed over the ground (as the engine's CFL)
                r /= 1.0 - min(4500.0, env.terrain.top) / env.grid.ztop
            rdz = self._rdz = cp.asarray(r)[:, None, None]
        cv = float((cp.abs(n.patch.state.w) * rdz).max()) * h
        if not math.isfinite(cv):
            raise FloatingPointError("non-finite vertical wind")
        return max(1, int(math.ceil(cv / CFL_W - 1e-9)))

    def _feedback(self, n: Nest, parent: Patch) -> None:
        """The nest's air into the parent's cells it covers, but for a margin along its edges."""
        import cupy as cp
        from gpuwm.core.diagnostics import update_diagnostics
        r = self.ratio
        m = max(2, -(-(WIDTH + 2) // r))                 # parent cells clear of the nest's boundary zone
        pr = n.parent_rect()
        q = parent.rect
        I0, J0 = pr.i0 - q.i0 + m, pr.j0 - q.j0 + m      # parent local cells fed back
        NI, NJ = pr.nx - 2 * m, pr.ny - 2 * m
        if NI <= 0 or NJ <= 0:
            return
        a0, b0 = m * r, m * r                            # the same cells' first nest cell
        pst, nst = parent.state, n.patch.state
        if self.vmaps is not None:
            self._feedback_levels(n, parent, I0, J0, NI, NJ, a0, b0)
            return
        terrain = _terrain_base(pst) and _terrain_base(nst)
        for name in FEEDBACK:
            src, dst = getattr(nst, name, None), getattr(pst, name, None)
            if src is None or dst is None:
                continue
            bn = BASE_OF.get(name) if terrain else None
            if bn:                                       # the full value, less the parent's own base
                s = (src + getattr(nst, bn))[..., b0:b0 + NJ * r, a0:a0 + NI * r]
                blk = s.reshape(s.shape[:-2] + (NJ, r, NI, r)).mean(axis=(-3, -1))
                dst[..., J0:J0 + NJ, I0:I0 + NI] = (blk - getattr(pst, bn)[..., J0:J0 + NJ, I0:I0 + NI]).astype(dst.dtype)
                continue
            if name == "u":                              # parent faces I0..I0+NI: the nest faces on them, mean along y
                s = src[:, b0:b0 + NJ * r, a0:a0 + NI * r + 1:r]
                dst[:, J0:J0 + NJ, I0:I0 + NI + 1] = s.reshape(s.shape[0], NJ, r, NI + 1).mean(axis=2)
            elif name == "v":
                s = src[:, b0:b0 + NJ * r + 1:r, a0:a0 + NI * r]
                dst[:, J0:J0 + NJ + 1, I0:I0 + NI] = s.reshape(s.shape[0], NJ + 1, NI, r).mean(axis=3)
            else:
                s = src[..., b0:b0 + NJ * r, a0:a0 + NI * r]
                lead = s.shape[:-2]
                blk = s.reshape(lead + (NJ, r, NI, r)).mean(axis=(-3, -1))
                dst[..., J0:J0 + NJ, I0:I0 + NI] = blk.astype(dst.dtype)
        update_diagnostics(pst)

    def _feedback_levels(self, n: Nest, parent: Patch, I0: int, J0: int, NI: int, NJ: int, a0: int, b0: int) -> None:
        """_feedback for a nest on its own levels: block means over each parent cell (faces: along them), then the
        nest's layers onto the parent's (VMaps.down: layer means; w levels interpolated); theta and the
        geopotential as full values, less the parent's own base."""
        from gpuwm.core.diagnostics import update_diagnostics
        r, vm = self.ratio, self.vmaps
        pst, nst = parent.state, n.patch.state
        terrain = _terrain_base(pst) and _terrain_base(nst)
        jb, ib = slice(b0, b0 + NJ * r), slice(a0, a0 + NI * r)
        for name in FEEDBACK:
            src, dst = getattr(nst, name, None), getattr(pst, name, None)
            if src is None or dst is None:
                continue
            full = name in BASE_OF and (name != "mup" or terrain)
            if name == "u":
                s = src[:, jb, a0:a0 + NI * r + 1:r]
                blk = s.reshape(s.shape[0], NJ, r, NI + 1).mean(axis=2)
                dst[:, J0:J0 + NJ, I0:I0 + NI + 1] = vm.down(blk).astype(dst.dtype)
                continue
            if name == "v":
                s = src[:, b0:b0 + NJ * r + 1:r, ib]
                blk = s.reshape(s.shape[0], NJ + 1, NI, r).mean(axis=3)
                dst[:, J0:J0 + NJ + 1, I0:I0 + NI] = vm.down(blk).astype(dst.dtype)
                continue
            s = src[..., jb, ib]
            if full:
                s = s + _base_part(nst, name, jb.start, jb.stop, ib.start, ib.stop)
            blk = s.reshape(s.shape[:-2] + (NJ, r, NI, r)).mean(axis=(-3, -1))
            if blk.ndim == 3:
                blk = vm.down(blk)
            if full:
                blk = blk - _base_part(pst, name, J0, J0 + NJ, I0, I0 + NI)
            dst[..., J0:J0 + NJ, I0:I0 + NI] = blk.astype(dst.dtype)
        update_diagnostics(pst)

    # ---------------------------------------------------------------- health, status
    def check(self) -> None:
        import cupy as cp
        if self.inner:
            self.inner.check()
        for n in list(self.items):
            st = n.patch.state
            w = float(cp.abs(st.w).max())
            if not math.isfinite(w) or w > 160.0 or not bool(cp.isfinite(st.thp).all()):
                self.drop(n, f"its air went bad (|w| {w:.0f} m/s)")

    def to_json(self) -> list:
        return [n.to_json() for n in self.all_items()]


def uh03_map(patch):
    """A patch's 0-3 km updraft helicity (sum of w x vertical vorticity x dz, updrafts only), m2/s2 per cell."""
    import cupy as cp
    st, env = patch.state, patch.env
    dx = env.grid.dx
    k3 = max(int(np.searchsorted(env.z_h, 3000.0)), 2)
    u = 0.5 * (st.u[:k3, :, 1:] + st.u[:k3, :, :-1])
    v = 0.5 * (st.v[:k3, 1:, :] + st.v[:k3, :-1, :])
    zeta = cp.gradient(v, dx, axis=2) - cp.gradient(u, dx, axis=1)
    w = cp.maximum(0.5 * (st.w[:k3] + st.w[1:k3 + 1]), 0.0)
    dz = cp.asarray(np.diff(np.asarray(env.z_w))[:k3], dtype=cp.float32)[:, None, None]
    return (w * zeta * dz).sum(axis=0)


class Rotation:
    """The 0-3 km updraft helicity of a set of patches, made once (on the host, max-pooled to ~2 km so a
    lookup is a slice): how strongly the storm at a point rotates, for ranking storms (a supercell's or a
    QLCS's low-level mesocyclone over a stronger but plain updraft)."""

    def __init__(self, patches):
        import cupy as cp
        self.maps = []
        for p in patches:
            try:
                uh = uh03_map(p)
            except Exception:
                continue
            dx = p.env.grid.dx
            b = max(1, int(round(2000.0 / dx)))
            ny, nx = (uh.shape[0] // b) * b, (uh.shape[1] // b) * b
            if ny == 0 or nx == 0:
                continue
            pooled = cp.asnumpy(uh[:ny, :nx].reshape(ny // b, b, nx // b, b).max(axis=(1, 3)))
            self.maps.append((p.world_x0(), p.world_y0(), b * dx, pooled))

    def near(self, x: float, y: float, r: float = 8000.0) -> float:
        """The largest UH03 within ``r`` m of ground point (x, y) (0 outside every patch)."""
        best = 0.0
        for x0, y0, d, a in self.maps:
            i0, i1 = int((x - r - x0) // d), int((x + r - x0) // d) + 1
            j0, j1 = int((y - r - y0) // d), int((y + r - y0) // d) + 1
            i0, j0 = max(i0, 0), max(j0, 0)
            i1, j1 = min(i1, a.shape[1]), min(j1, a.shape[0])
            if i1 > i0 and j1 > j0:
                best = max(best, float(a[j0:j1, i0:i1].max()))
        return best

    def score(self, w: float, x: float, y: float) -> float:
        """A storm's rank: its updraft, up to 3 x for a strongly rotating one (ROT_UH and more: x 3)."""
        return w * (1.0 + min(max(self.near(x, y), 0.0) / ROT_UH, 2.0))


def uh03_peak(patch):
    """(x, y, UH) ground m and m2/s2 of a patch's 0-3 km updraft helicity maximum (sum of w x vertical
    vorticity x dz, updrafts only): where its low-level mesocyclone is."""
    import cupy as cp
    dx = patch.env.grid.dx
    uh = uh03_map(patch)
    m = int(np.clip(int(round(8.0 * 333.0 / dx)), 1, 12))            # not within ~8 coarse cells of an edge
    core = uh[m:-m, m:-m]
    if core.size == 0:
        return None
    k = int(cp.argmax(core))
    j, i = divmod(k, core.shape[1])
    val = float(core[j, i])
    return patch.world_x0() + (i + m + 0.5) * dx, patch.world_y0() + (j + m + 0.5) * dx, val
