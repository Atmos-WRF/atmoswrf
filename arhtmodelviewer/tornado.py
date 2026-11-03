"""The tornado zoom: a small box dragged on the map around a tornado (or the mesocyclone where one may form) gets a
chain of nests (nest.py, pinned to it) down to 110, 55, 37, ~10 or ~1 m (DETAIL), and the chain follows the vortex.

The chain: the run's own nests where it has them (refine settings), then as many new levels as it takes (each 2-3x
finer, max_nests 0: only the pin) to reach the detail's spacing; the finest is the box's size, each level out OUT x
its next one in.  The finest has its own vertical levels (nest.VFineEnv, vertical_for): CM1-like, 8-25 m layers
at the ground stretching by at most STRETCH, with LES mixing.  Every FOLLOW_EVERY s the finest grid finds the vortex near where it was -- the strongest
near-ground vertical vorticity (lowest two levels, smoothed over 3 cells) within SEARCH_FRAC of the side -- and
every level's pin moves there (Nests.manage_pins).  No vortex (a box on a mesocyclone that has not made one yet):
the box drifts with its storm (the nearest followed updraft's motion) until one forms.

Its track records, near the vortex, the lowest surface pressure perturbation (where: the page's orange line, like
the "surface pressure minima" of Orf's renderings), the vorticity and the strongest ground-relative wind.

Trajectories: parcels released around it every SEED_EVERY s (a ring 0.6-1.8 km out, 10-400 m up) are carried
forward by the finest grid that holds them (ground-relative winds, RK2), each keeping its path for the 3-D view."""
from __future__ import annotations

import math

import numpy as np

FINE_MAX = 130.0       # m: the finest level ends at or below this...
FINE_MIN = 70.0        # ...and not below this (the model's grids are set up down to 100 m: environment.GRID_PRESETS)
OUT = 2.0              # each new outer level's side over the next one in
FOLLOW_EVERY = 12.0    # s of model time between vortex searches
SEARCH_FRAC = 0.3      # x the finest level's side: how far from the last place the vortex may be found
FOLLOW_ZETA = 0.01     # 1/s near the ground: weaker, and the box stays where it is
NEAR = 1200.0          # m around the vorticity maximum where its pressure minimum and strongest wind are looked for
DRIFT_REACH = 20000.0  # m: the followed updraft whose motion a box without a vortex drifts with
SNAP_KM = 12.0         # km: a new box goes onto the strongest low-level mesocyclone this near its centre
MESO_REACH = 4000.0    # m: how far a box without a vortex follows its mesocyclone between searches
MESO_UH = 40.0         # m2/s2 of 0-3 km updraft helicity: a low-level mesocyclone (nest.INNER_UH03)
TRACK_MAX = 2000       # points kept on the track
SEED_EVERY = 20.0      # s between parcel releases
SEED_N = 16            # parcels a release
PARCELS_MAX = 320
PATH_MAX = 150         # points on a parcel's path (one a parent step), then it is retired
PARCEL_DT = 3.0        # s: longest RK2 step of the parcels (the winds of the step just taken, held)
PIN = "tornado"


def _levels(eng) -> list:
    out, n = [], eng.nests
    while n is not None:
        out.append(n)
        n = n.inner
    return out


# detail: (finest spacing at most, at least (m), the box's usual side (km), the largest ratio between levels); the
# finest level gets its own vertical levels (vertical_for), so "ultra" and "extreme" are close to CM1 tornado runs
# near the ground.  "pc10" and "pc1" (~10 m, ~1 m: the "PC explosion" settings) are suction-vortex and
# debris-scale LES grids: 5:1 steps get there in few levels, their layers start at about their own spacing.  Any box
# size is taken (BOX_MIN_KM.. BOX_MAX_KM, at most FINE_CELLS_MAX cells a side on the finest grid); whether the card
# holds it is up to nest.Nests._room_for, which refuses a level that would not fit and says so in the Log
# "lite" (~350 m) and "medium" (~200 m): for weaker cards, the mesocyclone and a broad tornadic circulation, on the
# run's own levels and mixing like the refinement nests (OWN_LEVELS_MAX)
DETAIL = {"lite": (450.0, 300.0, 20.0, 3), "medium": (260.0, 120.0, 14.0, 3),
          "high": (FINE_MAX, FINE_MIN, 10.0, 3), "ultra": (65.0, 45.0, 6.0, 3), "extreme": (45.0, 30.0, 4.0, 3),
          "pc10": (12.0, 7.0, 2.0, 5), "pc1": (1.25, 0.75, 0.3, 5)}
OWN_LEVELS_MAX = 150.0    # m: a finest level coarser than this stays on the run's levels (no LES at 200-350 m)
OWN_LEVELS_BELOW = 20.0   # m: zoom levels finer than this get their own levels too, not only the finest
BOX_MIN_KM = 0.05
BOX_MAX_KM = 60.0
FINE_CELLS_MAX = 2048
STRETCH = 1.08         # most a layer may be deeper than the one under it (CM1's usual stretching)


def plan(dx: float, detail: str = "high") -> list[int]:
    """The ratios of the new levels under a grid of dx m to reach the detail's spacing: each step the largest ratio
    (up to the detail's) that stays at or above its finest spacing, 2 when none does."""
    hi, lo, _, rmax = DETAIL.get(detail, DETAIL["high"])
    out = []
    while dx > hi:
        r = next((r for r in range(rmax, 1, -1) if dx / r >= lo), 2)
        dx /= r
        out.append(r)
    return out


def vertical_for(dx: float, ztop: float, dz_top: float) -> tuple[int, float]:
    """(nz, lowest layer m) of a nest of dx m on its own levels: a lowest layer of a fifth of dx (8-25 m), the
    layers growing by at most STRETCH to the run's deepest, the run's top."""
    from .environment import stretched_levels
    dz_bot = min(25.0, max(min(8.0, dx), 0.2 * dx))      # under 8 m (the pc grids): about its own spacing
    for nz in range(40, 600, 2):
        dz = np.diff(stretched_levels(nz, ztop, dz_bot, dz_top))
        if (dz[1:] / dz[:-1]).max() <= STRETCH + 1e-6:
            return nz, dz_bot
    return 600, dz_bot


class TornadoZoom:
    def __init__(self, eng, x: float, y: float, size_km: float, detail: str = "high"):
        from .nest import Nests
        self.eng = eng
        self.detail = detail if detail in DETAIL else "high"
        self.x, self.y = float(x), float(y)
        have = _levels(eng)
        dx = have[-1].env.grid.dx if have else eng.env.grid.dx
        ratios = plan(dx, self.detail)
        fine_dx = dx / math.prod(ratios) if ratios else dx
        self.size_km = min(BOX_MAX_KM, FINE_CELLS_MAX * fine_dx / 1000.0, max(BOX_MIN_KM, float(size_km)))
        if not ratios and not have:
            raise ValueError(f"this run's {eng.env.grid.label} grid is already the finest the tornado zoom makes: "
                             "use the 3-D view on it instead")
        ratios, dxs, verts = self._plan_levels(dx, self.detail)
        self.size_km = self._fit_memory(have, ratios, dxs, verts, self.size_km)
        sizes = [self.size_km * OUT ** (len(ratios) - 1 - i) for i in range(len(ratios))]
        self.created = []
        last = have[-1] if have else None
        for r, s, vert in zip(ratios, sizes, verts):
            lvl = Nests(eng, r, s, 0, parent=last, vertical=vert)
            if last is None:
                eng.nests = lvl
            else:
                last.inner = lvl
            self.created.append(lvl)
            last = lvl
        self.chain = have + self.created
        self.track: list[list] = []
        self.found = None                    # the last search: x, y, zeta, dp, vmax (None: nothing yet)
        self.next_follow = eng.t
        self.next_seed = eng.t
        self.px = np.zeros(0)
        self.py = np.zeros(0)
        self.pz = np.zeros(0)
        self.paths: list[list] = []
        self._rng = np.random.default_rng(7)
        got = meso_near(list(eng.patches), self.x, self.y, max(SNAP_KM, self.size_km) * 1000.0)
        on = ""
        if got is not None:                      # auto-align: onto the low-level mesocyclone near the box
            moved = math.hypot(got[0] - self.x, got[1] - self.y) / 1000.0
            self.x, self.y = got[0], got[1]
            on = f", aligned {moved:.1f} km onto the low-level mesocyclone (0-3 km UH {got[2]:.0f} m2/s2)"
        self._pin(self.x, self.y)
        fe = self.chain[-1].env
        fine = fe.grid
        levels = (f", {fine.nz} levels from {float(fe.z_w[1]):.3g} m at the ground (LES)"
                  if self.chain[-1].vmaps is not None else "")
        eng.note("nest", f"tornado zoom ({self.detail}): {len(self.chain)} nest level(s) down to {fine.dx:.3g} m"
                         f"{levels}, {self.size_km:.3g} km box at {self.x / 1000:.1f}, {self.y / 1000:.1f} km{on}")

    def _plan_levels(self, dx: float, detail: str):
        """(ratios, spacings, vertical) of the new levels under a grid of dx m for ``detail``: the finest on its own
        levels (VFineEnv): isotropic near the ground, LES mixing; so is every level under OWN_LEVELS_BELOW (a ~1 m
        zoom's 6 m level on a 750 m run's 20-400 m layers blew up in its first step, and its NaNs took the 1.2 m
        grid inside it with them)."""
        g = self.eng.env.grid
        ratios = plan(dx, detail)
        dxs, verts, d = [], [], dx
        for k, r in enumerate(ratios):
            d /= r
            dxs.append(d)
            own = (k == len(ratios) - 1 and d <= OWN_LEVELS_MAX) or d < OWN_LEVELS_BELOW
            verts.append(vertical_for(d, g.ztop, g.dz_top) if own else None)
        return ratios, dxs, verts

    def change_detail(self, detail: str) -> None:
        """The zoom's detail changed where it is, without redrawing it: the same centre (the vortex it follows, not
        re-snapped), track and parcels.  The levels the two details share (same ratio, same vertical levels) are
        kept with all they resolved; those below are rebuilt from them.  The box keeps its size against the
        detail's usual one (a 1 m zoom's 0.15 km back on 110 m: ~5 km, not 0.15), made smaller if the GPU memory
        left cannot hold it."""
        from .nest import Nests
        eng = self.eng
        detail = detail if detail in DETAIL else "high"
        if detail == self.detail:
            return
        have = self.chain[:len(self.chain) - len(self.created)]
        dx0 = have[-1].env.grid.dx if have else eng.env.grid.dx
        ratios, dxs, verts = self._plan_levels(dx0, detail)
        if not ratios and not have:
            raise ValueError(f"this run's {eng.env.grid.label} grid is already the finest the tornado zoom makes")
        k = 0                                     # levels kept: same ratio, same vertical levels (own or the run's)
        while (k < min(len(ratios), len(self.created)) and self.created[k].ratio == ratios[k]
               and (self.created[k].vmaps is None) == (verts[k] is None)
               and (verts[k] is None or len(self.created[k].env.z_h) == verts[k][0])):
            k += 1
        old, size = self.detail, self.size_km * DETAIL[detail][2] / DETAIL[self.detail][2]
        fine_dx = dxs[-1] if dxs else dx0
        size = min(BOX_MAX_KM, FINE_CELLS_MAX * fine_dx / 1000.0, max(BOX_MIN_KM, size))
        if 0 < k < len(ratios):                   # the new levels fit inside the last one kept
            size = min(size, self.created[k - 1].size_km / OUT ** (len(ratios) - k))
        # fitted (or refused) before anything is taken down, counting what the levels that go hand back
        per = eng.nest_cell_bytes()
        freed = sum(n.patch.rect.cells * len(lvl.env.z_h) * per for lvl in self.created[k:] for n in lvl.items
                    if n.pin == PIN)
        if k < len(ratios):
            size = self._fit_memory([], ratios[k:], dxs[k:], verts[k:], size, extra=freed)
        for lvl in reversed(self.created[k:]):
            lvl.pins.pop(PIN, None)
            lvl.clear(f"the tornado zoom went to {detail}")
        parent = self.created[k - 1] if k else (have[-1] if have else None)
        if parent is None:
            eng.nests = None
        else:
            parent.inner = None
        self.created = self.created[:k]
        self.detail = detail
        sizes =[size * OUT ** (len(ratios) - 1 - i) for i in range(len(ratios))]
        last = parent
        for r, s, vert in zip(ratios[k:], sizes[k:], verts[k:]):
            lvl = Nests(eng, r, s, 0, parent=last, vertical=vert)
            if last is None:
                eng.nests = lvl
            else:
                last.inner = lvl
            self.created.append(lvl)
            last = lvl
        self.chain = have + self.created
        self.size_km = size
        self.found = None                         # the next search on the new finest grid
        self.next_follow = eng.t
        self._pin(self.x, self.y)
        fe = self.chain[-1].env
        eng.note("nest", f"tornado zoom {old} -> {detail} in place: down to {fe.grid.dx:.3g} m, {size:.3g} km box, "
                         f"{k} level(s) kept, {len(ratios) - k} rebuilt")

    def _chain_bytes(self, have, ratios, dxs, verts, size_km: float) -> float:
        """GPU bytes of the chain's pinned nests for a box of ``size_km`` (nest.Nests._room_for's measure): the run's
        own levels at their sizes, the new ones at the box's, each level out OUT x its next one in."""
        eng = self.eng
        per_cell = eng.nest_cell_bytes()
        total, n = 0.0, len(ratios)

        def side(km, parent_dx):                   # nest.Nests._side_cells
            c = int(round(km * 1000.0 / parent_dx))
            return max(8, -(-c // 4) * 4)

        for lvl in have:
            total += (side(lvl.size_km, lvl.base.grid.dx) * lvl.ratio) ** 2 * len(lvl.env.z_h) * per_cell
        for k, (r, d, vert) in enumerate(zip(ratios, dxs, verts)):
            nz = vert[0] if vert else len(eng.env.z_h)
            total += (side(size_km * OUT ** (n - 1 - k), d * r) * r) ** 2 * nz * per_cell
        return total

    def _fit_memory(self, have, ratios, dxs, verts, size_km: float, extra: float = 0.0) -> float:
        """The box's side, made smaller until the whole chain fits in the GPU memory left (a 10 m zoom on an 8.4 km box
        sat on "placing..." for good: its 40 m level alone wanted 6.4 GB of 3.5 free, the 8 m one tens of GB).  Too
        big even at BOX_MIN_KM: a ValueError saying so.  ``extra``: bytes about to be handed back (change_detail)."""
        room = self.eng.gpu_room() + extra
        if not math.isfinite(room):
            return size_km
        want = size_km
        for _ in range(40):
            need = self._chain_bytes(have, ratios, dxs, verts, size_km)
            if need <= room:
                break
            if size_km <= BOX_MIN_KM:
                raise ValueError(f"the {self.detail} tornado zoom does not fit in the GPU memory left even with the "
                                 f"smallest box ({need / 2**30:.1f} GB needed, {room / 2**30:.1f} GB free): use a coarser "
                                 "detail, or smaller patches")
            size_km = max(BOX_MIN_KM, size_km * min(0.9, math.sqrt(room / need)))
        if size_km < want - 1e-9:
            self.eng.note("memory", f"tornado zoom box made {size_km:.3g} km (from {want:.3g} km) so its "
                                    f"{len(have) + len(ratios)} levels fit in the GPU memory left ({room / 2**30:.1f} GB)")
        return size_km

    # ---------------------------------------------------------------- the chain
    def _pin(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        for lvl in self.chain:
            lvl.pins[PIN] = (x, y)
        if self.eng.nests is not None:
            self.eng.nests.manage_pins()

    def finest(self):
        """The finest level's pinned nest (its Patch), or None while it is not placed."""
        n = next((m for m in self.chain[-1].items if m.pin == PIN), None)
        return n.patch if n is not None else None

    def patches(self) -> list:
        """The chain's pinned patches, finest first, then the run's patches."""
        out = []
        for lvl in reversed(self.chain):
            out += [n.patch for n in lvl.items if n.pin == PIN]
        return out + list(self.eng.patches)

    def clear(self) -> None:
        for lvl in self.chain:
            lvl.pins.pop(PIN, None)
        for lvl in reversed(self.created):
            lvl.clear("the tornado zoom was removed")
        if self.created:
            first = self.created[0]
            if self.eng.nests is first:
                self.eng.nests = None
            elif first.parent is not None:
                first.parent.inner = None
        if self.eng.nests is not None:
            self.eng.nests.manage_pins()           # the run's own levels drop their pinned nests
        self.eng.note("nest", "tornado zoom removed")

    # ---------------------------------------------------------------- following
    def follow(self) -> None:
        self.next_follow = self.eng.t + FOLLOW_EVERY
        p = self.finest()
        if p is None:
            self._pin(self.x, self.y)            # not placed yet (or dropped by a rollback): again
            return
        got = vortex(p, self.x, self.y, SEARCH_FRAC * self.chain[-1]._side_cells() * self.chain[-1].base.grid.dx)
        if got is None:
            self._drift()
            return
        self.found = got
        x, y, zeta, px, py, dp, vmax = got
        self.track.append([round(self.eng.t, 1), round(px), round(py), round(dp, 2), round(zeta, 4), round(vmax, 1)])
        del self.track[:-TRACK_MAX]
        if zeta >= FOLLOW_ZETA:
            self._pin(x, y)
        else:
            self._drift()

    def _drift(self) -> None:
        """No vortex to hold on to (yet): the box goes onto its low-level mesocyclone (the 0-3 km updraft helicity
        maximum near it, on the finest grid that holds it), else moves with its storm (the nearest followed
        updraft's motion)."""
        eng = self.eng
        got = meso_near(self.patches(), self.x, self.y, MESO_REACH)
        if got is not None:
            self._pin(got[0], got[1])
            return
        near = [tr for tr in eng.tracks.live if math.hypot(tr.last[1] - self.x, tr.last[2] - self.y) <= DRIFT_REACH]
        if not near:
            return
        tr = min(near, key=lambda q: math.hypot(q.last[1] - self.x, q.last[2] - self.y))
        vx, vy = tr.velocity(eng.env.storm_motion)
        self._pin(self.x + vx * FOLLOW_EVERY, self.y + vy * FOLLOW_EVERY)

    # ---------------------------------------------------------------- trajectories
    def _seed(self) -> None:
        self.next_seed = self.eng.t + SEED_EVERY
        rng = self._rng
        a = rng.uniform(0.0, 2.0 * math.pi, SEED_N)
        r = rng.uniform(600.0, 1800.0, SEED_N)
        z = np.exp(rng.uniform(math.log(10.0), math.log(400.0), SEED_N))
        self.px = np.concatenate([self.px, self.x + r * np.cos(a)])
        self.py = np.concatenate([self.py, self.y + r * np.sin(a)])
        self.pz = np.concatenate([self.pz, z])
        self.paths += [[] for _ in range(SEED_N)]
        extra = len(self.paths) - PARCELS_MAX
        if extra > 0:
            self._keep(np.arange(len(self.paths)) >= extra)

    def _keep(self, k: np.ndarray) -> None:
        self.px, self.py, self.pz = self.px[k], self.py[k], self.pz[k]
        self.paths = [pth for pth, keep in zip(self.paths, k) if keep]

    def advect(self, dt: float) -> None:
        """The parcels over the parent step just taken (dt s), RK2 in a few sub-steps on the winds now."""
        if self.finest() is None:
            return
        if self.eng.t >= self.next_seed:
            self._seed()
        if not len(self.px):
            return
        pats = self.patches()
        n = max(1, int(math.ceil(dt / PARCEL_DT)))
        h = dt / n
        x, y, z = self.px, self.py, self.pz
        ok = np.ones(len(x), bool)
        for _ in range(n):
            u1, v1, w1 = winds(pats, x, y, z, self.eng.env.motion)
            xm, ym, zm = x + 0.5 * h * u1, y + 0.5 * h * v1, np.maximum(z + 0.5 * h * w1, 2.0)
            u2, v2, w2 = winds(pats, xm, ym, zm, self.eng.env.motion)
            bad = ~np.isfinite(u2)
            ok &= ~bad
            x = np.where(bad, x, x + h * u2)
            y = np.where(bad, y, y + h * v2)
            z = np.where(bad, z, np.maximum(z + h * w2, 2.0))
        self.px, self.py, self.pz = x, y, z
        for i, pth in enumerate(self.paths):
            pth.append((round(float(x[i])), round(float(y[i])), round(float(z[i]))))
        self._keep(ok & (np.array([len(p) for p in self.paths]) < PATH_MAX) & (z < 15000.0))

    # ---------------------------------------------------------------- the page
    def to_json(self) -> dict:
        p = self.finest()
        box = p.to_json() if p is not None else None
        fe = self.chain[-1].env
        waiting = next((lvl.mem_note for lvl in self.chain
                        if lvl.mem_note and not any(n.pin == PIN for n in lvl.items)), None)
        return {"x": self.x, "y": self.y, "size_km": self.size_km, "dx": fe.grid.dx, "detail": self.detail,
                "nz": fe.grid.nz, "dz0": float(fe.z_w[1]), "waiting": waiting if box is None else None,
                "box": [box["x0"], box["y0"], box["x1"], box["y1"]] if box else None,
                "found": ({"x": self.found[0], "y": self.found[1], "zeta": self.found[2], "dp": self.found[5],
                           "vmax": self.found[6]} if self.found else None),
                "track": self.track[-600:]}

    def paths_json(self) -> dict:
        return {"t": self.eng.t, "paths": [[c for pt in pth for c in pt] for pth in self.paths if len(pth) >= 2],
                "track": self.track}


def vortex(patch, x: float, y: float, radius: float):
    """Near ground point (x, y) in ``patch``: the strongest near-ground vertical vorticity (lowest two levels,
    3-cell mean) within ``radius`` and clear of the nest's boundary zone, then within NEAR of it the lowest
    surface pressure perturbation (from the patch's mean) and the strongest ground-relative wind.
    (x, y, zeta, px, py, dp hPa, vmax m/s), or None."""
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    st, env = patch.state, patch.env
    dx = env.grid.dx
    x0, y0 = patch.world_x0(), patch.world_y0()
    uc = 0.5 * (st.u[:2, :, 1:] + st.u[:2, :, :-1])
    vc = 0.5 * (st.v[:2, 1:, :] + st.v[:2, :-1, :])
    zeta = (cp.gradient(vc, dx, axis=2) - cp.gradient(uc, dx, axis=1)).mean(axis=0)
    zeta = cnd.uniform_filter(zeta, 3, mode="nearest")
    ny, nx = zeta.shape
    jj, ii = cp.mgrid[0:ny, 0:nx]
    X = x0 + (ii + 0.5) * dx
    Y = y0 + (jj + 0.5) * dx
    edge = 8
    inner = (ii >= edge) & (ii < nx - edge) & (jj >= edge) & (jj < ny - edge)
    look = inner & ((X - x) ** 2 + (Y - y) ** 2 <= radius ** 2)
    if not bool(look.any()):
        return None
    k = int(cp.argmax(cp.where(look, zeta, -cp.inf)))
    j, i = divmod(k, nx)
    zx, zy, zmax = float(X[j, i]), float(Y[j, i]), float(zeta[j, i])
    near = inner & ((X - zx) ** 2 + (Y - zy) ** 2 <= NEAR ** 2)
    p0 = st.p[0]
    pp = p0 - p0.mean()
    k2 = int(cp.argmin(cp.where(near, pp, cp.inf)))
    j2, i2 = divmod(k2, nx)
    mx, my = env.motion
    speed = cp.sqrt((uc[0] + mx) ** 2 + (vc[0] + my) ** 2)
    vmax = float(cp.where(near, speed, 0.0).max())
    return zx, zy, zmax, float(X[j2, i2]), float(Y[j2, i2]), float(pp[j2, i2]) / 100.0, vmax


def meso_near(patches, x: float, y: float, reach: float):
    """(x, y, UH) of the strongest 0-3 km updraft helicity (the low-level mesocyclone) within ``reach`` m of
    (x, y), on the first of ``patches`` (finest first) that holds (x, y); None below MESO_UH."""
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    for p in patches:
        st, env = p.state, p.env
        dx = env.grid.dx
        x0, y0 = p.world_x0(), p.world_y0()
        if not (x0 <= x <= x0 + p.rect.nx * dx and y0 <= y <= y0 + p.rect.ny * dx):
            continue
        k3 = max(int(np.searchsorted(env.z_h, 3000.0)), 2)
        u = 0.5 * (st.u[:k3, :, 1:] + st.u[:k3, :, :-1])
        v = 0.5 * (st.v[:k3, 1:, :] + st.v[:k3, :-1, :])
        zeta = cp.gradient(v, dx, axis=2) - cp.gradient(u, dx, axis=1)
        w = cp.maximum(0.5 * (st.w[:k3] + st.w[1:k3 + 1]), 0.0)
        dz = cp.asarray(np.diff(np.asarray(env.z_w))[:k3], dtype=cp.float32)[:, None, None]
        uh = (w * zeta * dz).sum(axis=0)
        # a few km mean: a mesocyclone, not one gust's spin (the finer the grid, the more cells)
        uh = cnd.uniform_filter(uh, max(1, int(round(1500.0 / dx))) | 1, mode="nearest")
        ny, nx = uh.shape
        jj, ii = cp.mgrid[0:ny, 0:nx]
        X, Y = x0 + (ii + 0.5) * dx, y0 + (jj + 0.5) * dx
        m = 4
        look = (ii >= m) & (ii < nx - m) & (jj >= m) & (jj < ny - m) & ((X - x) ** 2 + (Y - y) ** 2 <= reach ** 2)
        if not bool(look.any()):
            return None
        k = int(cp.argmax(cp.where(look, uh, -cp.inf)))
        j, i = divmod(k, nx)
        val = float(uh[j, i])
        return (float(X[j, i]), float(Y[j, i]), val) if val >= MESO_UH else None
    return None


def _levels_frac(z: np.ndarray, heights) -> np.ndarray:
    h = np.asarray(heights, np.float64)
    return np.interp(z, h, np.arange(len(h), dtype=np.float64))


def winds(patches, X, Y, Z, motion):
    """Ground-relative u, v, w (m/s) at ground points X, Y and heights Z (m above the ground's level surface):
    from the first of ``patches`` (finest first) that holds each point; NaN where none does."""
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    n = len(X)
    out = np.full((3, n), np.nan)
    left = np.ones(n, bool)
    for p in patches:
        if not left.any():
            break
        st, env = p.state, p.env
        dx = env.grid.dx
        gi = (X - p.world_x0()) / dx
        gj = (Y - p.world_y0()) / dx
        ny, nx = p.rect.ny, p.rect.nx
        sel = left & (gi >= 1.0) & (gi <= nx - 1.0) & (gj >= 1.0) & (gj <= ny - 1.0)
        if not sel.any():
            continue
        # one upload and one download a grid: mass-level k, w-level k, j, i (u on x faces, v on y faces)
        c = cp.asarray(np.stack([_levels_frac(Z[sel], env.z_h), _levels_frac(Z[sel], env.z_w), gj[sel], gi[sel]]),
                       dtype=cp.float32)
        km, kw, j, i = c[0], c[1], c[2], c[3]
        u = cnd.map_coordinates(st.u, cp.stack([km, j - 0.5, i]), order=1, mode="nearest")
        v = cnd.map_coordinates(st.v, cp.stack([km, j, i - 0.5]), order=1, mode="nearest")
        w = cnd.map_coordinates(st.w, cp.stack([kw, j - 0.5, i - 0.5]), order=1, mode="nearest")
        got = cp.asnumpy(cp.stack([u, v, w]))
        out[0, sel] = got[0] + motion[0]
        out[1, sel] = got[1] + motion[1]
        out[2, sel] = got[2]
        left &= ~sel
    return out
