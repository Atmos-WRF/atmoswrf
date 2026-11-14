"""The EF swath: the strongest wind a tornado left on the ground, on the grids that resolve it, as the map's
"efswath" product (EF-scale colours, web/js/colors.js) -- like a damage survey's EF map, from the model's winds.

Every model step each grid of radar.VORTEX_DX or finer (the nests, the tornado zoom's levels, a fine run's
patches) keeps the run-max of its lowest-level ground-relative wind, only within REACH of a tornado followed then
(the radar's TornadoTracker, moved on at its motion since the last refresh): the vortex's own swath and a little of
the rear-flank wind round it, not every gust front's.  Those maxima are put every FLUSH s (and when a grid is moved
or gone) onto tiles fixed to the ground at that grid's own spacing (one layer of tiles per spacing), which outlive
the grids.  The map product is made of COMP_M tiles, each at the spacing of the finest layer with EF0 or more in it
-- where the strongest winds are, the polygons are as fine as the finest grid that had them (a 10 m vortex core's
path at 10 m, a 37 m zoom's at 37 m), not one fixed spacing -- each pixel from the finest layer that has a wind
there, the coarser ones round it."""
from __future__ import annotations

import math

import numpy as np

TILE = 128                       # pixels a side of a layer's tile (at its own spacing)
COMP_M = 2560.0                  # m a side of the map product's tiles (world-fixed)
COMP_MAX = 1024                  # most pixels a side of one (a ~1 m grid's swath at 2.5 m)
REACH = 1000.0                   # m round a tornado
FINE_DX = 200.0                  # grids this fine or finer (radar.VORTEX_DX)
SCALE, OFFSET = 2.0, 1.0         # code = m/s * SCALE + OFFSET (products.CODECS "efswath"); 0 = none


class EFSwath:
    def __init__(self):
        self.layers: dict = {}           # dx (m, to 0.01) -> {"dx", "tiles": {(ti, tj): uint8 (TILE, TILE)}}
        self.bufs: dict = {}             # patch id -> {"key", "x0", "y0", "dx", "a": cupy (ny, nx) m/s}
        self.version = 0

    def step(self, grids, tors) -> None:
        """One model step: ``grids`` (patches), ``tors`` [(x, y) world m] -- the tornadoes now."""
        import cupy as cp
        from .products import low_wind
        live = set()
        for p in grids:
            dx = p.env.grid.dx
            if dx > FINE_DX or getattr(p, "state", None) is None:
                continue
            x0, y0 = p.world_x0(), p.world_y0()
            nx, ny = p.rect.nx, p.rect.ny
            near = [(x, y) for x, y in tors
                    if x0 - REACH <= x <= x0 + nx * dx + REACH and y0 - REACH <= y <= y0 + ny * dx + REACH]
            key = (p.rect.i0, p.rect.j0, nx, ny, round(x0, 1), round(y0, 1), dx)
            b = self.bufs.get(p.id)
            if b is not None and b["key"] != key:          # moved (re-placed, or a moving grid): its swath kept
                self._flush_one(b)
                b = None
            if not near:
                if b is not None:
                    live.add(p.id)
                continue
            if b is None:
                b = self.bufs[p.id] = {"key": key, "x0": x0, "y0": y0, "dx": dx,
                                       "a": cp.zeros((ny, nx), cp.float32)}
            live.add(p.id)
            spd = low_wind(p)[0]
            X = x0 + (cp.arange(nx, dtype=cp.float32) + 0.5) * dx
            Y = y0 + (cp.arange(ny, dtype=cp.float32) + 0.5) * dx
            mask = cp.zeros((ny, nx), bool)
            for x, y in near:
                mask |= ((X[None, :] - x) ** 2 + (Y[:, None] - y) ** 2) <= REACH * REACH
            cp.maximum(b["a"], cp.where(mask, spd, 0.0), out=b["a"])
        for pid in [k for k in self.bufs if k not in live]:   # a grid gone (dropped nest): its swath kept
            self._flush_one(self.bufs.pop(pid))

    def flush(self) -> None:
        """Every grid's maxima onto the tiles (the grids go on adding to theirs)."""
        for b in self.bufs.values():
            self._flush_one(b, keep=True)

    def _flush_one(self, b, keep: bool = False) -> None:
        import cupy as cp
        a = cp.asnumpy(b["a"])
        if keep:
            b["a"].fill(0.0)
        jj, ii = np.nonzero(a > 0.5)
        if not jj.size:
            return
        dx, x0, y0 = b["dx"], b["x0"], b["y0"]
        lay = self.layers.setdefault(round(dx, 2), {"dx": dx, "tiles": {}})
        bx0, bx1 = x0 + ii.min() * dx, x0 + (ii.max() + 1) * dx
        by0, by1 = y0 + jj.min() * dx, y0 + (jj.max() + 1) * dx
        T = TILE * dx                         # the layer's pixels are the grid's cells (a nest's x0 stays on them)
        for tj in range(int(math.floor(by0 / T)), int(math.floor(by1 / T)) + 1):
            for ti in range(int(math.floor(bx0 / T)), int(math.floor(bx1 / T)) + 1):
                px = ti * T + (np.arange(TILE) + 0.5) * dx           # pixel centres
                py = tj * T + (np.arange(TILE) + 0.5) * dx
                ci = np.floor((px - x0) / dx).astype(int)
                cj = np.floor((py - y0) / dx).astype(int)
                okx, oky = (ci >= 0) & (ci < a.shape[1]), (cj >= 0) & (cj < a.shape[0])
                if not okx.any() or not oky.any():
                    continue
                sub = a[np.clip(cj, 0, a.shape[0] - 1)][:, np.clip(ci, 0, a.shape[1] - 1)]
                sub = np.where(oky[:, None] & okx[None, :], sub, 0.0)
                if not (sub > 0.5).any():
                    continue
                code = np.where(sub > 0.5, np.clip(np.rint(sub * SCALE + OFFSET), 1, 255), 0).astype(np.uint8)
                tile = lay["tiles"].get((ti, tj))
                if tile is None:
                    tile = lay["tiles"][(ti, tj)] = np.zeros((TILE, TILE), np.uint8)
                np.maximum(tile, code, out=tile)
        self.version += 1

    @staticmethod
    def _sample(lay, px, py) -> np.ndarray:
        """Layer ``lay``'s codes at pixel centres px (nx), py (ny) world m: (ny, nx), 0 where it has none."""
        dx, T = lay["dx"], lay["tiles"]
        gi, gj = np.floor(px / dx).astype(np.int64), np.floor(py / dx).astype(np.int64)
        ti, tj = gi // TILE, gj // TILE
        out = np.zeros((len(py), len(px)), np.uint8)
        for a in np.unique(tj):
            rows = np.nonzero(tj == a)[0]
            for b in np.unique(ti):
                tile = T.get((int(b), int(a)))
                if tile is None:
                    continue
                cols = np.nonzero(ti == b)[0]
                out[np.ix_(rows, cols)] = tile[np.ix_(gj[rows] - a * TILE, gi[cols] - b * TILE)]
        return out

    def composite(self) -> list:
        """The map product's tiles: [(meta, uint8 (n, n))], each COMP_M a side at the spacing of the finest layer
        with EF0 or more in it (a tile with none: left out, the map draws nothing under EF0), each pixel from the
        finest layer that has a wind there."""
        lays = sorted(self.layers.values(), key=lambda L: L["dx"])            # finest first
        ef0 = int(round(29.0 * SCALE + OFFSET))
        keys = set()
        for L in lays:
            T = TILE * L["dx"]
            for (ti, tj), a in L["tiles"].items():
                jj, ii = np.nonzero(a)
                if not jj.size:
                    continue
                x0, y0 = ti * T + ii.min() * L["dx"], tj * T + jj.min() * L["dx"]
                x1, y1 = ti * T + (ii.max() + 1) * L["dx"], tj * T + (jj.max() + 1) * L["dx"]
                for cj in range(int(math.floor(y0 / COMP_M)), int(math.floor((y1 - 1e-6) / COMP_M)) + 1):
                    for ci in range(int(math.floor(x0 / COMP_M)), int(math.floor((x1 - 1e-6) / COMP_M)) + 1):
                        keys.add((ci, cj))
        out = []
        for ci, cj in sorted(keys):
            bx, by = ci * COMP_M, cj * COMP_M
            # this tile's spacing: the finest layer with EF0 here (sampled at its own spacing)
            n = None
            for L in lays:
                m = min(COMP_MAX, max(1, int(round(COMP_M / L["dx"]))))
                c = bx + (np.arange(m) + 0.5) * COMP_M / m
                if (self._sample(L, c, by + (np.arange(m) + 0.5) * COMP_M / m) >= ef0).any():
                    n = m
                    break
            if n is None:
                continue
            px = bx + (np.arange(n) + 0.5) * COMP_M / n
            py = by + (np.arange(n) + 0.5) * COMP_M / n
            acc = np.zeros((n, n), np.uint8)
            for L in lays:                                 # the finest that has a wind there, then coarser ones
                v = self._sample(L, px, py)
                np.copyto(acc, v, where=(acc == 0) & (v > 0))
            if acc.any():
                out.append(({"x0": bx, "y0": by, "x1": bx + COMP_M, "y1": by + COMP_M, "nx": n, "ny": n,
                             "dx": COMP_M / n}, acc))
        return out

    def pack(self, t: float) -> bytes | None:
        """The map product (products.pack_frame_product): the composite's tiles, in world m."""
        from .products import pack_frame_product
        lst = self.composite()
        if not lst:
            return None
        return pack_frame_product(t, "efswath", lst)

    def peak(self) -> float:
        """The strongest wind in the swath (m/s)."""
        m = max((int(a.max()) for L in self.layers.values() for a in L["tiles"].values()), default=0)
        return (m - OFFSET) / SCALE if m else 0.0
