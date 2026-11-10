"""The EF swath: the strongest wind a tornado left on the ground, on the grids that resolve it, as the map's
"efswath" product (EF-scale colours, web/js/colors.js) -- like a damage survey's EF map, from the model's winds.

Every model step each grid of radar.VORTEX_DX or finer (the nests, the tornado zoom's levels, a fine run's
patches) keeps the run-max of its lowest-level ground-relative wind, only within REACH of a tornado followed then
(the radar's TornadoTracker, moved on at its motion since the last refresh): the vortex's own swath and a little of
the rear-flank wind round it, not every gust front's.  Those maxima are put every FLUSH s (and when a grid is moved
or gone) onto tiles of DX m fixed to the ground, which outlive the grids."""
from __future__ import annotations

import math

import numpy as np

DX = 20.0                        # m per pixel of the swath
TILE = 128                       # pixels a side
REACH = 1000.0                   # m round a tornado
FINE_DX = 200.0                  # grids this fine or finer (radar.VORTEX_DX)
SCALE, OFFSET = 2.0, 1.0         # code = m/s * SCALE + OFFSET (products.CODECS "efswath"); 0 = none


class EFSwath:
    def __init__(self):
        self.tiles: dict = {}            # (ti, tj) -> uint8 (TILE, TILE)
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
        bx0, bx1 = x0 + ii.min() * dx, x0 + (ii.max() + 1) * dx
        by0, by1 = y0 + jj.min() * dx, y0 + (jj.max() + 1) * dx
        T = TILE * DX
        for tj in range(int(math.floor(by0 / T)), int(math.floor(by1 / T)) + 1):
            for ti in range(int(math.floor(bx0 / T)), int(math.floor(bx1 / T)) + 1):
                px = ti * T + (np.arange(TILE) + 0.5) * DX           # pixel centres
                py = tj * T + (np.arange(TILE) + 0.5) * DX
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
                tile = self.tiles.get((ti, tj))
                if tile is None:
                    tile = self.tiles[(ti, tj)] = np.zeros((TILE, TILE), np.uint8)
                np.maximum(tile, code, out=tile)
        self.version += 1

    def pack(self, t: float) -> bytes | None:
        """The map product (products.pack_frame_product): every tile, in world m."""
        from .products import pack_frame_product
        if not self.tiles:
            return None
        T = TILE * DX
        lst = [({"x0": ti * T, "y0": tj * T, "x1": (ti + 1) * T, "y1": (tj + 1) * T, "nx": TILE, "ny": TILE},
                a.copy()) for (ti, tj), a in self.tiles.items()]
        return pack_frame_product(t, "efswath", lst)

    def peak(self) -> float:
        """The strongest wind in the swath (m/s)."""
        if not self.tiles:
            return 0.0
        return max((float(a.max()) - OFFSET) / SCALE for a in self.tiles.values())
