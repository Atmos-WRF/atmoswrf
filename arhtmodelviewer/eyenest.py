"""Hurricane nests: a box of finer grid (250 m - 1 km) on each tracked cyclone of a hurricane run, kept on its centre
as the cyclone tracker (sim._cyclones_step, every CYCLONE_EVERY s) fixes it.

The chain is the tornado zoom's (tornado.py): pinned nests (nest.Nests with max_nests 0, only the pins), as many
levels as it takes from the run's grid to the spacing asked (each 2-4x finer, the steps chosen to land nearest it),
the finest the box asked, each level out OUT x its next one in.  Every level keeps the run's own vertical levels and
mixing (they are 250 m or coarser).  Each cyclone has its own pin (PIN + its id), so two cyclones get two boxes; a
cyclone that leaves the domain or is lost loses its box."""
from __future__ import annotations

import itertools
import math

PIN = "cyclone"
DX_MIN, DX_MAX = 250.0, 1000.0     # m: the spacings offered
KM_MIN, KM_MAX = 60.0, 800.0       # km: the finest box's side
OUT = 1.25                         # each outer level's side over the next one in
REPLACE = 0.06                     # x a level's side: how far off its cyclone a box may drift before it is moved
LEAD = 120.0                       # s of the cyclone's motion a box is placed ahead
RATIOS = (4, 3, 2)


def plan(dx: float, want: float) -> list[int]:
    """The ratios (coarse first, each 2-4, at most four levels) taking a grid of dx m nearest to ``want`` m (fewer
    levels when two land about as near); [] when dx is already about that fine."""
    best, score = [], abs(math.log(dx / want))
    for n in range(1, 5):
        for rs in itertools.combinations_with_replacement(RATIOS, n):     # non-increasing: 4s, then 3s, then 2s
            d = dx / math.prod(rs)
            if d < 0.75 * DX_MIN:
                continue
            s = abs(math.log(d / want)) + 0.04 * n
            if s < score - 1e-9:
                best, score = list(rs), s
    return best


class CycloneNests:
    def __init__(self, eng, dx: float, km: float):
        from .nest import Nests
        if eng.tornado is not None:
            raise ValueError("remove the tornado zoom first: hurricane nests are the outer levels it would sit in")
        if eng.nests is not None:
            raise ValueError("this run has refinement nests on its storms (Domain tab, Refinement): hurricane nests "
                             "need a run without them")
        self.eng = eng
        self.want_dx = min(DX_MAX, max(DX_MIN, float(dx)))
        g = eng.env.grid
        self.ratios = plan(g.dx, self.want_dx)
        if not self.ratios:
            raise ValueError(f"the run's {g.label} grid is already about {self.want_dx:g} m: no nest needed")
        self.dxs = []
        d = g.dx
        for r in self.ratios:
            d /= r
            self.dxs.append(d)
        self.km = self._fit_memory(min(KM_MAX, max(KM_MIN, float(km))))
        n = len(self.ratios)
        sizes = [self.km * OUT ** (n - 1 - i) for i in range(n)]
        self.levels = []
        last = None
        for r, s in zip(self.ratios, sizes):
            lvl = Nests(eng, r, s, 0, parent=last)
            lvl.replace_pin = REPLACE
            if last is None:
                eng.nests = lvl
            else:
                last.inner = lvl
            self.levels.append(lvl)
            last = lvl
        steps = " -> ".join(f"{q:.0f} m" for q in [g.dx] + self.dxs)
        eng.note("nest", f"hurricane nests: {steps}, a {self.km:.0f} km box on each cyclone (each level out "
                         f"{OUT:g}x the next)")

    @property
    def dx(self) -> float:
        return self.dxs[-1]

    def _bytes(self, km: float) -> float:
        per = self.eng.nest_cell_bytes()
        nz = len(self.eng.env.z_h)
        n, total, pdx = len(self.ratios), 0.0, self.eng.env.grid.dx
        for k, (r, d) in enumerate(zip(self.ratios, self.dxs)):
            c = int(round(km * OUT ** (n - 1 - k) * 1000.0 / pdx))
            c = max(8, -(-c // 4) * 4)
            total += (c * r) ** 2 * nz * per
            pdx = d
        return total

    def _fit_memory(self, km: float) -> float:
        """The box's side, smaller until one cyclone's chain fits in the GPU memory left."""
        room = self.eng.gpu_room()
        if not math.isfinite(room):
            return km
        want = km
        for _ in range(40):
            need = self._bytes(km)
            if need <= room:
                break
            if km <= KM_MIN:
                raise ValueError(f"a ~{self.want_dx:g} m hurricane nest does not fit in the GPU memory left even "
                                 f"{KM_MIN:g} km a side ({need / 2**30:.1f} GB needed, {room / 2**30:.1f} GB free): use a "
                                 "coarser spacing or a smaller domain")
            km = max(KM_MIN, km * min(0.9, math.sqrt(room / need)))
        if km < want - 1e-6:
            self.eng.note("memory", f"hurricane nest box made {km:.0f} km (from {want:.0f} km) to fit in the GPU "
                                    f"memory left ({room / 2**30:.1f} GB)")
        return km

    def follow(self, cyclones: list) -> None:
        """Each active cyclone's box onto its latest fix (LEAD s of its motion ahead); a lost one's box goes."""
        live = {}
        for c in cyclones:
            if c.get("active") and c.get("track"):
                live[f"{PIN}{c['id']}"] = (c["x"] + c["vx"] * LEAD, c["y"] + c["vy"] * LEAD)
        for lvl in self.levels:
            for k in [k for k in lvl.pins if k.startswith(PIN) and k not in live]:
                lvl.pins.pop(k)
            lvl.pins.update(live)
        if self.eng.nests is not None:
            self.eng.nests.manage_pins()

    def clear(self) -> None:
        for lvl in self.levels:
            lvl.pins.clear()
        for lvl in reversed(self.levels):
            lvl.clear("the hurricane nests were turned off")
        if self.eng.nests is self.levels[0]:
            self.eng.nests = None
        self.eng.note("nest", "hurricane nests removed")

    def to_json(self) -> dict:
        boxes = []
        fin = self.levels[-1]
        for n in fin.items:
            d = n.patch.to_json()
            boxes.append({"pin": n.pin, "x0": d["x0"], "y0": d["y0"], "x1": d["x1"], "y1": d["y1"]})
        return {"dx": self.dx, "ask_dx": self.want_dx, "km": self.km, "levels": [round(q, 1) for q in self.dxs],
                "boxes": boxes}
