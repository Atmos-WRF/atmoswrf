"""Storm interactions: updraft tracks and nudger cells.

Tracks: every model minute the engine finds the deep updrafts of each patch
(local maxima of the 3-7 km maximum w, 12 m/s and up, at least 6 km apart) and
links them to the tracks of the minute before, so each storm has an identity,
a path and a measured motion.

Nudgers: a small cell started so that it runs into a chosen storm at a chosen
time -- "nudger" tornadogenesis in storm chasers' words, a merger with a
nascent cell in Hastings & Richardson's (2016, MWR 144) idealized runs, whose
outcome depended on how close the two updrafts passed (under 10 km: a classic
supercell; 10-20 km: an updraft bridge, a bow on the forward flank or an HP
supercell on the rear flank; over 20 km on the forward flank: two cells).  Its
start point comes from relative motion: where the storm's updraft will be at
the meeting time, moved sideways by ``aim`` across the cell's path relative to
the storm (+ ahead of the updraft, its forward flank; - behind, its rear
flank), less the cell's own motion over the lead time.  Its push ends at least
5 minutes before contact, so the merger itself is the model's own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TRACK_KEEP = 2400.0       # s of path kept per track
TRACK_GAP = 300.0         # s a track may go unseen before it ends
MERGE_KM = 12.0           # a nudger whose updraft ends this close to the storm's merged into it


def updraft_peaks(patch, w_min: float = 12.0, sep_km: float = 6.0, z_lo: float = 3000.0,
                  z_hi: float = 7000.0, max_n: int = 40) -> list:
    """(x, y, w) ground m and m/s of the patch's deep updrafts: local maxima of
    the z_lo..z_hi maximum w over ``w_min``, at least ``sep_km`` apart."""
    import cupy as cp
    from cupyx.scipy import ndimage

    st, env = patch.state, patch.env
    dx = env.grid.dx
    k0 = int(np.searchsorted(env.z_w, z_lo))
    k1 = max(int(np.searchsorted(env.z_w, z_hi)), k0 + 1)
    wm = st.w[k0:k1].max(axis=0)
    size = max(3, int(round(sep_km * 1000.0 / dx)) | 1)
    peak = (wm == ndimage.maximum_filter(wm, size=size)) & (wm >= w_min)
    jj, ii = cp.nonzero(peak)
    if jj.size == 0:
        return []
    vals = wm[jj, ii]
    order = cp.argsort(-vals)[:max_n]
    jj, ii, vals = cp.asnumpy(jj[order]), cp.asnumpy(ii[order]), cp.asnumpy(vals[order])
    x0, y0 = patch.world_x0(), patch.world_y0()
    return [(x0 + (i + 0.5) * dx, y0 + (j + 0.5) * dx, float(w)) for i, j, w in zip(ii, jj, vals)]


@dataclass
class Track:
    """One deep updraft followed from minute to minute: points (t, x, y, w)."""
    id: int
    pts: list = field(default_factory=list)
    ended: float | None = None

    @property
    def born(self) -> float:
        return self.pts[0][0]

    @property
    def last(self):
        return self.pts[-1]

    def velocity(self, default=(0.0, 0.0), span: float = 900.0):
        """Ground motion (m/s): least squares over the last ``span`` s if that holds
        at least 5 points over 8 min, else ``default``."""
        t_end = self.pts[-1][0]
        a = np.array([p for p in self.pts if p[0] >= t_end - span])
        if len(a) < 5 or a[-1, 0] - a[0, 0] < 480.0:
            return tuple(default)
        t = a[:, 0] - a[:, 0].mean()
        den = float((t * t).sum())
        return (float((t * (a[:, 1] - a[:, 1].mean())).sum() / den),
                float((t * (a[:, 2] - a[:, 2].mean())).sum() / den))

    def strength(self, span: float = 300.0) -> float:
        t_end = self.pts[-1][0]
        return float(np.mean([p[3] for p in self.pts if p[0] >= t_end - span]))

    def to_json(self, default=(0.0, 0.0)) -> dict:
        t, x, y, w = self.last
        vx, vy = self.velocity(default)
        return {"id": self.id, "t": t, "x": x, "y": y, "w": w, "vx": vx, "vy": vy, "born": self.born,
                "ended": self.ended}


class Tracks:
    """Deep updraft tracks: each minute's peaks are linked to the live tracks by
    nearest neighbour within 6 km of where each track was heading, plus 8 m/s of
    drift since it was last seen, if they have 40 % of its strength.  A track that ends
    within MERGE_KM of a live one merged into it: that one is its successor, so a
    storm keeps its identity when two of its updrafts join."""

    def __init__(self):
        self.live: list[Track] = []
        self.done: list[Track] = []
        self.successor: dict = {}           # ended track id -> the track it merged into
        self._next = 1

    def get(self, tid: int | None) -> Track | None:
        for tr in self.live + self.done:
            if tr.id == tid:
                return tr
        return None

    def resolve(self, tid: int | None) -> Track | None:
        """The live track that carries ``tid`` on (itself, or what it merged into), or None."""
        seen = set()
        while tid is not None and tid not in seen:
            seen.add(tid)
            tr = self.get(tid)
            if tr is not None and tr.ended is None:
                return tr
            tid = self.successor.get(tid)
        return None

    def update(self, t: float, peaks: list, default_v=(0.0, 0.0)) -> None:
        pred = []
        for tr in self.live:
            lt, x, y, _ = tr.last
            vx, vy = tr.velocity(default_v)
            # a track unseen for a while may only continue on a peak of comparable strength: a lost
            # 60 m/s storm must not grab a new 12 m/s cell nearby (a nudger, a daughter cell)
            pred.append((x + vx * (t - lt), y + vy * (t - lt), 6000.0 + 8.0 * (t - lt), 0.4 * tr.strength()))
        pairs = sorted((math.hypot(px - x, py - y), -w, k, n)
                       for n, (x, y, w) in enumerate(peaks)
                       for k, (px, py, reach, w_min) in enumerate(pred)
                       if math.hypot(px - x, py - y) <= reach and w >= w_min)
        used_t, used_p = set(), set()
        for _, _, k, n in pairs:
            if k in used_t or n in used_p:
                continue
            used_t.add(k)
            used_p.add(n)
            self.live[k].pts.append((t,) + tuple(peaks[n]))
        for n, pk in enumerate(peaks):
            if n not in used_p:
                self.live.append(Track(self._next, [(t,) + tuple(pk)]))
                self._next += 1
        keep, ended = [], []
        for tr in self.live:
            if t - tr.last[0] > TRACK_GAP:
                tr.ended = tr.last[0]
                self.done.append(tr)
                ended.append(tr)
            else:
                while tr.pts[0][0] < t - TRACK_KEEP and len(tr.pts) > 1:
                    tr.pts.pop(0)
                keep.append(tr)
        self.live = keep
        for tr in ended:                      # merged into the nearest live updraft, if one is close
            near = [(_sep(tr, o), o.id) for o in keep if o.born <= tr.last[0]]
            if near and min(near)[0] <= MERGE_KM * 1000.0:
                self.successor[tr.id] = min(near)[1]
        if len(self.done) > 200:
            del self.done[:-200]

    def strongest(self, t: float, min_age: float = 600.0) -> Track | None:
        """The live track with the strongest updraft over the last 5 min, preferring
        storms at least ``min_age`` old (a new cell's first surge is not the storm)."""
        recent = [tr for tr in self.live if t - tr.last[0] <= 120.0]
        old = [tr for tr in recent if t - tr.born >= min_age] or recent
        return max(old, key=Track.strength, default=None)

    def nearest(self, x: float, y: float, t: float, within: float = 25000.0) -> Track | None:
        recent = [tr for tr in self.live if t - tr.last[0] <= 120.0]
        best = min(recent, key=lambda tr: math.hypot(tr.last[1] - x, tr.last[2] - y), default=None)
        if best is None or math.hypot(best.last[1] - x, best.last[2] - y) > within:
            return None
        return best


def _sep(a: Track, b: Track) -> float:
    """Distance (m) between two tracks when the one that ended first was last seen."""
    t = min(a.last[0], b.last[0])
    pa = min(a.pts, key=lambda p: abs(p[0] - t))
    pb = min(b.pts, key=lambda p: abs(p[0] - t))
    return math.hypot(pa[1] - pb[1], pa[2] - pb[2])


def plan(target_xy, target_v, cell_v, lead: float, aim: float):
    """(start, meeting point) ground m of a cell moving at ``cell_v`` that meets a storm
    now at ``target_xy`` moving at ``target_v`` after ``lead`` s, passing ``aim`` m from
    its updraft (+ ahead, - behind).  None if the cell moves too much like the storm."""
    rx, ry = cell_v[0] - target_v[0], cell_v[1] - target_v[1]
    rel = math.hypot(rx, ry)
    if rel < 2.0:
        return None
    px, py = -ry / rel, rx / rel                    # across the cell's path relative to the storm
    if px * target_v[0] + py * target_v[1] < 0.0:   # + is the side the storm moves toward
        px, py = -px, -py
    mx = target_xy[0] + target_v[0] * lead + aim * px
    my = target_xy[1] + target_v[1] * lead + aim * py
    return (mx - cell_v[0] * lead, my - cell_v[1] * lead), (mx, my)


@dataclass
class Nudger:
    """A nudger cell sent at a storm, and what became of it."""
    id: int
    sent: float                   # model s
    meet: float                   # planned contact (model s)
    target: int                   # the storm's track
    aim: float                    # m, + ahead of the updraft / - behind
    motion: tuple                 # the cell's ground motion (m/s)
    start: tuple                  # ground m
    meet_point: tuple             # ground m
    start_id: int = 0             # its StormStart
    push_end: float = 0.0
    cell: int | None = None       # its own updraft's track, once it has one
    closest: float = math.inf     # closest approach of the two updrafts (m), and when
    closest_t: float = math.nan
    status: str = "pushing"       # pushing, on its own, merged, missed, no updraft, storm gone
    done: bool = False

    def expected(self, t: float):
        dt = t - self.sent
        return self.start[0] + self.motion[0] * dt, self.start[1] + self.motion[1] * dt

    def watch(self, tracks: Tracks, t: float) -> str | None:
        """Follow the cell and the storm; a sentence for the log when something happens."""
        if self.done:
            return None
        old = tracks.get(self.target)
        tgt = tracks.resolve(self.target)            # the storm's updraft now (its own updrafts may join)
        cell = tracks.get(self.cell)
        if cell is not None and tgt is cell:
            # the storm's updraft merged into the nudger's: one updraft left, on the nudger's side
            # (Nixon's "handoff").  Its old track may be gone from the list of ended ones by now
            # (only the last 200 are kept): then the merger is dated now
            self.done, self.status = True, "merged"
            ended = old.ended if old is not None and old.ended is not None else t
            return (f"nudger #{self.id}: merged with the storm at +{ended / 60:.0f} min (planned "
                    f"+{self.meet / 60:.0f}); the joined updraft carries on from the nudger's side; {self._closest()}")
        if tgt is not None and tgt.id != self.target:
            self.target = tgt.id                     # follow it
        if self.cell is None and t >= self.sent + 300.0:
            ex, ey = self.expected(t)
            cands = [tr for tr in tracks.live if tr is not tgt and tr.born >= self.sent - 60.0
                     and t - tr.last[0] <= 120.0 and math.hypot(tr.last[1] - ex, tr.last[2] - ey) <= 10000.0]
            if cands:
                cell = min(cands, key=lambda tr: math.hypot(tr.last[1] - ex, tr.last[2] - ey))
                self.cell = cell.id
        if cell is not None and tgt is not None:
            where = {p[0]: p for p in tgt.pts}
            for p in cell.pts:                               # the minutes both were seen
                q = where.get(p[0])
                if q is not None:
                    d = math.hypot(p[1] - q[1], p[2] - q[2])
                    if d < self.closest:
                        self.closest, self.closest_t = d, p[0]
        if self.status == "pushing" and t >= self.push_end:
            self.status = "on its own"
            return (f"nudger #{self.id}: push over, " + (f"its updraft {cell.last[3]:.0f} m/s, "
                    f"{_sep(cell, tgt) / 1000:.0f} km from the storm's" if cell is not None and tgt is not None
                    else "no deep updraft of its own yet"))
        if tgt is None:
            self.done, self.status = True, "storm gone"
            return f"nudger #{self.id}: the storm it was aimed at has no deep updraft any more"
        if cell is not None and cell.ended is not None:
            self.done = True
            into = tracks.resolve(cell.id)
            if into is tgt:
                self.status = "merged"
                return (f"nudger #{self.id}: merged into the storm at +{cell.ended / 60:.0f} min "
                        f"(planned +{self.meet / 60:.0f}); {self._closest()}")
            self.status = "missed"
            if into is not None:
                return (f"nudger #{self.id}: merged into another updraft (track {into.id}), "
                        f"{_sep(cell, tgt) / 1000:.0f} km from the storm's; {self._closest()}")
            return (f"nudger #{self.id}: its updraft died {_sep(cell, tgt) / 1000:.0f} km from the storm's; "
                    f"{self._closest()}")
        if cell is None and t > self.meet + 600.0:
            self.done, self.status = True, "no updraft"
            return f"nudger #{self.id}: never grew a deep updraft (12 m/s at 3-7 km) of its own"
        return None

    def _closest(self) -> str:
        if not math.isfinite(self.closest):
            return "never seen apart from the storm"
        return (f"closest approach of the updrafts {self.closest / 1000:.1f} km at "
                f"+{self.closest_t / 60:.0f} min")

    def to_json(self) -> dict:
        return {"id": self.id, "sent": self.sent, "meet": self.meet, "target": self.target, "aim": self.aim,
                "motion": list(self.motion), "start": list(self.start), "meet_point": list(self.meet_point),
                "push_end": self.push_end, "cell": self.cell, "status": self.status,
                "closest": None if math.isinf(self.closest) else self.closest,
                "closest_t": None if math.isnan(self.closest_t) else self.closest_t}
