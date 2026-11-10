"""Surface boundaries and the air masses behind them.

A boundary is a polyline (ground metres at t = 0) that moves rigidly at its
own speed and heading -- except where boundaries meet: their meeting vertices
are one point, a surface low, that moves at the mean motion of the boundaries
meeting there (or a set motion), so a cold front and a warm front stay joined
and pivot about the low instead of each drifting off on its own.  On one side
of a boundary lies another air mass: a
temperature difference, a depth that grows with distance from the boundary
(the frontal slope) up to a maximum, a moisture deficit and its own surface
wind.  Every column's target state is the sounding plus the anomaly of
whichever air mass covers it, so

* new patch area starts with the air masses in it (hydrostatically
  re-balanced by ArWen's own moist initializer),
* patch edges are WRF specified/relaxation boundaries that carry the moving
  air masses (time intervals rebuilt as the boundaries move),
* deep inside an air mass the low levels can be held to it ("keep the air
  mass"), so fronts stay fronts for hours, while near the boundary the
  model's own density-current dynamics decide where the front really is.

Along a boundary the air mass is not uniform: its depth and wind contrast
(what makes a boundary lift air) are concentrated in drifting hot spots, so
storms fire from a few places at a time instead of as one solid line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

KINDS = {
    # dT: air-mass temperature difference at the ground (K); depth/slope/min: air-mass depth (m);
    # dq: fractional moisture deficit; wind: the air mass's own surface wind (from, m/s);
    # speed/heading: boundary motion (toward, deg); hw: half width of the frontal zone (m)
    "cold": dict(label="Cold front", dT=-8.0, depth=1500.0, slope=0.012, depth_min=400.0, dq=0.35,
                 q_ratio=1.0, wind_speed=10.0, wind_dir=320.0, speed=10.0, heading=110.0, hw=6000.0,
                 variation=0.8, init_rate=1.5, keep=True, side_motion=-1),
    "warm": dict(label="Warm front", dT=-4.0, depth=1000.0, slope=0.004, depth_min=200.0, dq=0.15,
                 q_ratio=1.0, wind_speed=6.0, wind_dir=80.0, speed=5.0, heading=0.0, hw=12000.0,
                 variation=0.7, init_rate=0.8, keep=True, side_motion=1),
    "dry": dict(label="Dryline", dT=2.0, depth=2000.0, slope=0.03, depth_min=1200.0, dq=0.6,
                q_ratio=1.0, wind_speed=10.0, wind_dir=250.0, speed=3.0, heading=90.0, hw=4000.0,
                variation=0.8, init_rate=1.5, keep=True, side_motion=-1),
    "stationary": dict(label="Stationary front", dT=-5.0, depth=1200.0, slope=0.008, depth_min=300.0, dq=0.25,
                       q_ratio=1.0, wind_speed=6.0, wind_dir=60.0, speed=0.0, heading=0.0, hw=8000.0,
                       variation=0.7, init_rate=1.0, keep=True, side_motion=-1),
    # the front from a mature low's centre out to its triple point, where the cold front has caught up with the
    # warm front and lifted the warm sector off the ground: cool air on both sides, the colder behind it (a
    # cold-type occlusion), so a weaker contrast than either front, and it moves slowly, pivoting with the low
    "occluded": dict(label="Occluded front", dT=-3.0, depth=1200.0, slope=0.008, depth_min=300.0, dq=0.2,
                     q_ratio=1.0, wind_speed=8.0, wind_dir=320.0, speed=5.0, heading=60.0, hw=9000.0,
                     variation=0.6, init_rate=0.4, keep=True, side_motion=-1),
    "outflow": dict(label="Outflow boundary", dT=-4.0, depth=700.0, slope=0.03, depth_min=300.0, dq=0.1,
                    q_ratio=1.0, wind_speed=8.0, wind_dir=None, speed=8.0, heading=180.0, hw=3000.0,
                    variation=0.6, init_rate=1.0, keep=False, side_motion=-1),
}

def auto_speed(kind: str, dT: float, depth: float) -> float:
    """How fast a boundary left to itself moves (m/s): a cold front or outflow is a density current,
    c = 0.7 sqrt(g' H) with g' = g |dT| / 300 K (Benjamin 1968; Simpson 1997 -- a 1.5 km, 8 K colder air mass
    ~14 m/s, a 700 m, 4 K outflow ~7 m/s); a warm front retreats at about a third of that (the cold air
    ahead is only eroded); an occluded front, cool air against cool air, at half; a dryline drifts ~3 m/s; a
    stationary front stays put."""
    if kind == "stationary":
        return 0.0
    if kind == "dry":
        return 3.0
    c = 0.7 * math.sqrt(9.81 * abs(dT) / 300.0 * max(depth, 0.0))
    return round(c / 3.0 if kind == "warm" else c / 2.0 if kind == "occluded" else c, 1)


# the warm side of each kind (AirMasses.maps): conv = inflow toward the boundary (m/s), back = how far the
# low-level wind backs (deg) and jet how much it strengthens (m/s) over pre_depth (m), pre_width (m) ahead
# of it, twice that (up to 80 deg) within ~a third of the zone of it, where storms ride it; wobble = the amplitude (m, x variation) of the bulges that run along it
WARM_SIDE = {
    "cold": dict(conv=3.0, back=30.0, jet=4.0, pre_width=60000.0, pre_depth=900.0, wobble=6000.0),
    "warm": dict(conv=2.0, back=45.0, jet=5.0, pre_width=80000.0, pre_depth=800.0, wobble=4000.0),
    "dry": dict(conv=3.0, back=20.0, jet=3.0, pre_width=35000.0, pre_depth=1100.0, wobble=5000.0),
    "stationary": dict(conv=2.5, back=45.0, jet=5.0, pre_width=70000.0, pre_depth=800.0, wobble=5000.0),
    "occluded": dict(conv=2.5, back=35.0, jet=4.0, pre_width=50000.0, pre_depth=900.0, wobble=4000.0),
    "outflow": dict(conv=2.0, back=35.0, jet=3.0, pre_width=30000.0, pre_depth=700.0, wobble=3000.0),
}
for _k, _d in KINDS.items():
    _d.update(WARM_SIDE[_k])

for _k, _d in KINDS.items():                # the defaults are what each kind does on its own
    _d["speed"] = auto_speed(_k, _d["dT"], _d["depth"])

END_FADE = 150000.0     # air masses fade out over this distance beyond a boundary's end points (m)
JOIN_DIST = 8000.0      # vertices of different boundaries this close meet at a surface low (m)
FRONT_KINDS = ("cold", "warm", "stationary")   # a meeting point with one of these is drawn as an L ...
# ... unless an occluded front meets there too: then it is the triple point and the L is the occluded front's
# free end, the low's centre (join_at_lows)
FREE_ZONE = 40000.0     # "keep the air mass" is at full strength this far behind the boundary (m)
FRONT_KEEP = 0.5        # ... and at this fraction of it near the boundary


def _xp(a):
    """numpy or cupy, whichever owns ``a``."""
    if type(a).__module__.startswith("cupy"):
        import cupy
        return cupy
    return np


def smoothstep(a, b, x):
    xp = _xp(x)
    t = xp.clip((xp.asarray(x, dtype=xp.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def prof(z, H):
    """1 inside an air mass of depth H, blending to 0 between 0.7 H and 1.15 H."""
    xp = _xp(H)
    H = xp.maximum(H, 50.0)
    t = xp.clip((z - 0.7 * H) / (0.45 * H), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


@dataclass
class Boundary:
    kind: str
    points: list                      # [[x, y], ...] ground metres at t = 0
    dT: float = 0.0
    depth: float = 1000.0
    slope: float = 0.01
    depth_min: float = 300.0
    dq: float = 0.0
    q_ratio: float = 1.0
    wind_speed: float = 0.0
    wind_dir: float | None = None     # direction the air-mass wind blows FROM (deg); None = straight out of it
    speed: float = 0.0
    heading: float = 0.0              # direction the boundary moves TOWARD (deg)
    hw: float = 5000.0
    variation: float = 0.7
    init_rate: float = 0.0            # storms per hour per 100 km of boundary
    init_start: float = 0.0           # s
    init_end: float = 4 * 3600.0      # s
    keep: bool = True
    side: int = 1                     # +1: air mass on the left of the polyline's direction
    seed: int = 1
    id: int = 0
    vel: list | None = None           # per-point velocity [[vx, vy], ...] (m/s); None = its own motion
    auto_speed: bool = True           # speed from its air mass (auto_speed), not set by hand
    mode: str = "auto"                # storms it fires: "cells", "line" (a squall line), "auto" (from the shear)
    conv: float = 0.0                 # warm side (WARM_SIDE): inflow toward it (m/s)
    back: float = 0.0                 # ... low-level wind backed by (deg)
    jet: float = 0.0                  # ... and strengthened by (m/s)
    pre_width: float = 50000.0        # ... over this far ahead of it (m)
    pre_depth: float = 1000.0         # ... and this deep (m)
    wobble: float = 0.0               # bulges running along it (m, x variation)

    @classmethod
    def new(cls, kind: str, points, **kw) -> "Boundary":
        d = dict(KINDS[kind])
        side_motion = d.pop("side_motion")
        d.pop("label")
        d.update({k: v for k, v in kw.items() if v is not None and k in cls.__dataclass_fields__})
        if kw.get("speed") is None or kw.get("auto_speed"):
            d["speed"], d["auto_speed"] = auto_speed(kind, float(d["dT"]), float(d["depth"])), True
        else:
            d["auto_speed"] = False
        b = cls(kind=kind, points=[[float(x), float(y)] for x, y in points], **{
            k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if "side" not in kw:
            b.side = b.default_side(side_motion)
        return b

    # ------------------------------------------------------------ geometry
    def velocity(self):
        h = math.radians(self.heading)
        return self.speed * math.sin(h), self.speed * math.cos(h)

    def point_velocities(self) -> np.ndarray:
        n = len(self.points)
        if self.vel is not None and len(self.vel) == n:
            return np.asarray(self.vel, dtype=np.float64).reshape(n, 2)
        return np.tile(np.asarray(self.velocity(), dtype=np.float64), (n, 1))

    def points_at(self, t: float) -> np.ndarray:
        p = np.asarray(self.points, dtype=np.float64)
        return p + self.point_velocities() * t

    def bulge(self, s, t: float):
        """How far (m) the boundary bulges ahead of its drawn line at arc length s, time t: frontal waves
        ~180 and ~70 km long that run along it and grow and fade over a few hours, so one stretch surges
        ahead while another lags (the kinks storms fire at and ride).  0 without variation."""
        amp = self.wobble * self.variation
        if amp <= 0:
            return 0.0 * s
        ph, sc = _rng_phases(self.seed * 13 + 5, 4)
        grow = 0.75 + 0.25 * math.sin(t / (7200.0 * sc[3]) + ph[3])
        v = (0.65 * np.sin(2 * np.pi * (s / (180e3 * sc[0]) - t / (3 * 3600.0 * sc[1])) + ph[0])
             + 0.35 * np.sin(2 * np.pi * (s / (70e3 * sc[2]) + t / (2 * 3600.0)) + ph[1]))
        return amp * grow * v

    def length(self) -> float:
        p = np.asarray(self.points, dtype=np.float64)
        return float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0

    def default_side(self, side_motion: int) -> int:
        """Air mass behind a cold front / dryline / outflow (opposite the motion),
        ahead of a warm front (along the motion)."""
        p = np.asarray(self.points, dtype=np.float64)
        if len(p) < 2:
            return 1
        d = p[-1] - p[0]
        left = np.array([-d[1], d[0]])
        vx, vy = self.velocity()
        if abs(vx) + abs(vy) < 1e-6:
            # stationary: the cold side is the poleward one (north / left of an eastward line)
            return 1 if left[1] >= 0 else -1
        s = np.sign(left @ np.array([vx, vy])) or 1.0
        return int(s * side_motion)

    def wind_anomaly(self, u_env0: float, v_env0: float):
        """Surface wind of the air mass minus the environment's (m/s)."""
        if self.wind_speed <= 0:
            return 0.0, 0.0
        if self.wind_dir is None:
            # outflow: blowing straight out of the air mass across the boundary
            p = np.asarray(self.points, dtype=np.float64)
            d = p[-1] - p[0]
            n = np.array([d[1], -d[0]]) * self.side      # points from the air mass outward
            n = n / max(np.hypot(*n), 1e-9)
            ua, va = self.wind_speed * n
        else:
            r = math.radians(self.wind_dir)
            ua, va = -self.wind_speed * math.sin(r), -self.wind_speed * math.cos(r)
        return ua - u_env0, va - v_env0

    def to_json(self) -> dict:
        d = asdict(self)
        d["label"] = KINDS[self.kind]["label"]
        return d


# ------------------------------------------------------------------ surface lows
@dataclass
class SurfaceLow:
    """Where boundaries meet: their meeting vertices, moving as one point.  With an occluded front among them
    it is a triple point, and the low itself is the occluded front's free end (``centre``, one member)."""
    x: float                          # ground metres at t = 0
    y: float
    vx: float
    vy: float
    members: list                     # [(boundary index, point index), ...]
    front: bool                       # a front meets here (drawn as an L)
    triple: bool = False              # the triple point of an occluded low (no L: that is at its centre)
    centre: bool = False              # an occluded low's centre: the occluded front's free end

    def position(self, t: float):
        return self.x + self.vx * t, self.y + self.vy * t

    def to_json(self, boundaries: list[Boundary]) -> dict:
        return {"x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy, "front": self.front,
                "triple": self.triple, "centre": self.centre,
                "members": [[boundaries[bi].id, i] for bi, i in self.members]}


def find_lows(boundaries: list[Boundary], t: float = 0.0, join: float = JOIN_DIST) -> list[list]:
    """Groups of vertices of different boundaries within ``join`` of each other
    at time t: [[(boundary index, point index), ...], ...]."""
    V = [(bi, i, p) for bi, b in enumerate(boundaries) for i, p in enumerate(b.points_at(t))]
    used = [False] * len(V)
    groups = []
    for a in range(len(V)):
        if used[a]:
            continue
        g = [a]
        for c in range(a + 1, len(V)):
            if used[c] or any(V[k][0] == V[c][0] for k in g):
                continue
            if math.hypot(*(V[c][2] - V[a][2])) < join:
                g.append(c)
        if len(g) < 2:
            continue
        for k in g:
            used[k] = True
        groups.append([(V[k][0], V[k][1]) for k in g])
    return groups


def low_motion(cfg: dict | None):
    """None: surface lows move with their fronts; else the set (vx, vy) m/s
    from cfg = {"auto": bool, "speed": m/s, "heading": deg toward}."""
    cfg = cfg or {}
    if cfg.get("auto", True):
        return None
    h = math.radians(float(cfg.get("heading", 0.0)))
    sp = max(0.0, float(cfg.get("speed", 0.0)))
    return sp * math.sin(h), sp * math.cos(h)


def join_at_lows(boundaries: list[Boundary], t: float = 0.0, motion=None) -> list[SurfaceLow]:
    """Join boundaries that meet (vertices within JOIN_DIST at time t): the
    meeting vertices snap to one point, a surface low, which moves at the mean
    motion of the boundaries meeting there (or ``motion`` = (vx, vy) m/s); the
    rest of each boundary moves with its own motion, so the fronts pivot about
    the low.  A meeting point with an occluded front is a triple point; the
    occluded front's other end is then the low's centre, moving at ``motion``
    (else the occluded front's own), so the low can sit well away from where
    its fronts meet.  Everything stays where it is at time t (the stored t = 0
    points are re-anchored)."""
    now = [b.points_at(t) for b in boundaries]
    vel = [np.tile(np.asarray(b.velocity(), dtype=np.float64), (len(b.points), 1)) for b in boundaries]
    lows = []
    joined = set()
    for g in find_lows(boundaries, t):
        p = np.mean([now[bi][i] for bi, i in g], axis=0)
        m = (np.mean([boundaries[bi].velocity() for bi, _ in g], axis=0) if motion is None
             else np.asarray(motion, dtype=np.float64))
        for bi, i in g:
            now[bi][i] = p
            vel[bi][i] = m
            joined.add((bi, i))
        occl = any(boundaries[bi].kind == "occluded" for bi, _ in g)
        front = any(boundaries[bi].kind in FRONT_KINDS for bi, _ in g)
        lows.append(SurfaceLow(x=float(p[0] - m[0] * t), y=float(p[1] - m[1] * t), vx=float(m[0]),
                               vy=float(m[1]), members=g, front=front and not occl, triple=front and occl))
    # an occluded front joined at one end: its other end is the low's centre
    for bi, b in enumerate(boundaries):
        n = len(b.points)
        if b.kind != "occluded" or n < 2:
            continue
        for i, j in ((0, n - 1), (n - 1, 0)):
            if (bi, i) in joined or (bi, j) not in joined:
                continue
            m = np.asarray(motion, dtype=np.float64) if motion is not None else vel[bi][i]
            vel[bi][i] = m
            p = now[bi][i]
            lows.append(SurfaceLow(x=float(p[0] - m[0] * t), y=float(p[1] - m[1] * t), vx=float(m[0]),
                                   vy=float(m[1]), members=[(bi, i)], front=True, centre=True))
    for b, p, v in zip(boundaries, now, vel):
        own = np.asarray(b.velocity(), dtype=np.float64)
        b.vel = None if np.allclose(v, own) else v.tolist()
        b.points = (p - v * t).tolist()
    return lows


def signed_distance(P: np.ndarray, X: np.ndarray, Y: np.ndarray, normal: bool = False):
    """Distance to a polyline, positive on its left; plus how far beyond its
    end points (along the extended end segments) and the arc length (and with
    ``normal`` the unit left normal (nx, ny) of the nearest segment)."""
    n = len(P)
    best = np.full(X.shape, np.inf)
    sign = np.ones(X.shape)
    over = np.zeros(X.shape)
    along = np.zeros(X.shape)
    nx = np.zeros(X.shape)
    ny = np.ones(X.shape)
    s0 = 0.0
    for i in range(n - 1):
        ax, ay = P[i]
        bx, by = P[i + 1]
        abx, aby = bx - ax, by - ay
        L = max(math.hypot(abx, aby), 1.0)
        t0 = ((X - ax) * abx + (Y - ay) * aby) / (L * L)
        t = t0
        if i > 0:
            t = np.maximum(t, 0.0)
        if i < n - 2:
            t = np.minimum(t, 1.0)
        px, py = ax + t * abx, ay + t * aby
        d = np.hypot(X - px, Y - py)
        sg = np.where(abx * (Y - ay) - aby * (X - ax) >= 0.0, 1.0, -1.0)
        ov = np.zeros(X.shape)
        if i == 0:
            ov = ov + np.maximum(-t0, 0.0) * L
        if i == n - 2:
            ov = ov + np.maximum(t0 - 1.0, 0.0) * L
        closer = d < best
        best = np.where(closer, d, best)
        sign = np.where(closer, sg, sign)
        over = np.where(closer, ov, over)
        along = np.where(closer, s0 + t * L, along)
        if normal:
            nx = np.where(closer, -aby / L, nx)
            ny = np.where(closer, abx / L, ny)
        s0 += L
    if normal:
        return best * sign, over, along, nx, ny
    return best * sign, over, along


def _rng_phases(seed: int, n: int):
    r = np.random.default_rng(seed)
    return r.uniform(0, 2 * np.pi, n), r.uniform(0.7, 1.3, n)


def along_variation(b: Boundary, s, t):
    """Smooth -1..1 variation along the boundary (wavelengths ~140 and 60 km), drifting over ~1.5 h."""
    if b.variation <= 0:
        return np.zeros_like(s)
    ph, sc = _rng_phases(b.seed * 7 + 3, 3)
    v = (0.6 * np.sin(2 * np.pi * s / (140e3 * sc[0]) + ph[0] + t / 5400.0)
         + 0.4 * np.sin(2 * np.pi * s / (60e3 * sc[1]) + ph[1] - t / 3600.0))
    return np.clip(v * 1.3, -1.0, 1.0)


def hot_spots(b: Boundary, s, t, spacing=50e3):
    """0..1 bumps about every ``spacing`` along the boundary (random spacing and
    strength, ~14 km wide), re-drawn every ~80 min with a cross-fade."""
    period = 4800.0
    e = math.floor(t / period)
    fade = float(smoothstep(0.0, 1.0, t / period - e))
    acc = np.zeros_like(s)
    k0 = int(np.floor(np.nanmin(s) / spacing)) - 1 if np.size(s) else 0
    k1 = int(np.floor(np.nanmax(s) / spacing)) + 2 if np.size(s) else 0
    for epoch, wgt in ((e, 1.0 - fade), (e + 1, fade)):
        if wgt <= 0:
            continue
        for k in range(k0, k1):
            r = np.random.default_rng(abs(hash((b.seed, k, epoch))) % (2 ** 32))
            centre = (k + 0.5 + 0.35 * (r.random() - 0.5) * 2) * spacing
            amp = 0.35 + 0.65 * r.random()
            acc += wgt * amp * np.exp(-((s - centre) / (0.14 * spacing)) ** 2)
    return np.clip(acc, 0.0, 1.0)


MAP_NAMES = ("dT", "H", "dq", "Hq", "du", "dv", "Hw", "nudge", "wnudge", "hold", "pin", "pin_lo")   # wnudge: winds only


class WarmSectorTexture:
    """Small mesoscale variations of the open warm sector's boundary layer: a few tenths of a
    kelvin warmer or cooler, a few percent moister or drier (correlated with the warmth) and a
    900-1400 m deep mixed layer -- no wind of its own: its ~1 m/s wind field made convergence lines
    that, under a weak cap, set off cells all over the domain (37 updrafts by 75 min vs 11 without
    it, 2026-10-24).  Each is a sum of plane waves 25-120 km long in random directions that drift
    with the 0-1 km wind and change over 1.5-4 h, so the air every storm feeds on is a little
    different.  ``amp`` scales them all (1: sd ~0.2 K, ~2 % moisture; at most 0.6 K, 7 %)."""

    MODES = 12
    FIELDS = 5                      # temperature, moisture, u, v, depth

    def __init__(self, seed: int, amp: float = 1.0, flow=(0.0, 0.0)):
        r = np.random.default_rng(seed)
        n = self.MODES
        lam = np.exp(r.uniform(math.log(25e3), math.log(120e3), (self.FIELDS, n)))
        ang = r.uniform(0.0, 2.0 * math.pi, (self.FIELDS, n))
        self.kx = 2.0 * math.pi / lam * np.cos(ang)
        self.ky = 2.0 * math.pi / lam * np.sin(ang)
        self.om = 2.0 * math.pi / r.uniform(5400.0, 14400.0, (self.FIELDS, n)) * r.choice([-1.0, 1.0], (self.FIELDS, n))
        self.ph = r.uniform(0.0, 2.0 * math.pi, (self.FIELDS, n))
        a = np.sqrt(lam)                                     # the longer waves carry more
        self.a = a * math.sqrt(2.0) / np.sqrt((a * a).sum(axis=1, keepdims=True))   # unit variance
        self.amp = float(amp)
        self.flow = (float(flow[0]), float(flow[1]))

    def _field(self, c: int, X, Y, t: float):
        xp = _xp(X)
        Xs, Ys = X - self.flow[0] * t, Y - self.flow[1] * t
        acc = 0.0 * Xs
        for k in range(self.MODES):
            acc = acc + self.a[c, k] * xp.cos(self.kx[c, k] * Xs + self.ky[c, k] * Ys + self.om[c, k] * t + self.ph[c, k])
        return acc

    def maps(self, X, Y, t: float) -> dict:
        xp = _xp(X)
        g = [self._field(c, X, Y, t) for c in range(self.FIELDS)]
        moist = 0.6 * g[0] + 0.8 * g[1]
        A = self.amp
        zero = 0.0 * g[2]
        return {"dT": xp.clip(0.25 * A * g[0], -0.6 * A, 0.6 * A),
                "dq": xp.clip(-0.025 * A * moist, -0.07 * A, 0.07 * A),
                "du": zero, "dv": zero,
                "H": 1150.0 + 250.0 * xp.clip(g[4], -1.0, 1.0)}

    def favour(self, X, Y, t: float):
        """0..1: how ready the air is for a storm (warm and moist toward 1)."""
        warm = self._field(0, X, Y, t)
        return smoothstep(-1.5, 1.5, 0.5 * (1.6 * warm + 0.8 * self._field(1, X, Y, t)))


class WindBrush:
    """A hodograph brush stroke: under it the low-level wind is backed by ``back`` degrees (veered when
    negative) and strengthened by ``add`` m/s along its new direction, over ``depth`` m (fading out by
    1.15 x depth, as an air mass's wind does).  Full within 40 % of ``radius`` of the stroke, feathered to
    nothing at ``radius``, and tapered toward both of the stroke's ends over ~2 radii (a third of a short
    stroke), so the changed winds blend into the sounding's."""

    def __init__(self, d: dict, env):
        self.P = np.asarray(d.get("points") or [], dtype=np.float64).reshape(-1, 2)
        self.radius = max(float(d.get("radius", 30000.0)), 2000.0)
        self.depth = min(max(float(d.get("depth", 1500.0)), 200.0), 6000.0)
        back, add = math.radians(float(d.get("back", 0.0))), float(d.get("add", 0.0))
        low = np.asarray(env.z_h) <= 0.5 * self.depth
        if not low.any():
            low = np.zeros(len(env.z_h), bool); low[0] = True
        u0, v0 = float(np.mean(env.u_ground[low])), float(np.mean(env.v_ground[low]))
        ur = u0 * math.cos(back) - v0 * math.sin(back)              # rotated counter-clockwise: backed
        vr = u0 * math.sin(back) + v0 * math.cos(back)
        sp = math.hypot(ur, vr)
        ex, ey = (ur / sp, vr / sp) if sp > 0.5 else (0.0, 1.0)
        self.du, self.dv = ur - u0 + add * ex, vr - v0 + add * ey
        seg = np.hypot(*np.diff(self.P, axis=0).T) if len(self.P) > 1 else np.zeros(0)
        self.length = float(seg.sum())

    def weight(self, X, Y):
        if len(self.P) == 0:
            return np.zeros(np.shape(X))
        if len(self.P) == 1:
            d = np.hypot(X - self.P[0, 0], Y - self.P[0, 1]); along = np.zeros(np.shape(X)); L = 1.0
        else:
            d, _, along = signed_distance(self.P, X, Y)
            d = np.abs(d)
            # past the stroke's ends: the distance to the end point
            a0, a1 = self.P[0], self.P[-1]
            d = np.where(along <= 0.0, np.hypot(X - a0[0], Y - a0[1]), d)
            d = np.where(along >= self.length, np.hypot(X - a1[0], Y - a1[1]), d)
            L = self.length
        w = 1.0 - smoothstep(0.4 * self.radius, self.radius, d)
        if len(self.P) > 1:
            taper = min(2.0 * self.radius, L / 3.0)
            a = np.clip(along, 0.0, L)
            w = w * smoothstep(-0.25 * taper, taper, a) * smoothstep(-0.25 * taper, taper, L - a)
        return w


class AirMasses:
    """The boundaries of a run and the anomaly fields they imply (and the warm sector's own
    small variations, ``texture``, wherever no air mass covers it, and the hodograph brushes' winds)."""

    def __init__(self, boundaries: list[Boundary], env, texture: WarmSectorTexture | None = None,
                 brushes: list | None = None, pockets=None):
        self.boundaries = list(boundaries)
        self.pockets = pockets if pockets else None     # pockets.SoundingPockets: more soundings eased in
        self.env = env
        self.texture = texture
        u0, v0 = float(env.u_ground[0]), float(env.v_ground[0])
        self._wind = [b.wind_anomaly(u0, v0) for b in self.boundaries]
        self._pre = [self._backed(b, env) for b in self.boundaries]
        self._near = [self._backed(b, env, near=True) for b in self.boundaries]
        self.brushes = [WindBrush(d, env) for d in (brushes or []) if d.get("points")]

    @staticmethod
    def _backed(b: Boundary, env, near: bool = False):
        """The warm side's low-level wind change ahead of a boundary (du, dv m/s): the sounding's wind over
        the lower half of pre_depth backed by ``back`` and strengthened by ``jet`` -- the isallobaric, pressure-
        trough-driven backing ahead of fronts that turns the low-level hodograph and piles up 0-1 km SRH;
        ``near``: right by the boundary, backed twice as far (at most 80 deg) with twice the jet (Markowski
        et al. 1998: SRH near boundaries often double the ambient)."""
        if b.back == 0 and b.jet == 0:
            return 0.0, 0.0
        back, jet = (min(2.0 * b.back, 80.0), 2.0 * b.jet) if near else (b.back, b.jet)
        low = np.asarray(env.z_h) <= 0.5 * max(b.pre_depth, 200.0)
        if not low.any():
            low = np.zeros(len(env.z_h), bool); low[0] = True
        u0, v0 = float(np.mean(env.u_ground[low])), float(np.mean(env.v_ground[low]))
        a = math.radians(back)
        ur, vr = u0 * math.cos(a) - v0 * math.sin(a), u0 * math.sin(a) + v0 * math.cos(a)
        sp = math.hypot(ur, vr)
        ex, ey = (ur / sp, vr / sp) if sp > 0.5 else (0.0, 1.0)
        return ur - u0 + jet * ex, vr - v0 + jet * ey

    def __bool__(self):
        return bool(self.boundaries) or self.texture is not None or bool(self.brushes) or self.pockets is not None

    def maps(self, X: np.ndarray, Y: np.ndarray, t: float) -> dict:
        """Per-column anomaly parameters at ground points (X, Y) (m) and time t (s)."""
        shape = np.shape(X)
        sT = np.zeros(shape)
        sH = np.zeros(shape)
        sQ = np.zeros(shape)
        sHq = np.zeros(shape)
        sU = np.zeros(shape)
        sV = np.zeros(shape)
        sw = np.zeros(shape)
        nud = np.zeros(shape)
        wnud = np.zeros(shape)
        hold = np.zeros(shape)
        pin = np.zeros(shape)
        pin_lo = np.zeros(shape)
        pU, pV, pH, pW = (np.zeros(shape) for _ in range(4))      # the warm sides ahead of the boundaries
        for b, (du0, dv0), (bu, bv), (nu, nv) in zip(self.boundaries, self._wind, self._pre, self._near):
            P = b.points_at(t)
            if len(P) < 2:
                continue
            d, over, along, nx, ny = signed_distance(P, X, Y, normal=True)
            d = d * b.side + b.bulge(along, t)
            ends = 1.0 - smoothstep(0.0, END_FADE, over)
            w = smoothstep(-b.hw, b.hw, d) * ends
            if b.pre_width > 0 and (b.conv or b.back or b.jet):
                # ahead of it: backed, stronger low-level wind across the zone, and inflow converging on
                # the boundary, strongest within ~a third of the zone (where storms fire and latch on)
                ahead = -d
                pf = (1.0 - smoothstep(0.35 * b.pre_width, b.pre_width, ahead)) * smoothstep(-2.0 * b.hw, 0.0, ahead) * ends
                cg = pf * np.exp(-np.maximum(ahead, 0.0) / (0.3 * b.pre_width))
                pU += pf * bu + cg * (nu - bu + b.conv * b.side * nx)
                pV += pf * bv + cg * (nv - bv + b.conv * b.side * ny)
                pH += pf * b.pre_depth
                pW += pf
            if not np.any(w > 0):
                continue
            fade = 1.0 - smoothstep(30e3, 90e3, d)              # variation fades deep inside
            av = b.variation * along_variation(b, along, t) * fade
            if b.variation > 0:
                # deeper air mass and a little more wind contrast at the hot spots
                hs = hot_spots(b, along, t)
                dep = 1.0 + fade * b.variation * (1.6 * hs - 0.5)
                wnd = 1.0 + fade * b.variation * (0.5 * hs - 0.2)
            else:
                dep = wnd = np.ones(shape)
            Hmin = b.depth_min * dep
            H = np.clip(b.slope * np.maximum(d + b.hw, 0.0) * (1.0 + 0.3 * av), Hmin, np.maximum(b.depth, Hmin))
            sT += w * b.dT * (1.0 + 0.25 * av)
            sQ += w * b.dq
            sU += w * du0 * wnd
            sV += w * dv0 * wnd
            sH += w * H
            sHq += w * H * b.q_ratio
            sw += w
            if b.keep:
                # held to the air mass: gently near the boundary (so the front follows
                # its set motion), firmly deep inside (so it stays an air mass for hours)
                deep = smoothstep(FREE_ZONE, FREE_ZONE + 40e3, d)
                nud = np.maximum(nud, w * (FRONT_KEEP + (1.0 - FRONT_KEEP) * deep))
                # and its frontal zone pinned to the set line: warm air blown across it is cooled back
                hold = np.maximum(hold, w * (1.0 - deep))
                if float(b.speed) == 0.0 and b.vel is None:
                    # a stationary one both ways: its cool air pushed across the line is warmed back too
                    # (down to its own dT; a storm's colder outflow is left alone)
                    zone = smoothstep(-b.hw, 0.0, d) * (1.0 - deep) * ends
                    pin = np.maximum(pin, zone)
                    pin_lo = np.where(zone > 0, np.minimum(pin_lo, min(float(b.dT), 0.0)), pin_lo)
        cover = np.clip(sw, 0.0, 1.0)                # how much of an air mass is there
        ov = 1.0 / np.maximum(sw, 1.0)          # where air masses overlap (a cold sector), blend them
        dT, dq, du, dv = sT * ov, sQ * ov, sU * ov, sV * ov
        if self.texture is not None:
            # the open warm sector's own variations, fading out under the air masses
            X = np.asarray(X, dtype=np.float64)
            Y = np.asarray(Y, dtype=np.float64)
            tx = self.texture.maps(X, Y, t)
            op = np.clip(1.0 - sw, 0.0, 1.0)
            dT, dq, du, dv = dT + op * tx["dT"], dq + op * tx["dq"], du + op * tx["du"], dv + op * tx["dv"]
            sH, sHq, sw = sH + op * tx["H"], sHq + op * tx["H"], sw + op
        iw = 1.0 / np.maximum(sw, 1e-6)
        Hw = np.maximum(sH * iw, 100.0)
        if self.brushes:
            # the hodograph brushes' winds on top, over their own depth, held there like a kept air mass
            X = np.asarray(X, dtype=np.float64)
            Y = np.asarray(Y, dtype=np.float64)
            bU, bV, bH, bW = (np.zeros(shape) for _ in range(4))
            for br in self.brushes:
                w = br.weight(X, Y)
                bU += w * br.du; bV += w * br.dv; bH += w * br.depth; bW += w
            du, dv = du + bU, dv + bV
            Hw = np.maximum((sH + bH) / np.maximum(sw + bW, 1e-6), 100.0)
            nud = np.maximum(nud, 0.6 * np.clip(bW, 0.0, 1.0))
        if np.any(pW > 0):
            # the warm sides' winds where no air mass covers them, held there like a kept air mass
            op = 1.0 - cover
            pn = 1.0 / np.maximum(pW, 1.0)
            du, dv = du + op * pU * pn, dv + op * pV * pn
            pw = np.clip(pW, 0.0, 1.0) * op
            Hw = (Hw * (1.0 - pw) + pw * pH / np.maximum(pW, 1e-6))
            wnud = np.maximum(wnud, 0.6 * pw)            # the winds only: storms keep their cold pools
        return {
            "dT": np.clip(dT, -20.0, 10.0),
            "H": np.maximum(sH * iw, 100.0),
            "dq": np.clip(dq, -0.25, 0.92),
            "Hq": np.maximum(sHq * iw, 100.0),
            "du": du, "dv": dv,
            "Hw": Hw,
            "nudge": nud,
            "wnudge": np.maximum(nud, wnud),
            "hold": np.clip(hold, 0.0, 1.0),
            "pin": np.clip(pin, 0.0, 1.0),
            "pin_lo": pin_lo,
        }

    def weight(self, X, Y, t: float):
        """0..1: how much of an air mass covers ground points (X, Y) at time t
        (0 in the open warm sector, 1 inside a cold / cool / dry air mass)."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        w = np.zeros(X.shape)
        for b in self.boundaries:
            P = b.points_at(t)
            if len(P) < 2:
                continue
            d, over, along = signed_distance(P, X, Y)
            w = np.maximum(w, smoothstep(-b.hw, b.hw, d * b.side + b.bulge(along, t)) * (1.0 - smoothstep(0.0, END_FADE, over)))
        return w

    @staticmethod
    def stack(m: dict) -> np.ndarray:
        return np.stack([m[k] for k in MAP_NAMES]).astype(np.float32)

    # ------------------------------------------------------- 3-D anomalies
    @staticmethod
    def fields_3d(m: dict, z: np.ndarray):
        """theta' (nz,...), qv factor (nz,...), du, dv (nz,...) for maps m at
        heights z (nz,).  The maps may be numpy or cupy arrays."""
        xp = _xp(m["dT"])
        zz = xp.asarray(np.asarray(z, dtype=np.float64)).reshape((-1,) + (1,) * m["dT"].ndim)
        pT = prof(zz, m["H"][None])
        pq = prof(zz, m["Hq"][None])
        pw = prof(zz, m["Hw"][None])
        return m["dT"][None] * pT, 1.0 - m["dq"][None] * pq, m["du"][None] * pw, m["dv"][None] * pw

    @staticmethod
    def to_device(m: dict) -> dict:
        import cupy as cp
        return {k: cp.asarray(v, dtype=cp.float64) for k, v in m.items()}

    def total_length_km(self) -> float:
        return sum(b.length() for b in self.boundaries) / 1000.0


def boundary_from_json(d: dict) -> Boundary:
    kind = d.get("kind", "cold")
    pts = d.get("points") or []
    kw = {k: d[k] for k in Boundary.__dataclass_fields__ if k in d and k not in ("kind", "points")}
    b = Boundary.new(kind, pts, **kw)
    if "side" in d:
        b.side = 1 if float(d["side"]) >= 0 else -1
    return b
