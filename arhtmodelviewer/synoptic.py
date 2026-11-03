"""Synoptic mode: the whole country's weather, simulated, instead of one sounding.

Storms then form where the simulated atmosphere makes them: ahead of troughs, on fronts and
drylines, in warm sectors the Gulf has moistened under an elevated mixed layer off the Mexican
Plateau, as the afternoon heats and the evening low-level jet ramps up.  Three layers, coarse to
fine:

1. The flow (qg.py): a two-layer quasi-geostrophic model of the westerlies on a 75 km grid.
   Troughs, ridges and surface cyclones grow out of the unstable jet, deepen and move on their
   own.  A long free run of it (the "bank", built once and cached) is the model's climate; a run
   starts from the day in it whose cyclone sits where the chosen scenario wants it, shifted east
   or west to put it there.
2. The surface (this module): a 25 km grid over the contiguous US carrying the boundary layer's
   potential temperature and moisture and the elevated mixed layer's potential temperature,
   advected by the low-level and 3 km winds.  Sources from the land cover: the Gulf and the
   oceans (sea-surface temperatures by month), moist farmland, the dry West, the hot Mexican
   Plateau and desert Southwest (the EML).  Fronts and drylines are where its gradients sharpen;
   surface pressure is the flow's plus the boundary layer's own warmth (thermal lows, frontal and
   lee troughs); winds are geostrophic, slowed and turned by friction, plus the isallobaric wind
   toward falling pressure.
3. Columns: a full sounding anywhere at any time: a mixed layer that grows through the day
   under the cap, a nocturnal inversion, the EML, a free troposphere matched to the flow's
   temperature, humidity from the large-scale ascent, an Ekman-turned boundary layer, a nocturnal
   low-level jet over the Plains and the jet stream aloft.

The storm patches (ArWen) see this through SynopticAirMass, a drop-in for boundaries.AirMasses:
new patch area starts in the simulated air, the patch edges are driven by it (WRF specified
boundaries), and the low levels inside are held gently to its diurnal cycle away from storms.
Storms are started by the simulation (sim._synoptic_step) where CAPE, CIN and lift allow it --
the 1 km patches exist only around storms, so convective initiation itself is parameterized.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import math
import threading

import numpy as np

from . import PROJECT, qg
from .sounding import (CP, EPS, G, KAPPA, P0, RD, Sounding, _moist_T, bunkers, dewpoint_from_qv, qv_sat,
                       qv_sat_ice, thetae_bolton)

CACHE = PROJECT / "runtime" / "cache"
BANK_VERSION = 8
BANK_FILE = CACHE / f"synoptic_bank_v{BANK_VERSION}.npz"
BANK_SPINUP_DAYS = 45
BANK_DAYS = 100
SPIN = 2.0 * 86400.0            # s the surface runs before the start, so moisture return and the EML exist

OMEGA = 7.292e-5
RHO = 1.2
DT = 900.0                      # s: one step of the whole synoptic model (= qg.QGModel.dt)
DX = 25e3                       # m: surface grid
GAMMA = 6.5e-3                  # free-troposphere lapse rate (K/m)
STAB = 3.3e-3                   # dry minus environmental lapse rate (K/m): QG vertical motion from heating
TH0 = 300.0
H_THERMAL = 1200.0              # m: depth over which the boundary layer's warmth lowers surface pressure
AHEAD = 7200.0                  # s the model runs ahead of the storms (their edge tables look 1 h ahead)

# internal column levels (m AGL): 50 m to 3 km, then 200 m to 22 km
ZC = np.concatenate([np.arange(0.0, 3000.0, 50.0), np.arange(3000.0, 22001.0, 200.0)])
# the maps' columns (CAPE, CIN, shear, helicity on the 50 km grid): 100 m to 3 km, then 400 m to 16 km
ZD = np.concatenate([np.arange(0.0, 3000.0, 100.0), np.arange(3000.0, 16001.0, 400.0)])

# moist physics (grid scale): the mid-troposphere's moisture is carried at ~4 km (620 hPa)
Z4, P4 = 4000.0, 62000.0
LV_CP = 2.501e6 / 1004.5          # K per (kg/kg) condensed
MID_MASS = 3000.0                 # kg/m^2 of air in the moist mid-level layer (~1.5-5.5 km)
# the Plains low-level jet, as a fraction of the 550 m geostrophic wind added along it: by day (the sloping
# heated boundary layer), at night (the inertial oscillation) and under a trough's large-scale ascent
LLJ_DAY, LLJ_NIGHT, LLJ_ASCENT = 0.25, 0.65, 0.5
RH_MAX = 0.80                     # most relative humidity (over water; over ice below 0 C) above the boundary layer
COL_MASS = 1.0e4                  # kg/m^2 of a whole column: latent heat warms it by LV_CP * rain / COL_MASS

# by month: jet latitude, Gulf SST (C), boundary-layer theta offset (K), EML source theta (K), 500 hPa T at the jet (K)
SEASON = {1: (33, 21.0, -8, 306, 247), 2: (34, 21.0, -6, 307, 248), 3: (36, 22.0, -3, 309, 251),
          4: (38, 23.0, 0, 312, 254), 5: (40, 25.0, 2, 314, 256), 6: (43, 27.5, 5, 315, 259),
          7: (46, 29.0, 6, 315, 261), 8: (46, 29.5, 6, 315, 261), 9: (44, 28.5, 4, 313, 259),
          10: (40, 26.5, 0, 311, 255), 11: (37, 24.0, -3, 309, 251), 12: (34, 22.0, -6, 307, 248)}

# scenarios: where the strongest surface low is at the peak time relative to the map centre (km), and which
# of the candidate lows there to take (0 = the weakest .. 1 = the deepest)
SCENARIOS = {
    "outbreak": dict(label="Tornado outbreak: a deepening low to the northwest, the centre in its warm sector",
                     low=(-550.0, 450.0), strength=0.85, gulf_open=True, veering=True),
    "squall": dict(label="Squall line: a strong cold front sweeping across the centre",
                   low=(-300.0, 600.0), strength=0.75, gulf_open=True),
    "dryline": dict(label="Dryline day: a weaker low far to the north, storms on the dryline",
                    low=(-750.0, 850.0), strength=0.45, gulf_open=True, veering=True),
    "triple": dict(label="Triple point & warm front: the low just west, storms on the warm front and at the triple point",
                   low=(-350.0, 200.0), strength=0.75, gulf_open=True, veering=True),
    "dixie": dict(label="Dixie Alley / cool season: a strong jet, high shear, modest instability (start in Mar or Nov, "
                        "centre in MS or AL)", low=(-250.0, 950.0), strength=0.95, gulf_open=True, veering=True),
    "derecho": dict(label="Ridge-riding MCS / derecho: a weak low far north, big instability, storms that grow into a line",
                    low=(0.0, 1300.0), strength=0.3, gulf_open=True),
    "random": dict(label="Random weather: any day of the model's climate", low=None, strength=None,
                   gulf_open=False),
}

DEFAULTS = {"on": False, "scenario": "outbreak", "peak_h": 6.0, "area_km": 0.0, "rate": 1.5,
            "max_storms": 10, "spacing_km": 45.0, "seed": None,
            "source": "sim",            # "sim": the simulated weather; "hrrr": NOAA's HRRR (hrrr.py)
            "hrrr": None}               # {"cycle": ISO time, "start_fh": first forecast hour, "hours": how many}


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ------------------------------------------------------------------ the bank
_BANK = {}
_BANK_LOCK = threading.Lock()


def bank(progress=None) -> dict:
    """The QG model's own climate: daily states of a long free run after a spin-up, and every 6 h the
    surface cyclones in it (index, i, j, depth, strength).  Built once (~1.5 min) and cached."""
    with _BANK_LOCK:
        if "b" in _BANK:
            return _BANK["b"]
        b = None
        if BANK_FILE.is_file():
            try:
                with np.load(BANK_FILE) as z:
                    b = {k: z[k] for k in z.files}
                if int(b["version"]) != BANK_VERSION:
                    b = None
            except Exception:
                b = None
        if b is None:
            b = _build_bank(progress)
            try:
                CACHE.mkdir(parents=True, exist_ok=True)
                np.savez(BANK_FILE, **b)
            except OSError:
                pass
        _BANK["b"] = b
        return b


def _build_bank(progress=None) -> dict:
    m = qg.QGModel()
    m.init_rest(seed=11)
    per_day = int(round(86400.0 / m.dt))
    total = BANK_SPINUP_DAYS + BANK_DAYS
    for d in range(BANK_SPINUP_DAYS):
        m.step(per_day)
        if progress:
            progress(d / total, "spinning up the jet stream")
    q1, q2, lows, psis, psi1s = [], [], [], [], []
    for d in range(BANK_DAYS):
        q1.append(m.q1h.copy())
        q2.append(m.q2h.copy())
        done = 0
        for k in range(4):
            if k:
                m.step(per_day // 4)
                done += per_day // 4
            fl = m.fields()
            ps = m.surface_psi(fl)
            psis.append(ps.astype(np.float32))
            psi1s.append(fl["psi1"].astype(np.float32))           # the upper flow: scoring a day's hodograph
            anom = ps - ps.mean(axis=1, keepdims=True)
            r = 8                                       # where the map's L is: the full field's lowest point nearby
            for i, j, depth in qg.find_lows(anom, m.dx, radius=700e3, min_depth=5e6):
                jj = (j + np.arange(-r, r + 1)) % m.ny
                ii = (i + np.arange(-r, r + 1)) % m.nx
                w = ps[np.ix_(jj, ii)]
                a, c = np.unravel_index(int(np.argmin(w)), w.shape)
                lows.append((d * 4 + k, i, j, depth, depth, (i + c - r) % m.nx, (j + a - r) % m.ny))
        m.step(per_day - done)
        if progress:
            progress((BANK_SPINUP_DAYS + d) / total, "recording the model's weather")
    return {"version": np.array(BANK_VERSION), "q1": np.array(q1), "q2": np.array(q2),
            "lows": np.array(lows, dtype=np.float64).reshape(-1, 7), "psis": np.array(psis), "psi1s": np.array(psi1s), "dt": np.array(m.dt)}


# ------------------------------------------------------------------ helpers
def _sample(F, fx, fy):
    """Bilinear sample of F (..., ny, nx) at fractional indices (clamped to the grid); flat gathers."""
    ny, nx = F.shape[-2:]
    fx, fy = np.broadcast_arrays(np.asarray(fx, np.float64), np.asarray(fy, np.float64))
    shape = fx.shape
    fx = np.clip(fx.ravel(), 0.0, nx - 1.000001)
    fy = np.clip(fy.ravel(), 0.0, ny - 1.000001)
    i0 = fx.astype(np.int64)
    j0 = fy.astype(np.int64)
    tx, ty = fx - i0, fy - j0
    idx = j0 * nx + i0
    flat = F.reshape(F.shape[:-2] + (ny * nx,))
    a, b = flat[..., idx], flat[..., idx + 1]
    c, d = flat[..., idx + nx], flat[..., idx + nx + 1]
    lo = a + (b - a) * tx
    out = lo + (c + (d - c) * tx - lo) * ty
    return out.reshape(F.shape[:-2] + shape)

def _sample_periodic(F, fx, fy):
    ny, nx = F.shape[-2:]
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    tx, ty = fx - i0, fy - j0
    i0 %= nx
    j0 %= ny
    i1, j1 = (i0 + 1) % nx, (j0 + 1) % ny
    return ((F[..., j0, i0] * (1 - tx) + F[..., j0, i1] * tx) * (1 - ty)
            + (F[..., j1, i0] * (1 - tx) + F[..., j1, i1] * tx) * ty)


def _smooth(a, sigma):
    """Gaussian smoothing (sigma in cells), edges reflected."""
    if sigma <= 0:
        return a
    r = max(1, int(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    out = a
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (r, r)
        p = np.pad(out, pad, mode="reflect")
        n = out.shape[ax]
        acc = np.zeros_like(out)
        for k in range(2 * r + 1):
            sl = [slice(None), slice(None)]
            sl[ax] = slice(k, k + n)
            acc += w[k] * p[tuple(sl)]
        out = acc
    return out


def _smooth_coarse(a, sigma, k=4):
    """Wide Gaussian smoothing (sigma in cells) done on a k-times coarser copy, then interpolated back: the
    same result as _smooth for sigma >> k at a fraction of its cost."""
    ny, nx = a.shape
    py, px = (-ny) % k, (-nx) % k
    b = np.pad(a, ((0, py), (0, px)), mode="edge")
    c = b.reshape(b.shape[0] // k, k, b.shape[1] // k, k).mean(axis=(1, 3))
    c = _smooth(c, sigma / k)
    fx = (np.arange(nx) + 0.5) / k - 0.5
    fy = (np.arange(ny) + 0.5) / k - 0.5
    FX, FY = np.meshgrid(fx, fy)
    return _sample(c, FX, FY)


class _Bilinear:
    """Bilinear interpolation at fixed fractional indices, precomputed (periodic or clamped)."""

    def __init__(self, shape, fx, fy, periodic):
        ny, nx = shape
        fx = np.asarray(fx, np.float64)
        fy = np.asarray(fy, np.float64)
        if not periodic:
            fx = np.clip(fx, 0.0, nx - 1.000001)
            fy = np.clip(fy, 0.0, ny - 1.000001)
        i0, j0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        tx, ty = fx - i0, fy - j0
        i1, j1 = i0 + 1, j0 + 1
        if periodic:
            i0, i1, j0, j1 = i0 % nx, i1 % nx, j0 % ny, j1 % ny
        self.idx = [(j * nx + i).ravel() for j, i in ((j0, i0), (j0, i1), (j1, i0), (j1, i1))]
        self.w = [((1 - tx) * (1 - ty)).ravel(), (tx * (1 - ty)).ravel(), ((1 - tx) * ty).ravel(), (tx * ty).ravel()]
        self.shape = np.shape(fx)

    def __call__(self, F):
        flat = F.reshape(F.shape[:-2] + (-1,))
        out = sum(flat[..., i] * w for i, w in zip(self.idx, self.w))
        return out.reshape(F.shape[:-2] + self.shape)


def _grad(a, d):
    gy, gx = np.gradient(a, d)
    return gx, gy


_MOIST = {}


def _moist_lookup(the, p):
    """Temperature (K) on the saturated pseudo-adiabat of theta_e ``the`` at pressure ``p`` (Pa): bilinear in
    a table of sounding._moist_T (theta_e 250-420 K by 0.5, 50-1080 hPa by 5), built once."""
    if "t" not in _MOIST:
        th = np.arange(250.0, 420.01, 0.5)
        pp = np.arange(108000.0, 4999.0, -500.0)
        tab = np.empty((len(pp), len(th)))
        guess = np.full(len(th), 300.0)
        for k, pk in enumerate(pp):
            for _ in range(4 if k == 0 else 1):
                guess = _moist_T(th, np.full(len(th), pk), guess)
            tab[k] = guess
        _MOIST["t"] = (th, pp[::-1].copy(), tab[::-1].copy())
    th, pp, tab = _MOIST["t"]
    fi = (np.clip(the, th[0], th[-1]) - th[0]) / (th[1] - th[0])
    fj = (np.clip(p, pp[0], pp[-1]) - pp[0]) / (pp[1] - pp[0])
    return _sample(tab, fi, fj)


def _local_hour(hour_utc, lon):
    return (hour_utc + np.asarray(lon) / 15.0) % 24.0


def _heating(h):
    """0..1: how far the day's heating of the mixed layer has gone (its daily mean ~0.35)."""
    return np.where(h < 8.0, 0.45 * (1.0 - h / 8.0),
                    np.where(h < 16.0, _ss((h - 8.0) / 8.0), 1.0 - 0.55 * _ss((h - 16.0) / 8.0)))


def _night(h):
    """0..1: the nocturnal surface inversion (full from 21 to 6 local)."""
    return np.where(h >= 18.0, _ss((h - 18.0) / 3.0), np.where(h < 6.0, 1.0, 1.0 - _ss((h - 6.0) / 3.0)))


def _jet_phase(h):
    """0..1: the evening transition and the nocturnal low-level jet (inertial oscillation): from ~2 h before
    sunset, as the surface layer stabilizes and decouples, to its peak after midnight."""
    return np.where(h >= 17.0, _ss((h - 17.0) / 5.0), np.where(h < 6.0, 1.0, 1.0 - _ss((h - 6.0) / 4.0)))


def _rotate(u, v, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return u * c - v * s, u * s + v * c


# ------------------------------------------------------------------ contouring
_EDGES = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)],
          9: [(0, 2)], 11: [(1, 2)], 12: [(3, 1)], 13: [(0, 1)], 14: [(3, 0)]}


def contour_segments(F, level, x0, y0, d, mask=None):
    """Marching squares: segments (N, 4) [xa, ya, xb, yb] (world units of x0/y0/d) of F = level,
    each oriented with the higher values on its left.  ``mask`` (ny, nx) bool drops cells where
    any corner is False."""
    a = F[:-1, :-1] - level
    b = F[:-1, 1:] - level
    c = F[1:, 1:] - level
    e = F[1:, :-1] - level
    case = (a > 0) * 1 + (b > 0) * 2 + (c > 0) * 4 + (e > 0) * 8
    if mask is not None:
        ok = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, 1:] & mask[1:, :-1]
        case = np.where(ok, case, 0)
    J, I = np.indices(case.shape)

    def pt(edge, sel):
        A, B, C, E, i, j = a[sel], b[sel], c[sel], e[sel], I[sel], J[sel]
        with np.errstate(divide="ignore", invalid="ignore"):
            if edge == 0:
                return i + np.nan_to_num(A / (A - B), nan=0.5), j + 0.0 * i
            if edge == 1:
                return i + 1.0, j + np.nan_to_num(B / (B - C), nan=0.5)
            if edge == 2:
                return i + np.nan_to_num(E / (E - C), nan=0.5), j + 1.0
            return i + 0.0 * j, j + np.nan_to_num(A / (A - E), nan=0.5)

    segs = []
    for cs, pairs in _EDGES.items():
        sel = case == cs
        if not sel.any():
            continue
        for p, q in pairs:
            xa, ya = pt(p, sel)
            xb, yb = pt(q, sel)
            segs.append(np.stack([xa, ya, xb, yb], axis=1))
    for cs in (5, 10):                   # saddles: decide by the centre value
        sel = case == cs
        if not sel.any():
            continue
        centre = 0.25 * (a + b + c + e)[sel] > 0
        for flip in (False, True):
            s2 = sel.copy()
            s2[sel] = centre != flip
            if not s2.any():
                continue
            hi_mid = not flip
            if cs == 5:
                pairs = [(3, 2), (0, 1)] if not hi_mid else [(3, 0), (1, 2)]
            else:
                pairs = [(0, 3), (2, 1)] if not hi_mid else [(0, 1), (2, 3)]
            for p, q in pairs:
                xa, ya = pt(p, s2)
                xb, yb = pt(q, s2)
                segs.append(np.stack([xa, ya, xb, yb], axis=1))
    if not segs:
        return np.zeros((0, 4))
    S = np.concatenate(segs)
    # orient: higher values on the left of a -> b
    i = np.clip(np.floor(0.5 * (S[:, 0] + S[:, 2])).astype(int), 0, F.shape[1] - 2)
    j = np.clip(np.floor(0.5 * (S[:, 1] + S[:, 3])).astype(int), 0, F.shape[0] - 2)
    gx = (b + c - a - e)[j, i]
    gy = (c + e - a - b)[j, i]
    cross = (S[:, 2] - S[:, 0]) * gy - (S[:, 3] - S[:, 1]) * gx
    flip = cross < 0
    S[flip] = S[flip][:, [2, 3, 0, 1]]
    S[:, [0, 2]] = x0 + S[:, [0, 2]] * d
    S[:, [1, 3]] = y0 + S[:, [1, 3]] * d
    return S


def join_segments(S, tol):
    """Chain oriented segments into polylines (lists of [x, y])."""
    if len(S) == 0:
        return []
    key = lambda x, y: (int(round(x / tol)), int(round(y / tol)))
    starts = {}
    for k, s in enumerate(S):
        starts.setdefault(key(s[0], s[1]), []).append(k)
    used = np.zeros(len(S), bool)
    lines = []
    for k in range(len(S)):
        if used[k]:
            continue
        used[k] = True
        line = [[S[k, 0], S[k, 1]], [S[k, 2], S[k, 3]]]
        while True:                                       # forward
            nxt = [m for m in starts.get(key(*line[-1]), []) if not used[m]]
            if not nxt:
                break
            m = nxt[0]
            used[m] = True
            line.append([S[m, 2], S[m, 3]])
        lines.append(line)
    # join lines whose end meets another's start
    merged = True
    while merged and len(lines) > 1:
        merged = False
        heads = {key(*l[0]): n for n, l in enumerate(lines)}
        for n, l in enumerate(lines):
            m = heads.get(key(*l[-1]))
            if m is not None and m != n:
                lines[n] = l + lines[m][1:]
                del lines[m]
                merged = True
                break
    return lines


def _polyline_length(p):
    p = np.asarray(p)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _chaikin(p, n=2):
    p = np.asarray(p, dtype=np.float64)
    for _ in range(n):
        if len(p) < 3:
            break
        q = np.empty((2 * (len(p) - 1), 2))
        q[0::2] = 0.75 * p[:-1] + 0.25 * p[1:]
        q[1::2] = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], q, p[-1:]])
    return p


def _pack(a, lo, hi):
    """uint8 codes (1..255; 0 = no data) of a field between lo and hi, base64."""
    c = np.clip(np.round(1 + 254 * (np.asarray(a, np.float64) - lo) / (hi - lo)), 1, 255)
    c = np.where(np.isfinite(a), c, 0).astype(np.uint8)
    return base64.b64encode(c.tobytes()).decode()


# ------------------------------------------------------------------ the world
class SynopticWorld:
    """The simulated weather of a run: advance_to(t), columns(X, Y, t, z), sounding(x, y, t),
    diagnostics(t), analysis(t), payload(t).  World coordinates are the run's (Lambert metres
    about the map centre); t is model seconds from the run's start time."""

    KEEP = 12                       # recent 15-min snapshots kept (3 h; a run reads at most 15 min back)
    DIURNAL_IN_ANALYSIS = True      # analysis(): the stored boundary-layer theta is a daily mean
    max_hour = 24                   # forecast hours a preview offers

    def __init__(self, proj, start: _dt.datetime, cfg: dict | None = None, seed: int = 1, progress=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.proj = proj
        if start.tzinfo is None:
            start = start.replace(tzinfo=_dt.timezone.utc)
        self.start = start
        self.seed = int(seed)
        self.lock = threading.RLock()
        self.progress = progress or (lambda f, msg: None)
        (self.jet_lat, self.sst_gulf, self.dtheta, self.eml_src,
         self.t500_jet) = SEASON[start.month]
        self.keep_hourly = True
        self._snaps = {}
        self._diag = {}
        self._anal = {}
        self.notes = []
        b = bank(self.progress)
        self.q = qg.QGModel()
        if abs(float(b["dt"]) - self.q.dt) > 1e-9 or abs(self.q.dt - DT) > 1e-9:
            raise RuntimeError("synoptic bank time step mismatch")
        self._grids()
        self._place(b)

    # ------------------------------------------------------------ geometry
    def _grids(self):
        q = self.q
        # the QG domain: the jet axis through the season's jet latitude at the map's longitude
        xj, yj = self.proj.fwd(self.jet_lat, self.proj.lon0)
        xc, _ = self.proj.fwd(38.0, -96.0)
        self.qx0 = float(xc) - 0.5 * q.Lx
        self.qy0 = float(yj) - 0.5 * q.Ly
        # the surface grid: the contiguous US with the Gulf, Mexico's plateau and the coastal waters
        lat_b = np.concatenate([np.linspace(20.5, 54.0, 60), np.full(60, 54.0), np.linspace(54.0, 20.5, 60),
                                np.full(60, 20.5)])
        lon_b = np.concatenate([np.full(60, -128.0), np.linspace(-128.0, -63.0, 60), np.full(60, -63.0),
                                np.linspace(-63.0, -128.0, 60)])
        bx, by = self.proj.fwd(lat_b, lon_b)
        x0, x1 = float(np.min(bx)), float(np.max(bx))
        y0, y1 = float(np.min(by)), float(np.max(by))
        x0, y0 = min(x0, -1500e3), min(y0, -1500e3)
        x1, y1 = max(x1, 1500e3), max(y1, 1500e3)
        self.nx = int(math.ceil((x1 - x0) / DX)) + 1
        self.ny = int(math.ceil((y1 - y0) / DX)) + 1
        self.x0, self.y0 = x0, y0
        xs = x0 + DX * np.arange(self.nx)
        ys = y0 + DX * np.arange(self.ny)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.lat, self.lon = self.proj.inv(self.X, self.Y)
        self.lat = self.lat.astype(np.float64)
        self.lon = (self.lon + 180.0) % 360.0 - 180.0
        self.f = np.maximum(2 * OMEGA * np.sin(np.radians(self.lat)), 5e-5)
        # QG grid indices of the surface points
        self.qfx = (self.X - self.qx0) / q.dx - 0.5
        self.qfy = (self.Y - self.qy0) / q.dx - 0.5
        # land and water from the land cover (25 km boxes of its 1 km cells)
        water = None
        try:
            from .landcover import load
            lc = load()
            if lc is not None:
                acc = np.zeros(self.X.shape)
                n = 0
                for oy in (-0.375, -0.125, 0.125, 0.375):
                    for ox in (-0.375, -0.125, 0.125, 0.375):
                        la, lo = self.proj.inv(self.X + ox * DX, self.Y + oy * DX)
                        c = lc.category(la, lo)
                        acc += np.isin(c, (17, 21)) | (c == 0)
                        n += 1
                water = acc / n
        except Exception:
            water = None
        if water is None:
            water = ((self.lat < 29.5) & (self.lon > -97.5) & (self.lon < -81.0)).astype(float)
            water = np.maximum(water, (self.lon < -124.5).astype(float))
            water = np.maximum(water, ((self.lon > -75.0) & (self.lat < 40)).astype(float))
            self.notes.append("no land-cover data: a rough land/sea map is used")
        self.water = _smooth(water, 0.7)
        self.land = 1.0 - self.water
        lat, lon = self.lat, self.lon
        # warm seas: the Gulf, the Caribbean and the Atlantic along the Gulf Stream; cold Pacific
        self.pacific = self.water * _ss((-112.0 - lon) / 6.0)
        self.sst = np.where(lon < -112.0,
                            self.sst_gulf - 7.0 - 0.45 * np.maximum(lat - 28.0, 0.0),
                            self.sst_gulf + 1.0 - 0.9 * np.maximum(lat - 26.0, 0.0))
        # dry West (relative humidity of the land's own moisture) and the hot sources of the EML
        self.rh_land = 0.30 + 0.45 * _ss((lon + 106.0) / 12.0)
        self.desert = self.land * _ss((36.5 - lat) / 4.0) * _ss((lon + 116.0) / 3.0) * _ss((-100.5 - lon) / 3.0)
        self.desert = np.maximum(self.desert, self.land * _ss((42.0 - lat) / 3.0) * _ss((lat - 33.0) / 2.0)
                                 * _ss((lon + 113.0) / 2.0) * _ss((-104.5 - lon) / 2.0) * 0.8)
        # the High Plains and the plateau: the model is flat, so their height (1-1.6 km: the same temperature at
        # 850 hPa instead of 950) is a warmer mixed-layer theta -- what puts the hotter air west of the dryline
        # and the lee trough under it
        self.high = self.land * _ss((-98.5 - lon) / 6.0) * _ss((lon + 116.0) / 4.0)
        self.plains = self.land * _ss((lon + 106.0) / 3.0) * _ss((-88.0 - lon) / 4.0) * _ss((lat - 25.0) / 3.0) \
            * _ss((50.0 - lat) / 4.0)
        # all the land east of the Rockies: where a trough's ascent drives a low-level jet (the Southeast's
        # cool-season outbreaks ride one), not only the Plains' nocturnal one
        self.east = self.land * _ss((lon + 106.0) / 3.0) * _ss((-72.0 - lon) / 3.0) * _ss((lat - 25.0) / 3.0) \
            * _ss((50.0 - lat) / 4.0)
        edge = np.minimum(np.minimum(np.arange(self.nx)[None, :], self.nx - 1 - np.arange(self.nx)[None, :]),
                          np.minimum(np.arange(self.ny)[:, None], self.ny - 1 - np.arange(self.ny)[:, None]))
        self.edge = 1.0 - _ss(edge / 8.0)                     # the inflow sponge at the grid's edges
        # diagnostics grid: every second surface point (50 km)
        self.dsl = (slice(0, None, 2), slice(0, None, 2))
        # the QG fields onto the surface grid at fixed points: the weights once, not every step
        self._q2s = _Bilinear((q.ny, q.nx), self.qfx, self.qfy, periodic=True)
        # latent heat back onto the QG grid: each QG point samples the surface grid (nothing outside it)
        QX, QY = np.meshgrid((np.arange(q.nx) + 0.5) * q.dx + self.qx0, (np.arange(q.ny) + 0.5) * q.dx + self.qy0)
        sx, sy = (QX - self.x0) / DX, (QY - self.y0) / DX
        self._s2q_in = ((sx >= 0) & (sx <= self.nx - 1) & (sy >= 0) & (sy <= self.ny - 1)).astype(np.float64)
        self._s2q = _Bilinear((self.ny, self.nx), sx, sy, periodic=False)
        # the storm patches' own weather (their real microphysics), fed back by the run (ingest_storms)
        self.fb_dth = np.zeros(self.X.shape)
        self.fb_dq = np.zeros(self.X.shape)
        self.fb_rain = np.zeros(self.X.shape)
        self.fb_w = np.zeros(self.X.shape)

    # ------------------------------------------------------------ the start
    def _place(self, b):
        """Pick the bank day and east-west shift that put the scenario's low where it wants it at the
        peak time, run the flow and the surface up to the start, and keep the start's snapshot."""
        cfg = self.cfg
        rng = np.random.default_rng(self.seed)
        sc = SCENARIOS.get(cfg.get("scenario") or "outbreak", SCENARIOS["outbreak"])
        peak = round(float(cfg.get("peak_h", 6.0)) * 4.0) / 4.0 * 3600.0
        peak = min(max(peak, 0.0), 36 * 3600.0)
        lows = b["lows"]
        q = self.q
        need = SPIN + peak + 86400.0                      # history the start needs in the bank
        ok = lows[:, 0] * 21600.0 >= need
        lows = lows[ok]
        if sc["low"] is None or not len(lows):
            k = int(rng.integers(int(math.ceil(need / 21600.0)), b["q1"].shape[0] * 4))
            shift = float(rng.uniform(0.0, q.Lx))
            self.scenario_note = "random weather"
            self.target = None
        else:
            tx, ty = sc["low"][0] * 1000.0, sc["low"][1] * 1000.0
            wy = (lows[:, 6] + 0.5) * q.dx + self.qy0
            near = np.abs(wy - ty) < 300e3
            cand = lows[near] if near.sum() >= 5 else lows[np.argsort(np.abs(wy - ty))[:20]]
            # each candidate scored as a forecaster would: as strong as the scenario wants, and (for the
            # Gulf scenarios) with southerly return flow from the Gulf to the centre over the 2 days before
            rank = np.argsort(np.argsort(cand[:, 4])) / max(len(cand) - 1, 1)
            scored = []
            for row, pct in zip(cand, rank):
                wx = (row[5] + 0.5) * q.dx + self.qx0
                shift = (tx - wx) % q.Lx
                ret = self._return_flow(b, int(row[0]), shift) if sc.get("gulf_open") else 0.0
                hodo = self._hodograph_score(b, int(row[0]), shift) if sc.get("veering") else 0.0
                scored.append((-2.0 * abs(pct - sc["strength"]) + min(ret, 10.0) / 4.0 + 3.0 * hodo, row, shift, ret))
            # one of the five best, the better ones likelier: every seed a different day of the same kind
            scored.sort(key=lambda s: -s[0])
            top = scored[:5]
            wts = np.exp((np.array([s[0] for s in top]) - top[0][0]) / 1.0)
            _, row, shift, ret = top[int(rng.choice(len(top), p=wts / wts.sum()))]
            k = int(row[0])
            self.target = (tx, ty)
            self.scenario_note = (f"{sc['label']}: a {row[4] * RHO * qg.F0 / 100.0:.0f} hPa cyclone "
                                  f"placed {peak / 3600:.0f} h after the start"
                                  + (f", {ret:.0f} m/s of Gulf return flow the 2 days before" if sc.get("gulf_open") else ""))
        t_peak = k * 21600.0
        t_start = t_peak - peak
        d0 = int(math.floor((t_start - SPIN) / 86400.0))
        d0 = min(max(d0, 0), b["q1"].shape[0] - 1)
        q.t = d0 * 86400.0
        q.q1h, q.q2h = b["q1"][d0].copy(), b["q2"][d0].copy()
        q.translate(shift)
        n_spin = int(round((t_start - q.t) / DT))
        self.n = -n_spin                                  # step index; 0 = the run's start
        self._init_surface()
        for k in range(n_spin):
            self._step()
            if k % 24 == 0:
                self.progress(0.95 + 0.05 * k / max(n_spin, 1), "moistening the Gulf air and building the cap")
        self.progress(1.0, "ready")

    def _hodograph_score(self, b, k: int, shift: float) -> float:
        """0..1: how much the day at bank record k (shifted east by ``shift``) looks like a tornado day's wind
        profile at the map centre: the wind veering 40-100 degrees from the surface to 500 hPa (south-southeast
        under southwest), 20-45 m/s at 500 hPa -- not the whole column blowing one way."""
        q = self.q
        fx = np.array([(0.0 - self.qx0 - shift) / q.dx - 0.5])
        fy = np.array([(0.0 - self.qy0) / q.dx - 0.5])

        def wind(ps):
            u = -(np.roll(ps, -1, 0) - np.roll(ps, 1, 0)) / (2 * q.dx)
            v = (np.roll(ps, -1, 1) - np.roll(ps, 1, 1)) / (2 * q.dx)
            return float(_sample_periodic(u, fx, fy)[0]), float(_sample_periodic(v, fx, fy)[0])

        ps, p1 = b["psis"][k].astype(np.float64), b["psi1s"][k].astype(np.float64)
        us, vs = wind(ps)
        u5, v5 = wind(0.5 * (ps + p1))
        ds = math.degrees(math.atan2(-us, -vs)) % 360.0                 # direction blowing from
        d5 = math.degrees(math.atan2(-u5, -v5)) % 360.0
        veer = (d5 - ds + 540.0) % 360.0 - 180.0                         # + = veering with height
        s5 = math.hypot(u5, v5)
        return float(math.exp(-((veer - 70.0) / 35.0) ** 2) * min(1.0, max(0.0, (s5 - 12.0) / 10.0))
                     * (1.0 if s5 < 50.0 else 0.6))

    def _return_flow(self, b, k: int, shift: float) -> float:
        """Mean southerly surface geostrophic wind (m/s) from the Gulf coast to the map centre over the 48 h
        before bank record k, with the flow shifted east by ``shift``: how much Gulf air it brought."""
        q = self.q
        ys = np.arange(-900e3, 1.0, 150e3)
        fx = (np.zeros_like(ys) - self.qx0 - shift) / q.dx - 0.5
        fy = (ys - self.qy0) / q.dx - 0.5
        vs = []
        for kk in range(max(k - 8, 0), k + 1):
            ps = b["psis"][kk].astype(np.float64)
            v = (np.roll(ps, -1, 1) - np.roll(ps, 1, 1)) / (2 * q.dx)
            vs.append(float(_sample_periodic(v, fx, fy).mean()))
        return float(np.mean(vs))

    # ------------------------------------------------------------ the flow on the surface grid
    def _flow(self):
        """The QG fields sampled on the surface grid, the QG vertical motion, pressure and its tendency."""
        q = self.q
        f = q.fields()
        psi_s = q.surface_psi(f)
        a = qg.SURFACE_A
        us = (1 + a) * f["u2"] - a * f["u1"]
        vs = (1 + a) * f["v2"] - a * f["v1"]
        us = us - qg.SURFACE_ZONAL_DAMP * us.mean(axis=1, keepdims=True)      # as surface_psi
        vs = vs - qg.SURFACE_ZONAL_DAMP * vs.mean(axis=1, keepdims=True)
        # QG vertical motion at mid-levels: w = -(DT/Dt - heating) / (dry - environmental lapse rate)
        tm = f["tmid"]
        prev = getattr(self, "_tmid_prev", None)
        dtdt = (tm - prev) / DT if prev is not None else 0.0 * tm
        self._tmid_prev = tm
        um, vm = 0.5 * (f["u1"] + f["u2"]), 0.5 * (f["v1"] + f["v2"])
        tx = (np.roll(tm, -1, 1) - np.roll(tm, 1, 1)) / (2 * q.dx)
        ty = (np.roll(tm, -1, 0) - np.roll(tm, 1, 0)) / (2 * q.dx)
        teq = qg.F0 * (2 * q.psic_eq - 2 * q.psic_ref) / (qg.RD * math.log(qg.P2 / qg.P1))
        heat = -(tm - teq) / q.tau_rad + getattr(self, "_qlat_qg", 0.0)     # radiation + the rain's latent heat
        w = -(dtdt + um * tx + vm * ty - heat) / STAB
        w = np.clip(w, -0.08, 0.08)
        stack = np.stack([psi_s, us, vs, f["u2"], f["v2"], f["u1"], f["v1"], tm, w])
        s = self._q2s(stack)
        return dict(zip(("psi_s", "uqs", "vqs", "u2", "v2", "u1", "v1", "tmid", "w"), s))

    # ------------------------------------------------------------ climate the surface relaxes to
    def _theta_eq(self, tmid):
        land = (293.0 + self.dtheta + 0.6 * tmid + 4.0 * self.desert + 7.0 * self.high)
        sea = self.sst + 273.15 - 0.5
        return self.land * land + self.water * sea

    def _q_eq(self, th):
        T = th * (95000.0 / P0) ** KAPPA
        land = self.rh_land * qv_sat(95000.0, T)
        land = land * (1.0 - 0.45 * self.desert)
        sea = 0.80 * qv_sat(101000.0, self.sst + 273.15)
        return self.land * land + self.water * sea

    def _theta_ft(self, tmid, z, p):
        T = self.t500_jet + tmid + GAMMA * (5600.0 - z)
        return T * (P0 / p) ** KAPPA

    def _init_surface(self):
        fl = self._flow()
        self.th = self._theta_eq(fl["tmid"])
        self.qv = self._q_eq(self.th)
        if SCENARIOS.get(self.cfg.get("scenario"), {}).get("gulf_open"):
            # the Gulf has been open for days (as before most outbreaks): its air over the south-central and
            # southeastern US at the start of the spin-up; the flow decides where it goes from there
            reg = 0.95 * self.land * _ss((40.0 - self.lat) / 4.0) * _ss((self.lon + 102.0) / 3.0)
            gulf = 0.8 * qv_sat(101000.0, self.sst_gulf + 273.15)
            self.qv = self.qv * (1.0 - reg) + reg * np.maximum(self.qv, gulf)
        self.te = self._theta_ft(fl["tmid"], 1500.0, 85000.0)
        # the mid-troposphere's moisture: half saturated, moister over the warm seas
        qs4 = qv_sat(P4, self.t500_jet + fl["tmid"] + GAMMA * (5600.0 - Z4))
        self.qm = qs4 * (0.45 + 0.2 * self.water)
        self.cv = np.zeros(self.X.shape)                # convective activity 0..1 (parameterized convection)
        self._moist = {k: np.zeros(self.X.shape) for k in ("prl", "prc", "ctt", "cld")}
        self._qlat_qg = 0.0
        self._ps_prev = None
        self._fl = fl
        self._store(fl)

    # ------------------------------------------------------------ one step
    def calendar(self, t: float) -> _dt.datetime:
        return self.start + _dt.timedelta(seconds=float(t))

    def _step(self):
        """One DT of the flow and the surface."""
        self.q.step(1)
        self.n += 1
        fl = self._flow()
        t = self.n * DT
        now = self.calendar(t)
        hour = now.hour + now.minute / 60.0 + now.second / 3600.0
        hl = _local_hour(hour, self.lon)
        S = _heating(hl)
        # winds: the flow's surface geostrophic wind + the thermal part from the boundary layer's warmth
        th_eff = self.th + self.land * (4.5 + 3.0 * self.desert) * (S - 0.35)
        p_th = -RHO * G * H_THERMAL * (th_eff - _smooth_coarse(th_eff, 12.0)) / TH0
        gx, gy = _grad(p_th, DX)
        uth, vth = -gy / (RHO * self.f), gx / (RHO * self.f)
        uth, vth = np.clip(uth, -15, 15), np.clip(vth, -15, 15)
        ps = 101325.0 + RHO * qg.F0 * fl["psi_s"] + p_th
        # the isallobaric wind toward falling pressure
        if self._ps_prev is not None:
            pt = _smooth((ps - self._ps_prev) / DT, 6.0)
            gx, gy = _grad(pt, DX)
            uia, via = -gx / (RHO * self.f ** 2), -gy / (RHO * self.f ** 2)
            sp = np.hypot(uia, via)
            fac = np.minimum(1.0, 10.0 / np.maximum(sp, 1e-6))
            uia, via = uia * fac, via * fac
        else:
            pt = 0.0 * ps
            uia = via = 0.0 * ps
        self._ps_prev = ps
        ug0, vg0 = fl["uqs"] + uth, fl["vqs"] + vth
        # friction: slower and turned toward low pressure, more over land
        ang = 35.0 * self.land + 12.0 * self.water
        fr = 0.65 * self.land + 0.85 * self.water
        usf, vsf = _rotate(ug0, vg0, ang)
        usf, vsf = fr * usf, fr * vsf
        us, vs = usf + 0.8 * uia, vsf + 0.8 * via
        # advection: the mixed layer by the 0-1.5 km wind, the EML by the 3 km wind
        u15 = ug0 + (fl["u2"] - fl["uqs"]) * 0.6 - uth * 0.6 + 0.4 * uia
        v15 = vg0 + (fl["v2"] - fl["vqs"]) * 0.6 - vth * 0.6 + 0.4 * via
        ua, va = 0.7 * us + 0.3 * u15, 0.7 * vs + 0.3 * v15
        u3 = fl["u2"] + (fl["u1"] - fl["u2"]) * (500.0 / 7500.0)
        v3 = fl["v2"] + (fl["v1"] - fl["v2"]) * (500.0 / 7500.0)
        self.th, self.qv = self._advect([self.th, self.qv], ua, va)
        # (the EML and the mid-level moisture ride the 3-4 km wind together: _moist_physics)
        # sources
        th_eq = self._theta_eq(fl["tmid"])
        tau = 2 * 86400.0 * self.land + 12 * 3600.0 * self.water
        tau = tau / (1.0 + 4.0 * self.edge)
        self.th += (th_eq - self.th) * (DT / tau)
        q_eq = self._q_eq(self.th)
        tq = (6 * 86400.0 * (1.0 - self.desert) + 1.5 * 86400.0 * self.desert) * self.land + 12 * 3600.0 * self.water
        tq = tq / (1.0 + 4.0 * self.edge)
        self.qv += (q_eq - self.qv) * (DT / tq)
        self.qv = np.clip(self.qv, 1e-4, 0.95 * qv_sat(100000.0, self.th * (100000.0 / P0) ** KAPPA))
        # the EML: heated over the plateau and the desert in the day, eroded by ascent and over the sea
        te_none = self._theta_ft(fl["tmid"], 1500.0, 85000.0)
        day = _ss((hl - 9.0) / 3.0) * _ss((19.0 - hl) / 3.0)
        src = self.desert * day
        te_src = self.eml_src + 3.0 + 0.5 * (fl["tmid"])      # the plateau's afternoon mixed layer (~317 K in May)
        self.te += src * (te_src - self.te) * (DT / (8 * 3600.0))
        decay = 1.0 / (48 * 3600.0) * (1.0 + np.maximum(fl["w"], 0.0) / 0.04) + self.water / (18 * 3600.0)
        decay = decay + self.edge / (6 * 3600.0)
        self.te += (1.0 - src) * (te_none - self.te) * np.minimum(DT * decay, 1.0)
        self.te = np.maximum(self.te, te_none - 2.0)
        self._moist_physics(fl, us, vs, th_eff, ps, te_none)
        fl.update(ps=ps, pt=pt, us=us, vs=vs, usf=usf, vsf=vsf, ug0=ug0, vg0=vg0, uth=uth, vth=vth,
                  uia=uia, via=via)
        self._fl = fl
        self._store(fl)

    def _moist_physics(self, fl, us, vs, th_eff, ps, te_none):
        """Water in the large-scale weather: moisture carried at mid levels, lifted out of the boundary layer
        by the troughs' ascent and dried by sinking air (the dry slot); grid-scale condensation into
        stratiform rain where it saturates; parameterized convection where the air is unstable, uncapped
        and lifted -- except where a storm patch computes its storms with real microphysics, whose cold
        pools and rain come back through ingest_storms instead.  Rain falling into dry air below cools and
        moistens the boundary layer (the rain-cooled wedge north of a warm front, an MCS's cold pool), and
        the latent heat of all of it warms the troughs' ascent regions in the QG flow (deeper lows,
        stronger ridges downstream)."""
        k_dt = DT / 3600.0
        # mid-level moisture rides the 4 km wind
        a4 = (Z4 - qg.Z2) / (qg.Z1 - qg.Z2)
        u4 = fl["u2"] + (fl["u1"] - fl["u2"]) * a4
        v4 = fl["v2"] + (fl["v1"] - fl["v2"]) * a4
        self.te, self.qm, self.cv = self._advect([self.te, self.qm, self.cv], u4, v4)
        T4 = self.t500_jet + fl["tmid"] + GAMMA * (5600.0 - Z4)
        qs4 = qv_sat(P4, T4)
        w = fl["w"]
        # ascent lifts boundary-layer air into the mid levels -- not through a cap: under the EML the moist layer
        # is sealed in and what rises above it is the dry EML air -- diluted on the way; descent brings dry air down
        uncapped = _ss((th_eff + 1.5 - self.te) / 3.0)
        up = np.clip(np.maximum(w, 0.0) / 3000.0 * DT, 0.0, 1.0) * (0.1 + 0.9 * uncapped)
        dn = np.clip(np.maximum(-w, 0.0) / 3000.0 * DT, 0.0, 1.0)
        self.qm += up * np.maximum(0.6 * self.qv - self.qm, 0.0) - dn * np.maximum(self.qm - 0.12 * qs4, 0.0)
        self.qm += (0.4 * qs4 - self.qm) * self.edge * (DT / (6 * 3600.0))
        # the storm patches' own weather (real microphysics) where they run: no parameterized convection there
        kfb = min(1.0, DT / 1800.0)
        self.th += self.fb_w * self.fb_dth * kfb
        self.qv = np.maximum(self.qv + self.fb_w * self.fb_dq * kfb, 1e-4)
        self.fb_dth *= 1.0 - kfb
        self.fb_dq *= 1.0 - kfb
        explicit = np.clip(self.fb_w, 0.0, 1.0)
        # parameterized convection: boundary-layer theta-e above the mid-levels' saturated theta-e, the cap
        # (the EML) overcome, and something lifting the air
        T_bl = th_eff * (ps / P0) ** KAPPA
        the_bl = thetae_bolton(ps, T_bl, self.qv)
        the_4 = thetae_bolton(P4, T4, qs4)
        unstable = _ss((the_bl - the_4) / 8.0)
        dudx, _ = _grad(us, DX)
        _, dvdy = _grad(vs, DX)
        conv = _smooth(-(dudx + dvdy), 1.5)
        lift = np.clip(0.2 + conv / 2.0e-5 + w / 0.03, 0.0, 1.0)
        trigger = unstable * uncapped * lift * self.land ** 0.5 * (1.0 - explicit)
        self.cv += (trigger - self.cv) * np.minimum(1.0, DT / (trigger * 2400.0 + (1 - trigger) * 5400.0 + 1.0))
        self.cv = np.clip(self.cv, 0.0, 1.0) * (1.0 - explicit)
        prc = 8.0 * self.cv * np.clip(self.qv / 0.014, 0.0, 1.5)              # mm/h
        # convection's effects: downdrafts cool and dry the boundary layer (cold pools), detrainment moistens
        # the mid levels, overturning wipes out the cap
        mm = prc * k_dt
        self.th -= 0.6 * mm * (0.5 + _ss((T_bl - dewpoint_from_qv(ps, self.qv) - 3.0) / 12.0))
        self.qv = np.maximum(self.qv * (1.0 - 0.012 * mm), 1e-4)
        self.qm += (qs4 - self.qm) * np.minimum(1.0, self.cv * DT / 3600.0)
        self.te += (te_none - self.te) * np.minimum(1.0, self.cv * DT / 3600.0)
        # grid-scale condensation: what is above saturation rains out over ~30 min
        cond = np.maximum(self.qm - qs4, 0.0) * min(1.0, DT / 1800.0)
        self.qm -= cond
        prl = cond * MID_MASS / DT * 3600.0                                  # mm/h
        # rain evaporating into the dry air below: cooler, moister boundary layer
        rh_bl = np.clip(self.qv / qv_sat(ps, T_bl), 0.0, 1.0)
        evap = np.minimum(prl * k_dt, 3.0) * (1.0 - rh_bl) * self.land
        self.th -= 0.5 * evap
        self.qv += evap * 0.5 / LV_CP
        # latent heat: the column warms by LV_CP * rain mass / column mass; into the QG thickness
        rain_rate = (prl + prc) / 3600.0                                     # kg/m^2/s
        q_lat = np.minimum(LV_CP * rain_rate / COL_MASS, 2.0 / 3600.0)        # K/s, at most 2 K/h
        q_lat = _smooth(q_lat, 1.5)
        q_qg = self._s2q(q_lat) * self._s2q_in
        if self.n > 0:          # from the start on: the spin-up keeps the day the scenario picked from the bank
            self.q.heat(q_qg, DT)
            self._qlat_qg = q_qg
        # clouds for the satellite view: cloud-top temperature (deep convection at the tropopause, layer cloud
        # at ~7 km, else the ground)
        z_trop = np.clip(11000.0 + 150.0 * fl["tmid"], 9000.0, 14500.0)
        t_trop = self.t500_jet + fl["tmid"] - 7.5e-3 * (z_trop - 5600.0) - 4.0     # overshooting tops a little colder
        t7 = self.t500_jet + fl["tmid"] - GAMMA * 1400.0
        cld = np.clip((self.qm / qs4 - 0.7) / 0.25, 0.0, 1.0)
        deep = np.clip(np.maximum(self.cv, explicit * _ss(self.fb_rain / 4.0)) * 1.4, 0.0, 1.0)
        tsfc = T_bl
        ctt = tsfc + (t7 - tsfc) * cld
        ctt = ctt + (t_trop - ctt) * deep
        self.fb_w *= math.exp(-DT / 2700.0)
        self.fb_rain *= math.exp(-DT / 1800.0)
        self._moist = {"prl": prl, "prc": prc + self.fb_rain, "ctt": ctt, "cld": np.maximum(cld, deep)}

    def ingest_storms(self, X, Y, th0, q0, rain, z0: float, t: float) -> None:
        """The storm patches' own lowest-level air and rain (their real microphysics: cold pools, outflow,
        moistening) at ground points (X, Y) (m) at model time t, block-averaged on ~25 km.  Their departure from
        this model's air is laid into the boundary layer over the next half hour, carried ahead by the wind to
        this model's time (it runs ahead of the storms), and parameterized convection stays off there."""
        X = np.asarray(X, np.float64).ravel()
        Y = np.asarray(Y, np.float64).ravel()
        if X.size == 0:
            return
        with self.lock:
            c = self.columns(X, Y, t, np.array([z0]))
            dth = np.asarray(th0, np.float64).ravel() - c["th"][:, 0]
            dq = np.asarray(q0, np.float64).ravel() - c["qv"][:, 0]
            lag = max(self.n * DT - t, 0.0)
            F = self._snaps[self.n]
            fx, fy = (X - self.x0) / DX, (Y - self.y0) / DX
            u = _sample(F[self.FIELDS.index("us")], fx, fy)
            v = _sample(F[self.FIELDS.index("vs")], fx, fy)
            i = np.clip(np.rint(fx + u * lag / DX).astype(int), 0, self.nx - 1)
            j = np.clip(np.rint(fy + v * lag / DX).astype(int), 0, self.ny - 1)
            self.fb_dth[j, i] = np.clip(dth, -15.0, 5.0)
            self.fb_dq[j, i] = np.clip(dq, -0.008, 0.008)
            self.fb_rain[j, i] = np.maximum(self.fb_rain[j, i], np.asarray(rain, np.float64).ravel())
            self.fb_w[j, i] = 1.0

    def _advect(self, fields, u, v):
        """Semi-Lagrangian (midpoint departure, bilinear), the grid's edges clamped: every field in ``fields``
        carried by the same wind, sampled together."""
        k = DT / DX
        xi, yj = np.arange(self.nx)[None, :], np.arange(self.ny)[:, None]
        um, vm = _sample(np.stack([u, v]), xi - 0.5 * k * u, yj - 0.5 * k * v)
        return list(_sample(np.stack(fields), xi - k * um, yj - k * vm))

    FIELDS = ("th", "qv", "te", "tmid", "w", "ps", "pt", "us", "vs", "usf", "vsf", "ug0", "vg0", "uth", "vth",
              "uia", "via", "uqs", "vqs", "u2", "v2", "u1", "v1", "qm", "cv", "prl", "prc", "ctt", "cld")

    def _store(self, fl):
        fl = dict(fl)
        fl.update(th=self.th, qv=self.qv, te=self.te, qm=self.qm, cv=self.cv, **self._moist)
        for k in ("ps", "pt", "us", "vs", "usf", "vsf", "ug0", "vg0", "uth", "vth", "uia", "via"):
            if k not in fl:
                if k == "ps":
                    fl[k] = 101325.0 + RHO * qg.F0 * fl["psi_s"]
                elif k in ("ug0", "usf", "us"):
                    fl[k] = fl["uqs"]
                elif k in ("vg0", "vsf", "vs"):
                    fl[k] = fl["vqs"]
                else:
                    fl[k] = 0.0 * self.th
        snap = np.stack([np.asarray(fl[k], dtype=np.float32) for k in self.FIELDS])
        self._snaps[self.n] = snap              # ~4 MB each
        if self.n <= 0:                         # the spin-up: only what the start needs
            for n in [n for n in self._snaps if n < self.n - 1]:
                del self._snaps[n]
            return
        for n in [n for n in self._snaps if n < self.n - self.KEEP and not (self.keep_hourly and n % 4 == 0)]:
            del self._snaps[n]
        if self.keep_hourly:                    # the page's preview: hourly snapshots of the last 24 h
            for n in [n for n in self._snaps if n < self.n - 96]:
                del self._snaps[n]

    # ------------------------------------------------------------ time
    def advance_to(self, t: float) -> None:
        with self.lock:
            n = int(math.ceil(t / DT - 1e-9))
            while self.n < n:
                self._step()

    def _fields_at(self, t: float):
        """Interpolated snapshot (nf, ny, nx) at time t (advancing the model if needed)."""
        self.advance_to(t)
        with self.lock:
            a = t / DT
            n0 = int(math.floor(a + 1e-9))
            w = a - n0
            s0 = self._snaps.get(n0)
            if s0 is None:
                have = sorted(self._snaps)
                n0 = min(have, key=lambda n: abs(n - a))
                return self._snaps[n0]
            if w < 1e-6:
                return s0
            s1 = self._snaps.get(n0 + 1)
            if s1 is None:
                return s0
            # a patch's edges ask for the same times again and again (4 sides x 3 fields x 5 times): the blend
            # of the two snapshots is kept for the last few times
            fc = self.__dict__.setdefault("_fcache", {})
            key = (round(t, 3), n0)
            if key not in fc:
                if len(fc) > 6:
                    fc.clear()
                fc[key] = (1.0 - w) * s0 + w * s1
            return fc[key]

    # ------------------------------------------------------------ columns
    def _point_fields(self, X, Y, t):
        F = self._fields_at(t)
        fx = (np.asarray(X, np.float64) - self.x0) / DX
        fy = (np.asarray(Y, np.float64) - self.y0) / DX
        s = _sample(F, fx, fy)
        P = dict(zip(self.FIELDS, s.astype(np.float64)))
        geo = self.__dict__.get("_geo")
        if geo is None:
            geo = self._geo = np.stack([self.lon, self.lat, self.land, self.desert, self.plains, self.east])
        g = _sample(geo, fx, fy)
        P.update(lon=g[0], lat=g[1], land=g[2], desert=g[3], plains=g[4], east=g[5])
        return P

    def columns(self, X, Y, t: float, z=None) -> dict:
        """Columns at ground points (X, Y) (m, any shape -> flattened N) and time t: z (nz,) m AGL, and
        p (Pa), T (K), th (K), qv (kg/kg), u, v (m/s, ground-relative) as (N, nz)."""
        X = np.asarray(X, np.float64).ravel()
        Y = np.asarray(Y, np.float64).ravel()
        P = self._point_fields(X, Y, t)
        if z is None:
            return dict(self._build(P, t), z=ZC)
        # straight on the levels asked for (the storm patches' 50, or one): 3-150x less work than ZC and an
        # interpolation; the ground (0 m) goes first for the hydrostatic integration and is dropped after
        z = np.asarray(z, np.float64)
        zc = np.concatenate([[0.0], z]) if z[0] > 0.0 else z
        C = self._build(P, t, zc)
        cut = 1 if z[0] > 0.0 else 0
        out = {"z": z}
        for name in ("p", "T", "th", "qv", "u", "v"):
            out[name] = C[name][:, cut:]
        return out

    def _build(self, P, t, zc=ZC):
        """Columns on the levels zc (default ZC) from point fields P."""
        now = self.calendar(t)
        hour = now.hour + now.minute / 60.0 + now.second / 3600.0
        hl = _local_hour(hour, P["lon"])
        S, Nn, J = _heating(hl), _night(hl), _jet_phase(hl)
        land = np.clip(P["land"], 0.0, 1.0)
        N = len(hl)
        ps = P["ps"]
        T_bl = P["th"] * (ps / P0) ** KAPPA
        dd = np.maximum(T_bl - dewpoint_from_qv(ps, P["qv"]), 0.0)
        dry = np.clip((dd - 5.0) / 20.0, 0.0, 1.0)
        th_ml = P["th"] + land * (4.5 + 4.5 * dry + 3.0 * P["desert"]) * (S - 0.35)
        q_ml = P["qv"] * (1.0 - 0.15 * land * S * dry)
        dth_n = land * Nn * (3.0 + 4.0 * dry)
        h = np.clip(300.0 + S * (500.0 + 140.0 * dd), 250.0, 3800.0)
        # a dry mixed layer stops at its cloud base (~125 m per K of dewpoint depression): above it is cumulus
        h = np.minimum(h, np.maximum(118.0 * dd, 300.0))
        capped = P["te"] > th_ml + 0.5
        h = np.where(capped, np.minimum(h, 1100.0 + 600.0 * dry), h)
        tm = P["tmid"]
        z_trop = np.clip(11000.0 + 150.0 * tm, 9000.0, 14500.0)
        T500 = self.t500_jet + tm
        # humidity aloft from the simulated mid-level moisture (relative to saturation at 4 km); convection
        # moistens the whole troposphere (anvils)
        T4 = T500 + GAMMA * (5600.0 - Z4)
        rh4 = np.clip(P["qm"] / qv_sat(P4, T4), 0.03, 1.0)
        deep = np.clip(P["cv"] * 1.5, 0.0, 1.0)
        nz = len(zc)
        p = np.empty((N, nz))
        T = np.empty((N, nz))
        th = np.empty((N, nz))
        qv = np.empty((N, nz))
        p[:, 0] = ps
        q_top = q_ml * 0.88
        for k in range(nz):
            z = zc[k]
            if k > 0:
                dz = z - zc[k - 1]
                Tv0 = T[:, k - 1] * (1 + qv[:, k - 1] / EPS) / (1 + qv[:, k - 1])
                p[:, k] = p[:, k - 1] * np.exp(-G * dz / (RD * Tv0))
            pk = p[:, k]
            Tft = np.where(z <= z_trop, T500 + GAMMA * (5600.0 - z), T500 + GAMMA * (5600.0 - z_trop))
            th_ft = Tft * (P0 / pk) ** KAPPA
            cap_w = 1.0 - _ss((z - 5000.0) / 1500.0)
            th_above = np.maximum(np.maximum(th_ft, np.where(cap_w > 0, P["te"] * cap_w + th_ft * (1 - cap_w), 0.0)),
                                  th_ml + 3e-3 * np.maximum(z - h, 0.0))
            th_bl = th_ml - dth_n * np.clip(1.0 - z / 350.0, 0.0, 1.0) ** 2
            wgt = np.clip((z - h) / 200.0, 0.0, 1.0)
            thk = th_bl * (1.0 - wgt) + th_above * wgt
            if k > 0:
                above = z > h + 200.0
                thk = np.where(above, np.maximum(thk, th[:, k - 1]), thk)
            th[:, k] = thk
            Tk = thk * (pk / P0) ** KAPPA
            T[:, k] = Tk
            # moisture
            aloft = 1.0 - 0.45 * min(max((z - 6000.0) / 5000.0, 0.0), 1.0)
            # never near saturation: the storm patches take this air as their environment, without the clouds
            # that go with it, and air at 90-100 % would condense or snow out at the slightest lift into one huge
            # featureless echo (the maps' clouds and rain come from the mid-level moisture field, not from here)
            rh = np.clip(np.maximum(rh4 * aloft, deep * 0.75), 0.03, RH_MAX)
            in_eml = (P["te"] * cap_w > th_ft + 0.5) & (z > h)
            rh = np.where(in_eml, np.minimum(rh, 0.3), rh)
            q_ft = rh * qv_sat(pk, Tk)
            q_bl = q_ml * (1.0 - 0.12 * np.minimum(z / h, 1.0))
            ew = np.clip((z - h) / 300.0, 0.0, 1.0)
            qk = q_bl * (1.0 - ew) + np.minimum(q_ft, q_top) * ew
            qk = np.minimum(qk, qv_sat(pk, Tk))
            qk = np.where(z > h + 150.0, np.minimum(qk, RH_MAX * qv_sat(pk, Tk)), qk)     # above the mixed layer too
            qk = np.where(Tk < 273.15, np.minimum(qk, RH_MAX * qv_sat_ice(pk, Tk)), qk)
            qk = np.where(z > z_trop, np.minimum(qk, 3e-6), qk)
            qv[:, k] = np.maximum(qk, 1e-7)
            if k > 0:                           # corrector: the layer's mean virtual temperature
                Tv1 = Tk * (1 + qv[:, k] / EPS) / (1 + qv[:, k])
                p[:, k] = p[:, k - 1] * np.exp(-G * (z - zc[k - 1]) / (RD * 0.5 * (Tv0 + Tv1)))
        # winds
        z = zc[None, :]
        zt = z_trop[:, None]

        def prof(lo, mid, hi):
            a = lo[:, None] + (mid - lo)[:, None] * np.minimum(z / qg.Z2, 1.0)
            b = mid[:, None] + (hi - mid)[:, None] * (np.minimum(z, zt) - qg.Z2) / (qg.Z1 - qg.Z2)
            v = np.where(z <= qg.Z2, a, b)
            return v

        ug = prof(P["uqs"], P["u2"], P["u1"])
        vg = prof(P["vqs"], P["v2"], P["v1"])
        aloft = 1.0 - 0.6 * _ss((z - zt) / 5000.0)
        ug, vg = ug * aloft, vg * aloft
        thermal = np.clip(1.0 - z / 1500.0, 0.0, 1.0)
        ug = ug + P["uth"][:, None] * thermal
        vg = vg + P["vth"][:, None] * thermal
        he = (220.0 + 180.0 * S * land)[:, None]
        ek = np.exp(-z / he)
        # friction over land by the time of day: once the surface layer stabilizes in the evening the wind
        # at the ground slows and backs further (35 -> 45 deg, 0.65 -> 0.5 of geostrophic) while the air above
        # it is let go -- the evening transition that raises low-level shear and helicity
        fr = 0.65 - 0.15 * J
        ua, va = _rotate(P["ug0"], P["vg0"], 35.0 + 10.0 * J)
        usf = land * fr * ua + (1.0 - land) * P["usf"]
        vsf = land * fr * va + (1.0 - land) * P["vsf"]
        u = ug + (usf - P["ug0"])[:, None] * ek + P["uia"][:, None] * np.exp(-z / 1200.0)
        v = vg + (vsf - P["vg0"])[:, None] * ek + P["via"][:, None] * np.exp(-z / 1200.0)
        # nocturnal low-level jet over the Plains, along the 550 m geostrophic wind, mostly in southerly flow
        # the geostrophic wind at 550 m, exactly (whatever levels were asked for: one, or fifty)
        a5, th5 = 550.0 / qg.Z2, 1.0 - 550.0 / 1500.0
        u5 = P["uqs"] + (P["u2"] - P["uqs"]) * a5 + P["uth"] * th5
        v5 = P["vqs"] + (P["v2"] - P["vqs"]) * a5 + P["vth"] * th5
        sp = np.hypot(u5, v5)
        south = np.clip(v5 / np.maximum(sp, 0.5), 0.0, 1.0) ** 0.5
        # supergeostrophic by ~15 % in the day (the sloping Plains' baroclinic boundary layer), ~65 % at
        # night (the inertial oscillation once the boundary layer decouples), up to 35 % more under the
        # large-scale ascent ahead of a trough (lee cyclogenesis); lower at night
        night = P["plains"] + 0.5 * np.maximum(P["east"] - P["plains"], 0.0)       # weaker off the Plains
        amp = (P["plains"] * LLJ_DAY + night * LLJ_NIGHT * J
               + P["east"] * LLJ_ASCENT * np.clip(P["w"] / 0.03, 0.0, 1.0)) * sp * south
        zj = (750.0 - 250.0 * J)[:, None]
        jp = np.exp(-((z - zj) / 380.0) ** 2)
        u = u + (amp * u5 / np.maximum(sp, 0.5))[:, None] * jp
        v = v + (amp * v5 / np.maximum(sp, 0.5))[:, None] * jp
        return {"p": p, "T": T, "th": th, "qv": qv, "u": u, "v": v, "h": h, "z_trop": z_trop}

    def sounding(self, x: float, y: float, t: float, title: str | None = None) -> Sounding:
        c = self.columns(np.array([x]), np.array([y]), t)
        top = ZC <= 20000.0
        now = self.calendar(t)
        lat, lon = self.proj.inv(x, y)
        return Sounding(ZC[top].copy(), c["p"][0, top], c["T"][0, top], c["qv"][0, top], c["u"][0, top],
                        c["v"][0, top],
                        title or f"Synoptic {float(lat):.2f}, {float(lon):.2f} {now:%Y-%m-%d %H%MZ}",
                        "synoptic").cleaned()

    def storm_motion(self, x: float, y: float, t: float):
        """Bunkers right-mover motion (m/s) of the column at (x, y)."""
        zz = np.arange(0.0, 6001.0, 50.0)
        c = self.columns(np.array([x]), np.array([y]), t, zz)
        rm, _, _ = bunkers(zz, c["u"][0], c["v"][0])
        return float(rm[0]), float(rm[1])

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self, t: float) -> dict:
        """Severe-weather fields on the 50 km grid at time t (rounded to the 15-min step): MLCAPE, MLCIN,
        LCL, 0-1 km SRH, 0-6 km shear, STP, 2 m T and Td, MSLP, 500 hPa height and wind, surface
        convergence, and the initiation rate (storms per 100,000 km^2 per hour)."""
        n = int(round(t / DT))
        with self.lock:
            if n in self._diag:
                return self._diag[n]
        t = n * DT
        F = self._fields_at(t)
        Xd, Yd = self.X[self.dsl], self.Y[self.dsl]
        P = self._point_fields(Xd.ravel(), Yd.ravel(), t)
        C = self._build(P, t, ZD)
        shape = Xd.shape
        p, T, qv, u, v = C["p"], C["T"], C["qv"], C["u"], C["v"]
        top = ZD <= 16000.0
        zz = ZD[top]
        p, T, qv, u, v = p[:, top], T[:, top], qv[:, top], u[:, top], v[:, top]
        # mixed-layer parcel: the lowest 100 hPa
        ml = p >= (p[:, :1] - 10000.0)
        th = T * (P0 / p) ** KAPPA
        thm = (th * ml).sum(1) / ml.sum(1)
        qm = (qv * ml).sum(1) / ml.sum(1)
        p0 = p[:, 0]
        t0 = thm * (p0 / P0) ** KAPPA
        e = np.maximum(qm * p0 / (EPS + qm), 1.0)
        tlcl = 2840.0 / (3.5 * np.log(t0) - np.log(e / 100.0) - 4.805) + 55.0
        plcl = P0 * (tlcl / thm) ** (1.0 / KAPPA)
        the = thetae_bolton(p0, t0, qm)
        tp = thm[:, None] * (p / P0) ** KAPPA
        sat = p < plcl[:, None]
        if sat.any():
            tp[sat] = _moist_lookup(np.broadcast_to(the[:, None], p.shape)[sat], p[sat])
        qp = np.where(sat, qv_sat(p, tp), qm[:, None])
        tvp = tp * (1 + qp / EPS) / (1 + qp)
        tve = T * (1 + qv / EPS) / (1 + qv)
        b = G * (tvp - tve) / tve
        dz = np.gradient(zz)[None, :]
        # the LCL height: between the levels around its pressure (log-linear), all columns at once
        rows = np.arange(len(plcl))
        kl = np.clip(np.argmax(p < plcl[:, None], axis=1), 1, len(zz) - 1)
        lp = np.log(p)
        fr = np.clip((lp[rows, kl - 1] - np.log(plcl)) / (lp[rows, kl - 1] - lp[rows, kl]), 0.0, 1.0)
        zlcl = np.where(plcl >= p[:, 0], 0.0, zz[kl - 1] + fr * (zz[kl] - zz[kl - 1]))
        above = zz[None, :] >= zlcl[:, None] - 1.0
        posa = (b > 0) & above
        has = posa.any(1)
        klfc = np.where(has, np.argmax(posa, axis=1), len(zz) - 1)
        kk = np.arange(len(zz))[None, :]
        cape = np.where(has, (np.maximum(b, 0) * dz * (kk >= klfc[:, None])).sum(1), 0.0)
        low5 = zz[None, :] <= 5000.0
        cin = np.where(has, (np.minimum(b, 0) * dz * (kk < klfc[:, None])).sum(1),
                       (np.minimum(b, 0) * dz * low5).sum(1))
        cin = np.maximum(cin, -500.0)
        # kinematics
        sel6 = zz <= 6000.0
        um, vm = u[:, sel6].mean(1), v[:, sel6].mean(1)
        lo = zz <= 500.0
        hi = (zz >= 5500.0) & (zz <= 6000.0)
        su, sv = u[:, hi].mean(1) - u[:, lo].mean(1), v[:, hi].mean(1) - v[:, lo].mean(1)
        mag = np.maximum(np.hypot(su, sv), 1e-6)
        rmu, rmv = um + 7.5 * sv / mag, vm - 7.5 * su / mag
        s1 = zz <= 1000.0
        uu, vv = u[:, s1] - rmu[:, None], v[:, s1] - rmv[:, None]
        srh1 = (uu[:, 1:] * vv[:, :-1] - uu[:, :-1] * vv[:, 1:]).sum(1)
        k6 = int(np.argmin(np.abs(zz - 6000.0)))
        sh6 = np.hypot(u[:, k6] - u[:, 0], v[:, k6] - v[:, 0])
        stp = ((cape / 1500.0) * np.clip((2000.0 - zlcl) / 1000.0, 0, 1) * (srh1 / 150.0)
               * np.where(sh6 >= 12.5, np.clip(sh6 / 20.0, 0, 1.5), 0.0) * np.clip((200.0 + cin) / 150.0, 0, 1))
        stp = np.maximum(stp, 0.0)
        # 500 hPa
        ln = np.log(p)
        k5 = np.argmax(p < 50000.0, axis=1)
        a = (ln[np.arange(len(k5)), k5 - 1] - math.log(50000.0)) / (ln[np.arange(len(k5)), k5 - 1] - ln[np.arange(len(k5)), k5])
        z500 = zz[k5 - 1] + a * (zz[k5] - zz[k5 - 1])
        u500 = u[np.arange(len(k5)), k5 - 1] * (1 - a) + u[np.arange(len(k5)), k5] * a
        v500 = v[np.arange(len(k5)), k5 - 1] * (1 - a) + v[np.arange(len(k5)), k5] * a
        # surface
        t2 = T[:, 1] - 273.15
        td2 = dewpoint_from_qv(p[:, 1], qv[:, 1]) - 273.15
        ps = F[self.FIELDS.index("ps")]
        mslp = _smooth(ps, 1.5)[self.dsl] / 100.0
        us_f = F[self.FIELDS.index("us")]
        vs_f = F[self.FIELDS.index("vs")]
        dudx, _ = _grad(us_f, DX)
        _, dvdy = _grad(vs_f, DX)
        conv = _smooth(-(dudx + dvdy), 1.5)[self.dsl]
        R = lambda a: np.asarray(a, np.float64).reshape(shape)
        out = {"t": t, "mlcape": R(cape), "mlcin": R(cin), "lcl": R(zlcl), "srh1": R(srh1), "shear6": R(sh6),
               "stp": R(stp), "t2m": R(t2), "td2m": R(td2), "mslp": mslp, "z500": R(z500),
               "u500": R(u500), "v500": R(v500), "conv": conv, "rm": (R(rmu), R(rmv)),
               "usfc": us_f[self.dsl], "vsfc": vs_f[self.dsl], "w": F[self.FIELDS.index("w")][self.dsl],
               "rain": F[self.FIELDS.index("prl")][self.dsl] + F[self.FIELDS.index("prc")][self.dsl],
               "ctt": F[self.FIELDS.index("ctt")][self.dsl], "cld": F[self.FIELDS.index("cld")][self.dsl],
               "land": self.land[self.dsl], "heat": R(_heating(_local_hour(
                   self.calendar(t).hour + self.calendar(t).minute / 60.0, P["lon"])))}
        an = self.analysis(t)
        out["front_near"] = self._near_lines(an["fronts"], Xd, Yd)
        out["ci"] = self._ci(out)
        with self.lock:
            self._diag[n] = out
            for m in [m for m in self._diag if abs(m - n) > 8 and m % 4]:
                del self._diag[m]
            if len(self._diag) > 60:
                for m in sorted(self._diag, key=lambda m: abs(m - n))[60:]:
                    del self._diag[m]
        return out

    def _near_lines(self, fronts, X, Y, reach=90e3):
        """0..1: how close each point is to a front or dryline (1 on it, 0 beyond ``reach``), on its warm or
        moist side or within 20 km behind it."""
        near = np.zeros(X.shape)
        from .boundaries import signed_distance
        for fr in fronts:
            P = np.asarray(fr["points"]) * 1000.0
            if len(P) < 2:
                continue
            box = (P[:, 0].min() - reach, P[:, 1].min() - reach, P[:, 0].max() + reach, P[:, 1].max() + reach)
            sel = (X >= box[0]) & (X <= box[2]) & (Y >= box[1]) & (Y <= box[3])
            if not sel.any():
                continue
            d, over, _ = signed_distance(P, X[sel], Y[sel])
            ahead = d * fr["side"]              # + on the side it faces (warm / moist)
            v = np.where((ahead > -20e3) & (over <= 0), 1.0 - np.clip(np.abs(ahead) / reach, 0, 1), 0.0)
            near[sel] = np.maximum(near[sel], v)
        return near

    def _ci(self, d):
        """Initiation rate (storms per 100,000 km^2 per hour) from CAPE, CIN and lift."""
        lift = (0.7 * np.clip(d["conv"] / 2e-5, 0, 3) + 0.35 * np.clip(d["w"] / 0.02, 0, 2)
                + 1.6 * d["front_near"])
        cin_f = np.clip((d["mlcin"] + 25.0 + 60.0 * lift) / 40.0, 0.0, 1.0)
        # a threshold, not a ramp: the trough's broad ascent alone (lift ~0.5) seldom starts a storm; a dryline,
        # front or convergence line (lift 1.5-3) does
        lift_f = np.clip((lift - 0.4) / 1.2, 0.0, 1.5)
        # strong forcing in strong shear (a cold-season front, the high-shear low-CAPE squall lines): storms go
        # with a few hundred J/kg, day or night; weakly forced storms need the CAPE and the day's heating
        forced = np.clip(lift_f, 0.0, 1.0) * np.clip((d["shear6"] - 15.0) / 15.0, 0.0, 1.0)
        # weakly forced: none under 150 J/kg, half at 650, all from 1150; strongly forced in strong shear: from 50,
        # half at 350 (it was half at 1000 and 450: the low-CAPE days hardly started storms)
        cape_f = _ss((d["mlcape"] - 150.0 + 100.0 * forced) / (1000.0 - 400.0 * forced))
        heat = d["land"] * (0.3 + 0.7 * d["heat"]) + (1 - d["land"]) * 0.15
        heat = np.maximum(heat, 0.35 + 0.5 * forced)
        return 1.6 * cape_f * cin_f * lift_f * heat

    # ------------------------------------------------------------ analysis (fronts, centres, contours)
    def analysis(self, t: float) -> dict:
        n = int(round(t / DT))
        with self.lock:
            if n in self._anal:
                return self._anal[n]
        t = n * DT
        F = self._fields_at(t)
        get = lambda k: F[self.FIELDS.index(k)].astype(np.float64)
        now = self.calendar(t)
        hl = _local_hour(now.hour + now.minute / 60.0, self.lon)
        th = get("th")
        if self.DIURNAL_IN_ANALYSIS:            # the surface model's theta is a daily mean: add the hour's heating
            th = th + self.land * (4.5 + 3.0 * self.desert) * (_heating(hl) - 0.35)
        qv = get("qv")
        ths = _smooth(th, 2.5)
        qs = _smooth(qv, 2.5)
        gx, gy = _grad(ths, DX)
        mag = np.hypot(gx, gy)
        mx, my = _grad(mag, DX)
        tfp = -(mx * gx + my * gy) / np.maximum(mag, 1e-9)
        us, vs = get("us"), get("vs")
        inner = (self.edge < 0.5)
        fronts = []
        # fronts: the zero of the thermal front parameter on the cold side of the strongest gradient
        strong = (mag > 2.2e-5) & inner & (self.water < 0.6)
        S = contour_segments(tfp, 0.0, self.x0, self.y0, DX, mask=strong)
        if len(S):
            # keep the crossing at the strongest gradient (TFP rises toward the warm side there); the
            # other zero, the warm edge of the zone, is where the gradient has already faded
            cx, cy = 0.5 * (S[:, 0] + S[:, 2]), 0.5 * (S[:, 1] + S[:, 3])
            fi, fj = (cx - self.x0) / DX, (cy - self.y0) / DX
            ngx, ngy = _sample(gx, fi, fj), _sample(gy, fi, fj)
            tx, ty = _grad(tfp, DX)
            dtn = _sample(tx, fi, fj) * ngx + _sample(ty, fi, fj) * ngy
            S = S[dtn > 0]
        lines = join_segments(S, DX * 0.01) if len(S) else []
        for line in lines:
            P = _chaikin(np.asarray(line), 2)
            if _polyline_length(P) < 300e3:
                continue
            fi, fj = (P[:, 0] - self.x0) / DX, (P[:, 1] - self.y0) / DX
            ngx, ngy = _sample(gx, fi, fj), _sample(gy, fi, fj)
            nrm = np.hypot(ngx, ngy) + 1e-12
            # a few cells toward the cold side: the air behind it pushes (cold front) or retreats (warm)
            ci, cj = fi - 2.5 * ngx / nrm, fj - 2.5 * ngy / nrm
            adv = (_sample(us, ci, cj) * ngx + _sample(vs, ci, cj) * ngy) / nrm   # + = toward the warm side
            dtheta = float(np.mean(nrm) * 150e3)
            fronts.extend(self._split_kinds(P, adv, ngx / nrm, ngy / nrm, dtheta))
        # drylines: sharp moisture gradient, little or reversed temperature contrast, west of ~94 W over land
        qx, qy = _grad(qs, DX)
        qmag = np.hypot(qx, qy)
        dry = ((qmag > 1.8e-8) & (qs > 0.0065) & (self.lon < -94.0) & (self.lon > -110.0) & inner
               & (self.land > 0.8) & ((gx * qx + gy * qy) < 0.3 * mag * qmag))
        qmx, qmy = _grad(qmag, DX)
        qfp = -(qmx * qx + qmy * qy) / np.maximum(qmag, 1e-12)
        S = contour_segments(qfp, 0.0, self.x0, self.y0, DX, mask=dry)
        if len(S):
            cx, cy = 0.5 * (S[:, 0] + S[:, 2]), 0.5 * (S[:, 1] + S[:, 3])
            fi, fj = (cx - self.x0) / DX, (cy - self.y0) / DX
            tx, ty = _grad(qfp, DX)
            dtn = _sample(tx, fi, fj) * _sample(qx, fi, fj) + _sample(ty, fi, fj) * _sample(qy, fi, fj)
            S = S[dtn > 0]
        for line in (join_segments(S, DX * 0.01) if len(S) else []):
            P = _chaikin(np.asarray(line), 2)
            if _polyline_length(P) < 200e3:
                continue
            fronts.append(self._front("dry", P, *self._side_toward(P, qx, qy)))
        # pressure centres
        mslp = _smooth(get("ps"), 4.0) / 100.0
        centres = []
        r = 16                                              # 400 km
        pad = np.pad(mslp, r, mode="edge")
        mn = mslp.copy()
        mxp = mslp.copy()
        for dj in range(-r, r + 1, 2):
            for di in range(-r, r + 1, 2):
                if di * di + dj * dj > r * r:
                    continue
                sh = pad[r + dj:r + dj + self.ny, r + di:r + di + self.nx]
                mn = np.minimum(mn, sh)
                mxp = np.maximum(mxp, sh)
        for kind, sel in (("L", mslp <= mn + 1e-6), ("H", mslp >= mxp - 1e-6)):
            pts = np.argwhere(sel & (self.edge < 0.3))
            # a centre is closed: 600 km out the pressure is 2 hPa higher (a low) or lower (a high) all round
            a = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
            ok = []
            for j, i in pts:
                ring = _sample(mslp, i + 24.0 * np.cos(a), j + 24.0 * np.sin(a))
                diff = (ring - mslp[j, i]) if kind == "L" else (mslp[j, i] - ring)
                ok.append(diff.min() > 0.0 and diff.mean() > 2.0)
            pts = [q for q, k in zip(pts, ok) if k]
            # the most extreme first; a flat bottom gives several points: one per 500 km
            pts = sorted(pts, key=lambda q: mslp[q[0], q[1]] * (1 if kind == "L" else -1))
            kept = []
            for j, i in pts:
                x, y = float(self.X[j, i]), float(self.Y[j, i])
                if any(math.hypot(x - a, y - b) < 500e3 for a, b in kept):
                    continue
                kept.append((x, y))
                centres.append({"kind": kind, "x": round(x / 1000, 1), "y": round(y / 1000, 1),
                                "value": round(float(mslp[j, i]))})
        out = {"t": t, "fronts": fronts, "centres": centres}
        with self.lock:
            self._anal[n] = out
            for m in [m for m in self._anal if abs(m - n) > 8 and m % 4]:
                del self._anal[m]
        return out

    def _side_toward(self, P, gx, gy):
        """Which side of polyline P (+1 left, -1 right) the gradient (gx, gy) points to."""
        fi, fj = (P[:, 0] - self.x0) / DX, (P[:, 1] - self.y0) / DX
        nx_, ny_ = _sample(gx, fi, fj), _sample(gy, fi, fj)
        d = np.gradient(P, axis=0)
        left = np.stack([-d[:, 1], d[:, 0]], axis=1)
        s = float(np.sum(left[:, 0] * nx_ + left[:, 1] * ny_))
        return (1 if s >= 0 else -1),

    def _split_kinds(self, P, adv, nx_, ny_, dtheta):
        """A front polyline cut where its kind changes: cold where the cold air advances (> 1.5 m/s),
        warm where it retreats, stationary between; pieces under 150 km join their neighbours."""
        kind = np.where(adv > 1.5, "cold", np.where(adv < -1.5, "warm", "stationary"))
        out = []
        start = 0
        for k in range(1, len(P) + 1):
            if k == len(P) or kind[k] != kind[start]:
                seg = P[max(start - 1, 0):k]
                if len(seg) >= 2 and (_polyline_length(seg) >= 150e3 or not out):
                    d = np.gradient(seg, axis=0)
                    left = np.stack([-d[:, 1], d[:, 0]], axis=1)
                    s = float(np.sum(left[:, 0] * nx_[max(start - 1, 0):k] + left[:, 1] * ny_[max(start - 1, 0):k]))
                    out.append(self._front(str(kind[start]), seg, 1 if s >= 0 else -1, dtheta))
                elif out and len(seg) >= 2:
                    out[-1]["points"].extend([[round(float(x) / 1000, 1), round(float(y) / 1000, 1)]
                                              for x, y in seg[1:]])
                start = k
        return out

    @staticmethod
    def _front(kind, P, side, strength=0.0):
        """side: +1 when the warm (moist) air is on the left of the polyline's direction."""
        return {"kind": kind, "side": int(side), "strength": round(float(strength), 1),
                "points": [[round(float(x) / 1000, 1), round(float(y) / 1000, 1)] for x, y in P]}

    # ------------------------------------------------------------ for the page
    def payload(self, t: float) -> dict:
        """Everything the map draws at time t (fields packed as bytes, km coordinates); kept per 15-min step."""
        n = int(round(t / DT))
        cache = self.__dict__.setdefault("_pay", {})
        if n in cache:
            return cache[n]
        out = self._payload(n * DT)
        with self.lock:
            cache[n] = out
            for m in sorted(cache, key=lambda m: abs(m - n))[40:]:
                del cache[m]
        return out

    def _payload(self, t: float) -> dict:
        d = self.diagnostics(t)
        an = self.analysis(t)
        ny, nx = d["mlcape"].shape
        x0 = float(self.X[self.dsl][0, 0]) / 1000.0
        y0 = float(self.Y[self.dsl][0, 0]) / 1000.0
        dkm = 2 * DX / 1000.0
        F = {
            "t2m": (d["t2m"], -10.0, 40.0), "td2m": (d["td2m"], -15.0, 25.0), "mlcape": (d["mlcape"], 0.0, 5000.0),
            "mlcin": (d["mlcin"], -300.0, 0.0), "stp": (d["stp"], 0.0, 10.0), "srh1": (d["srh1"], -100.0, 600.0),
            "shear6": (d["shear6"], 0.0, 40.0), "ci": (d["ci"], 0.0, 3.0),
            "wspd500": (np.hypot(d["u500"], d["v500"]), 0.0, 60.0),
            # radar: reflectivity from the rain rate as a WSR-88D assumes it (Marshall-Palmer, Z = 200 R^1.6)
            "dbz": (np.where(d["rain"] > 0.1, 10.0 * np.log10(200.0 * np.maximum(d["rain"], 1e-3) ** 1.6), np.nan),
                    0.0, 75.0),
            # satellite: the infrared cloud-top (or ground) temperature
            "ctt": (d["ctt"] - 273.15, -85.0, 40.0),
        }
        fields = {k: {"lo": lo, "hi": hi, "data": _pack(a, lo, hi)} for k, (a, lo, hi) in F.items()}
        mslp = d["mslp"]
        cont = {}
        segs = []
        for lev in range(960, 1048, 4):
            s = contour_segments(mslp, lev, x0, y0, dkm)
            if len(s):
                segs.append({"v": lev, "s": np.round(s, 1).ravel().tolist()})
        cont["mslp"] = segs
        z5 = _smooth(d["z500"], 1.0)
        segs = []
        for lev in range(4800, 6000, 60):
            s = contour_segments(z5, lev, x0, y0, dkm)
            if len(s):
                segs.append({"v": lev, "s": np.round(s, 1).ravel().tolist()})
        cont["z500"] = segs
        # wind barbs (500 hPa and surface) every 200 km
        sl = (slice(2, None, 4), slice(2, None, 4))
        X = (x0 + dkm * np.arange(nx))[None, :] + 0 * mslp
        Y = (y0 + dkm * np.arange(ny))[:, None] + 0 * mslp
        barbs = {"x": np.round(X[sl].ravel(), 1).tolist(), "y": np.round(Y[sl].ravel(), 1).tolist(),
                 "u500": np.round(d["u500"][sl].ravel(), 1).tolist(), "v500": np.round(d["v500"][sl].ravel(), 1).tolist(),
                 "usfc": np.round(d["usfc"][sl].ravel(), 1).tolist(), "vsfc": np.round(d["vsfc"][sl].ravel(), 1).tolist()}
        now = self.calendar(d["t"])
        c = self.centre_diag(d)
        return {"t": d["t"], "valid": now.strftime("%Y-%m-%dT%H:%MZ"), "valid_ms": int(now.timestamp() * 1000),
                "grid": {"x0": x0, "y0": y0, "dx": dkm, "nx": nx, "ny": ny}, "fields": fields, "contours": cont,
                "fronts": an["fronts"], "centres": an["centres"], "barbs": barbs, "centre": c,
                "max_hour": int(self.max_hour), "kind": "hrrr" if hasattr(self, "cycle") else "synoptic",
                "scenario": self.scenario_note, "notes": self.notes}

    def centre_diag(self, d) -> dict:
        """The diagnostics at the map centre (nearest 50 km point)."""
        Xd, Yd = self.X[self.dsl], self.Y[self.dsl]
        j, i = np.unravel_index(int(np.argmin(Xd ** 2 + Yd ** 2)), Xd.shape)
        g = lambda k: round(float(d[k][j, i]), 1)
        return {"mlcape": g("mlcape"), "mlcin": g("mlcin"), "srh1": g("srh1"), "shear6": g("shear6"),
                "stp": round(float(d["stp"][j, i]), 2), "t2m": g("t2m"), "td2m": g("td2m"), "lcl": g("lcl"),
                "mslp": g("mslp")}

    def summary(self, t: float) -> dict:
        """Small: fronts and pressure centres for the run's state (the page draws them every poll)."""
        an = self.analysis(t)
        return {"t": an["t"], "fronts": an["fronts"], "centres": an["centres"], "scenario": self.scenario_note}


# ------------------------------------------------------------------ worlds, cached at their start
_WORLDS: dict = {}
_WORLDS_LOCK = threading.Lock()


def world_key(proj, start: _dt.datetime, cfg: dict, seed: int) -> str:
    cfg = cfg or {}
    if cfg.get("source") == "hrrr":
        h = cfg.get("hrrr") or {}
        return json.dumps([h.get("model") or "hrrr", round(proj.lat0, 4), round(proj.lon0, 4), str(h.get("cycle")),
                           int(h.get("start_fh", 0)), int(h.get("hours", 12))])
    return json.dumps([round(proj.lat0, 4), round(proj.lon0, 4), start.isoformat(),
                       cfg.get("scenario", "outbreak"), round(float(cfg.get("peak_h", 6.0)), 2), int(seed)])


def world_for(proj, start: _dt.datetime, cfg: dict, seed: int, progress=None) -> SynopticWorld:
    """A fresh copy of the world for these settings at its start: the one the page previewed is the one the
    run gets, built once (the last three kept)."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=_dt.timezone.utc)
    key = world_key(proj, start, cfg, seed)
    with _WORLDS_LOCK:
        w0 = _WORLDS.get(key)
    if w0 is None:
        if (cfg or {}).get("source") == "hrrr" and ((cfg or {}).get("hrrr") or {}).get("model") == "era5":
            from .era5 import Era5World            # ERA5's environment: any past day since 1940
            w0 = Era5World(proj, start, cfg, seed, progress=progress)
        elif (cfg or {}).get("source") == "hrrr":
            from .hrrr import HrrrWorld            # the HRRR's environment instead of the simulated weather
            w0 = HrrrWorld(proj, start, cfg, seed, progress=progress)
        else:
            w0 = SynopticWorld(proj, start, cfg, seed, progress=progress)
        with _WORLDS_LOCK:
            _WORLDS[key] = w0
            while len(_WORLDS) > 3:
                del _WORLDS[next(iter(_WORLDS))]
    w = w0.clone()
    w.cfg = {**DEFAULTS, **(cfg or {})}
    return w


def _clone(self) -> "SynopticWorld":
    """An independent copy (its own lock), at the same time as this one."""
    import copy
    w = object.__new__(type(self))
    shared = getattr(self, "_SHARED", ())
    for k, v in self.__dict__.items():
        if k in ("lock", "progress"):
            continue
        # read-only data (the HRRR's hours) is shared, not copied
        w.__dict__[k] = v if k in shared else copy.deepcopy(v)
    w.lock = threading.RLock()
    w.progress = lambda f, msg: None
    return w


SynopticWorld.clone = _clone


# ------------------------------------------------------------------ for the storm patches
class SynopticAirMass:
    """boundaries.AirMasses for the synoptic mode: every column's target state is the simulated
    atmosphere's at that place and time, in full (theta', qv, u, v at every model level), relative to
    the run's base sounding (the simulated column at the map centre at the start)."""

    synoptic = True
    KEEP_TH = 0.15                  # x the engine's keep rate: the interior's low levels follow the day's cycle
    KEEP_QV = 0.10
    KEEP_TOP = 1500.0               # m

    def __init__(self, world: SynopticWorld, env):
        self.world, self.env = world, env
        self.boundaries = []
        self.texture = None
        self.brushes = []
        self.thb = np.asarray(env.base.thb, dtype=np.float64)
        self.qv_env = np.maximum(env.qv_f(env.z_h), 1e-9)
        self.pb = np.asarray(env.base.pb, dtype=np.float64)         # the patches' own pressure on their levels

    def _world_columns(self, X, Y, t, z):
        """The world's columns on the patches' levels: a world with terrain (the HRRR) hands them over at the
        patches' own pressure (columns_for_model), the flat simulated weather as they are."""
        if hasattr(self.world, "columns_for_model"):
            return self.world.columns_for_model(X, Y, t, z, self.pb)
        return self.world.columns(X, Y, t, z)

    def __bool__(self):
        return True

    def total_length_km(self) -> float:
        return 0.0

    def weight(self, X, Y, t):
        return np.zeros(np.shape(X))

    def _cols(self, X, Y, t):
        """Columns on the model's levels at ground points (X, Y) (2-D meshgrid or anything): computed on
        every ~10 km of a regular grid and interpolated between (the air varies on 25 km and more)."""
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        z = np.asarray(self.env.z_h, np.float64)
        if X.ndim == 2 and X.shape[0] > 2 and X.shape[1] > 2 and np.allclose(X[0], X[-1]) and np.allclose(Y[:, 0], Y[:, -1]):
            s = max(1, int(round(10000.0 / self.env.grid.dx)))
            ny, nx = X.shape
            iy = np.unique(np.r_[np.arange(0, ny, s), ny - 1])
            ix = np.unique(np.r_[np.arange(0, nx, s), nx - 1])
            c = self._world_columns(X[np.ix_(iy, ix)], Y[np.ix_(iy, ix)], t, z)
            out = {}
            fy = np.interp(np.arange(ny), iy, np.arange(len(iy)))
            fx = np.interp(np.arange(nx), ix, np.arange(len(ix)))
            FX, FY = np.meshgrid(fx, fy)
            for k in ("th", "qv", "u", "v"):
                a = c[k].reshape(len(iy), len(ix), len(z)).transpose(2, 0, 1)
                out[k] = _sample(a, FX, FY)
            return out
        c = self._world_columns(X.ravel(), Y.ravel(), t, z)
        return {k: c[k].T.reshape((len(z),) + X.shape) for k in ("th", "qv", "u", "v")}

    def maps(self, X, Y, t: float) -> dict:
        c = self._cols(X, Y, t)
        sh = (-1,) + (1,) * np.ndim(X)
        thp = c["th"] - self.thb.reshape(sh)
        qfac = c["qv"] / self.qv_env.reshape(sh)
        du = c["u"] - np.asarray(self.env.u_ground).reshape(sh)
        dv = c["v"] - np.asarray(self.env.v_ground).reshape(sh)
        one = np.ones(np.shape(X))
        return {"dT": thp[0], "H": 1000.0 * one, "dq": 1.0 - qfac[0], "Hq": 1000.0 * one, "du": du[0],
                "dv": dv[0], "Hw": 1000.0 * one, "nudge": 0.0 * one, "wnudge": 0.0 * one,
                "hold": 0.0 * one, "pin": 0.0 * one, "pin_lo": 0.0 * one, "_3d": (thp, qfac, du, dv)}

    def fields_3d(self, m: dict, z=None):
        return m["_3d"]

    @staticmethod
    def to_device(m: dict) -> dict:
        import cupy as cp
        out = {k: cp.asarray(v, dtype=cp.float64) for k, v in m.items() if k != "_3d"}
        out["_3d"] = tuple(cp.asarray(a, dtype=cp.float64) for a in m["_3d"])
        return out

    def patch_targets(self, rect, t: float) -> dict:
        """The patch's target fields on a coarse grid (every ~10 km of mass points; the relaxation kernel
        interpolates): theta', qv and the grid-relative u, v (nz, cy, cx), float32 on the device."""
        import cupy as cp
        env = self.env
        s = max(1, int(round(10000.0 / env.grid.dx)))
        cx, cy = -(-rect.nx // s), -(-rect.ny // s)
        ox, oy = env.offset_at(t)
        xs = (rect.i0 + np.minimum(np.arange(cx) * s + 0.5 * s, rect.nx - 0.5)) * env.grid.dx + ox
        ys = (rect.j0 + np.minimum(np.arange(cy) * s + 0.5 * s, rect.ny - 0.5)) * env.grid.dx + oy
        X, Y = np.meshgrid(xs, ys)
        c = self._world_columns(X.ravel(), Y.ravel(), t, env.z_h)
        nz = len(env.z_h)
        f = lambda a: cp.asarray(a.T.reshape(nz, cy, cx), dtype=cp.float32)
        return {"thp": f(c["th"] - self.thb[None, :]), "qv": f(c["qv"]), "u": f(c["u"] - env.motion[0]),
                "v": f(c["v"] - env.motion[1]), "stride": float(s), "cx": cx, "cy": cy}
