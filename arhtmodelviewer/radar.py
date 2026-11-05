"""Simulated WSR-88D radars that scan the ArWen patches.

* Reflectivity is ArWen's own WRF ``refl10cm_hm`` (the Morrison scheme's
  10-cm reflectivity, with melting graupel/hail), interpolated in linear Z.
* Fall speed, ZDR, rho_hv and KDP come from the Morrison moments (rain
  number and mass give the drop size; rimed ice is hail-like with
  morr_rimed_ice = 1): Rayleigh, exponential size distributions.
* Beams follow the 4/3-earth model along great circles from the real site,
  are filled over the beam's depth (5 sub-beams) and width (the neighbouring
  radials), weighted by the two-way antenna pattern, and
  carry the model wind along the local beam direction minus the
  reflectivity-weighted fall speed.  Spectrum width comes from the
  sub-grid TKE plus the velocity spread inside the beam.
* Sweeps follow the real VCP layouts and are sampled when the model reaches
  each sweep's scan time; a finished volume becomes NEXRAD Level II.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, replace

import numpy as np

from . import geo

KE_RE = 4.0 / 3.0 * 6371000.0
BEAMWIDTH = 0.95                     # deg (WSR-88D)
SUB_OFFSETS = (-0.5, -0.25, 0.0, 0.25, 0.5)
# A target off the beam's axis echoes by the antenna's two-way power pattern, exp(-8 ln2 (angle /
# beamwidth)^2): 0.71 at a quarter beamwidth, 0.25 at half (the one-way pattern's 0.84 and 0.50
# made the beam about 1.4x too wide).  Across azimuth the antenna turns while it dwells, which
# widens the beam there (Torres & Curtis 2007): 1.02 deg for 0.5-degree super-resolution radials,
# 1.39 for 1-degree ones.
TWO_WAY = 8.0 * math.log(2.0)
EFFECTIVE_BEAMWIDTH = {True: 1.02, False: 1.39}      # deg across azimuth, by super-resolution

# Level II gate layout
REF_FIRST, REF_GATE, REF_NGATES = 2125.0, 250.0, 1832       # to ~460 km
DOP_FIRST, DOP_GATE, DOP_NGATES = 2125.0, 250.0, 1192       # to ~300 km

# (name, scale, offset, word bits) -- code = value * scale + offset
MOMENTS = {
    "REF": (2.0, 66.0, 8),
    "VEL": (2.0, 129.0, 8),
    "SW": (2.0, 129.0, 8),
    "ZDR": (16.0, 128.0, 8),
    "PHI": (2.8361, 2.0, 16),
    "RHO": (300.0, -60.5, 8),
}
# Nyquist written to Level II with dealiased velocities: past anything a sweep can hold (+-63.5 m/s
# in 0.5 m/s steps), so a viewer's own dealiasing (GR2Analyst's) finds nothing to unfold
DEALIASED_NYQUIST = 127.0
VEL_DEALIASED_SCALE = 1.0     # codes per m/s of dealiased velocity: 1 m/s steps reach +-127 m/s


# ------------------------------------------------------------------ VCPs
@dataclass
class Cut:
    elev: float
    kind: str          # "surv" (REF + dual-pol), "dop" (VEL/SW + REF), "both"
    seconds: float     # scan time
    nyquist: float     # m/s: the PRF's true Nyquist velocity (lambda PRF / 4)
    super_res: bool
    # gate layout (WSR-88D unless given): first gate and spacing (m), gates of REF and of the Doppler
    # moments, and how far the cut is sampled to find the echoes that fold into it (range_fold)
    first: float = REF_FIRST
    gate: float = REF_GATE
    n_ref: int = 0
    n_dop: int = DOP_NGATES
    n_far: int = REF_NGATES
    radials: int = 0   # 0: 720 with super-resolution, else 360
    wavelength: float = 0.1071   # m (S band; C band 0.0532)
    unamb: float = 0.0           # m: the unambiguous range when not the PRF's own (dual-PRF C-band radars)

    def __post_init__(self):
        if not self.n_ref:
            self.n_ref = REF_NGATES if self.kind in ("surv", "both") else DOP_NGATES

    @property
    def nrad(self) -> int:
        return self.radials or (720 if self.super_res else 360)

    @property
    def unamb_m(self) -> float:
        """Unambiguous range of the cut's PRF (m): c / (2 PRF) = c lambda / (8 Va)."""
        if self.kind == "surv":
            return self.first + self.gate * self.n_ref
        if self.unamb > 0.0:
            return self.unamb
        return 2.99792458e8 * self.wavelength / (8.0 * self.nyquist)


def _vcp_cuts(number: int, sails: int = 0) -> list[Cut]:
    def split(e, s1, s2, sr=True):
        return [Cut(e, "surv", s1, 8.9, sr), Cut(e, "dop", s2, 28.5, sr)]

    if number in (212, 12):
        cuts = (split(0.5, 21.5, 20.0) + split(0.9, 21.5, 20.0) + split(1.3, 21.5, 20.0)
                + [Cut(1.8, "both", 13.5, 26.4, True), Cut(2.4, "both", 13.5, 26.4, True),
                   Cut(3.1, "both", 13.5, 26.4, True), Cut(4.0, "both", 13.0, 26.4, True)]
                + [Cut(e, "both", 11.5, 33.0, False) for e in (5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5)])
    elif number == 215:
        cuts = (split(0.5, 23.0, 21.0) + split(0.9, 23.0, 21.0) + split(1.3, 23.0, 21.0)
                + [Cut(e, "both", 14.0, 26.4, True) for e in (1.8, 2.4, 3.1, 4.0)]
                + [Cut(e, "both", 12.0, 30.0, False) for e in (5.1, 6.4, 8.0, 10.0, 12.0, 14.0, 16.7, 19.5)])
    elif number == 35:
        cuts = (split(0.5, 33.0, 30.0) + split(0.9, 33.0, 30.0) + split(1.3, 33.0, 30.0)
                + [Cut(e, "both", 30.0, 26.4, True) for e in (1.8, 2.4, 3.1, 4.0, 5.1, 6.4)])
    else:
        raise ValueError(f"VCP {number} is not supported (212, 12, 215, 35)")
    if sails:
        # SAILS / MESO-SAILS: extra 0.5 deg split cuts spread through the volume
        base = cuts
        n = len(base)
        out = []
        insert_at = [int(round(n * (k + 1) / (sails + 1))) for k in range(sails)]
        for i, c in enumerate(base):
            out.append(c)
            if i + 1 in insert_at:
                out.extend(split(0.5, 21.5, 20.0))
        cuts = out
    return cuts


# TDWR (Terminal Doppler Weather Radar): C band (5.32 cm), a 0.55-degree beam, 150 m gates and 1-degree
# radials.  Its "hazardous weather" mode: a long-range, low-PRF reflectivity scan at the base tilt (to 460 km,
# 300 m gates, no Doppler), then Doppler tilts to 89.7 km at a high PRF -- a Nyquist of only ~18-23 m/s, so
# its velocities fold, and an unambiguous range of only ~90-110 km, so storms beyond it fold back in as
# range-folded (purple) gates -- with the base Doppler tilt repeated every few tilts (once a minute in real
# scanning).  No dual polarization.
TDWR_VCP = 90
TDWR_LAMBDA = 0.0532


def _tdwr_cuts() -> list[Cut]:
    def dop(e, nyq):
        return Cut(e, "dop", 10.0, nyq, False, first=150.0, gate=150.0, n_ref=598, n_dop=598, n_far=3066,
                   radials=360, wavelength=TDWR_LAMBDA)
    lr = Cut(0.3, "surv", 12.0, 8.0, False, first=300.0, gate=300.0, n_ref=1533, n_dop=0, n_far=1533,
             radials=360, wavelength=TDWR_LAMBDA)
    base = [dop(0.3, 19.6)]
    cuts = [lr] + base
    for k, (e, nyq) in enumerate(((0.6, 21.3), (1.0, 18.5), (1.6, 19.6), (2.4, 21.3), (3.3, 18.5), (4.4, 19.6),
                                  (5.9, 21.3), (7.8, 18.5), (10.3, 19.6), (13.6, 21.3), (18.0, 18.5))):
        cuts.append(dop(e, nyq))
        if k in (2, 6):                                   # the base tilt again, about once a minute
            cuts.append(dop(0.3, 19.6))
    return cuts


# DOW (Doppler on Wheels, the DOW6 / DOW7 of the University of Illinois' facility): X band (9.35-9.5 GHz, 3.2 cm),
# a 2.4 m dish's 0.93-degree beam, dual polarization, two 250 kW transmitters.  Its tornado mode: full 360-degree
# PPIs at 50 deg/s (7.2 s a sweep) with 0.5-degree radials and 75 m gates (VORTEX2's setting) to 60 km, 0.5 to 16
# degrees in about 1.5 minutes.  Dual-PRF processing keeps the low PRF's ~75 km unambiguous range and unfolds the
# velocities to an extended Nyquist.  No attenuation (the user's choice; real X band loses much behind heavy cores).
DOW_VCP = 81
DOW_LAMBDA = 0.0318
DOW_IDS = ("DOW6", "DOW7", "DOW8")
DOW_TILTS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)
DOW_GATES = (25.0, 50.0, 75.0)     # m: the gate lengths offered (the DOWs run 15-600 m)
DOW_RANGE = 60000.0                 # m sampled


def _dow_cuts(gate: float = 75.0) -> list[Cut]:
    gate = min(DOW_GATES, key=lambda g: abs(g - float(gate or 75.0)))
    n = int(round(DOW_RANGE / gate))
    return [Cut(e, "both", 7.2, 40.0, True, first=gate, gate=gate, n_ref=n, n_dop=n, n_far=n,
                radials=720, wavelength=DOW_LAMBDA, unamb=75000.0) for e in DOW_TILTS]


# The legacy WSR-88D (the 1990s to 2008, before super-resolution (Build 10, 2008) and dual polarization (2011-13)):
# 1-degree radials everywhere, reflectivity in 1 km gates to 460 km and velocity / spectrum width in 250 m gates to
# 230 km, no dual pol, and the old precipitation VCPs -- 11 (14 tilts in 5 min, for severe storms) and 21 (9 tilts in
# 6 min) -- whose two lowest tilts are split (a low-PRF surveillance sweep for reflectivity, then a high-PRF Doppler
# sweep for velocity).  Level II of the time: message type 1 radials (level2.encode_volume_legacy).
LEGACY_VCPS = (11, 21)
LEGACY_REF_GATES = 4             # 250 m samples averaged into each 1 km reflectivity gate
LEGACY_N_DOP = 920               # 250 m Doppler gates: to ~230 km
# as the legacy RDAs laid their gates out (checked on KTLX 1999-05-03 and 2005-05-12 archives): reflectivity's first
# gate centred at 0 m (460 x 1 km), the Doppler moments' at -375 m (920 x 250 m) -- the gates nearer than the
# model's first (REF_FIRST, 2125 m) are written empty
LEGACY_REF_FIRST, LEGACY_REF_N = 0.0, 460
LEGACY_DOP_FIRST = -375.0


def _legacy_cuts(number: int) -> list[Cut]:
    def c(e, kind, s, nyq):
        return Cut(e, kind, s, nyq, False, n_dop=LEGACY_N_DOP, radials=360)

    if number == 21:
        cuts = ([c(0.5, "surv", 34.0, 8.9), c(0.5, "dop", 30.0, 22.9), c(1.45, "surv", 34.0, 8.9),
                 c(1.45, "dop", 30.0, 22.9)]
                + [c(e, "both", 33.0, 22.9) for e in (2.4, 3.35, 4.3, 6.0)]
                + [c(e, "both", 36.0, 32.9) for e in (9.9, 14.6, 19.5)])
    else:
        cuts = ([c(0.5, "surv", 17.0, 8.9), c(0.5, "dop", 18.0, 27.1), c(1.45, "surv", 17.0, 8.9),
                 c(1.45, "dop", 18.0, 27.1)]
                + [c(e, "both", 20.0, 26.4) for e in (2.4, 3.35, 4.3, 5.25, 6.2)]
                + [c(e, "both", 18.6, 34.3) for e in (7.5, 8.7, 10.0, 12.0, 14.0, 16.7, 19.5)])
    return cuts


def legacy_ref(dbz, has):
    """Reflectivity in 1 km gates the way a legacy RDA made it -- the mean (linear Z) of the 4 x 250 m samples in
    each that hold echo (at least 2 of them) -- on its gates (LEGACY_REF_FIRST, LEGACY_REF_N): (dbz, has)."""
    import cupy as cp
    n, ng = dbz.shape
    k = LEGACY_REF_GATES
    g = 1000.0 * k / 4.0
    # the first whole 1 km gate whose samples (REF_FIRST + 250 i) all lie inside it, and the samples it starts at
    k0 = int(math.ceil((REF_FIRST - REF_GATE / 2 + g / 2 - LEGACY_REF_FIRST) / g))
    s0 = int(round((LEGACY_REF_FIRST + k0 * g - g / 2 + REF_GATE / 2 - REF_FIRST) / REF_GATE))
    m = min((ng - s0) // k, LEGACY_REF_N - k0)
    h = has[:, s0:s0 + m * k].reshape(n, m, k)
    z = cp.where(h, 10.0 ** (0.1 * dbz[:, s0:s0 + m * k].reshape(n, m, k)), 0.0).sum(axis=2)
    cnt = h.sum(axis=2)
    out = cp.full((n, k0 + m), -99.0, cp.float32)
    ok = cp.zeros((n, k0 + m), bool)
    out[:, k0:] = 10.0 * cp.log10(cp.maximum(z / cp.maximum(cnt, 1), 1e-10))
    ok[:, k0:] = cnt >= 2
    return out, ok


VCP_INFO = {
    11: "VCP 11 (1990s) - severe weather, 14 tilts in 5 min",
    21: "VCP 21 (1990s) - precipitation, 9 tilts in 6 min",
    212: "VCP 212 - precipitation, 14 tilts (SAILS-capable)",
    12: "VCP 12 - precipitation, 14 tilts",
    215: "VCP 215 - general precipitation, 15 tilts",
    35: "VCP 35 - clear air / light precipitation",
    TDWR_VCP: "TDWR hazardous mode - C band, 15 scans, base tilt every few tilts",
    DOW_VCP: "DOW tornado mode - X band, 12 tilts 0.5-16 deg, 75 m gates, ~1.5 min",
}


@dataclass
class Radar:
    icao: str
    lat: float
    lon: float
    elev_m: float = 300.0          # site elevation (m MSL), recorded in Level II
    vcp: int = 212
    sails: int = 1
    alias: bool = False            # fold velocities at the cut's Nyquist (raw, as a real RDA sends them)
    volume_s: float = 150.0        # s per volume, every sweep sped up alike; 0 = the VCP's own timing
    floor_dbz: float = -5.0        # gates below this are 'below threshold'
    noise: bool = True             # scatter each gate's moments like a real radar's estimates (add_noise)
    clear_air: float = 1.0         # insects / Bragg echo in the boundary layer and noise speckle (clear_air_z); 0 = none
    dwell_factor: float = 1.0      # VCP scan time / this radar's (faster volumes: fewer pulses per gate)
    tower_m: float = 20.0
    kind: str = "wsr88d"           # "wsr88d" (S band, dual pol), "tdwr" (C band, single pol), "cband"
                                   # (an international C-band radar: dual pol, 1-degree beam, dual-PRF velocity)
                                   # or "dow" (a Doppler on Wheels: X band, 0.93 deg, 75 m gates, parked anywhere)
    gate_m: float = 75.0           # a DOW's gate length (m)
    beamwidth: float = BEAMWIDTH   # deg
    sens_50km: float = -5.0        # dBZ at 50 km with SNR 0 dB
    x: float = 0.0                 # world metres (filled by the engine)
    y: float = 0.0
    export: bool = True
    # schedule
    vol_start: float | None = None
    next_cut: int = 0
    cuts: list = field(default_factory=list)
    vol_sweeps: list = field(default_factory=list)
    latest: dict = field(default_factory=dict)   # last sweep per (elev, moment) for display
    volumes_written: int = 0

    def __post_init__(self):
        if self.kind == "tdwr":
            self.vcp, self.sails, self.beamwidth = TDWR_VCP, 0, 0.55
            self.sens_50km = -2.0                         # a smaller antenna, shorter pulses
            cuts = _tdwr_cuts()
        elif self.kind == "dow":
            # its own scan, not sped up to the run's radar volume time; a ~4 m mast on the truck
            self.vcp, self.sails, self.beamwidth, self.tower_m = DOW_VCP, 0, 0.93, 4.0
            self.sens_50km = 5.0                          # -5 dBZ at 15 km
            self.volume_s = 0.0
            cuts = _dow_cuts(self.gate_m)
        elif self.kind == "legacy":
            self.vcp = self.vcp if self.vcp in LEGACY_VCPS else 11
            self.sails = 0
            cuts = _legacy_cuts(self.vcp)
        else:
            cuts = _vcp_cuts(self.vcp, self.sails if self.vcp in (212, 215) else 0)
            if self.kind == "cband":
                # Europe's, Canada's, Japan's ... C-band radars scan much as a WSR-88D does here, but their
                # shorter wavelength attenuates through heavy cores; dual-PRF processing unfolds their
                # velocities to an extended Nyquist and keeps the long unambiguous range of the low PRF
                self.beamwidth, self.sens_50km = 1.0, -3.0
                cuts = [replace(c, wavelength=TDWR_LAMBDA, unamb=c.unamb_m) for c in cuts]
        if self.volume_s > 0:
            # faster than a WSR-88D can turn (like a phased-array radar), for frequent volumes:
            # the VCP's tilts and order, each sweep shortened by the same factor
            f = self.volume_s / sum(c.seconds for c in cuts)
            cuts = [replace(c, seconds=c.seconds * f) for c in cuts]
            self.dwell_factor = 1.0 / f
        self.cuts = cuts

    @property
    def dualpol(self) -> bool:
        return self.kind not in ("tdwr", "legacy")

    @property
    def volume_seconds(self) -> float:
        return sum(c.seconds for c in self.cuts)

    def nyquist(self, cut: Cut) -> float:
        """The Nyquist velocity recorded in Level II for this cut."""
        return cut.nyquist if self.alias else DEALIASED_NYQUIST

    def to_json(self) -> dict:
        return {"icao": self.icao, "kind": self.kind, "lat": self.lat, "lon": self.lon, "elev": self.elev_m,
                "vcp": self.vcp, "sails": self.sails, "dealias": not self.alias,
                "volume_min": round(self.volume_s / 60.0, 2),
                "floor": self.floor_dbz, "noise": self.noise, "clear_air": self.clear_air,
                "x": self.x, "y": self.y, "export": self.export, "gate": self.cuts[0].gate if self.cuts else None,
                "volumes": self.volumes_written,
                "volume_seconds": round(self.volume_seconds, 1)}


# --------------------------------------------------------------- kernels
_SRC = r"""
#define PI_F 3.14159265f
__device__ __forceinline__ float clampf(float x, float a, float b) { return fminf(fmaxf(x, a), b); }

// ---------------------------------------------------------------------------
// Per-cell radar fields from Morrison moments (Rayleigh, exponential PSDs).
// in : dbz (ArWen refl10cm), T (K), rho (kg/m3), qr, nr, qs, ns, qg, ng (per kg)
// out: zh (linear mm6/m3), zv (linear), rz (rho_hv * sqrt(zh zv): adds up over a beam to the
//      mixture's rho_hv = sum(rho_i sqrt(zh_i zv_i)) / sqrt(sum zh_i * sum zv_i)), kdp (deg/km),
//      vt (m/s, Z-weighted)
extern "C" __global__ void radar_fields(
    const float* dbz, const float* T, const float* rho,
    const float* qr, const float* nr, const float* qs, const float* ns,
    const float* qg, const float* ng,
    float* zh, float* zv, float* rz, float* kdp, float* vt, long long n)
{
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    // dbz == nullptr: no scheme reflectivity -- Z is the Rayleigh sum of the species below
    float Z = dbz ? ((dbz[id] > -30.0f) ? exp10f(0.1f * dbz[id]) : 0.0f) : -1.0f;
    float r = fmaxf(rho[id], 0.05f);
    float t = T[id];
    float dens = powf(1.2f / r, 0.4f);
    // rain: N0 = Nv*lambda, lambda = (pi rho_w Nv / (rho q))^(1/3)
    float zr = 0.0f, zdr_r = 1.0f, rh_r = 0.995f, vr = 0.0f, kd = 0.0f;
    float q = qr[id], nn = nr[id];
    if (q > 1.0e-6f && nn > 1.0f) {
        float nv = nn * r;
        float lam = cbrtf(PI_F * 1000.0f * nv / (r * q));
        float lc = clampf(lam, 357.0f, 5.0e4f);       // Morrison's own bounds: mean diameter 20 um - 2.8 mm
        if (lc != lam) { lam = lc; nv = lam * lam * lam * r * q / (PI_F * 1000.0f); }   // keep the mass
        zr = 720.0f * nv / (lam * lam * lam * lam * lam * lam) * 1.0e18f;
        // median volume diameter (mm) for ZDR: the moments spread as a gamma with shape 2 rather
        // than an exponential (D0 = 1.45 x the mean-mass diameter, not 2.02x), at most 5 mm.
        // Two-moment schemes over-sort rain (Dawson et al. 2014): taken as exponential, a few big
        // drops read as D0 up to ~10 mm and ZDR 5+ dB across whole forward flanks, where real
        // radars show 0.5-2.5 dB and 3-5 dB only in the arc
        float d0 = fminf(2.63f / lam * 1000.0f, 5.0f);
        // invert Brandes et al. (2004): D0 = 0.717 + 1.479 Z - 0.725 Z^2 + 0.171 Z^3
        float lo = 0.0f, hi = 5.5f;
        for (int it = 0; it < 18; ++it) {
            float m = 0.5f * (lo + hi);
            float d = 0.717f + 1.479f * m - 0.725f * m * m + 0.171f * m * m * m;
            if (d < d0) lo = m; else hi = m;
        }
        // light rain and drizzle still read 0.25-0.5 dB (real echo edges are blue-cyan, not grey ~0)
        float zdb = (d0 < 0.717f) ? 0.25f + 0.25f * d0 / 0.717f : fmaxf(0.5f * (lo + hi), 0.5f);
        zdr_r = exp10f(0.1f * zdb);
        rh_r = 0.997f - 0.010f * clampf((d0 - 2.0f) / 3.0f, 0.0f, 1.0f);
        float x = lam / (lam + 600.0f);
        float x2 = x * x;
        vr = (9.65f - 10.3f * x2 * x2 * x2 * x) * dens;          // Z-weighted
        float vm = (9.65f - 10.3f * x2 * x2) * dens;             // mass-weighted
        float R = 3600.0f * r * q * vm;                          // mm/h
        kd = powf(R / 44.0f, 1.0f / 0.822f);
    }
    // snow (rho_s = 100, D^3 mass), dry: |K|^2 0.176/0.93 and density scaling
    float zs = 0.0f;
    q = qs[id]; nn = ns[id];
    if (q > 1.0e-6f && nn > 1.0f) {
        float nv = nn * r;
        float lam = cbrtf(PI_F * 100.0f * nv / (r * q));
        float lc = clampf(lam, 300.0f, 2.0e4f);
        if (lc != lam) { lam = lc; nv = lam * lam * lam * r * q / (PI_F * 100.0f); }
        zs = 0.224f * 0.0119f * 720.0f * nv / powf(lam, 6.0f) * 1.0e18f;
        if (t > 273.15f) zs *= 4.0f;   // melting snow: bright band
    }
    // hail-like rimed ice (rho 900, v = 114.5 D^0.5)
    float zg = 0.0f, vg = 0.0f, d0g = 0.0f;
    q = qg[id]; nn = ng[id];
    if (q > 1.0e-6f && nn > 1.0e-3f) {
        float nv = nn * r;
        float lam = cbrtf(PI_F * 900.0f * nv / (r * q));
        float lc = clampf(lam, 100.0f, 2.0e4f);
        if (lc != lam) { lam = lc; nv = lam * lam * lam * r * q / (PI_F * 900.0f); }
        d0g = 3.67f / lam * 1000.0f;                  // median volume diameter (mm)
        float wet = (t > 273.15f) ? 1.0f : 0.0f;
        float kfac = wet > 0.0f ? 0.81f : 0.224f * 0.963f;
        zg = kfac * 720.0f * nv / powf(lam, 6.0f) * 1.0e18f;
        zg = fminf(zg, 1.0e7f);      // Rayleigh overstates big hail at S band: ~70 dBZ at most
        vg = 114.5f * 1.0587f * sqrtf(7.0f) / sqrtf(lam) * sqrtf(1.2f / r);   // ~ a G(7.5)/G(7) lam^-0.5
    }
    float zsum = zr + zs + zg;
    if (Z < 0.0f) Z = zsum;
    if (Z <= 0.0f || zsum <= 0.0f) {
        zh[id] = Z; zv[id] = Z; rz[id] = 0.99f * Z; kdp[id] = kd; vt[id] = 0.0f;
        return;
    }
    // the scheme's own Z above the species' sum where there is rimed ice: its graupel and hail are
    // bigger than the fixed-intercept guess (Thompson's, by ~10 dB in cores), so the excess is theirs
    // and hail cores keep their low rho_hv and ZDR instead of reading as rain
    if (zg > 0.0f && Z > zsum) { zg += Z - zsum; zsum = Z; }
    // ArWen's Z sets the magnitude; the species mix sets ZDR / rho_hv / fall speed
    float fr = zr / zsum, fs = zs / zsum, fg = zg / zsum;
    bool melt = t > 273.15f;
    float zdr_s = exp10f(0.1f * (melt ? 1.5f : 0.25f));
    float rh_s = melt ? 0.93f : 0.99f;                          // bright band / dry aggregates
    // hail at S band (Ryzhkov et al. 2013; Kumjian 2013): small graupel and hail ~0.98, large hail
    // tumbles (ZDR ~0, slightly negative when big) and decorrelates (rho_hv 0.85-0.92); melting hail
    // under ~1 cm wears a water torus that makes it oblate (ZDR 2-3 dB) and lowers rho_hv further
    float big = clampf((d0g - 5.0f) / 20.0f, 0.0f, 1.0f);     // 5 mm .. 25 mm
    // ... and hail as big as the echo says: a rimed-ice core past 52 dBZ is large hail by 65
    big = fmaxf(big, clampf((10.0f * log10f(Z) - 52.0f) / 13.0f, 0.0f, 1.0f) * fminf(1.0f, 2.0f * fg));
    float zdb_g = melt ? 2.5f - 2.2f * fmaxf(clampf((d0g - 6.0f) / 14.0f, 0.0f, 1.0f), big)
                       : 0.2f - 0.4f * clampf((d0g - 10.0f) / 20.0f, 0.0f, 1.0f);
    float zdr_g = exp10f(0.1f * zdb_g);
    float rh_g = melt ? 0.97f - 0.10f * big : 0.985f - 0.07f * big;
    // the mixture: rho_hv of species with different ZDR is below their own (no fudge factor)
    float zvr = fr / zdr_r + fs / zdr_s + fg / zdr_g;           // Zv / Zh
    float cross = fr * rh_r * rsqrtf(zdr_r) + fs * rh_s * rsqrtf(zdr_s) + fg * rh_g * rsqrtf(zdr_g);
    zh[id] = Z;
    zv[id] = Z * zvr;
    rz[id] = Z * cross;                                         // rho_hv * sqrt(zh zv) of the mix
    kdp[id] = kd;
    vt[id] = fr * vr + fs * 1.0f * dens + fg * vg;
}

// ---------------------------------------------------------------------------
// Sample one sweep from one patch.  Gate (ray, gate) at range r0 + g*dr.
struct SweepArgs {
    float lat0, lon0;              // radar (rad)
    float h0;                      // antenna height AGL (m)
    float elev;                    // rad
    float az0, daz;                // rad, first radial centre and spacing
    float r0, dr;                  // m
    int nrad, ngate;
    // projection
    float pn, pF, prho0, plon0, pR;
    // patch
    float x0, y0, dx;              // west/south edge (world m)
    int nx, ny, nz;
    float ztop;
    int nvort;                     // sub-grid tornadoes (add_debris): x, y, peak wind, core radius, reach, full-strength top,
                                   // how far their winds have grown in (0..1)
    int nvk, nvr;                  // their tables of the model's own tangential wind: levels, radii
    float ca_z0, ca_top;           // clear air (insects, Bragg): linear Z near the ground (0: none), layer top (m)
    float bw;                      // the radar's beamwidth (rad)
    float gx0, gy0, gdx;           // terrain raster around the radar (beam_block): first cell centre (world m), spacing
    int gn;                        // its cells a side (0: flat ground)
    int hole;                      // draw the tornadoes' weak-echo holes (vortex_hole): DOWs only
};

__device__ __forceinline__ float lerp(float a, float b, float t) { return a + (b - a) * t; }

// Where a gate's centre beam is over the ground: distance along the ground, lat/lon, world x/y and the
// projection's rotation there (Lambert)
struct GatePos { float s, lat, dlon, x, y, th; };
__device__ GatePos gate_pos(const SweepArgs& a, float R, float az)
{
    GatePos p;
    float hc = sqrtf(R * R + KE * KE + 2.0f * R * KE * sinf(a.elev)) - KE + a.h0;
    p.s = KE * asinf(R * cosf(a.elev) / (KE + hc));
    float del = p.s / a.pR;
    float sl0 = sinf(a.lat0), cl0 = cosf(a.lat0);
    float sla = sl0 * cosf(del) + cl0 * sinf(del) * cosf(az);
    p.lat = asinf(clampf(sla, -1.0f, 1.0f));
    p.dlon = atan2f(sinf(az) * sinf(del) * cl0, cosf(del) - sl0 * sla);
    float rho = a.pR * a.pF / powf(tanf(0.25f * PI_F + 0.5f * p.lat), a.pn);
    p.th = a.pn * (a.lon0 + p.dlon - a.plon0);
    p.x = rho * sinf(p.th);
    p.y = a.prho0 - rho * cosf(p.th);
    return p;
}

// Ground height (m above the model's zero) on the radar's terrain raster, bilinear; 0 off it
__device__ float ground_at(const float* g, const SweepArgs& a, float x, float y)
{
    float fi = (x - a.gx0) / a.gdx, fj = (y - a.gy0) / a.gdx;
    if (!(fi >= 0.0f) || !(fj >= 0.0f) || !(fi < (float)(a.gn - 1)) || !(fj < (float)(a.gn - 1))) return 0.0f;
    int i = (int)fi, j = (int)fj;
    float ti = fi - i, tj = fj - j;
    const float* r0 = g + (long long)j * a.gn + i;
    return lerp(lerp(r0[0], r0[1], ti), lerp(r0[a.gn], r0[a.gn + 1], ti), tj);
}

// Terrain blocking each radial (one thread a radial, out along it): bit sb of blk[gate] is set once sub-beam sb
// has met the ground -- the beam is blocked from there on, its power lost (partial blocking dims the echo)
extern "C" __global__ void beam_block(SweepArgs a, const float* g, unsigned char* blk)
{
    int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= a.nrad) return;
    float az = a.az0 + ray * a.daz;
    unsigned int b = 0;
    for (int gi = 0; gi < a.ngate; ++gi) {
        float R = a.r0 + gi * a.dr;
        if (b != 31u) {
            GatePos p = gate_pos(a, R, az);
            float hg = ground_at(g, a, p.x, p.y);
            for (int sb = 0; sb < 5; ++sb) {
                float el = a.elev + SUBOFF[sb] * a.bw;
                float h = sqrtf(R * R + KE * KE + 2.0f * R * KE * sinf(el)) - KE + a.h0;
                if (h < hg) b |= 1u << sb;
            }
        }
        blk[(long long)ray * a.ngate + gi] = (unsigned char)b;
    }
}

// Clear-air echo (clear_air_z below, the same formula): insects and Bragg scatter in the boundary
// layer, full strength to 0.4 of its top, 20 dB weaker at the top, gone 600 m above, patchy by +-3 dB
__device__ float clear_air(float z0, float top, float h, float x, float y)
{
    if (z0 <= 0.0f || h > top + 600.0f) return 0.0f;
    float a = fminf(fmaxf((h - 0.4f * top) / (0.6f * top), 0.0f), 1.0f);
    float b = fminf(fmaxf((h - top) / 600.0f, 0.0f), 1.0f);
    float db = -20.0f * a * a * (3.0f - 2.0f * a) - 15.0f * b * b * (3.0f - 2.0f * b)
             + 3.0f * (0.6f * sinf(x * 2.7318e-4f + 1.3f) * sinf(y * 2.0268e-4f + 0.4f)
                       + 0.4f * sinf((x + y) * 5.7120e-4f + 2.1f));
    return z0 * exp10f(0.1f * db);
}

// A tornado the grid cannot hold.  Within its reach the grid smears it into a vortex 3-4 cells
// wide; the radar sees the tornado instead: the model's axisymmetric tangential wind about it
// (vtab: per tornado, per model level, on nvr radii from 0 to the reach) is replaced by the
// tornado's (Rankine inside the core, falling off as r^-0.8 outside) where that is stronger, wholly inside half the reach
// and blending back into the model's at the reach; full strength up to its top, gone 2 km higher.
// The model's inflow, the storm's motion, the shear and any asymmetry stay.  Its peak inbound and
// outbound winds sit in neighbouring radials beyond ~50 km: the gate-to-gate couplet of a TVS.  A
// tornado just found grows in, and one lost dies away, over ~45 s (V[6]), not at once.
__device__ void add_tornado_wind(const float* vort, const float* vtab, int nvort, int nvk, int nvr,
                                 float x, float y, float h, int k, float tk, float* u, float* v)
{
    for (int q = 0; q < nvort; ++q) {
        const float* V = vort + 7 * q;
        float ex = x - V[0], ey = y - V[1];
        float r = sqrtf(ex * ex + ey * ey);
        if (r >= V[4] || r < 1.0f) continue;
        float fz = h <= V[5] ? 1.0f : fmaxf(0.0f, 1.0f - (h - V[5]) / 2000.0f);
        if (fz <= 0.0f) continue;
        float vt = r < V[3] ? V[2] * r / V[3] : V[2] * powf(V[3] / r, 0.8f);
        float fr = r / V[4] * (nvr - 1);
        int j = min((int)fr, nvr - 2);
        float tj = fr - j;
        int k0 = min(k, nvk - 1), k1 = min(k + 1, nvk - 1);
        const float* T = vtab + (long long)q * nvk * nvr;
        float vm = lerp(lerp(T[k0 * nvr + j], T[k0 * nvr + j + 1], tj),
                        lerp(T[k1 * nvr + j], T[k1 * nvr + j + 1], tj), tk);
        float s = fmaxf(0.0f, 2.0f * r / V[4] - 1.0f);
        // never weaker than the model's own vortex there: the tornado sharpens it, it does not take it away
        float d = V[6] * fz * (1.0f - s * s * (3.0f - 2.0f * s)) * fmaxf(vt - vm, 0.0f);
        *u -= d * ey / r;                  // cyclonic (counter-clockwise)
        *v += d * ex / r;
    }
}

// The tornado's weak-echo hole: its winds centrifuge rain and debris out of the core, so the echo has an eye
// there (Bluestein et al.'s and Wurman's close-range data: ~1-1.2 core radii across, 10-20 dB deep, through the
// tornado's depth, the debris ring around it).  The fraction of the echo left at (x, y, h): 1 outside, down to
// 10^(-depth/10) on the axis, depth 8 dB for a 35 m/s tornado to 22 dB from 80 m/s, grown in with its winds
// (V[6]) and fading out above its full-strength top like them.  ZDR and CC stay (Zh, Zv and rho Z alike).
// Drawn for DOWs only (SweepArgs.hole, the user's choice): a DOW a few km away sees the eye.
__device__ float vortex_hole(const float* vort, int nvort, float x, float y, float h)
{
    float keep = 1.0f;
    for (int q = 0; q < nvort; ++q) {
        const float* V = vort + 7 * q;
        float ex = x - V[0], ey = y - V[1];
        float rh = HOLE_R * V[3];
        float r2 = (ex * ex + ey * ey) / (rh * rh);
        if (r2 >= 9.0f) continue;
        float fz = h <= V[5] ? 1.0f : fmaxf(0.0f, 1.0f - (h - V[5]) / 2000.0f);
        float st = fminf(fmaxf((V[2] - 35.0f) / 45.0f, 0.0f), 1.0f);
        float depth = (HOLE_DB0 + (HOLE_DB1 - HOLE_DB0) * st) * V[6] * fz;
        keep *= 1.0f - (1.0f - exp10f(-0.1f * depth)) * expf(-r2);
    }
    return keep;
}

__device__ float tri(const float* f, int nx, int ny, int i, int j, int k, float fi, float fj, float fk, int nz)
{
    long long s = (long long)nx * ny;
    long long b0 = (long long)k * s + (long long)j * nx + i;
    int k1 = (k + 1 < nz) ? 1 : 0;
    long long b1 = b0 + k1 * s;
    float c00 = lerp(f[b0], f[b0 + 1], fi), c01 = lerp(f[b0 + nx], f[b0 + nx + 1], fi);
    float c10 = lerp(f[b1], f[b1 + 1], fi), c11 = lerp(f[b1 + nx], f[b1 + nx + 1], fi);
    return lerp(lerp(c00, c01, fj), lerp(c10, c11, fj), fk);
}

// lzh, lzv, lrz: the natural logs of Zh, Zv and rho_hv sqrt(Zh Zv) (PatchRadarFields.finish): between
// the model's cells the echo is interpolated in dB, as its values span powers of ten.  Interpolated
// as power, a 60 dBZ cell next to a 20 dBZ one reads 57 dBZ halfway: every echo swelled by about half a
// cell on each side, hooks blunted and inflow notches filled.  The beam still adds up power.
extern "C" __global__ void sample_sweep(
    SweepArgs a, const float* zh_h, const float* lzh, const float* lzv, const float* lrz,
    const float* kdp, const float* vt, const float* uc, const float* vc, const float* wc,
    const float* tke, const float* vort, const float* vtab,
    const float* hg, const unsigned char* blk,   // over terrain: the patch's ground (m above the model's zero), beam_block's
    float* acc)   // per gate: [wsum_all, Zsum, Zvsum, rZsum, VZ, V2Z, kdp*w, tkeZ, air weight, air VR]
{
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)a.nrad * a.ngate;
    if (id >= n) return;
    int ray = id / a.ngate, g = id % a.ngate;
    float R = a.r0 + g * a.dr;
    float az = a.az0 + ray * a.daz;
    // centre beam: ground distance, gate lat/lon (great circle on R) and world metres (Lambert)
    GatePos gp = gate_pos(a, R, az);
    float s = gp.s, lat = gp.lat, dlon = gp.dlon, th = gp.th, x = gp.x, y = gp.y;
    float sl0 = sinf(a.lat0), cl0 = cosf(a.lat0);
    float fi = (x - a.x0) / a.dx - 0.5f, fj = (y - a.y0) / a.dx - 0.5f;
    float* o = acc + id * 10;
    // beam direction at the gate, in grid coordinates
    float bb = atan2f(sinf(-dlon) * cl0, cosf(lat) * sl0 - sinf(lat) * cl0 * cosf(-dlon)) + PI_F;
    float azg = bb - th;
    float sa = sinf(azg), ca = cosf(azg);
    float elloc = a.elev + s / KE;
    float ce = cosf(elloc), se = sinf(elloc);
    // Integer bounds (NaN-safe): the interpolation reads cells i..i+1, j..j+1, so i and j
    // must be below nx-1 and ny-1.  A float test against nx - 1.000001f rounds to nx - 1
    // for large patches and let a gate on the last row read past the end of the array.
    if (!(fi >= 0.0f) || !(fj >= 0.0f) || !(fi < (float)a.nx) || !(fj < (float)a.ny)) return;
    int i = (int)fi, j = (int)fj;
    if (i >= a.nx - 1 || j >= a.ny - 1) return;
    float ti = fi - i, tj = fj - j;
    // the ground under the gate; the model's levels are squeezed over it (sigma), so the beam's height above it is
    // stretched back to the flat levels zh_h
    float hgr = hg ? lerp(lerp(hg[j * a.nx + i], hg[j * a.nx + i + 1], ti),
                          lerp(hg[(j + 1) * a.nx + i], hg[(j + 1) * a.nx + i + 1], ti), tj) : 0.0f;
    unsigned int blocked = blk ? blk[id] : 0u;
    for (int sb = 0; sb < 5; ++sb) {
        float off = SUBOFF[sb] * a.bw;
        float wgt = expf(-TWO_WAY * SUBOFF[sb] * SUBOFF[sb]);
        float el = a.elev + off;
        if ((blocked >> sb) & 1u) continue;               // this sub-beam hit the terrain nearer the radar
        float hm = sqrtf(R * R + KE * KE + 2.0f * R * KE * sinf(el)) - KE + a.h0;
        if (hm < hgr || hm > a.ztop) continue;            // below ground / above the model
        float h = hg ? (hm - hgr) / (1.0f - hgr / a.ztop) : hm;
        // vertical index
        int k = 0; float tk = 0.0f;
        if (h <= zh_h[0]) { k = 0; tk = 0.0f; }
        else if (h >= zh_h[a.nz - 1]) { k = a.nz - 1; tk = 0.0f; }
        else {
            int lo = 0, hi = a.nz - 1;
            while (hi - lo > 1) { int m = (lo + hi) >> 1; if (zh_h[m] <= h) lo = m; else hi = m; }
            k = lo; tk = (h - zh_h[lo]) / (zh_h[lo + 1] - zh_h[lo]);
        }
        o[0] += wgt;                                   // this gate is inside a patch
        float lz = tri(lzh, a.nx, a.ny, i, j, k, ti, tj, tk, a.nz);
        float Za = clear_air(a.ca_z0, a.ca_top, h, x, y);
        if (lz < LZ_MIN && Za < 1.0e-4f) continue;    // below -40 dBZ and no clear air: no echo
        float Z = 0.0f, Zv = 0.0f, rZ = 0.0f;
        if (lz >= LZ_MIN) {
            Z = expf(lz);
            Zv = expf(tri(lzv, a.nx, a.ny, i, j, k, ti, tj, tk, a.nz));
            rZ = expf(tri(lrz, a.nx, a.ny, i, j, k, ti, tj, tk, a.nz));
        }
        float kd = tri(kdp, a.nx, a.ny, i, j, k, ti, tj, tk, a.nz);
        float v_t = tri(vt, a.nx, a.ny, i, j, k, ti, tj, tk, a.nz);
        float u = tri(uc, a.nx, a.ny, i, j, k, ti, tj, tk, a.nz);
        float v = tri(vc, a.nx, a.ny, i, j, k, ti, tj, tk, a.nz);
        float w = tri(wc, a.nx, a.ny, i, j, k, ti, tj, tk, a.nz);
        float tk2 = tke ? tri(tke, a.nx, a.ny, i, j, k, ti, tj, tk, a.nz) : 0.0f;
        if (a.nvort > 0) {
            add_tornado_wind(vort, vtab, a.nvort, a.nvk, a.nvr, x, y, h, k, tk, &u, &v);
            if (a.hole) {
                float keep = vortex_hole(vort, a.nvort, x, y, h);
                Z *= keep; Zv *= keep; rZ *= keep;
            }
        }
        if (Z == 0.0f) { Zv = 0.0f; rZ = 0.0f; }
        float va = (u * sa + v * ca) * ce + w * se;      // the air's own: insects and Bragg do not fall
        float vr = va - v_t * se;
        float wz = wgt * Z, wa = wgt * Za;
        o[1] += wz; o[2] += wgt * Zv; o[3] += wgt * rZ;
        o[4] += wz * vr; o[5] += wz * vr * vr; o[6] += wgt * kd; o[7] += wz * tk2;
        // the clear air's own radial wind, weighted through the boundary layer: what the
        // contamination layer's insects carry (add_contamination)
        o[8] += wa; o[9] += wa * va;
    }
}
"""


def _sweep_struct_dtype():
    return np.dtype([
        ("lat0", "f4"), ("lon0", "f4"), ("h0", "f4"), ("elev", "f4"), ("az0", "f4"), ("daz", "f4"),
        ("r0", "f4"), ("dr", "f4"), ("nrad", "i4"), ("ngate", "i4"),
        ("pn", "f4"), ("pF", "f4"), ("prho0", "f4"), ("plon0", "f4"), ("pR", "f4"),
        ("x0", "f4"), ("y0", "f4"), ("dx", "f4"), ("nx", "i4"), ("ny", "i4"), ("nz", "i4"),
        ("ztop", "f4"), ("nvort", "i4"), ("nvk", "i4"), ("nvr", "i4"), ("ca_z0", "f4"), ("ca_top", "f4"), ("bw", "f4"),
        ("gx0", "f4"), ("gy0", "f4"), ("gdx", "f4"), ("gn", "i4"), ("hole", "i4"),
    ], align=True)


GROUND_REACH = 470000.0           # m: the terrain raster around a radar (beam_block) reaches this far
GROUND_DX = 1000.0                # m: its spacing (the terrain data is ~1 km)


def ground_raster(radar: Radar, terrain):
    """The terrain around a radar for beam_block (model ground heights, m above the model's zero, on the device)
    and the radar's own ground there: (raster, x0, y0, dx, n, ground).  Made once per radar and terrain."""
    import cupy as cp
    key = (id(terrain), round(radar.x), round(radar.y))
    got = getattr(radar, "_ground", None)
    if got is None or got[0] != key:
        n = int(2 * GROUND_REACH / GROUND_DX) + 1
        x0, y0 = radar.x - GROUND_REACH, radar.y - GROUND_REACH
        X, Y = np.meshgrid(x0 + np.arange(n) * GROUND_DX, y0 + np.arange(n) * GROUND_DX)
        g = cp.asarray(terrain.height(X, Y), dtype=cp.float32)
        h = float(terrain.height(np.array([radar.x]), np.array([radar.y]))[0])
        radar._ground = got = (key, g, x0, y0, GROUND_DX, n, h)
    return got[1:]


_MOD = {}
HOLE_R = 1.1                       # x the tornado's core radius: the weak-echo hole's e-folding radius (vortex_hole)
HOLE_DB = (8.0, 22.0)              # dB deep on its axis: a 35 m/s tornado .. 80 m/s and up


def _module():
    import cupy as cp
    if "m" not in _MOD:
        src = ("#define KE %.1ff\n#define BW %.8ff\n#define TWO_WAY %.6ff\n#define LZ_MIN %.6ff\n"
               "#define HOLE_R %.3ff\n#define HOLE_DB0 %.2ff\n#define HOLE_DB1 %.2ff\n"
               "__constant__ float SUBOFF[5] = {%s};\n"
               % (KE_RE, math.radians(BEAMWIDTH), TWO_WAY, math.log(Z_FLOOR) + 0.01, HOLE_R, HOLE_DB[0], HOLE_DB[1],
                  ", ".join(f"{o:.3f}f" for o in SUB_OFFSETS))) + _SRC
        _MOD["m"] = cp.RawModule(code=src, options=("--std=c++14", "--use_fast_math"))
    return _MOD["m"]


def _launch(name, n, args, tpb=256):
    _module().get_function(name)(((int(n) + tpb - 1) // tpb,), (tpb,), args)


# ----------------------------------------------------------- patch fields
def _diag_number(q, rho, rho_x: float, n0):
    """Number per kg of an exponential size distribution with intercept n0 (m^-4)
    holding q (kg/kg) of particles of density rho_x: N = n0 / lambda."""
    import cupy as cp
    qq = cp.maximum(q, 1.0e-12)
    lam = (math.pi * rho_x * n0 / (rho * qq)) ** 0.25
    return cp.where(q > 1.0e-9, n0 / lam / rho, 0.0)


def radar_species(st, mp: int, T, rho):
    """(qr, nr, qs, ns, qg, ng) per kg for any ArWen microphysics scheme.  Hail
    joins graupel (the radar's rimed-ice class); number moments a scheme does
    not predict come from WSM / Lin-style fixed-intercept distributions."""
    import cupy as cp

    def get(*names):
        for n in names:
            a = getattr(st, n, None)
            if a is not None:
                return a
        return None

    zero = cp.zeros_like(T)
    qr, nr = get("qr"), get("nr", "qnr")
    qs, ns = get("qs"), get("ns", "qns")
    qg, ng = get("qg"), get("ng", "qng")
    qh, nh = get("qh"), get("nh", "qnh")
    if mp == 3 and qr is not None:                   # WSM3: precipitation is snow below freezing
        cold = T <= 273.15
        qs, ns = cp.where(cold, qr, 0.0), None
        qr = cp.where(cold, 0.0, qr)
    if get("qir") is not None:                       # P3: one ice category, split by rime fraction
        fr = cp.clip(st.qir / cp.maximum(st.qi, 1.0e-12), 0.0, 1.0)
        qs, ns, qg, ng = st.qi * (1.0 - fr), st.ni * (1.0 - fr), st.qi * fr, st.ni * fr
    if mp == 5 and qs is None and get("qi") is not None:   # Ferrier-Aligo: total ice + rime factor
        rimef = get("f_rimef")
        heavy = (rimef > 5.0) if rimef is not None else cp.zeros(T.shape, bool)
        qs, qg = cp.where(heavy, 0.0, st.qi), cp.where(heavy, st.qi, 0.0)
        ns = ng = None
    qr = zero if qr is None else qr
    if nr is None:
        nr = _diag_number(qr, rho, 1000.0, 8.0e6)
    if qs is None:
        qs, ns = zero, zero
    elif ns is None:
        n0s = cp.minimum(2.0e6 * cp.exp(0.12 * (273.15 - T)), 1.0e11)
        ns = _diag_number(qs, rho, 100.0, n0s)
    if qg is None:
        qg, ng = zero, zero
    elif ng is None:
        ng = _diag_number(qg, rho, 917.0, 4.0e4) if mp == 2 else _diag_number(qg, rho, 500.0, 4.0e6)
    if qh is not None:
        if nh is None:
            nh = _diag_number(qh, rho, 900.0, 4.0e4)
        qg, ng = qg + qh, ng + nh
    f32 = lambda a: cp.ascontiguousarray(a, dtype=cp.float32)
    return tuple(f32(a) for a in (qr, nr, qs, ns, qg, ng))


def _wrf_refl(st, cfg):
    """gpuwm's REFL_10CM for the state's scheme (a copy: it lives in a reused scratch slot), or None
    for a scheme it has no between-steps operator for."""
    import cupy as cp
    try:
        from gpuwm.da.obsop import simulated_reflectivity
        return cp.ascontiguousarray(simulated_reflectivity(st, cfg), dtype=cp.float32).copy()
    except Exception:
        return None


Z_FLOOR = 1.0e-4                  # mm6/m3 (-40 dBZ): the log fields' value where there is no echo

# Wet graupel and hail.  WRF's Morrison reflectivity (refl10cm_hm) sees rimed ice dry: ice's dielectric factor
# (|K|^2 0.176 against water's 0.93) at its own density, ~7 dB under a water drop of its size for hail and more for
# graupel.  A real core's 60-70 dBZ is wet hail -- melting below the freezing level, wet growth just above it --
# which scatters like water.  Rain alone tops out near 55 dBZ, so a run's cores sat on a flat ~52-55 dBZ plateau
# with nothing above.  WET_T: the rimed ice's wet share, 0 at WET_T[0], all of it from WET_T[1] (melting) on; it
# then reads as water drops of its size, the gain at most WET_GAIN_DB (a melting bright band's ~10 dB).  Exponential
# sizes (Morrison's graupel and hail), Rayleigh; DBZ_MAX caps what Rayleigh overstates for the largest hail at S band.
WET_T = (263.15, 273.15)
WET_GAIN_DB = 10.0
DBZ_MAX = 75.0


def wet_rimed_ice(dbz, qg, ng, T, rho, hail: bool):
    """``dbz`` (WRF's dry refl10cm) with the wet part of the rimed ice read as water (WET_T, WET_GAIN_DB)."""
    import cupy as cp
    rho_g = 900.0 if hail else 400.0                         # Morrison's hail / graupel bulk density
    q = cp.maximum(qg, 0.0) * rho                            # kg/m3
    n = cp.maximum(ng, 0.0) * rho                            # 1/m3
    has = (q > 1.0e-6) & (n > 1.0e-3)
    lam = cp.clip((math.pi * rho_g * cp.maximum(n, 1.0e-3) / cp.maximum(q, 1.0e-12)) ** (1.0 / 3.0), 500.0, 1.0e5)
    z_wet = 720.0 * n / lam ** 6 * 1.0e18                    # mm6/m3: drops of its sizes (D^6 moment)
    z_dry = 0.224 * (rho_g / 1000.0) ** 2 * z_wet             # as refl10cm sees it
    gain = min(10.0 ** (WET_GAIN_DB / 10.0), 1.0 / (0.224 * (rho_g / 1000.0) ** 2))
    wet = cp.clip((T - WET_T[0]) / (WET_T[1] - WET_T[0]), 0.0, 1.0)
    add = cp.where(has, z_dry * (gain - 1.0) * wet, 0.0)
    z = 10.0 ** (dbz.astype(cp.float32) / 10.0) + add
    out = cp.where(add > 0.0, cp.minimum(10.0 * cp.log10(cp.maximum(z, 1.0e-6)), DBZ_MAX), dbz)
    return cp.ascontiguousarray(out, dtype=cp.float32)


class PatchRadarFields:
    """Radar-ready fields of one patch at one moment (linear Z etc., and their logs for sample_sweep).

    A run builds them in two steps (build_fields): the echo and the circulations of every patch, then,
    once the tornadoes have been followed from the last refresh (TornadoTracker), their winds and
    debris (finish).  Built alone (tests, map products) a patch finishes at once, each tornado's
    debris fully grown.  ``refine``: join hooks to their tips and sharpen the echo (connect_hooks,
    sharpen_echo)."""

    def __init__(self, patch, finish: bool = True, refine: bool = False):
        import cupy as cp
        from gpuwm.core.morrison import launch_refl10cm_wrf

        st, env = patch.state, patch.env
        mp = int(env.physics.mp_physics)
        nz, ny, nx = st.p.shape
        self.patch_id = patch.id
        self.x0, self.y0 = patch.world_x0(), patch.world_y0()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = env.grid.dx
        thb = st.thb if st.thb.ndim == 3 else st.thb[:, None, None]
        pii = (st.p / 1.0e5) ** (287.0 / 1004.5)
        T = ((thb + st.thp) * pii).astype(cp.float32)
        rho = (st.p / (287.0 * T * (1.0 + 0.608 * st.qv))).astype(cp.float32)
        if mp == 10:
            # ArWen's own Morrison reflectivity (WRF refl10cm_hm)
            dbz = cp.empty((nz, ny, nx), dtype=cp.float32)
            launch_refl10cm_wrf(st.qv, st.qr, st.nr, st.qs, st.ns, st.qg, st.ng, T, st.p, dbz,
                                morr_rimed_ice=env.physics.morr_rimed_ice)
            dbz = wet_rimed_ice(dbz, st.qg, st.ng, T, rho, bool(env.physics.morr_rimed_ice))
            species = (st.qr, st.nr, st.qs, st.ns, st.qg, st.ng)
        else:
            # the scheme's own WRF 10-cm reflectivity (Thompson's calc_refl10cm with its graupel number,
            # WSM6, WDM6, Milbrandt, P3, NSSL ...); the fixed-intercept species below only set the
            # mix for ZDR / rho_hv / fall speed.  They alone read Thompson's big graupel ~10 dB low.
            dbz = _wrf_refl(st, patch.cfg)
            if mp == 18 and dbz is not None:
                # NSSL's radardd02 floors at 0 dBZ, not -35 (gpuwm da/obsop._nssl_reflectivity): clear air read
                # as echo above the radars' floor, so the whole domain showed up as a square with noisy velocity
                dbz = cp.where(dbz <= 0.0, cp.float32(-35.0), dbz)
            species = radar_species(st, mp, T, rho)
        self.T = T
        shp = (nz, ny, nx)
        self.zh = cp.empty(shp, cp.float32)
        self.zv = cp.empty(shp, cp.float32)
        self.rz = cp.empty(shp, cp.float32)
        self.kdp = cp.empty(shp, cp.float32)
        self.vt = cp.empty(shp, cp.float32)
        n = nz * ny * nx
        _launch("radar_fields", n, (dbz if dbz is not None else np.intp(0), T, rho, *species,
                                    self.zh, self.zv, self.rz, self.kdp, self.vt, np.int64(n)))
        if dbz is None:
            dbz = cp.where(self.zh > 1.0e-3, 10.0 * cp.log10(cp.maximum(self.zh, 1.0e-3)), -35.0).astype(cp.float32)
        self.dbz = dbz
        mx, my = env.motion                          # ground-relative winds on a moving grid
        self.uc = cp.ascontiguousarray(0.5 * (st.u[:, :, :-1] + st.u[:, :, 1:]) + mx)
        self.vc = cp.ascontiguousarray(0.5 * (st.v[:, :-1, :] + st.v[:, 1:, :]) + my)
        self.wc = cp.ascontiguousarray(0.5 * (st.w[:-1] + st.w[1:]))
        tke = getattr(st, "tke", None)
        self.tke = cp.ascontiguousarray(tke[:nz]) if tke is not None and tke.shape[0] >= nz else None
        self.z_h = env.column_device("z_h") if hasattr(env, "column_device") else cp.asarray(
            env.z_h, dtype=cp.float32)
        self.ztop = float(env.z_w[-1])
        # over terrain: each column's ground (m above the model's zero), its levels squeezed over it (sample_sweep)
        self.hgt = (cp.ascontiguousarray(st.ht, dtype=cp.float32)
                    if getattr(env, "terrain", None) is not None else None)
        self.tornadoes = None                    # (n, 7) float32: sub-grid tornadoes (add_debris)
        self.tornado_winds = None                # (n, levels, radii): the model's tangential wind about them
        self.clouds = []                         # the tornadoes whose winds and debris finish() adds
        self.debris = False
        self.clean_box = None                    # (box, clean log fields) where the debris changed them (finish)
        self.lzh = self.lzv = self.lrz = None    # logs of zh, zv, rz (finish)
        self._seed = (patch.id * 1000003 + int(getattr(env, "clock", 0.0))) & 0x7FFFFFFF
        # the circulations: tornadoes, and the low-level rotation a hook wraps around.  Storm-scale features (km
        # across) only on grids of FEATURE_MIN_DX and coarser: on a ~10 / ~1 m zoom level their windows ran to
        # thousands of cells (33 s of every refresh at 1.2 m), on a box smaller than the features; the coarser
        # levels around it carry them
        self.thp0 = st.thp[0]                    # the lowest level's theta': is there RFD / outflow air
        if self.dx >= FEATURE_MIN_DX:
            self.vrot = lowlevel_vrot(self.uc, self.vc, env.z_h, self.dx)
            add_fine_lines(self, env, (patch.id * 7919 + int(getattr(env, "clock", 0.0))) & 0x7FFFFFFF)
            add_zdr_arc(self, env, (patch.id * 6271 + int(getattr(env, "clock", 0.0))) & 0x7FFFFFFF)
            add_v_notch(self, env)
        else:
            self.vrot = None
        self.found = find_tornadoes(self, env)
        if refine:
            connect_hooks(self, env)
            sharpen_echo(self)
        if finish:
            self.finish(env, [Tornado.grown(d) for d in self.found])

    def finish(self, env, tornadoes) -> None:
        """The tornadoes' winds and debris (``tornadoes``: TornadoTracker's, or Tornado.grown), then the
        log fields sample_sweep interpolates."""
        import cupy as cp
        self.clouds = list(tornadoes)
        self.tornadoes = self.tornado_winds = None
        # a DOW scanning (env.radar_clean): it is to see no debris balls, so the echo as it was before the debris
        # is kept where the debris changed it (clean_box), for sample_cut to swap in for it
        before = (self.zh.copy(), self.zv.copy(), self.rz.copy()) if getattr(env, "radar_clean", False) and self.clouds else None
        self.debris = add_debris(self, env, self._seed)
        self.lzh = cp.log(cp.maximum(self.zh, Z_FLOOR))
        self.lzv = cp.log(cp.maximum(self.zv, Z_FLOOR))
        self.lrz = cp.log(cp.maximum(self.rz, Z_FLOOR))
        self.clean_box = None
        if before is not None and self.debris:
            changed = (self.zh != before[0]) | (self.zv != before[1]) | (self.rz != before[2])
            kk = cp.nonzero(changed.any(axis=(1, 2)))[0]
            if kk.size:
                col = changed.any(axis=0)
                jj, ii = cp.nonzero(col.any(axis=1))[0], cp.nonzero(col.any(axis=0))[0]
                box = (slice(0, int(kk.max()) + 1), slice(int(jj.min()), int(jj.max()) + 1),
                       slice(int(ii.min()), int(ii.max()) + 1))
                # as a difference in the log fields: what _crossfade adds to them later stays (log weights)
                self.clean_box = (box, tuple(cp.log(cp.maximum(a[box], Z_FLOOR)) - lf[box] for a, lf in
                                             zip(before, (self.lzh, self.lzv, self.lrz))))
            del changed
        del before
        self.vrot = None

    def without_debris(self):
        """Context: the log fields as they were before the debris balls (a DOW's sweep), put back after."""
        import contextlib

        @contextlib.contextmanager
        def swap():
            if self.clean_box is None:
                yield
                return
            box, delta = self.clean_box
            arrs = (self.lzh, self.lzv, self.lrz)
            held = [a[box].copy() for a in arrs]
            try:
                for a, d in zip(arrs, delta):
                    a[box] += d
                yield
            finally:
                for a, h in zip(arrs, held):
                    a[box] = h
        return swap()


def build_fields(patches, env, tracker=None, refine: bool = False, nests=(), holes=()) -> dict:
    """Every patch's radar fields at one moment of a run ({patch id: PatchRadarFields}): the echo and
    the circulations, then the tornadoes followed by ``tracker`` (their debris grown or fallen out
    since its last refresh) with their winds and debris.  ``nests``: refinement nests' patches (every level,
    each with its ``nest_level``), sampled instead of the coarser grids inside ``holes`` (ground x0, y0, x1, y1,
    level, id, d0, ramp each: nest.Nests.holes): the finest grid at a point is seen, crossfaded from the
    coarser one just inside each nest (_crossfade), so no seam shows at its edge."""
    level = {p.id: 0 for p in patches}
    level.update({p.id: int(getattr(p, "nest_level", 1)) for p in nests})
    fields = {p.id: PatchRadarFields(p, finish=False, refine=refine) for p in list(patches) + list(nests)}
    for pid, f in fields.items():
        over = [h for h in holes if h[4] > level[pid]]
        f.found = [d for d in f.found if not _in_holes(d["x"], d["y"], over)]
    found = _merge_tornadoes([d for f in fields.values() for d in f.found])
    tors = (tracker.update(float(getattr(env, "clock", 0.0)), found) if tracker is not None
            else [Tornado.grown(d) for d in found])
    for pid, f in fields.items():
        f.finish(env, tors)
        _crossfade(f, [h for h in holes if h[4] > level[pid]], next((h for h in holes if h[5] == pid), None))
    return fields


def _in_holes(x: float, y: float, holes) -> bool:
    return any(h[0] <= x < h[2] and h[1] <= y < h[3] for h in holes)


def _hole_weight(f: "PatchRadarFields", hole):
    """0..1 on the field's columns: how much the nest of ``hole`` takes over there (0 outside it and in its
    boundary zone, 1 from d0 + ramp inside its edge), or None where the field and the nest do not meet."""
    import cupy as cp
    x0, y0, x1, y1, _, _, d0, ramp = hole
    if x1 <= f.x0 or x0 >= f.x0 + f.nx * f.dx or y1 <= f.y0 or y0 >= f.y0 + f.ny * f.dx:
        return None
    X = f.x0 + (cp.arange(f.nx, dtype=cp.float32) + 0.5) * f.dx
    Y = f.y0 + (cp.arange(f.ny, dtype=cp.float32) + 0.5) * f.dx
    dx_in = cp.minimum(X - x0, x1 - X)[None, :]
    dy_in = cp.minimum(Y - y0, y1 - Y)[:, None]
    d = cp.minimum(dx_in, dy_in)                     # m inside the nest's edge (negative outside)
    return cp.clip((d - d0) / ramp, 0.0, 1.0)


def _crossfade(f: "PatchRadarFields", over, own) -> None:
    """Each grid's share of the echo where nests overlap it: a coarser grid fades out across a ramp just inside
    each finer nest (``over``) as that nest (``own``, for a nest's own field) fades in, so the shares add to 1
    and reflectivity, velocity and dual-pol run on across the edge.  A hard cut left a low-echo line there:
    the sampler interpolates log Z, and the masked cells pulled the coarse grid's values down for a cell
    outside the nest.  Applied as log(weight) on the log fields (Z, Zv and rho's Z scale with it) and to KDP."""
    import cupy as cp
    w = None
    for h in over:
        a = _hole_weight(f, h)
        if a is not None:
            w = (1.0 - a) if w is None else w * (1.0 - a)
    if own is not None:
        a = _hole_weight(f, own)
        if a is not None:
            w = a if w is None else w * a
    if w is None:
        return
    floor = math.log(Z_FLOOR)
    lw = cp.where(w > 1.0e-6, cp.log(cp.maximum(w, 1.0e-6)), -1.0e30).astype(cp.float32)[None]
    for a in (f.lzh, f.lzv, f.lrz):
        a[...] = cp.maximum(a + lw, floor)
    f.kdp *= w.astype(f.kdp.dtype)[None]


# Outflow boundaries as a radar sees them: a faint, noisy fine line (FINE_DBZ) along a cold pool's
# leading edge, where the low-level convergence piles up insects and turbulent (Bragg-scattering) air --
# the gust fronts that run out ahead of storms on real low-level scans.  Where the air ~250 m up converges
# (FINE_CONV) across a sharp surface theta' gradient (FINE_GRAD), in the lowest ~1.5 km, with insects'
# dual-pol (ZDR ~3 dB, CC ~0.6); grainy along the line, so it reads as a broken line, not a stripe.
# Subtle and close in only: a shallow line that a radar's low beam catches within ~20 km and loses by
# FINE_RANGE[1] (the beam rising over the layer and the sensitivity falling off with range), one cell wide
# (the ridge of the convergence), and only where a strong cold pool (FINE_COLD) pushes hard into the
# inflow: ordinary outflow edges show nothing.
FINE_CONV = (2.0e-3, 5.0e-3)     # 1/s of convergence: none .. full
FINE_GRAD = (0.3, 0.8)           # K/km of surface theta' gradient: none .. full (a 1 km grid smears a gust
                                 # front to ~0.3-0.5 K/km, so only the sharpest show)
FINE_COLD = (3.0, 6.0)           # K: the cold pool's theta' deficit within ~3 km behind the edge: none .. full
FINE_DBZ = (8.0, 25.0)           # dBZ: faintest .. strongest (capped at 25)
FINE_TOP = (700.0, 1200.0)       # m: full below, none above
FINE_RANGE = (20000.0, 45000.0)  # m from the nearest radar: full within, gone beyond


def add_fine_lines(f: "PatchRadarFields", env, seed: int) -> None:
    """The outflow boundaries' fine lines in the patch's Zh, Zv and rho_hv (in place)."""
    import cupy as cp
    from cupyx.scipy import ndimage
    dx = f.dx
    zh_host = np.asarray(env.z_h)
    kz = min(int(np.searchsorted(zh_host, 250.0)), f.nz - 1)
    u, v = f.uc[kz], f.vc[kz]
    conv = -(cp.gradient(u, dx, axis=1) + cp.gradient(v, dx, axis=0))
    T = f.thp0.astype(cp.float32)
    gy, gx = cp.gradient(T, dx)
    grad = cp.hypot(gx, gy) * 1000.0                                      # K/km
    a = cp.clip((conv - FINE_CONV[0]) / (FINE_CONV[1] - FINE_CONV[0]), 0.0, 1.0)
    b = cp.clip((grad - FINE_GRAD[0]) / (FINE_GRAD[1] - FINE_GRAD[0]), 0.0, 1.0)
    # on the warm side of the edge: the cold air itself is not the line
    behind = ndimage.minimum_filter(T, size=7)
    warm = cp.clip((T - behind - 0.3) / 1.0, 0.0, 1.0)
    cold = cp.clip((-behind - FINE_COLD[0]) / (FINE_COLD[1] - FINE_COLD[0]), 0.0, 1.0)
    line = a * b * cold * (0.4 + 0.6 * warm)
    if float(line.max()) < 0.05:
        return
    # thin: only the ridge across the line, one cell wide
    line = cp.where(line >= 0.9 * ndimage.maximum_filter(line, size=3), line, 0.0)
    sites = getattr(env, "radar_xy", None)                                # the run's radars (sim)
    if sites:
        ny, nx = line.shape
        X = f.x0 + (cp.arange(nx, dtype=cp.float32) + 0.5) * dx
        Y = f.y0 + (cp.arange(ny, dtype=cp.float32) + 0.5) * dx
        d = cp.full(line.shape, cp.inf, dtype=cp.float32)
        for sx, sy in sites:
            d = cp.minimum(d, cp.hypot(X[None, :] - sx, Y[:, None] - sy))
        line = line * cp.clip((FINE_RANGE[1] - d) / (FINE_RANGE[1] - FINE_RANGE[0]), 0.0, 1.0)
        if float(line.max()) < 0.05:
            return
    rs = cp.random.RandomState(seed)
    grain = 0.45 + 0.9 * rs.random_sample(line.shape, dtype=cp.float32)          # a broken, noisy line
    s = cp.clip(line * grain, 0.0, 1.0)
    z = cp.where(s > 0.08, 10.0 ** (0.1 * (FINE_DBZ[0] + (FINE_DBZ[1] - FINE_DBZ[0]) * s)), 0.0)
    kt = min(int(np.searchsorted(zh_host, FINE_TOP[1])) + 1, f.nz)
    wz = cp.asarray(np.clip((FINE_TOP[1] - zh_host[:kt]) / (FINE_TOP[1] - FINE_TOP[0]), 0.0, 1.0), cp.float32)
    add = (z[None] * wz[:, None, None]).astype(cp.float32)
    f.zh[:kt] += add
    f.zv[:kt] += add / 2.0                                                # ZDR ~3 dB, as insects give
    f.rz[:kt] += 0.6 * add / math.sqrt(2.0)


# The ZDR arc (Kumjian & Ryzhkov 2008, 2009; Dawson et al. 2015): along the inflow edge of a supercell's forward
# flank, from the hook downstream, the storm-relative inflow sorts the rain so only a few big drops fall there --
# ZDR 3-5+ dB in modest (25-50 dBZ) echo, with high CC, in the lowest 1-3 km; stronger the more low-level
# helicity the storm ingests.  The grid's own drop sorting cannot be trusted for it (two-moment schemes over-sort
# everywhere; radar_fields caps D0, which left no arc at all: 0.5 deg ZDR topped out near 2.8 dB), so it is
# added where a storm has one: a rotating low-level updraft (ARC_VROT) with its rear-flank downdraft's outflow
# developing or under way next to it (ARC_RFD: theta' near the ground) -- no RFD, no arc.
ARC_VROT = (4.0, 10.0)        # m/s of low-level rotation (lowlevel_vrot): none .. full (a 500 m grid's strong
                              # supercell reads 6-8 m/s here; its 2-5 km UH 1000-2000)
ARC_UPDRAFT = 2.0             # m/s at ~1 km near it
ARC_RFD = (-0.5, -2.5)        # K: the coldest theta' within ARC_RFD_KM at the ground: none .. full
ARC_RFD_KM = 8.0
ARC_REACH = (2000.0, 22000.0) # m from the circulation: full from the first, faded out by the second
ARC_DBZ = (18.0, 25.0, 50.0, 56.0)   # dBZ: in from the first, full 25-50, out by the last (no arc in the hail)
ARC_GRAD = 1.0                # dB/km falling toward the inflow: full edge
ARC_ZDR = (2.2, 4.6)          # dB: the arc's ZDR, weakest .. strongest (5-6 right by the hook, ARC_HOOK)
ARC_WIDTH = 3500.0           # m: the band's half width across the edge (real arcs are 5-10 km wide)
ARC_HOOK = 5000.0            # m from the circulation: the arc's strongest, up to +1 dB
ARC_TOP = (1500.0, 3000.0)    # m: full below, none above
ARC_MAX = 4                   # circulations per patch


def add_zdr_arc(f: "PatchRadarFields", env, seed: int) -> None:
    """Each mature supercell's ZDR arc in the patch's Zv and rho_hv (in place; Zh unchanged)."""
    import cupy as cp
    from cupyx.scipy import ndimage
    if f.vrot is None:
        return
    dx = f.dx
    zh_host = np.asarray(env.z_h)
    k1 = min(int(np.searchsorted(zh_host, 1000.0)), f.nz - 1)
    m = max(3, int(round(2.0 * max(TVS_RADIUS, 2.5 * dx) / dx)) | 1)
    rot = cp.clip((f.vrot - ARC_VROT[0]) / (ARC_VROT[1] - ARC_VROT[0]), 0.0, 1.0)
    rot = rot * (ndimage.maximum_filter(f.wc[k1], size=m) >= ARC_UPDRAFT)
    n = max(1, int(round(ARC_RFD_KM * 1000.0 / dx)))
    cold = ndimage.minimum_filter(f.thp0.astype(cp.float32), size=2 * n + 1)
    rfd = cp.clip((cold - ARC_RFD[0]) / (ARC_RFD[1] - ARC_RFD[0]), 0.0, 1.0)
    score = rot * rfd
    peak = (score >= 0.15) & (score >= ndimage.maximum_filter(score, size=int(round(12000.0 / dx)) | 1))
    if not bool(peak.any()):
        return
    jj, ii = cp.nonzero(peak)
    order = cp.argsort(-score[jj, ii])[:ARC_MAX]
    centres = [(int(j), int(i), float(score[j, i])) for j, i in zip(cp.asnumpy(jj[order]), cp.asnumpy(ii[order]))]
    # storm-relative winds of the sounding: the low-level inflow (0-1 km) and where the forward flank lies (3-6 km)
    zz = zh_host
    cx, cy = env.storm_motion
    lo, mid = zz <= 1000.0, (zz >= 3000.0) & (zz <= 6000.0)
    lo = lo if lo.any() else zz <= zz[1]
    inf = np.array([np.mean(env.u_ground[lo]) - cx, np.mean(env.v_ground[lo]) - cy])
    fwd = np.array([np.mean(env.u_ground[mid]) - cx, np.mean(env.v_ground[mid]) - cy]) if mid.any() else -inf
    if np.hypot(*inf) < 2.0:
        return
    srh = float(getattr(env, "diag", {}).get("srh1", 150.0)) if isinstance(getattr(env, "diag", None), dict) else 150.0
    gain = float(np.clip(0.55 + srh / 600.0, 0.6, 1.25))          # more low-level helicity, a stronger arc
    # the inflow flank: right of the storm's motion (its southern / eastern side), leaning toward where the
    # low-level storm-relative inflow comes from -- for a fast storm that is its front, not its side
    sm = math.hypot(cx, cy)
    right = np.array([cy, -cx]) / sm if sm > 2.0 else -inf / np.hypot(*inf)
    n_in = right + 0.5 * (-inf / np.hypot(*inf))
    n_in = n_in / max(np.hypot(*n_in), 1e-6)
    fw = fwd / max(np.hypot(*fwd), 1e-6)
    kz = min(int(np.searchsorted(zh_host, 500.0)), f.nz - 1)
    dbz = 10.0 * cp.log10(cp.maximum(f.zh[kz], 1e-3))
    gy, gx = cp.gradient(dbz, dx / 1000.0)
    edge = cp.clip(-(gx * float(n_in[0]) + gy * float(n_in[1])) / ARC_GRAD, 0.0, 1.0)
    # a broad band reaching into the echo from its edge, not just the steepest cells
    edge = ndimage.gaussian_filter(ndimage.maximum_filter(edge, size=max(3, int(round(ARC_WIDTH / dx)) | 1)),
                                   sigma=max(1.0, 0.4 * ARC_WIDTH / dx))
    edge = cp.clip(edge * 1.6, 0.0, 1.0)
    ny, nx = dbz.shape
    X = (cp.arange(nx, dtype=cp.float32) + 0.5) * dx
    Y = (cp.arange(ny, dtype=cp.float32) + 0.5) * dx
    arc = cp.zeros((ny, nx), cp.float32)
    hook = cp.zeros((ny, nx), cp.float32)
    for j, i, s in centres:
        rx, ry = X[None, :] - (i + 0.5) * dx, Y[:, None] - (j + 0.5) * dx
        d = cp.hypot(rx, ry)
        reach = cp.clip((ARC_REACH[1] - d) / (ARC_REACH[1] - ARC_REACH[0]), 0.0, 1.0)
        # on the forward flank's side of the circulation, and on its inflow side of the echo
        ahead = cp.clip((rx * float(fw[0]) + ry * float(fw[1]) + 3000.0) / 4000.0, 0.0, 1.0)
        inside = cp.clip((rx * float(n_in[0]) + ry * float(n_in[1]) + 8000.0) / 6000.0, 0.0, 1.0)
        arc = cp.maximum(arc, s * reach * ahead * inside)
        hook = cp.maximum(hook, s * cp.clip(1.0 - d / ARC_HOOK, 0.0, 1.0))
    a, b, c, e = ARC_DBZ
    band = cp.clip((dbz - a) / (b - a), 0.0, 1.0) * cp.clip((e - dbz) / (e - c), 0.0, 1.0)
    w = arc * band * (0.35 + 0.65 * edge)                          # strongest at the edge, some way into the echo
    if float(w.max()) < 0.05:
        return
    rs = cp.random.RandomState(seed)
    grain = 0.75 + 0.5 * rs.random_sample(w.shape, dtype=cp.float32)
    w = ndimage.gaussian_filter(w, sigma=max(1.0, 1000.0 / dx))     # soft edges: no hard line round the arc
    w = cp.clip(w * grain * gain, 0.0, 1.0)
    zdr_db = ARC_ZDR[0] + (ARC_ZDR[1] - ARC_ZDR[0]) * w + 1.0 * hook * cp.clip(w / 0.3, 0.0, 1.0)
    kt = min(int(np.searchsorted(zh_host, ARC_TOP[1])) + 1, f.nz)
    wz = cp.asarray(np.clip((ARC_TOP[1] - zh_host[:kt]) / (ARC_TOP[1] - ARC_TOP[0]), 0.0, 1.0), cp.float32)
    W = (cp.clip(w / 0.15, 0.0, 1.0)[None] * wz[:, None, None])   # how much of the column is the arc's
    zh = f.zh[:kt]
    cur = 10.0 * cp.log10(cp.maximum(zh, 1e-6) / cp.maximum(f.zv[:kt], 1e-6))
    new = cp.maximum(cur, zdr_db[None])
    db = cur + W * (new - cur)
    rho = cp.clip(f.rz[:kt] / cp.maximum(cp.sqrt(zh * f.zv[:kt]), 1e-6), 0.0, 1.0)
    rho = rho + W * (0.975 - rho) * (rho < 0.975)                  # big drops: a little under rain's 0.99
    f.zv[:kt] = (zh / 10.0 ** (0.1 * db)).astype(cp.float32)
    f.rz[:kt] = (rho * cp.sqrt(zh * f.zv[:kt])).astype(cp.float32)


# The V-notch (Lemon et al. 1978; Browning & Donaldson 1963): the strong upper-level storm-relative flow parts
# round a strong supercell's updraft like a stream round a rock, so the forward flank's echo aloft splits into two
# lobes with a weaker channel down the middle, opening downstream from just past the core.  The grid's anvil-level
# flow smears it out, so it is drawn in above VNOTCH_BASE for updrafts past VNOTCH_W: the echo along the axis of
# the 6-10 km storm-relative wind weakened by up to VNOTCH_DB, the channel widening downstream (VNOTCH_ANGLE).
VNOTCH_W = (18.0, 35.0)       # m/s of updraft at ~5 km: none .. full
VNOTCH_BASE = (1500.0, 3500.0)   # m: none below, full above
VNOTCH_DB = 22.0              # dB taken off the channel's axis at full strength
VNOTCH_ANGLE = 18.0           # deg: the channel's half-angle
VNOTCH_LEN = (2500.0, 40000.0)   # m downstream of the updraft: where it starts .. fades out
VNOTCH_MAX = 4


def add_v_notch(f: "PatchRadarFields", env) -> None:
    """Each strong supercell's V-notch in the patch's Zh, Zv and rho_hv (in place, all scaled alike)."""
    import cupy as cp
    from cupyx.scipy import ndimage
    dx = f.dx
    zh_host = np.asarray(env.z_h)
    k5 = min(int(np.searchsorted(zh_host, 5000.0)), f.nz - 1)
    w5 = ndimage.maximum_filter(f.wc[k5], size=max(3, int(round(6000.0 / dx)) | 1))
    s = cp.clip((w5 - VNOTCH_W[0]) / (VNOTCH_W[1] - VNOTCH_W[0]), 0.0, 1.0)
    peak = (s > 0.05) & (f.wc[k5] >= w5)
    if not bool(peak.any()):
        return
    jj, ii = cp.nonzero(peak)
    order = cp.argsort(-f.wc[k5][jj, ii])[:VNOTCH_MAX]
    cx, cy = env.storm_motion
    hi = (zh_host >= 6000.0) & (zh_host <= 10000.0)
    if not hi.any():
        return
    u, v = float(np.mean(env.u_ground[hi])) - cx, float(np.mean(env.v_ground[hi])) - cy
    sp = math.hypot(u, v)
    if sp < 8.0:                                    # weak anvil-level flow: no obstacle flow to part
        return
    ux, uy = u / sp, v / sp
    X = (cp.arange(f.nx, dtype=cp.float32) + 0.5) * dx
    Y = (cp.arange(f.ny, dtype=cp.float32) + 0.5) * dx
    cut = cp.zeros((f.ny, f.nx), cp.float32)
    tan = math.tan(math.radians(VNOTCH_ANGLE))
    for j, i in zip(cp.asnumpy(jj[order]).tolist(), cp.asnumpy(ii[order]).tolist()):
        st = float(s[j, i]) * min(1.0, sp / 20.0)
        ex, ey = X[None, :] - (i + 0.5) * dx, Y[:, None] - (j + 0.5) * dx
        along = ex * ux + ey * uy
        cross = -ex * uy + ey * ux
        half = 800.0 + tan * cp.maximum(along, 0.0)
        a0, a1 = VNOTCH_LEN
        on = cp.clip((along - a0) / 3000.0, 0.0, 1.0) * cp.clip((a1 - along) / (0.4 * a1), 0.0, 1.0)
        cut = cp.maximum(cut, st * on * cp.exp(-0.5 * (cross / half) ** 2))
    if float(cut.max()) < 0.05:
        return
    cut = ndimage.gaussian_filter(cut, sigma=max(1.0, 700.0 / dx))
    k0 = int(np.searchsorted(zh_host, VNOTCH_BASE[0]))
    if k0 >= f.nz:
        return
    wz = cp.asarray(np.clip((zh_host[k0:f.nz] - VNOTCH_BASE[0]) / (VNOTCH_BASE[1] - VNOTCH_BASE[0]), 0.0, 1.0),
                    cp.float32)
    fac = (10.0 ** (-0.1 * VNOTCH_DB * cut))[None] ** wz[:, None, None]
    f.zh[k0:] *= fac
    f.zv[k0:] *= fac
    f.rz[k0:] *= fac


# Tornado debris (the TDS, the debris ball).  Where the lowest model level holds a tornado, lofted
# debris joins the beam as one more scatterer: reflective, tumbling (ZDR ~0, scattered) and badly
# decorrelated (rho_hv 0.2-0.8; Ryzhkov et al. 2005, Bodine et al. 2013).  A tornado is either a
# vortex the grid resolves (a tight core of strong vorticity and wind), or one below the grid's reach
# under a TVS-like circulation: the mean vorticity within a circle gives the rotational velocity a
# radar would measure (Vrot = zeta R / 2, the strongest for circles up to TVS_RADIUS and in the
# lowest km, where a 0.5-degree beam looks), and under a low-level updraft that is stretching it,
# there is a tornado from TVS_VROT up, of DEBRIS_V0 + TVS_GAIN x (Vrot - TVS_VROT) (Vrot 14 m/s -> 35 m/s,
# EF0-1; 20 -> 46, EF1; 25 -> 55, EF2; 30 -> 64, EF3; 40 -> 82, EF4; after Smith et al. 2015).  A 0.5-1 km
# grid smooths a mesocyclone's low-level spin (the logged Vrot of a 4600 J/kg, 70 kt supercell at 500 m
# peaked near 20 m/s), so the threshold sits below the ~20 m/s of observed tornadic couplets.  The radar sees its winds in place of the model's smeared vortex (sample_cut,
# add_tornado_wind) and the debris it lofts.
# How much debris depends on that wind and on what is under it: the land cover (landcover.py: forests
# the most, then built-up land, farm-and-woodland, crops and grass, water next to nothing) and the
# towns of the population data on top.  Observed TDSs reach 45-60 dBZ, debris balls of violent
# tornadoes 65-70 (Bodine et al. 2013; Van Den Broeke & Jauernic 2014); the beam's CC only drops
# where the debris echo rivals the rain's (the mixture rho_hv of sample_cut), which it now does in
# most hooks (it used to top out at 35-60 dBZ and vanish in a 55-60 dBZ hook).
DEBRIS_V0 = 35.0              # m/s at the lowest level: lofting starts (about EF1)
DEBRIS_V1 = 80.0              # m/s: full strength (violent)
DEBRIS_STRONG = 50.0          # m/s: only strong tornadoes (EF2+, ~110 mph) show a debris ball
DEBRIS_MAX_R = 1200.0         # m: the ball's (and its CC drop's) radius, at most: 1.5 mi across
DEBRIS_ZETA_500M = 0.08       # 1/s at a 500 m grid, scaled with it: a resolved vortex's tight core
DEBRIS_RHO = (0.7, 0.5)          # rho_hv of the debris alone: 0.7 - 0.5 x load, ~0.6 light .. 0.2 the heaviest
                                 # (observed TDSs: 0.2-0.7, Ryzhkov et al. 2005; 0.8 - 0.45 x load left them too high)
DEBRIS_CLEAR = (0.75, 0.2, 0.6)  # the share of the core's rain echo the debris takes over: 75% for the
                                 # weakest, 95% from strength 0.6 (EF3) up, as the ball builds up (g): enough that
                                 # even an EF0-1's CC drop shows as a solid patch, not speckle
TVS_RADIUS = 1500.0           # m (2.5 cells at least): the circle Vrot is measured over
FEATURE_MIN_DX = 25.0         # m: finer grids (a ~10 / ~1 m zoom's) skip the storm-scale radar features
FEATURE_CELLS = 100           # the most cells a tornado search window reaches out (a 1 km window at 1.2 m: 833)
TVS_VROT = 14.0               # m/s (27 kt of model Vrot): the weakest circulation that makes a tornado
TVS_GAIN = 1.8                # the tornado's peak wind per m/s of Vrot
TVS_UPDRAFT = 4.0             # m/s at ~1 km within TVS_RADIUS: the circulation is being stretched
# Only mature storms make tornadoes.  A storm that has just fired has strong low-level rotation too --
# the environment's helicity tilted up by its first updraft, and the storm start's own push stretching
# it -- but no rear-flank downdraft yet, and real tornadoes come late in a supercell's life, out of the
# RFD's outflow wrapped round the low-level mesocyclone.  A circulation is a tornado only with
# outflow air (theta' TORNADO_COLD or colder) within TORNADO_COLD_KM of it and, in a run, under a storm
# whose deep updraft has been followed for TORNADO_MIN_AGE and that no storm-start push is still
# forcing (env.tornado_ok, set by the simulation); and TornadoTracker gives it winds and debris only
# once it has been found for TORNADO_CONFIRM, so a one-refresh spin-up never drops one.
# A strong circulation needs less: the RFDs of strongly tornadic supercells are often only slightly cold
# (Markowski et al. 2002: theta deficits of ~1 K or less), and a strong storm's RFD here read -0.6 to -0.8 K
# under Vrot 23-28 m/s, so the gate turned the run's strongest tornado away for an hour and split the rest
# into 1-3 min pieces as the coldest air within 5 km crossed -1 K from refresh to refresh.  A tornado
# already followed (env.tornado_near) keeps its outflow test passed while it is found again.
TORNADO_COLD = -1.0           # K
TORNADO_COLD_KM = 5.0
TORNADO_COLD_STRONG = -0.3    # K: enough for a circulation of TORNADO_STRONG_VROT or more
TORNADO_STRONG_VROT = 20.0    # m/s of model Vrot (39 kt)
TORNADO_COLD = -1.0           # K
TORNADO_COLD_KM = 5.0
TORNADO_MIN_AGE = 1500.0      # s
TORNADO_CONFIRM = 120.0       # s


# Estimated strength along a tornado's track (Simulation.tornado_tracks: the map's colours, the tornado list and the
# damage survey all use it), the way an NWS office rates a tornado off the radar before a survey: the 0.5 deg
# rotational velocity sets it (WSR-88D tornado-intensity studies, Smith et al. 2015, 2020: 33 kt EF1, 42 EF2,
# 52 EF3, 63 EF4, 74 EF5), a tornadic debris signature (CC under 0.8: debris lofted) makes it at least EF1 (under
# 0.6: EF2), and without one it is held to EF3.  The model's own peak wind only stands in when there is no Vrot.
# (Until 2026-11-03 the Vrot rating, on gentler steps, was averaged with the model wind's, which ran a step low.)
EF_WIND = (29.0, 38.0, 49.0, 60.0, 74.0, 89.0)      # m/s: EF0 .. EF5 lower bounds (3 s gusts)
EF_VROT_KT = (20.0, 33.0, 42.0, 52.0, 63.0, 74.0)   # kt of 0.5 deg Vrot: EF0 .. EF5
TDS_CC = 0.80                                       # CC this low in the tornado's echo: a debris signature
TDS_STRONG_CC = 0.60


def ef_estimate(wind: float | None, vrot: float | None, cc: float | None) -> int:
    """EF rating (0-5, -1 = not a tornado) from the radar's Vrot (m/s) and CC; the model's peak wind (m/s) when
    there is no Vrot."""
    def band(v, edges):
        return -1 if v is None else max([k for k, e in enumerate(edges) if v >= e], default=-1)
    est = band(vrot * 1.943844 + 1e-6, EF_VROT_KT) if vrot is not None else band(wind, EF_WIND)
    tds = cc is not None and cc < TDS_CC
    if tds:
        est = max(est, 1 if cc >= TDS_STRONG_CC else 2)
    else:
        est = min(est, 3)
    return min(est, 5)


def lowlevel_vrot(uc, vc, z_h, dx: float):
    """The rotational velocity (m/s) a radar would measure at each column: the strongest mean
    tangential wind on a circle about it (the vertical vorticity averaged over the disc, times half
    its radius) for radii up to TVS_RADIUS (2.5 cells at least), at its strongest in the lowest
    kilometre (where a 0.5-degree beam sees a TVS).  A compact vortex reads its own peak wind, a
    broad mesocyclone what turns within TVS_RADIUS.  ``uc``, ``vc``: the cell-centred winds
    (nz, ny, nx); ``z_h``: their heights (host)."""
    import cupy as cp
    from cupyx.scipy import ndimage
    R = max(TVS_RADIUS, 2.5 * dx)
    # discs, not squares: the square of side 2R (rounded up to odd cells) took in 1.3-1.7x the
    # circle's area and read a circulation's Vrot 25-40% low
    discs = []
    for rad in sorted({max(dx, R / 3.0), max(dx, 2.0 * R / 3.0), R}):
        n = math.ceil(rad / dx - 1e-6)
        j, i = np.mgrid[-n:n + 1, -n:n + 1]
        d = ((i * i + j * j) * dx * dx <= rad * rad * (1.0 + 1e-6)).astype(np.float32)
        discs.append((rad, cp.asarray(d / d.sum())))
    top = max(int(np.searchsorted(np.asarray(z_h), 1000.0)), 1)
    best = None
    for k in range(min(top, uc.shape[0])):
        zeta = cp.gradient(vc[k], dx, axis=1) - cp.gradient(uc[k], dx, axis=0)
        for rad, disc in discs:
            v = ndimage.convolve(zeta, disc, mode="nearest") * (0.5 * rad)
            best = v if best is None else cp.maximum(best, v)
    return best


TORNADO_RADII = 9             # radii (0 .. reach) of the model's tangential wind a tornado replaces


def _peak_xy(a, j: int, i: int, f: "PatchRadarFields") -> tuple:
    """World x, y of a 2-D field's peak at cell (j, i), to a fraction of a cell: a parabola through the
    peak and its neighbours along each axis.  A circulation centred between cells (a 1 km grid's
    usual case) no longer snaps to one of them, 0.5-0.7 cells off its couplet."""
    import cupy as cp
    blk = cp.asnumpy(a[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2]).astype(np.float64)
    cj, ci = min(j, 1), min(i, 1)

    def off(lo, c, hi):
        den = lo - 2.0 * c + hi
        return 0.0 if den >= 0.0 else max(-0.5, min(0.5, 0.5 * (lo - hi) / den))
    di = off(blk[cj, ci - 1], blk[cj, ci], blk[cj, ci + 1]) if 0 < ci < blk.shape[1] - 1 else 0.0
    dj = off(blk[cj - 1, ci], blk[cj, ci], blk[cj + 1, ci]) if 0 < cj < blk.shape[0] - 1 else 0.0
    return f.x0 + (i + 0.5 + di) * f.dx, f.y0 + (j + 0.5 + dj) * f.dx


def find_tornadoes(f: "PatchRadarFields", env) -> list:
    """The patch's tornadoes now, strongest first: vortices the grid resolves (a tight core of strong
    vorticity at the lowest level with 35+ m/s within 1 km of it) and ones below its reach under a
    TVS-like circulation (above).  Each sits on its circulation's centre -- the vorticity peak, or the
    strongest disc-mean rotation -- where the radar's couplet is, not on the fast flank where the
    storm's motion adds to the spin (the old debris cloud followed the 35 m/s winds there and sat 1-3 km
    off the couplet).  [{x, y, s: strength 0..1 over 35..80 m/s, ve: a sub-grid tornado's peak wind or None}]"""
    import cupy as cp
    from cupyx.scipy import ndimage
    dx = f.dx
    found = []
    u0, v0 = f.uc[0], f.vc[0]                       # ground-relative, lowest level
    zeta = cp.gradient(v0, dx, axis=1) - cp.gradient(u0, dx, axis=0)
    n = min(max(1, int(round(1000.0 / dx))), FEATURE_CELLS)
    # the core threshold grows as the grid resolves more of the vortex, until it resolves all of it (~30 m):
    # at 1.2 m it read 33 1/s, which no tornado reaches, and the finest zoom level found none
    peak = (zeta >= DEBRIS_ZETA_500M * 500.0 / max(dx, 30.0)) & (zeta == ndimage.maximum_filter(zeta, size=2 * n + 1))
    if bool(peak.any()):
        wmax = ndimage.maximum_filter(cp.hypot(u0, v0), size=2 * n + 1)
        jj, ii = cp.nonzero(peak)
        order = cp.argsort(-zeta[jj, ii])[:8]
        for j, i in zip(cp.asnumpy(jj[order]).tolist(), cp.asnumpy(ii[order]).tolist()):
            s = (float(wmax[j, i]) - DEBRIS_V0) / (DEBRIS_V1 - DEBRIS_V0)
            if s > 0.0:
                x, y = _peak_xy(zeta, j, i, f)
                vr = float(f.vrot[j, i]) if f.vrot is not None else float(wmax[j, i]) / TVS_GAIN
                found.append({"x": x, "y": y, "s": min(s, 1.0), "ve": None, "vrot": vr})
    # below the grid's reach: a TVS-like circulation in the lowest km, under a low-level updraft that
    # is stretching it, the tornado at its strongest rotation
    if dx < FEATURE_MIN_DX:                         # a grid this fine resolves its tornadoes: none below its reach
        return [d for d in _merge_tornadoes(found) if _tornado_allowed(f, env, d)]
    vrot = f.vrot if f.vrot is not None else lowlevel_vrot(f.uc, f.vc, env.z_h, dx)
    R = max(TVS_RADIUS, 2.5 * dx)
    m = max(3, int(round(2.0 * R / dx)) | 1)
    k1 = min(int(np.searchsorted(env.z_h, 1000.0)), f.nz - 1)
    lift = ndimage.maximum_filter(f.wc[k1], size=m) >= TVS_UPDRAFT
    vmax = ndimage.maximum_filter(vrot, size=2 * m + 1)
    centre = (vrot >= TVS_VROT) & lift & (vrot >= vmax)
    if bool(centre.any()):
        jj, ii = cp.nonzero(centre)
        order = cp.argsort(-vrot[jj, ii])[:8]
        for j, i in zip(cp.asnumpy(jj[order]).tolist(), cp.asnumpy(ii[order]).tolist()):
            ve = min(DEBRIS_V0 + TVS_GAIN * (float(vrot[j, i]) - TVS_VROT), 90.0)
            if ve >= DEBRIS_V0:
                x, y = _peak_xy(vrot, j, i, f)
                found.append({"x": x, "y": y, "s": (ve - DEBRIS_V0) / (DEBRIS_V1 - DEBRIS_V0), "ve": ve,
                              "vrot": float(vrot[j, i])})
    return [d for d in _merge_tornadoes(found) if _tornado_allowed(f, env, d)]


def _tornado_allowed(f: "PatchRadarFields", env, d: dict) -> bool:
    """A circulation is a tornado only with outflow air near it, under a mature storm (TORNADO_COLD)."""
    import cupy as cp
    thp0 = getattr(f, "thp0", None)
    if thp0 is not None:
        n = max(1, int(round(TORNADO_COLD_KM * 1000.0 / f.dx)))
        i, j = int((d["x"] - f.x0) / f.dx), int((d["y"] - f.y0) / f.dx)
        blk = thp0[max(j - n, 0):j + n + 1, max(i - n, 0):i + n + 1]
        cold = float(cp.min(blk)) if blk.size else 99.0
        need = TORNADO_COLD_STRONG if (d.get("vrot") or 0.0) >= TORNADO_STRONG_VROT else TORNADO_COLD
        near = getattr(env, "tornado_near", None)
        if cold > need and not (near is not None and near(d["x"], d["y"])):
            return False
    ok = getattr(env, "tornado_ok", None)
    return ok is None or bool(ok(d["x"], d["y"]))


def _merge_tornadoes(found: list, within: float = 2000.0) -> list:
    """One tornado per circulation: those within ``within`` m of a stronger one join it (a resolved
    vortex and the TVS of the same circulation; neighbouring patches)."""
    out = []
    for d in sorted(found, key=lambda d: -d["s"]):
        near = next((o for o in out if math.hypot(o["x"] - d["x"], o["y"] - d["y"]) < within), None)
        if near is None:
            out.append(dict(d))
        elif d["ve"] is not None:
            near["ve"] = max(near["ve"] or 0.0, d["ve"])
        if near is not None:
            near["vrot"] = max(near.get("vrot") or 0.0, d.get("vrot") or 0.0)
    return out


# A tornado's debris ball builds up over minutes as it lofts debris and falls out after the tornado
# ends, and its sub-grid winds grow in and die away over ~45 s: followed from radar refresh to refresh
# (TornadoTracker), a debris ball no longer pops up at full strength (it started at 35 dBZ) nor
# flickers on and off with each refresh's detection.  Its echo builds evenly in dB, from DEBRIS_DB
# below its full strength to full over DEBRIS_BUILD (growing from half its size and deepening; a CC
# dip that spreads and deepens over three or four low-level scans), and fades the same way over
# DEBRIS_FALL once the tornado is lost.  (Growing evenly in Z, as debris loads would, it read nearly
# full one volume after it began: dBZ is a log scale.)  Meanwhile its strength follows the
# tornado's (DEBRIS_UP, DEBRIS_DOWN).
DEBRIS_DEPTH = 6500.0         # m: the debris' e-folding depth of a violent tornado's ball, past 500 m
DEBRIS_TOP = 13716.0          # m (45,000 ft): no debris above, faded out over the 1.5 km below
DEBRIS_RHO_TOP = 0.97         # the debris' CC by DEBRIS_TOP: its drop shallower the higher it is lofted
DEBRIS_BUILD = 300.0          # s: a new tornado's debris ball, from barely there to full
DEBRIS_FALL = 480.0           # s: ... and back once the tornado is lost
DEBRIS_DB = 30.0              # dB below full strength when barely there
DEBRIS_UP = 100.0             # s: the debris following a strengthening tornado (e-folding time)
DEBRIS_DOWN = 300.0           # s: ... a weakening one
TVS_RAMP = 45.0               # s: a sub-grid tornado's winds growing in / dying away
TRACK_REACH = 3000.0          # m (+ 30 m/s x the time since): a tornado found again continues its track


@dataclass
class Tornado:
    """One tornado from radar refresh to refresh: where it is and its motion, its strength now (s,
    0..1 over 35..80 m/s), a sub-grid one's peak wind (ve), the strength of its debris (a), and how
    far its debris ball has built up (g, 0..1) and its sub-grid winds have grown in (aw, 0..1)."""
    x: float
    y: float
    s: float = 0.0
    ve: float | None = None
    a: float = 0.0
    g: float = 0.0
    aw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    found: bool = True
    seen: float = 0.0             # s it has been found for, without a gap (TORNADO_CONFIRM)
    vrot: float = 0.0             # m/s: the model's low-level rotation at it (lowlevel_vrot)
    id: int = 0                   # TornadoTracker's number for it (its track)
    ej: float = 0.0               # 0..1: how much debris it has thrown out around itself (_debris_ejection)

    @classmethod
    def grown(cls, d: dict) -> "Tornado":
        """A tornado found once, its debris and winds fully grown (fields built alone)."""
        return cls(d["x"], d["y"], s=d["s"], ve=d["ve"], a=d["s"], g=1.0, aw=1.0, seen=TORNADO_CONFIRM,
                   vrot=d.get("vrot") or 0.0)


class TornadoTracker:
    """A run's tornadoes from radar refresh to refresh.  Each refresh's tornadoes continue the nearest
    track; a debris ball builds up while its tornado is found (DEBRIS_BUILD), its strength following
    the tornado's (DEBRIS_UP, DEBRIS_DOWN), and falls out once it is lost (DEBRIS_FALL, drifting on at
    the tornado's last motion); sub-grid winds grow in and die away (TVS_RAMP).  A watchdog rollback
    (time going back) starts it again."""

    def __init__(self):
        self.items: list[Tornado] = []
        self.t = None
        self._next = 1

    def update(self, t: float, found: list) -> list:
        if self.t is not None and t < self.t - 1.0:
            self.items, self.t = [], None
        dt = 0.0 if self.t is None else max(t - self.t, 0.0)
        self.t = t
        for c in self.items:                         # where each was heading
            c.x += c.vx * dt
            c.y += c.vy * dt
        pairs = sorted((math.hypot(c.x - d["x"], c.y - d["y"]), ci, di)
                       for ci, c in enumerate(self.items) for di, d in enumerate(found))
        reach = TRACK_REACH + 30.0 * dt
        used_c, used_d = set(), set()
        for dist, ci, di in pairs:
            if dist > reach or ci in used_c or di in used_d:
                continue
            used_c.add(ci)
            used_d.add(di)
            c, d = self.items[ci], found[di]
            if dt > 0.0:                             # its motion, from where it was found last time
                vx = (d["x"] - (c.x - c.vx * dt)) / dt
                vy = (d["y"] - (c.y - c.vy * dt)) / dt
                sp = math.hypot(vx, vy)
                if sp > 45.0:
                    vx, vy = vx * 45.0 / sp, vy * 45.0 / sp
                w = min(1.0, dt / 120.0)
                c.vx, c.vy = c.vx + w * (vx - c.vx), c.vy + w * (vy - c.vy)
            c.x, c.y, c.s, c.found = d["x"], d["y"], d["s"], True
            c.vrot = d.get("vrot") or c.vrot
            if d["ve"] is not None:
                c.ve = d["ve"]
        for ci, c in enumerate(self.items):
            if ci not in used_c:
                c.s, c.found = 0.0, False
            c.seen = c.seen + dt if c.found else 0.0
            if dt > 0.0:
                if c.found and c.seen < TORNADO_CONFIRM and c.g <= 0.0:
                    continue                         # not yet confirmed: no winds, no debris
                if c.found:
                    c.a += (c.s - c.a) * (1.0 - math.exp(-dt / (DEBRIS_UP if c.s > c.a else DEBRIS_DOWN)))
                    c.g = min(1.0, c.g + dt / DEBRIS_BUILD)
                    if c.a >= EJECT_MIN:            # a strong tornado keeps throwing debris out
                        c.ej = min(1.0, c.ej + dt / EJECT_BUILD * min(1.0, c.a / 0.6))
                else:                                # its debris falling out where it drifts
                    c.g = max(0.0, c.g - dt / DEBRIS_FALL)
                if not c.found or c.a < EJECT_MIN:
                    c.ej = max(0.0, c.ej - dt / EJECT_FALL)
                aim = 1.0 if c.found and c.ve is not None else 0.0
                c.aw += (aim - c.aw) * (1.0 - math.exp(-dt / TVS_RAMP))
        # tornadoes found for the first time: nothing lofted yet, no winds of their own yet
        for di, d in enumerate(found):
            if di not in used_d:
                self.items.append(Tornado(d["x"], d["y"], s=d["s"], ve=d["ve"], a=d["s"], vrot=d.get("vrot") or 0.0,
                                          id=self._next))
                self._next += 1
        self.items = [c for c in self.items if c.found or c.g > 0.0 or c.aw > 0.01 or c.ej > 0.01]
        # what the radar sees: confirmed tornadoes, and the debris and winds of lost ones dying away
        return [c for c in self.items if c.seen >= TORNADO_CONFIRM or c.g > 0.0 or c.aw > 0.01 or c.ej > 0.01]


def tornado_winds(f: "PatchRadarFields", tors: list, z_h):
    """sample_sweep's sub-grid tornadoes: (n, 7) float32 [x, y, peak wind, core radius, reach,
    full-strength top, grown-in fraction], and the model's own mean tangential wind about each
    (n, levels, TORNADO_RADII), up to 2 km above the highest top, which their winds replace.
    ``z_h``: the levels' heights (host)."""
    import cupy as cp
    from cupyx.scipy import ndimage
    rows = []
    for c in tors:
        s = min(max((c.ve - DEBRIS_V0) / (DEBRIS_V1 - DEBRIS_V0), 0.0), 1.0)
        rows.append((c.x, c.y, c.ve, 150.0 + 250.0 * s, max(TVS_RADIUS, 2.5 * f.dx), 1000.0 + 2000.0 * s,
                     min(max(c.aw, 0.0), 1.0)))
    tor = np.array(rows, np.float32)
    nk = min(int(np.searchsorted(np.asarray(z_h), float(tor[:, 5].max()) + 2000.0)) + 2, f.nz)
    na = 16
    th = np.arange(na) * (2.0 * math.pi / na)
    sin_t, cos_t = cp.asarray(np.sin(th), cp.float32), cp.asarray(np.cos(th), cp.float32)
    tables = []
    for x, y, _, _, reach, _, _ in tor:
        rr = np.linspace(0.0, float(reach), TORNADO_RADII)
        gi = ((x + rr[:, None] * np.cos(th)) - f.x0) / f.dx - 0.5        # (radii, azimuths)
        gj = ((y + rr[:, None] * np.sin(th)) - f.y0) / f.dx - 0.5
        pts = TORNADO_RADII * na
        coords = cp.asarray(np.stack([np.repeat(np.arange(nk), pts), np.tile(gj.ravel(), nk),
                                      np.tile(gi.ravel(), nk)]), dtype=cp.float32)
        u = ndimage.map_coordinates(f.uc[:nk], coords, order=1, mode="nearest").reshape(nk, TORNADO_RADII, na)
        v = ndimage.map_coordinates(f.vc[:nk], coords, order=1, mode="nearest").reshape(nk, TORNADO_RADII, na)
        tables.append((v * cos_t - u * sin_t).mean(axis=2))           # counter-clockwise
    return tor, cp.ascontiguousarray(cp.stack(tables), dtype=cp.float32)


# Hooks.  A 1-km model draws the rain wrapping round a low-level mesocyclone as a thin, weak spiral
# (25-35 dBZ), often with a weak-echo hole 2-5 km wide at its centre where the tornado is; a radar
# shows a bright band (40-50 dBZ) running into a tip on the tornado.  Near strong low-level rotation,
# in the lowest ~2-3 km: gaps narrower than ~2 x HOOK_JOIN are closed, and the hole at the
# circulation's centre up to ~2 x HOOK_TIP (a grey-scale closing in dBZ: what is narrower than the
# disc fills in with the echo around it; edges and anything wider -- an inflow notch, the weak echo
# region aloft -- stay as they are); then the echo there is brightened, HOOK_GAIN of the way up to
# HOOK_BELOW_CORE under the storm's core, the tip HOOK_TIP_GAIN of the way (clear air is left alone,
# so the notch stays clear and the hook's edges sharpen).  The tip is on the circulation's centre,
# where find_tornadoes puts the tornado, its couplet and its debris ball: the ball sits in the tip,
# never in a hole or out on its own.
HOOK_JOIN = 1500.0            # m: radius of the closing disc
HOOK_TIP = 2500.0             # m: ... at the circulation's centre (full within it, none beyond twice it)
HOOK_REACH = 8000.0           # m from the circulation's centre (full within half of it)
HOOK_VROT = (10.0, 18.0)      # m/s of low-level rotation (lowlevel_vrot): none .. full
HOOK_UPDRAFT = 2.0            # m/s at ~1 km within TVS_RADIUS of it (a mesocyclone, not gust-front shear)
HOOK_TOP = (2000.0, 3000.0)   # m: full below, nothing above
HOOK_GAIN = 0.65              # of the way from an echo's dBZ up to ...
HOOK_TIP_GAIN = 0.9
HOOK_BELOW_CORE = 6.0         # dB below the storm's core: the strongest echo within HOOK_CORE at that height
HOOK_CORE = 12000.0           # m
HOOK_ECHO = (5.0, 25.0)       # dBZ: echo weaker than the first left alone (clear air), from the second brightened fully
# Sharper echo everywhere: each cell's dBZ pushed away from its neighbours' smooth mean (an unsharp
# mask, sigma one cell) but never past the highest or lowest of its eight neighbours, so it makes no new
# extremes, halos or rings.  The model's numerical smoothing spreads gradients over several cells; this
# gives back some of the sharpness of a radar's hook edges and notches, more on coarse grids.
SHARPEN = ((250.0, 0.4), (500.0, 0.55), (1000.0, 0.75))      # (dx m, amount)


def _rescale(f, k1: int, z_new, fill_zdr_db: float = 0.5) -> None:
    """New Zh for levels below k1, the cells' own ZDR and CC kept (echo where there was none: rain's)."""
    import cupy as cp
    zh, zv, rz = f.zh[:k1], f.zv[:k1], f.rz[:k1]
    had = zh > 1.0e-3
    ratio = cp.where(had, z_new / cp.maximum(zh, 1.0e-3), 1.0)
    zv_new = cp.where(had, zv * ratio, z_new / 10.0 ** (0.1 * fill_zdr_db))
    rz_new = cp.where(had, rz * ratio, 0.99 * cp.sqrt(z_new * zv_new))
    f.zh[:k1], f.zv[:k1], f.rz[:k1] = z_new, zv_new, rz_new


def _closing(D, rad_cells: float):
    """Grey-scale closing of each level of D (levels, y, x) with a disc of ``rad_cells`` (one at least)."""
    import cupy as cp
    from cupyx.scipy import ndimage
    rad = max(1.0, rad_cells)
    n = int(math.ceil(rad - 1e-6))
    j2, i2 = np.mgrid[-n:n + 1, -n:n + 1]
    disc = cp.asarray((i2 * i2 + j2 * j2 <= rad * rad + 1e-6)[None])
    return ndimage.minimum_filter(ndimage.maximum_filter(D, footprint=disc), footprint=disc)


def connect_hooks(f: "PatchRadarFields", env) -> None:
    """The hook joined to its tip on the circulation's centre and brightened, around strong low-level
    rotation (see HOOK_JOIN)."""
    import cupy as cp
    from cupyx.scipy import ndimage
    vrot, dx = f.vrot, f.dx
    if vrot is None:
        return
    pk = ndimage.maximum_filter(vrot, size=max(3, int(round(4000.0 / dx)) | 1))
    # the mesocyclone, by an updraft at ~1 km beside it (not shear along a gust front)
    m = max(3, int(round(2.0 * max(TVS_RADIUS, 2.5 * dx) / dx)) | 1)
    kw = min(int(np.searchsorted(env.z_h, 1000.0)), f.nz - 1)
    anchors = (vrot >= HOOK_VROT[0]) & (vrot >= pk) & (ndimage.maximum_filter(f.wc[kw], size=m) >= HOOK_UPDRAFT)
    if not bool(anchors.any()):
        return
    jj, ii = cp.nonzero(anchors)
    order = cp.argsort(-vrot[jj, ii])[:8]
    X = f.x0 + (cp.arange(f.nx, dtype=cp.float32) + 0.5) * dx
    Y = f.y0 + (cp.arange(f.ny, dtype=cp.float32) + 0.5) * dx
    W = cp.zeros((f.ny, f.nx), cp.float32)                   # the hook
    Wt = cp.zeros((f.ny, f.nx), cp.float32)                  # its tip
    for j, i in zip(cp.asnumpy(jj[order]).tolist(), cp.asnumpy(ii[order]).tolist()):
        strength = min(1.0, (float(vrot[j, i]) - HOOK_VROT[0]) / (HOOK_VROT[1] - HOOK_VROT[0]))
        x, y = _peak_xy(vrot, j, i, f)                       # where find_tornadoes puts a tornado
        r = cp.sqrt((X[None, :] - x) ** 2 + (Y[:, None] - y) ** 2)
        t = cp.clip(2.0 - 2.0 * r / HOOK_REACH, 0.0, 1.0)
        W = cp.maximum(W, strength * t * t * (3.0 - 2.0 * t))
        t = cp.clip(2.0 - r / HOOK_TIP, 0.0, 1.0)
        Wt = cp.maximum(Wt, strength * t * t * (3.0 - 2.0 * t))
    zh_lev = np.asarray(env.z_h)
    k1 = int(np.searchsorted(zh_lev, HOOK_TOP[1]))
    if k1 <= 0:
        return
    wz = cp.asarray(np.clip((HOOK_TOP[1] - zh_lev[:k1]) / (HOOK_TOP[1] - HOOK_TOP[0]), 0.0, 1.0),
                    cp.float32)[:, None, None]
    W, Wt = W[None] * wz, Wt[None] * wz
    D = 10.0 * cp.log10(cp.maximum(f.zh[:k1], 1.0e-3))
    # joined: narrow gaps, and the hole at the centre, filled with the echo around them
    D1 = D + cp.maximum(W * (_closing(D, HOOK_JOIN / dx) - D), Wt * (_closing(D, HOOK_TIP / dx) - D))
    # brightened toward the core
    n = max(1, int(round(HOOK_CORE / dx)))
    core = ndimage.maximum_filter(D1, size=(1, 2 * n + 1, 2 * n + 1))
    t = cp.clip((D1 - HOOK_ECHO[0]) / (HOOK_ECHO[1] - HOOK_ECHO[0]), 0.0, 1.0)
    D1 += (cp.maximum(HOOK_GAIN * W, HOOK_TIP_GAIN * Wt) * t * t * (3.0 - 2.0 * t)
           * cp.maximum(core - HOOK_BELOW_CORE - D1, 0.0))
    del core, t
    add = D1 - D
    if float(add.max()) <= 0.05:
        return
    _rescale(f, k1, cp.where(add > 0.01, 10.0 ** (0.1 * D1), f.zh[:k1]).astype(cp.float32))


def sharpen_echo(f: "PatchRadarFields") -> None:
    """Crisper echo edges at every height (see SHARPEN); never echo where there was none."""
    import cupy as cp
    from cupyx.scipy import ndimage
    amount = float(np.interp(f.dx, [x for x, _ in SHARPEN], [a for _, a in SHARPEN]))
    D = 10.0 * cp.log10(cp.maximum(f.zh, 1.0e-3))
    G = ndimage.gaussian_filter(D, sigma=(0.0, 1.0, 1.0))
    lo = ndimage.minimum_filter(D, size=(1, 3, 3))
    hi = ndimage.maximum_filter(D, size=(1, 3, 3))
    D2 = cp.clip(D + amount * (D - G), lo, hi)
    del G, lo, hi
    keep = f.zh > 1.0e-3
    _rescale(f, f.nz, cp.where(keep, 10.0 ** (0.1 * D2), 0.0).astype(cp.float32))


def add_debris(f: "PatchRadarFields", env, seed: int) -> bool:
    """The tornadoes' sub-grid winds (TVS couplets: f.tornadoes for sample_sweep) and their debris in
    Zh, Zv and rho_hv (in place), from f.clouds (TornadoTracker's tornadoes, or this patch's found
    ones fully grown).  True when there is debris."""
    import cupy as cp

    dx = f.dx
    x1, y1 = f.x0 + f.nx * dx, f.y0 + f.ny * dx
    inside = lambda c, m=0.0: f.x0 - m <= c.x <= x1 + m and f.y0 - m <= c.y <= y1 + m     # noqa: E731
    tors = [c for c in f.clouds if c.ve is not None and c.aw > 0.02 and inside(c)]
    if tors:
        f.tornadoes, f.tornado_winds = tornado_winds(f, tors, env.z_h)
    strong = (DEBRIS_STRONG - DEBRIS_V0) / (DEBRIS_V1 - DEBRIS_V0)
    ejected = _debris_ejection(f, env, [c for c in f.clouds if c.ej > 0.02 and inside(c, EJECT_LEN[1] + 5000.0)], seed)
    deb = [c for c in f.clouds if c.g > 0.0 and c.a > 0.004 and max(c.s, c.a) >= strong and inside(c, 4000.0)]
    if not deb:
        return ejected
    # the debris ball: on the tornado (its couplet, the hook's tip), strongest there and fading outward,
    # small and sized by the tornado -- under 1 km across for an EF2, ~1.6 km (1 mi) for a violent one, and
    # nothing past DEBRIS_MAX_R (1.5 mi across), where its Gaussian tail used to carry the CC drop out to
    # ~4 km -- from half that size as it builds up (g)
    X = f.x0 + (cp.arange(f.nx, dtype=cp.float32) + 0.5) * dx
    Y = f.y0 + (cp.arange(f.ny, dtype=cp.float32) + 0.5) * dx
    s = cp.zeros((f.ny, f.nx), cp.float32)
    g = cp.zeros((f.ny, f.nx), cp.float32)
    fp = cp.zeros((f.ny, f.nx), cp.float32)          # the ball's footprint, 1 on its plateau
    clear = cp.zeros((f.ny, f.nx), cp.float32)       # the share of the rain echo its debris takes over
    top = cp.zeros((f.ny, f.nx), cp.float32)         # ... up to the tornado's full-strength top
    for c in deb:
        k = 0.5 + 0.5 * c.g
        rp, sg = (200.0 + 250.0 * c.a) * k, math.hypot((100.0 + 130.0 * c.a) * k, 0.35 * dx)
        r = cp.sqrt((X[None, :] - c.x) ** 2 + (Y[:, None] - c.y) ** 2)
        fc = cp.exp(-cp.maximum(r - rp, 0.0) ** 2 / (2.0 * sg * sg))
        fc = fc * cp.clip((DEBRIS_MAX_R - r) / 300.0, 0.0, 1.0)
        sc = c.a * fc
        g = cp.where(sc > s, cp.float32(c.g), g)
        s = cp.maximum(s, sc)
        fp = cp.maximum(fp, fc)
        cc = (DEBRIS_CLEAR[0] + DEBRIS_CLEAR[1] * min(c.a / DEBRIS_CLEAR[2], 1.0)) * min(max(c.g, 0.0), 1.0) * fc
        top = cp.where(cc > clear, cp.float32(1000.0 + 2000.0 * c.a), top)
        clear = cp.maximum(clear, cc)
    lofted = s > 0.004
    if not bool(lofted.any()):
        return False
    # what is under it, only where there is debris: the land cover, and the towns on top
    jj, ii = cp.nonzero(lofted)
    j0, j1, i0, i1 = int(jj.min()), int(jj.max()) + 1, int(ii.min()), int(ii.max()) + 1
    X, Y = np.meshgrid(f.x0 + (np.arange(i0, i1) + 0.5) * dx, f.y0 + (np.arange(j0, j1) + 0.5) * dx)
    box = np.full(X.shape, 0.5, np.float32)
    lc, latlon = getattr(env, "landcover", None), getattr(env, "latlon", None)
    if lc is not None and latlon is not None:
        box = lc.debris_load(*latlon(X, Y)).astype(np.float32)
    towns = getattr(env, "towns", None)
    if towns is not None and len(towns):
        near = towns[(towns[:, 0] > X.min() - 2e4) & (towns[:, 0] < X.max() + 2e4)
                     & (towns[:, 1] > Y.min() - 2e4) & (towns[:, 1] < Y.max() + 2e4)]
        for tx, ty, pop in near:
            rad = max(400.0, 800.0 * (float(pop) / 1000.0) ** 0.4)       # town radius: 1 k -> 0.8 km, 100 k -> 5 km
            dens = min(1.0, math.log10(max(float(pop), 10.0)) / 5.0)
            box += dens * np.exp(-((X - tx) ** 2 + (Y - ty) ** 2) / (rad * rad))
    dry = _dry(f, env, lofted)
    avail = cp.zeros(s.shape, cp.float32)
    avail[j0:j1, i0:i1] = cp.asarray(np.minimum(box, 1.6))
    # in proportion to the tornado, the ground scaling it by about +-40% (forest 1, crops 0.8, water
    # 0.6, a town up to 1.3): ~45 dBZ for an EF0-1, 50-55 dBZ and CC 0.6-0.8 for an EF2-3, 65-70 dBZ and
    # CC ~0.35 for a violent one over forest or town (Bodine et al. 2013) -- a few dB over the hook's
    # brightened tip (connect_hooks) from EF1 up, so the ball shows in it
    load = cp.minimum((0.15 + 0.85 * s) * (0.55 + 0.45 * avail / 0.95), 1.0)
    # the echo fading to nothing at the ball's edge (it used to start at 35 dBZ there; by the footprint,
    # not the strength, which took 7 dB off a weak tornado's whole ball), and building up evenly in dB
    # from DEBRIS_DB below full (from nothing in the first 10% of it)
    grow = cp.clip(fp / 0.25, 0.0, 1.0)
    built = 10.0 ** (-0.1 * DEBRIS_DB * (1.0 - g)) * cp.minimum(g / 0.1, 1.0)
    z0 = cp.where(lofted, built * grow * grow * (3.0 - 2.0 * grow) * 10.0 ** (0.1 * (46.0 + 26.0 * load)), 0.0) * dry
    clear = clear * dry                                               # over water: no debris, the rain stays
    # lofted higher by stronger winds, over time: ~2 km for an EF1-2, ~4 km for an EF3, ~7 km for a violent
    # tornado (observed TDS tops: 2-3 km for weak tornadoes, 6-9 km for EF4-5; Van Den Broeke & Jauernic 2014)
    depth = 500.0 + DEBRIS_DEPTH * s ** 1.4 * g
    zh_lev = cp.asarray(f.z_h, dtype=cp.float32)
    kt = int(min(int(cp.searchsorted(zh_lev, float(3.0 * depth.max()))) + 1, f.nz))
    # the tornado's core: rain flung out, debris in its place (DEBRIS_CLEAR), up to its full-strength top
    # and tapering out by twice that.  The debris echo there is at least the rain it displaced, so the
    # ball is never a hole in the hook's tip; where the debris outshines the rain (strong tornadoes) the
    # ball stands out above the tip and its CC drops to the debris' own
    kt = max(kt, int(min(int(cp.searchsorted(zh_lev, float(2.0 * top.max()))) + 1, f.nz)))
    kt = max(1, min(kt, int(cp.searchsorted(zh_lev, DEBRIS_TOP)) + 1))
    zk = zh_lev[:kt, None, None]
    up = cp.clip(zk / DEBRIS_TOP, 0.0, 1.0)                           # 0 at the ground, 1 at 45 kft
    top_fade = cp.clip((DEBRIS_TOP - zk) / 1500.0, 0.0, 1.0)
    zd = z0[None] * cp.exp(-zk / depth[None]) * top_fade
    # the rain the debris displaces: less of it higher up, where the lofted debris thins out
    cl = clear[None] * cp.clip(2.0 - zk / cp.maximum(top[None], 1.0), 0.0, 1.0) * (1.0 - 0.6 * up) * top_fade
    # the column leans into the storm's updraft as it rises, as real debris columns do (DEBRIS_TILT)
    tilt = _debris_tilt(f, deb, cp.asnumpy(zh_lev[:kt]))
    if tilt is not None:
        zd, cl = _shift_levels(zd, tilt), _shift_levels(cl, tilt)
    if bool((cl > 0.01).any()):
        taken = cl * f.zh[:kt]
        keep = 1.0 - cl
        f.zh[:kt] *= keep                                              # the rain's ZDR and CC stay its own
        f.zv[:kt] *= keep
        f.rz[:kt] *= keep
        zd = cp.maximum(zd, taken)
        del taken, keep
    rs = cp.random.RandomState(seed)
    zdr = 10.0 ** (0.1 * cp.clip(rs.standard_normal(zd.shape, dtype=cp.float32), -3.0, 3.0))
    # the debris' own rho_hv, light loads to the heaviest (DEBRIS_RHO)
    rho = cp.clip(DEBRIS_RHO[0] - DEBRIS_RHO[1] * load[None] + 0.06 * rs.standard_normal(zd.shape, dtype=cp.float32),
                  0.15, 0.95)
    # lighter, sparser debris aloft: the CC drop gets shallower with height, nearly gone by 45 kft
    rho = rho + (DEBRIS_RHO_TOP - rho) * up ** 0.8
    if tilt is not None:
        rho = _shift_levels(rho, tilt, fill=0.9)
    f.zh[:kt] += zd
    f.zv[:kt] += zd / zdr
    f.rz[:kt] += rho * zd / cp.sqrt(zdr)
    return True


# A debris column is not upright: the tornado's parent updraft lifts and carries the debris, so the TDS
# leans into the updraft with height (a few km over its top in strong tornadoes).  Each level of the
# column shifts toward the strongest updraft at ~3 km within DEBRIS_TILT_REACH of the strongest tornado,
# DEBRIS_TILT km per km of height, never past the updraft itself.
DEBRIS_TILT = 0.6               # horizontal km per km of height (~30 degrees off vertical)
DEBRIS_TILT_REACH = 10000.0     # m: where the updraft is looked for
DEBRIS_TILT_W = 5.0             # m/s at ~3 km: weaker, and the column stays upright


def _debris_tilt(f: "PatchRadarFields", deb: list, z_h) -> list | None:
    """Per level (cells y, x) the column's offset toward the updraft, or None."""
    import cupy as cp
    c = max(deb, key=lambda q: q.a)
    dx = f.dx
    k3 = min(int(np.searchsorted(cp.asnumpy(cp.asarray(f.z_h)), 3000.0)), f.nz - 1)
    i, j = int((c.x - f.x0) / dx), int((c.y - f.y0) / dx)
    n = int(DEBRIS_TILT_REACH / dx)
    j0, i0 = max(j - n, 0), max(i - n, 0)
    w = f.wc[k3][j0:j + n + 1, i0:i + n + 1]
    if w.size == 0 or float(w.max()) < DEBRIS_TILT_W:
        return None
    jj, ii = np.unravel_index(int(cp.argmax(w)), w.shape)
    ux, uy = f.x0 + (i0 + ii + 0.5) * dx, f.y0 + (j0 + jj + 0.5) * dx
    d = math.hypot(ux - c.x, uy - c.y)
    if d < dx:
        return None
    ex, ey = (ux - c.x) / d, (uy - c.y) / d
    return [(ey * min(d, DEBRIS_TILT * float(z)) / dx, ex * min(d, DEBRIS_TILT * float(z)) / dx) for z in z_h]


def _shift_levels(a, tilt: list, fill: float = 0.0):
    """Each level of ``a`` (levels, ny, nx) moved by its (y, x) offset in cells (linear, sub-cell)."""
    import cupy as cp
    from cupyx.scipy import ndimage
    out = cp.empty_like(a)
    for k, (oy, ox) in enumerate(tilt[:a.shape[0]]):
        out[k] = a[k] if abs(oy) + abs(ox) < 0.05 else ndimage.shift(a[k], (oy, ox), order=1, mode="constant", cval=fill)
    return out


# Debris ejection (the loops of strong, long-lived tornadoes: KGWX, KICT, KPAH 2021-12-11): besides the ball on
# the tornado, debris it has lofted is thrown out and carried off by the storm-relative wind, so a ragged area
# of lowered CC (0.4-0.85, speckled) grows round the tornado's rear and left -- a few km after ~10 min, 10-15 km
# after half an hour -- and fades over ~15 min once the tornado weakens or ends.  Light debris mixed into the
# hook's rain: the CC drop shows where it rivals the rain's echo, ZDR noisy about 0.
EJECT_MIN = 0.3               # debris strength (a) from which a tornado throws debris out (about EF2-3)
EJECT_BUILD = 1500.0          # s of such a tornado for the full spread
EJECT_FALL = 900.0            # s: ... and for it to fall out once it stops
EJECT_LEN = (2500.0, 14000.0) # m downstream (storm-relative) of the tornado: barely started .. full
EJECT_DBZ = (32.0, 48.0)      # dBZ of the thrown-out debris: far edge .. next to the tornado
EJECT_CLEAR = 0.6             # the share of the rain echo it takes over where thickest (as DEBRIS_CLEAR)
EJECT_TOP = (1200.0, 3000.0)  # m: how high it is spread, weakest .. strongest


WATER = 17                    # land-cover category of open water: no debris lofted from it, none shown over it


def _dry(f: "PatchRadarFields", env, mask):
    """0 over open water, 1 elsewhere (cupy, the patch's shape), looked up only inside ``mask``'s bounding box
    (1 where there is no land cover)."""
    import cupy as cp
    from cupyx.scipy import ndimage
    lc, latlon = getattr(env, "landcover", None), getattr(env, "latlon", None)
    dry = cp.ones((f.ny, f.nx), cp.float32)
    if lc is None or latlon is None or not bool(mask.any()):
        return dry
    jj, ii = cp.nonzero(mask)
    j0, j1, i0, i1 = int(jj.min()), int(jj.max()) + 1, int(ii.min()), int(ii.max()) + 1
    X, Y = np.meshgrid(f.x0 + (np.arange(i0, i1) + 0.5) * f.dx, f.y0 + (np.arange(j0, j1) + 0.5) * f.dx)
    wet = (lc.category(*latlon(X, Y)) == WATER).astype(np.float32)
    dry[j0:j1, i0:i1] = 1.0 - cp.asarray(wet)
    return ndimage.minimum_filter(dry, size=3)        # a gate or two off the shore too


def _debris_ejection(f: "PatchRadarFields", env, tors: list, seed: int) -> bool:
    """The debris thrown out by strong, long-lived tornadoes (``tors``: c.ej > 0), added to Zh, Zv, rho_hv."""
    import cupy as cp
    from cupyx.scipy import ndimage
    if not tors:
        return False
    zh_host = np.asarray(env.z_h)
    dx = f.dx
    k15 = min(int(np.searchsorted(zh_host, 1500.0)), f.nz - 1)
    X = f.x0 + (cp.arange(f.nx, dtype=cp.float32) + 0.5) * dx
    Y = f.y0 + (cp.arange(f.ny, dtype=cp.float32) + 0.5) * dx
    kt = min(int(np.searchsorted(zh_host, EJECT_TOP[1] * 1.5)) + 1, f.nz)
    zlev = cp.asarray(zh_host[:kt], dtype=cp.float32)[:, None, None]
    rs = cp.random.RandomState(seed ^ 0x5EED)
    add = cp.zeros((kt, f.ny, f.nx), cp.float32)
    n = max(1, int(round(8000.0 / dx)))
    for c in tors:
        i, j = int((c.x - f.x0) / dx), int((c.y - f.y0) / dx)
        if not (0 <= i < f.nx and 0 <= j < f.ny):
            continue
        # the air the debris rides: the model's own ~1.5 km wind round the tornado, relative to the tornado
        u = float(f.uc[k15][max(j - n, 0):j + n + 1, max(i - n, 0):i + n + 1].mean()) - c.vx
        v = float(f.vc[k15][max(j - n, 0):j + n + 1, max(i - n, 0):i + n + 1].mean()) - c.vy
        sp = math.hypot(u, v)
        ux, uy = (u / sp, v / sp) if sp > 1.0 else (-c.vx / max(math.hypot(c.vx, c.vy), 1e-3), -c.vy / max(math.hypot(c.vx, c.vy), 1e-3))
        e = min(c.ej, 1.0)
        L = EJECT_LEN[0] + (EJECT_LEN[1] - EJECT_LEN[0]) * e
        ex, ey = X[None, :] - c.x, Y[:, None] - c.y
        along = ex * ux + ey * uy
        cross = -ex * uy + ey * ux
        q = cp.clip(along / L, 0.0, 1.0)
        width = 900.0 + 0.45 * cp.maximum(along, 0.0)                        # it fans out as it goes
        lat = cp.exp(-0.5 * (cross / width) ** 2)
        reach = cp.clip((along + 1500.0) / 1500.0, 0.0, 1.0) * cp.clip((L - along) / (0.3 * L), 0.0, 1.0)
        area = lat * reach
        top = EJECT_TOP[0] + (EJECT_TOP[1] - EJECT_TOP[0]) * min(c.a, 1.0)
        vert = cp.clip((top * (1.0 - 0.4 * q[None]) - zlev) / (0.4 * top), 0.0, 1.0)
        dbz = EJECT_DBZ[0] + (EJECT_DBZ[1] - EJECT_DBZ[0]) * (1.0 - q) * (0.6 + 0.4 * e)
        z = (10.0 ** (0.1 * dbz) * area * e)[None] * vert
        add = cp.maximum(add, z.astype(cp.float32))
    if float(add.max()) <= 1.0e-2:
        return False
    # clumpy, not a smooth plume: debris comes in streamers and clusters a few gates across
    clump = ndimage.uniform_filter(rs.random_sample(add.shape[1:], dtype=cp.float32), size=3)
    clump = cp.clip((clump - 0.35) / 0.3, 0.0, 1.0) * (0.3 + 1.4 * rs.random_sample(add.shape[1:], dtype=cp.float32))
    add = add * (clump * _dry(f, env, add.max(axis=0) > 1.0e-2))[None]
    # debris mixed through heavy rain still lowers CC: where it is thick it takes over part of the rain's echo
    share = cp.clip(add / 10.0 ** (0.1 * EJECT_DBZ[1]), 0.0, 1.0) * EJECT_CLEAR
    take = share * f.zh[:kt]
    for a_ in (f.zh, f.zv, f.rz):
        a_[:kt] *= 1.0 - share
    add = cp.maximum(add, take)
    zdr = 10.0 ** (0.1 * cp.clip(1.2 * rs.standard_normal(add.shape, dtype=cp.float32), -3.0, 3.0))
    rho = cp.clip(0.62 + 0.14 * rs.standard_normal(add.shape, dtype=cp.float32), 0.3, 0.88)
    f.zh[:kt] += add
    f.zv[:kt] += add / zdr
    f.rz[:kt] += rho * add / cp.sqrt(zdr)
    return True


# --------------------------------------------------------------- clear air
# The clear air's own wind: what the contamination layer's insects carry (add_contamination).  The
# beam's radial wind is weighted through the boundary layer by clear_air_z's profile -- the model's
# winds inside patches (sample_sweep, acc[..., 8:10]), the sounding's elsewhere (_clear_air_outside).
# Its echo is add_contamination's (RadarPainter's layer), not this profile's.
CLEAR_AIR_DBZ = 5.0
CLEAR_AIR_TOP = 1500.0           # m above ground
CLEAR_AIR_SPECKLE = 0.004        # fraction of empty gates that show noise speckle (x clear_air)


def clear_air_z0(radar: Radar) -> float:
    a = max(float(radar.clear_air), 0.0)
    return a * 10.0 ** (0.1 * CLEAR_AIR_DBZ) if a > 0 else 0.0


def clear_air_z(z0: float, top: float, h, x, y, xp):
    """sample_sweep's clear_air(), array version: linear Z at heights h (m AGL), world (x, y)."""
    a = xp.clip((h - 0.4 * top) / (0.6 * top), 0.0, 1.0)
    b = xp.clip((h - top) / 600.0, 0.0, 1.0)
    db = (-20.0 * a * a * (3.0 - 2.0 * a) - 15.0 * b * b * (3.0 - 2.0 * b)
          + 3.0 * (0.6 * xp.sin(x * 2.7318e-4 + 1.3) * xp.sin(y * 2.0268e-4 + 0.4)
                   + 0.4 * xp.sin((x + y) * 5.7120e-4 + 2.1)))
    return xp.where((h >= 0.0) & (h <= top + 600.0), z0 * 10.0 ** (0.1 * db), 0.0)


def _clear_air_outside(acc, radar: Radar, cut: Cut, proj, az0_deg: float, wind) -> None:
    """Clear-air echo of the gates no patch covered (acc[..., 0] == 0), carrying the sounding's
    ground-relative wind: ``wind`` = (heights, u, v) (m, m/s)."""
    import cupy as cp
    z0 = clear_air_z0(radar)
    if z0 <= 0 or wind is None:
        return
    nrad, ngate = acc.shape[:2]
    out = ~(acc[..., 0] > 0)
    if not bool(out.any()):
        return
    zw, ug, vg = (np.asarray(a, dtype=np.float64) for a in wind)
    R = cut.first + np.arange(ngate) * cut.gate
    conv = math.radians(float(proj.convergence(radar.lon)))
    azg = cp.asarray(np.radians(az0_deg + (np.arange(nrad) + 0.5) * 360.0 / nrad) - conv)[:, None]
    sa, ca = cp.sin(azg), cp.cos(azg)
    for off in SUB_OFFSETS:
        el = math.radians(cut.elev + off * radar.beamwidth)
        h = np.sqrt(R * R + KE_RE * KE_RE + 2.0 * R * KE_RE * math.sin(el)) - KE_RE + radar.tower_m
        s = KE_RE * np.arcsin(np.clip(R * math.cos(el) / (KE_RE + h), -1.0, 1.0))
        wgt = math.exp(-TWO_WAY * off * off)
        ce = np.cos(el + s / KE_RE)
        hc, sc = cp.asarray(h)[None, :], cp.asarray(s)[None, :]
        Za = clear_air_z(z0, CLEAR_AIR_TOP, hc, radar.x + sc * sa, radar.y + sc * ca, cp) * out
        ue = cp.asarray(np.interp(h, zw, ug) * ce)[None, :]
        ve = cp.asarray(np.interp(h, zw, vg) * ce)[None, :]
        va = ue * sa + ve * ca
        wa = (wgt * Za).astype(cp.float32)
        acc[..., 8] += wa
        acc[..., 9] += wa * va



# --------------------------------------------------------------- contamination (RadarPainter's)
# The clutter / clear-air layer of RadarPainter (app.js getClutter, worker.js buildLevel2), gate by gate:
# patchy biological and clear-air returns within CONTAM["radius"] km of the radar, denser toward it
# (two octaves of value noise at "scale" and 2.2 km, plus a per-gate random), min..max dBZ, carrying the
# low-level wind (here the model's and the sounding's own, add_contamination's air velocity) with a
# little scatter, SW 1.5-6, ZDR 1.5-5.5, CC 0.35-0.85; a stationary ground-clutter core within "core" km:
# sparse gates of 8-36 dBZ, ~0 velocity, ZDR -3..5, CC 0.4-0.85.  Full on tilts up to 0.6 deg, 30 dB
# weaker and gone by 2.2 deg.  Storm echo wins on reflectivity; the other moments blend from the clutter's
# to the storm's over "blend" dB of reflectivity difference, so storm edges do not pop out.  The patches
# drift with the low-level wind and the fine grain is new every scan; the clutter core stays put.
# The radar's clear_air amount scales the density (1 = RadarPainter's defaults, 0 = off).
# the user's RadarPainter settings (2026-10-24): radius 70 km, density 1, -12..36 dBZ, a 31 km core at density 0
# (no stationary core), 1 km patches, 10 dB velocity blend; a new random pattern every scan
CONTAM = dict(radius=70.0, density=1.0, max=36.0, min=-12.0, core=31.0, core_den=0.0, scale=1.0, blend=10.0)
# On top of it, what a real radar's clear air does: insects and Bragg scatter fill only the boundary
# layer (BIO_TOP: the echo fades as the beam climbs out of it, so higher tilts show a smaller ring), the
# bloom grows after sunset (+5 dB and denser at night) and migrating birds then add their own airspeed
# (BIRD_SPEED, north in spring, south in fall), insects aligned with the wind give their highest ZDR
# where the beam crosses it broadside, and the towns within TOWN_CLUTTER_KM are fixed clusters of strong,
# still ground clutter on the low tilts.
BIO_TOP = 1600.0                 # m above ground
BIRD_SPEED = 9.0                 # m/s
TOWN_CLUTTER_KM = 60.0
CLUTTER_FILTERED = 20.0          # dBZ of storm echo: clutter is filtered out of it (no blend)


_HASH01_KERNEL = None


def _hash01_gpu(ix, iy, seed: int):
    """_hash01 on the GPU in one kernel (the same 64-bit wrap-around arithmetic, so the same values): the array
    version's eight passes, called a few dozen times a sweep, made add_contamination ~40 ms of every sweep."""
    import cupy as cp
    global _HASH01_KERNEL
    if _HASH01_KERNEL is None:
        _HASH01_KERNEL = cp.ElementwiseKernel(
            "int64 ix, int64 iy, int64 seed", "float32 out",
            """unsigned long long h = ((unsigned long long)ix * 374761393ULL + (unsigned long long)iy * 668265263ULL
                                       + (unsigned long long)seed * 2246822519ULL) & 0xFFFFFFFFULL;
               h = ((h ^ (h >> 13)) * 1274126177ULL) & 0xFFFFFFFFULL;
               h = h ^ (h >> 16);
               out = (float)(h & 0xFFFFFFULL) / 16777216.0f;""",
            "arht_hash01")
    return _HASH01_KERNEL(ix.astype(cp.int64, copy=False), iy.astype(cp.int64, copy=False), cp.int64(seed))


def _hash01(ix, iy, seed: int, xp):
    """0..1 hash of integer lattice points (any shape) and a seed."""
    if xp is not np:
        return _hash01_gpu(ix, iy, seed)
    h = (ix.astype(xp.int64) * 374761393 + iy.astype(xp.int64) * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(xp.float32) / float(1 << 24)


def _value_noise(x, y, seed: int, xp):
    x0, y0 = xp.floor(x), xp.floor(y)
    fx, fy = x - x0, y - y0
    ux, uy = fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy)
    ix, iy = x0.astype(xp.int64), y0.astype(xp.int64)
    a, b = _hash01(ix, iy, seed, xp), _hash01(ix + 1, iy, seed, xp)
    c, d = _hash01(ix, iy + 1, seed, xp), _hash01(ix + 1, iy + 1, seed, xp)
    return (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uy


def _fbm(x, y, seed: int, octaves: int, xp):
    """RadarPainter's fbm: octaves at 2.03 x / 1.97 y the last, half the weight, seed + 101 each."""
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        tot = tot + amp * _value_noise(x, y, seed + 101 * o, xp)
        norm += amp
        x, y, amp = x * 2.03, y * 1.97, amp * 0.5
    return tot / norm


_TOWN_CLUTTER: dict = {}          # (radar, tilt geometry) -> the towns' fixed clutter (add_contamination)


def add_contamination(radar: Radar, cut: Cut, data: dict, t: float, drift=(0.0, 0.0), az0_deg: float = 0.0,
                      when=None, towns=None) -> None:
    """RadarPainter's clutter / clear-air layer on one sampled cut (in place, after add_noise), with what a
    real WSR-88D adds to it: ``t`` the scan's model time (s); ``drift`` the 0-1 km ground-relative wind
    (m/s) its patches move with and its insects line up with; ``when`` the scan's UTC datetime (the bloom
    grows after sunset, migrating birds fly at night in spring and fall); ``towns`` (n, 3) km east and
    north of the radar and population (fixed ground clutter).  See CONTAM and BIO_TOP."""
    import cupy as cp
    amount = float(radar.clear_air)
    cw = min(1.0, max(0.0, 1.0 - (cut.elev - 0.6) / 1.6))                     # ground clutter: low tilts only
    if amount <= 0:
        return
    c = CONTAM
    if radar.kind == "tdwr":
        # a TDWR scans a few tenths of a degree up next to an airport and a city: much more ground clutter
        c = dict(CONTAM, core=20.0, core_den=0.35)
    ref, has = data["REF"], data["_has"]
    nrad, ng = ref.shape
    R = cut.first + np.arange(ng) * cut.gate
    el = math.radians(cut.elev)
    h = np.sqrt(R * R + KE_RE * KE_RE + 2.0 * R * KE_RE * math.sin(el)) - KE_RE + radar.tower_m
    s = KE_RE * np.arcsin(np.clip(R * math.cos(el) / (KE_RE + h), -1.0, 1.0)) / 1000.0       # ground km
    # insects and Bragg scatter fill the boundary layer: an echo 35 dB weaker by the time the beam is
    # BIO_TOP x 1.05 up (the ring shrinks with tilt, as a real one does), none above
    loss = 35.0 * np.clip((h - 0.45 * BIO_TOP) / (0.6 * BIO_TOP), 0.0, 1.0)
    keep = int(np.searchsorted(s, c["radius"])) + 1
    if (loss >= 35.0).any():
        keep = min(keep, int(np.argmax(loss >= 35.0)) + 1)
    if cw > 0 and towns is not None and len(towns):
        keep = max(keep, int(np.searchsorted(s, TOWN_CLUTTER_KM + 3.0)) + 1)
    keep = min(keep, ng)
    if keep <= 1:
        return
    sk = cp.asarray(s[:keep], dtype=cp.float32)[None, :]
    lk = cp.asarray(loss[:keep], dtype=cp.float32)[None, :]
    azr = np.radians(az0_deg + (np.arange(nrad) + 0.5) * 360.0 / nrad)
    az = cp.asarray(azr, dtype=cp.float32)[:, None]
    x, y = sk * cp.sin(az), sk * cp.cos(az)                                     # km east, north of the radar
    base = zlib.crc32(radar.icao.encode()) & 0x7FFFFFFF
    scan = (base ^ int(t * 1000.0)) & 0x7FFFFFFF
    gi, gj = cp.arange(nrad)[:, None], cp.arange(keep)[None, :]
    hs = lambda k: _hash01(gi + 0 * gj, gj + 0 * gi, scan + k, cp)              # fresh every scan
    hg = lambda k: _hash01(cp.floor(x / 0.25), cp.floor(y / 0.25), base + k, cp)  # fixed on the ground
    # night: the bloom grows after sunset (insects, then migrating birds), strongest around 1 am local
    night, mig = 0.0, (0.0, 0.0)
    if when is not None:
        local = (when.hour + when.minute / 60.0 + radar.lon / 15.0) % 24.0
        night = 0.5 * (1.0 + math.cos(2.0 * math.pi * (local - 1.0) / 24.0))
        night = max(0.0, min(1.0, 1.6 * night - 0.6))
        if when.month in (3, 4, 5, 8, 9, 10, 11):                         # birds aloft: north in spring, south in fall
            head = math.radians(20.0 if when.month <= 5 else 200.0)
            mig = (BIRD_SPEED * night * math.sin(head), BIRD_SPEED * night * math.cos(head))
    r = sk
    rad, core = c["radius"], c["core"]
    env = cp.clip(1.0 - (r / rad) ** 1.6, 0.0, 1.0)
    n1 = _fbm(x / c["scale"], y / c["scale"], scan, 3, cp)                 # a new pattern every scan
    n2 = _fbm(x / 2.2, y / 2.2, scan + 7, 2, cp)
    sp = hs(99)
    field = 0.55 * n1 + 0.32 * n2 + 0.13 * sp
    th = 1.0 - min(0.98, c["density"] * amount * (1.0 + 0.5 * night)) * env * 0.95
    hit = (r < rad) & (field > th)
    q = cp.clip((field - th) / cp.maximum(1.0 - th, 0.02), 0.0, 1.0) ** 0.65
    cref = c["min"] + (c["max"] - c["min"]) * q * (0.55 + 0.45 * env) + (sp - 0.5) * 4.0 + 5.0 * night - lk
    air = data.get("_air_vel")
    air = cp.nan_to_num(air[:, :keep], nan=0.0) if air is not None else cp.zeros_like(cref)
    bird = (mig[0] * cp.sin(az) + mig[1] * cp.cos(az)) * math.cos(el)          # birds' own airspeed, radial
    cvel = air * (0.7 + 0.6 * n1) + bird + (hs(3) - 0.5) * 2.5
    csw = 1.5 + 3.0 * hs(5) + 1.5 * n2
    # insects line up with the wind: highest ZDR where the beam crosses it broadside
    wdir = math.atan2(drift[0], drift[1])
    cross = cp.sin(az - wdir) ** 2
    # (1-3 dB typical, ~4.5 at most broadside: 5-7 dB everywhere read as noise next to a storm's 0-3 dB rain)
    czdr = 0.6 + 3.2 * cross * (0.6 + 0.4 * n2) + (hs(11) - 0.5) * 1.5
    ccc = 0.35 + 0.5 * hs(13)
    hit = hit & (cref > -30.0)
    # the stationary ground-clutter core, and the towns near the radar (buildings, towers)
    rc = cp.clip(r / core, 0.0, 1.0)
    pc = c["core_den"] * (1.0 - rc) * (0.5 + 0.5 * n2)
    gc_hit = (r < core) & (hg(21) < pc)
    gc = 8.0 + 28.0 * hg(23) * (1.0 - rc * 0.6) + (hs(41) - 0.5) * 3.0
    if towns is not None and len(towns):
        # the towns' clutter stands still: worked out once per radar and tilt geometry, not every scan
        tkey = (radar.icao, round(float(cut.elev), 3), nrad, keep, float(cut.first), float(cut.gate),
                round(float(az0_deg), 3), len(towns))
        town = _TOWN_CLUTTER.get(tkey)
        if town is None:
            t_db = cp.full(x.shape, -99.0, dtype=cp.float32)
            t_hit = cp.zeros(x.shape, dtype=bool)
            h51, h53 = hg(51), hg(53)
            near = towns[np.hypot(towns[:, 0], towns[:, 1]) < TOWN_CLUTTER_KM]
            for tx, ty, pop in near[np.argsort(-near[:, 2])][:40]:
                tr = float(np.clip(0.35 + 0.45 * math.log10(max(pop, 300.0) / 1000.0), 0.3, 2.5))     # km
                d = cp.hypot(x - float(tx), y - float(ty))
                in_town = (d < tr) & (h51 < 0.55 * (1.0 - d / tr) + 0.15)
                tdb = 26.0 + 22.0 * h53 * (1.0 - 0.5 * d / tr) + 4.0 * math.log10(max(pop, 300.0) / 3000.0)
                t_db = cp.where(in_town, cp.maximum(t_db, tdb), t_db)
                t_hit = t_hit | in_town
            if len(_TOWN_CLUTTER) > 64:
                _TOWN_CLUTTER.clear()
            town = _TOWN_CLUTTER[tkey] = (t_db, t_hit)
        gc = cp.where(town[1], cp.maximum(gc, town[0]), gc)
        gc_hit = gc_hit | town[1]
    gc = gc - (1.0 - cw) * 30.0
    gc_hit = gc_hit & (gc > -30.0) & (cw > 0)
    take = gc_hit & ~(gc < cp.where(hit, cref, -99.0))
    cref = cp.where(take, gc, cref)
    cvel = cp.where(take, (hs(25) - 0.5) * 1.6, cvel)
    csw = cp.where(take, 0.5 + 1.5 * hs(27), csw)
    czdr = cp.where(take, -3.0 + 8.0 * hs(29), czdr)
    ccc = cp.where(take, 0.4 + 0.45 * hs(31), ccc)
    hit = hit | gc_hit
    # storm echo wins on reflectivity; the rest blend by how far it is above the clutter -- at weak echo
    # edges only: in precipitation (CLUTTER_FILTERED dBZ and up) the clutter filter takes the clutter out,
    # so a storm over a town keeps its own reflectivity and CC (it used to read as a debris signature)
    sref = cp.where(has[:, :keep], ref[:, :keep], -32.0)
    w = cp.clip((sref - cref) / c["blend"] + 0.5, 0.0, 1.0)
    w = cp.where(has[:, :keep], w * w * (3.0 - 2.0 * w), 0.0)
    w = cp.where(sref >= CLUTTER_FILTERED, 1.0, w)
    for name, cv in (("VEL", cvel), ("SW", csw), ("ZDR", czdr), ("RHO", ccc)):
        a = data[name]
        a[:, :keep] = cp.where(hit, w * a[:, :keep] + (1.0 - w) * cv, a[:, :keep])
    ref[:, :keep] = cp.where(hit & (~has[:, :keep] | ((cref > sref) & (sref < CLUTTER_FILTERED))), cref,
                             ref[:, :keep])
    has[:, :keep] |= hit
    data["_has"] = has
    contam = cp.zeros(has.shape, dtype=bool)
    contam[:, :keep] = hit
    data["_contam"] = contam                     # shown below the echo floor, as RadarPainter shows it

# Ground clutter from the terrain: where the beam meets the hills it returns from their faces, a fixed
# pattern on the ground with no motion (CC low, ZDR wild), much stronger than the flat ground's.  The
# clutter filter leaves only a residue (TERRAIN_CLUTTER dBZ for a whole beam at 20 km, weaker with range as a
# surface target in a growing beam) and takes it out of precipitation (CLUTTER_FILTERED), as for the towns.
TERRAIN_CLUTTER = 48.0
TERRAIN_FACE_M = 2500.0          # m down range from where the beam meets a slope that still returns clutter


def add_terrain_clutter(radar: Radar, cut: Cut, data: dict, t: float) -> None:
    """The hills' clutter on one sampled cut (sample_cut's "_terrain"; in place, after add_contamination)."""
    import cupy as cp
    f = data.get("_terrain")
    if f is None or float(radar.clear_air) <= 0:
        return
    ref, has = data["REF"], data["_has"]
    nrad, ng = ref.shape
    from cupyx.scipy import ndimage as cnd
    f = f[:, :ng]
    if not bool((f > 0.02).any()):
        return
    # the slope facing the radar returns all the way up from where the beam first meets it (TERRAIN_FACE_M down
    # range, weakening), and the beam is a radial or so wide
    n = max(2, int(round(TERRAIN_FACE_M / cut.gate)))
    k = cp.asarray(np.exp(-np.arange(n) / (0.5 * n)), cp.float32)
    face = f.copy()
    for d in range(1, n):
        face[:, d:] = cp.maximum(face[:, d:], f[:, :-d] * k[d])
    face = cnd.maximum_filter1d(face, 3, axis=0, mode="wrap")
    R = cp.asarray(cut.first + np.arange(ng) * cut.gate, cp.float32)[None, :]
    azr = cp.asarray(np.radians((np.arange(nrad) + 0.5) * 360.0 / nrad), cp.float32)[:, None]
    x, y = R * cp.sin(azr) / 1000.0, R * cp.cos(azr) / 1000.0
    base = zlib.crc32(radar.icao.encode()) & 0x7FFFFFFF
    scan = (base ^ int(t * 1000.0)) & 0x7FFFFFFF
    gi, gj = cp.arange(nrad)[:, None], cp.arange(ng)[None, :]
    hs = lambda k: _hash01(gi + 0 * gj, gj + 0 * gi, scan + k, cp)               # fresh every scan
    hg = lambda k: _value_noise(x / 0.6, y / 0.6, base + k, cp)                   # fixed on the ground, blotchy
    patch = hg(59)
    hit = (face > 0.02) & (patch > 0.38 + 0.4 * (1.0 - cp.minimum(face / 0.3, 1.0)))     # grazing: sparse
    cref = (TERRAIN_CLUTTER + 10.0 * cp.log10(cp.maximum(face, 1e-3)) - 10.0 * cp.log10(cp.maximum(R, 2.0e4) / 2.0e4)
            + 18.0 * (patch - 0.65) + 8.0 * (_hash01(cp.floor(x / 0.25), cp.floor(y / 0.25), base + 61, cp) - 0.5)
            + 3.0 * (hs(63) - 0.5))
    hit = hit & (cref > radar.floor_dbz)
    sref = cp.where(has, ref, -99.0)
    take = hit & (sref < CLUTTER_FILTERED) & (cref > sref)
    ref[...] = cp.where(take, cref, ref)
    for name, val in (("VEL", (hs(65) - 0.5) * 1.2), ("SW", 0.4 + 1.2 * hs(67)), ("ZDR", -4.0 + 9.0 * hs(69)),
                      ("RHO", 0.35 + 0.45 * hs(71))):
        a = data.get(name)
        if a is not None:
            a[...] = cp.where(take, val, a)
    has |= take
    data["_has"] = has
    if "_contam" in data:
        data["_contam"] = data["_contam"] | take


# --------------------------------------------------------------- sampling
def _beam_width(acc, daz_deg: float, bw: float = BEAMWIDTH):
    """The beam's width across azimuth: each gate's sums blended with the neighbouring
    radials' by the two-way pattern of the effective beam (TWO_WAY), which a gate on the
    beam's centre line alone misses.  The next 0.5-degree radial weighs 0.26 (the beam is
    1.02 degrees wide), the next 1-degree one 0.06.  At range this joins a hook's model-scale
    fragments into the band a real radar shows, while a tornado's couplet stays gate-to-gate.
    Returns the blended sums and their total weight."""
    import cupy as cp
    width = (EFFECTIVE_BEAMWIDTH[daz_deg < 0.75] if abs(bw - BEAMWIDTH) < 1e-6
             else math.sqrt(bw * bw + (0.9 * daz_deg) ** 2))
    out, total = acc.copy(), 1.0
    for k in (1, 2):
        w = math.exp(-TWO_WAY * (k * daz_deg / width) ** 2)
        if w < 0.02:
            break
        out += w * (cp.roll(acc, k, axis=0) + cp.roll(acc, -k, axis=0))
        total += 2.0 * w
    return out, total


def sample_cut(radar: Radar, cut: Cut, proj: geo.Lambert, fields: list[PatchRadarFields],
               az0_deg: float = 0.0, moments: tuple | None = None, wind=None, terrain=None) -> dict:
    """Sample one cut from every patch; returns float arrays per moment.  ``wind`` (heights,
    u, v: the sounding's ground-relative wind) gives clear air outside the patches its motion.
    ``terrain`` (terrain.Terrain): the beam starts at the radar's own ground, the hills block it.
    A DOW sees no debris balls (the user's choice): its sweeps sample the echo as it was before them;
    the tornadoes' winds (the couplets) it does see."""
    if radar.kind == "dow":
        import contextlib
        with contextlib.ExitStack() as stack:
            for f in (fields.values() if isinstance(fields, dict) else fields):
                if getattr(f, "clean_box", None) is not None:
                    stack.enter_context(f.without_debris())
            return _sample_cut(radar, cut, proj, fields, az0_deg, moments, wind, terrain)
    return _sample_cut(radar, cut, proj, fields, az0_deg, moments, wind, terrain)


def _sample_cut(radar: Radar, cut: Cut, proj: geo.Lambert, fields, az0_deg: float = 0.0,
                moments: tuple | None = None, wind=None, terrain=None) -> dict:
    import cupy as cp

    nrad = cut.nrad
    daz = 360.0 / nrad
    # sampled to n_far: the echoes beyond a Doppler cut's own range fold into it (range_fold)
    ngate = max(cut.n_ref, cut.n_far if cut.kind in ("dop", "both") else 0)
    acc = cp.zeros((nrad, ngate, 10), dtype=cp.float32)
    args = np.zeros((), dtype=_sweep_struct_dtype())
    args["lat0"] = math.radians(radar.lat)
    args["lon0"] = math.radians(radar.lon)
    args["h0"] = radar.tower_m
    args["elev"] = math.radians(cut.elev)
    args["az0"] = math.radians(az0_deg + 0.5 * daz)
    args["daz"] = math.radians(daz)
    args["r0"] = cut.first
    args["dr"] = cut.gate
    args["bw"] = math.radians(radar.beamwidth)
    args["hole"] = 1 if radar.kind == "dow" else 0        # vortex holes: the DOW's beam only (the user's choice)
    args["nrad"] = nrad
    args["ngate"] = ngate
    args["pn"] = proj.n
    args["pF"] = proj.F
    args["prho0"] = proj.rho0
    args["plon0"] = math.radians(proj.lon0)
    args["pR"] = geo.R_EARTH
    args["ca_z0"] = clear_air_z0(radar)
    args["ca_top"] = CLEAR_AIR_TOP
    blk = None
    if terrain is not None:
        g, args["gx0"], args["gy0"], args["gdx"], args["gn"], ground = ground_raster(radar, terrain)
        args["h0"] = ground + radar.tower_m
        blk = cp.zeros((nrad, ngate), dtype=cp.uint8)
        _launch("beam_block", nrad, (args, g, blk), tpb=64)
    for f in fields:
        args["x0"], args["y0"], args["dx"] = f.x0, f.y0, f.dx
        args["nx"], args["ny"], args["nz"] = f.nx, f.ny, f.nz
        args["ztop"] = f.ztop
        vort, tab = getattr(f, "tornadoes", None), getattr(f, "tornado_winds", None)
        args["nvort"] = 0 if vort is None else len(vort)
        args["nvk"], args["nvr"] = tab.shape[1:] if args["nvort"] else (0, 0)
        if f.lzh is None:                        # fields built without finish(): their logs now
            f.lzh, f.lzv, f.lrz = (cp.log(cp.maximum(a, Z_FLOOR)) for a in (f.zh, f.zv, f.rz))
        _launch("sample_sweep", nrad * ngate,
                (args, f.z_h, f.lzh, f.lzv, f.lrz, f.kdp, f.vt, f.uc, f.vc, f.wc,
                 f.tke if f.tke is not None else np.intp(0),
                 cp.asarray(vort) if args["nvort"] else np.intp(0),
                 tab if args["nvort"] else np.intp(0),
                 f.hgt if (blk is not None and f.hgt is not None) else np.intp(0),
                 blk if blk is not None else np.intp(0), acc))
    _clear_air_outside(acc, radar, cut, proj, az0_deg, wind)
    acc, across = _beam_width(acc, daz, radar.beamwidth)
    # beam weights of the 5 sub-beams that are above ground (all, normally), and across the beam
    wsum = float(sum(math.exp(-TWO_WAY * o * o) for o in SUB_OFFSETS)) * across
    Zs = acc[..., 1]
    has = Zs > 1.0e-6
    Z = Zs / wsum
    dbz = cp.where(has, 10.0 * cp.log10(cp.maximum(Z, 1e-10)), -99.0)
    if cut.wavelength < 0.04:
        pass                                  # a DOW: no attenuation (the user's choice)
    elif cut.wavelength < 0.08:
        dbz = cp.where(has, dbz - c_band_attenuation(dbz, has, cut.gate), dbz)
    out = {"REF": dbz}
    Zsafe = cp.maximum(Zs, 1e-12)
    vel = acc[..., 4] / Zsafe
    var = cp.maximum(acc[..., 5] / Zsafe - vel * vel, 0.0)
    tke = acc[..., 7] / Zsafe
    sw = cp.sqrt(var + 0.6667 * tke + 0.8 ** 2)   # radial variance of isotropic SGS turbulence
    out["VEL"] = vel
    out["SW"] = sw
    zdr = 10.0 * cp.log10(cp.maximum(Zs, 1e-12) / cp.maximum(acc[..., 2], 1e-12))
    out["ZDR"] = cp.clip(zdr, -7.9, 7.9)
    # the beam's mixture: rho_hv = sum(rho sqrt(Zh Zv)) / sqrt(sum Zh * sum Zv), lower where the
    # beam spans different ZDR (hail core edges, the ZDR arc) as a real beam's is
    rho = cp.clip(acc[..., 3] / cp.sqrt(Zsafe * cp.maximum(acc[..., 2], 1e-12)), 0.2, 1.05)
    out["RHO"] = rho
    kdp = acc[..., 6] / wsum
    phi = 60.0 + 2.0 * cp.cumsum(kdp, axis=1) * (cut.gate / 1000.0)
    out["PHI"] = cp.mod(phi, 360.0)
    out["_has"] = has
    out["_ngate"] = cut.n_ref
    aw = acc[..., 8]
    out["_air_vel"] = cp.where(aw > 1.0e-6, acc[..., 9] / cp.maximum(aw, 1.0e-6), cp.nan)
    if blk is not None:
        # the share of the beam's power meeting the ground at each gate: the sub-beams blocked there and not
        # before (add_terrain_clutter)
        wsb = np.array([math.exp(-TWO_WAY * o * o) for o in SUB_OFFSETS])
        lut = cp.asarray([sum(wsb[k] for k in range(5) if b >> k & 1) / wsb.sum() for b in range(32)], cp.float32)
        prev = cp.concatenate([cp.zeros((nrad, 1), cp.uint8), blk[:, :-1]], axis=1)
        out["_terrain"] = lut[(blk & ~prev).astype(cp.int32)]
    return out


# Measurement noise.  A radar's moments are estimates from a few dozen pulses per gate, so they
# scatter about the true value, the more so as the signal-to-noise ratio falls (with range and toward
# an echo's edge): the grain of real reflectivity, speckled CC in rain, low CC and random velocities
# at weak echo edges (Doviak & Zrnic 1993; Melnikov & Zrnic 2007).  Standard deviations at high SNR
# for a super-resolution sweep at VCP timing (reflectivity ~1 dB), for faster volumes scaled by the
# 4th root of the fewer pulses: a radar that fast is a phased array, whose beam multiplexing wins
# back part of them (the square root gave 2-3 dB of speckle at 1-2.5 min volumes, which broke up
# hook echoes' outlines).  Velocity: ~1 m/s at a 3-4 m/s spectrum width on a super-resolution radial
# (25 pulses at ~1 kHz; Doviak & Zrnic 1993, 6.22).  It was 1.2 x sqrt(SW / 2), 1.5-2 m/s in storms: a
# speckle that broke inbound and outbound regions into patches, though the sampled winds are smooth.
SENS_50KM = -5.0                  # dBZ at 50 km with SNR 0 dB in each polarization channel
SNR_THRESHOLD = 2.0               # dB: weaker gates are below threshold (no data), as the RDA censors them
NOISE_HIGH_SNR = {"REF": 2.0, "VEL": 0.8, "SW": 1.0, "ZDR": 0.8, "PHI": 3.0}   # ZDR: real 0.5 deg speckles ~1 dB gate to gate


def add_noise(radar: Radar, cut: Cut, data: dict, seed: int) -> None:
    """Scatter one sampled cut's moments (``sample_cut``'s arrays, in place) like real estimates."""
    import cupy as cp

    rs = cp.random.RandomState(seed)
    Z, has = data["REF"], data["_has"]
    shape = Z.shape
    r_km = (cut.first + cp.arange(shape[1], dtype=cp.float32) * cut.gate) / 1000.0
    snr_db = Z - (radar.sens_50km + 20.0 * cp.log10(r_km / 50.0))[None, :]
    has = has & (snr_db >= SNR_THRESHOLD)
    data["_has"] = has
    inv = 10.0 ** (-cp.clip(snr_db, SNR_THRESHOLD, 60.0) / 10.0)        # 1 / SNR
    k = min(max(radar.dwell_factor, 0.5), 9.0) ** 0.25 * (1.0 if cut.super_res else 1.0 / math.sqrt(2.0))
    g = lambda: rs.standard_normal(shape, dtype=cp.float32)
    s = NOISE_HIGH_SNR
    data["REF"] = cp.where(has, Z + s["REF"] * k * cp.sqrt(1.0 + 2.0 * inv) * g(), Z)
    sw = data["SW"]
    spread = cp.sqrt(1.0 + 4.0 * inv)
    vel = data["VEL"] + s["VEL"] * k * cp.sqrt(cp.maximum(sw, 1.0) / 2.0) * spread * g()
    # at low SNR the velocity estimate is noise: any value across the Nyquist interval
    junk = rs.random_sample(shape, dtype=cp.float32) < 1.0 / (1.0 + (1.0 / (inv * 1.5)) ** 2)
    vn = cut.nyquist
    data["VEL"] = cp.where(junk, (rs.random_sample(shape, dtype=cp.float32) * 2.0 - 1.0) * vn, vel)
    data["SW"] = cp.maximum(sw + s["SW"] * k * spread * g() + cp.minimum(2.0 * inv, 4.0), 0.0)
    data["ZDR"] = cp.clip(data["ZDR"] + s["ZDR"] * k * spread * g(), -7.9, 7.9)
    # CC: noise power the processor does not fully remove pulls it down at low SNR, and its
    # scatter grows as it falls (sd ~ (1 - rho^2))
    rho = data["RHO"] * cp.sqrt(1.0 / (1.0 + inv))
    # pure rain reads a steady 0.98-1.0: little scatter at high SNR, never past 1
    data["RHO"] = cp.clip(rho + (0.25 * (1.0 - rho * rho) + 0.002) * k * (1.0 + 2.0 * inv) * g(), 0.2, 1.0)
    data["PHI"] = cp.mod(data["PHI"] + s["PHI"] * k * spread * g(), 360.0)
    # clear-air noise: a sprinkle of empty gates where noise alone crossed the threshold -- just over
    # it, with random velocities, wide spectrum width, low CC and any ZDR
    if radar.clear_air > 0:
        u = lambda: rs.random_sample(shape, dtype=cp.float32)
        spk = (~has) & (u() < CLEAR_AIR_SPECKLE * min(float(radar.clear_air), 2.0)) & (r_km < 230.0)[None, :]
        sens = (radar.sens_50km + 20.0 * cp.log10(r_km / 50.0))[None, :]
        data["REF"] = cp.where(spk, sens + SNR_THRESHOLD + 4.0 * u(), data["REF"])
        data["VEL"] = cp.where(spk, (2.0 * u() - 1.0) * cut.nyquist, data["VEL"])
        data["SW"] = cp.where(spk, 4.0 + 4.0 * u(), data["SW"])
        data["ZDR"] = cp.where(spk, cp.clip(0.3 + 1.5 * rs.standard_normal(shape, dtype=cp.float32), -4.0, 4.0), data["ZDR"])
        data["RHO"] = cp.where(spk, 0.2 + 0.4 * u(), data["RHO"])
        data["_has"] = has | spk


# C-band attenuation: a TDWR's beam loses power through heavy rain and hail, so echo behind a core reads
# too weak or vanishes (the "attenuation shadow" TDWR users know).  Specific attenuation from the rain rate
# implied by Z (Z = 300 R^1.4, A = 0.0018 R^1.05 dB/km one way at 5.3 cm), more in hail cores (past 55 dBZ,
# Mie scattering), integrated two-way along the ray in front of each gate, at most ATTEN_MAX dB.
ATTEN_MAX = 40.0


def c_band_attenuation(dbz, has, gate_m: float):
    """Two-way path-integrated attenuation (dB) in front of each gate of a C-band ray."""
    import cupy as cp
    d = cp.where(has, dbz, -30.0)
    R = cp.minimum((10.0 ** (0.1 * d) / 300.0) ** (1.0 / 1.4), 200.0)
    A = 0.0018 * R ** 1.05 * (1.0 + 0.15 * cp.clip(d - 55.0, 0.0, 15.0))
    pia = 2.0 * cp.cumsum(A, axis=1) * (gate_m / 1000.0)
    pia = cp.concatenate([cp.zeros_like(pia[:, :1]), pia[:, :-1]], axis=1)
    return cp.minimum(pia, ATTEN_MAX)



# Range folding (the purple "RF" of a Doppler scan).  A Doppler cut's high PRF has a short unambiguous range
# (Cut.unamb_m: ~120-150 km for a WSR-88D's Doppler cuts, ~90-110 km for a TDWR), so echoes a whole number
# of unambiguous ranges apart arrive at the same time and overlay.  As the RDA does with its long-range
# surveillance scan, the strongest of them (in received power, Z / r^2) keeps its velocity if it beats every
# other one there by RF_TOVER; the others -- and all of them when none does -- are flagged range folded.
# Distant storms thus show as purple arcs and rings closer in, at their range minus the unambiguous range.
# Phase coding on the split cuts (SZ-2) recovers more of the second trip: a lower threshold there.
RF_TOVER = {"dop": 3.0, "both": 6.0}      # dB


def range_fold(radar: Radar, cut: Cut, data: dict) -> None:
    """The range-folded gates of a Doppler cut (data["_rf"]), from its echoes out to Cut.n_far."""
    import cupy as cp
    if cut.kind not in ("dop", "both") or not cut.n_dop:
        return
    ref, has = data["REF"], data["_has"]
    nrad, ng = ref.shape
    ra = max(int(round(cut.unamb_m / cut.gate)), 1)              # gates per unambiguous range
    trips = -(-ng // ra)
    if trips < 2:
        return
    r = (cut.first + cp.arange(ng, dtype=cp.float32) * cut.gate)[None, :]
    p = cp.where(has, 10.0 * cp.log10(cp.maximum(10.0 ** (0.1 * ref) / (r * r), 1e-30)), -999.0)
    pad = cp.full((nrad, trips * ra), -999.0, dtype=cp.float32)
    pad[:, :ng] = p
    pt = pad.reshape(nrad, trips, ra)                             # the trips that overlay at each apparent gate
    order = cp.sort(pt, axis=1)
    best, second = order[:, -1:, :], order[:, -2:-1, :]
    keep = (pt >= best) & (best - second >= RF_TOVER.get(cut.kind, 5.0))
    rf = (pt > -999.0) & ~keep & (second > -999.0)
    rf = rf.reshape(nrad, trips * ra)[:, :ng]
    data["_rf"] = rf & (cp.arange(ng)[None, :] < cut.n_dop)


def encode(radar: Radar, cut: Cut, data: dict) -> dict:
    """uint8/uint16 Level II codes per moment for this cut's kind."""
    import cupy as cp

    has = data["_has"] & (data["REF"] >= radar.floor_dbz)
    if "_contam" in data:                        # the clutter layer keeps its weak gates (to -30 dBZ)
        has = has | (data["_has"] & data["_contam"])
    ngate = data["_ngate"]
    if cut.kind == "surv":
        names = ["REF", "ZDR", "PHI", "RHO"]
    elif cut.kind == "dop":
        names = ["REF", "VEL", "SW"]
    else:
        names = ["REF", "VEL", "SW", "ZDR", "PHI", "RHO"]
    if not radar.dualpol:
        names = [m for m in names if m in ("REF", "VEL", "SW")]
    legacy = radar.kind == "legacy"
    if legacy and cut.kind == "dop":                # a legacy split cut's Doppler sweep: velocity only
        names = ["VEL", "SW"]
    rf = data.get("_rf")
    out = {}
    for m in names:
        scale, offset, bits = MOMENTS[m]
        val = data[m]
        ng = ngate if m == "REF" else (min(ngate, cut.n_dop) if m in ("VEL", "SW") else min(ngate, DOP_NGATES))
        v = val[:, :ng]
        h = has[:, :ng]
        rfm = rf[:, :ng] if rf is not None else None
        if legacy and m == "REF":                   # 1 km gates (legacy_ref)
            v, h = legacy_ref(v, h)
            ng = v.shape[1]
        if legacy and m in ("VEL", "SW"):           # from -375 m (LEGACY_DOP_FIRST): empty gates in front
            pad = int(round((REF_FIRST - LEGACY_DOP_FIRST) / REF_GATE))
            keep = min(ng, LEGACY_N_DOP - pad)
            v = cp.concatenate([cp.zeros((v.shape[0], pad), v.dtype), v[:, :keep]], axis=1)
            h = cp.concatenate([cp.zeros((h.shape[0], pad), bool), h[:, :keep]], axis=1)
            if rfm is not None:
                rfm = cp.concatenate([cp.zeros((rfm.shape[0], pad), bool), rfm[:, :keep]], axis=1)
            ng = pad + keep
        if m == "VEL" and radar.alias:
            vn = cut.nyquist
            v = cp.mod(v + vn, 2.0 * vn) - vn
        # dealiased velocity in 1 m/s steps (+-127 m/s, 284 mph) in every sweep of the volume, and message 5
        # says so (level2.encode_volume); 0.5 m/s steps saturated at +-63.5 (142 mph).  Switching only single
        # sweeps made GR2Analyst drop their velocity, so frames lost velocity at random tilts
        if m == "VEL" and not radar.alias:
            scale = VEL_DEALIASED_SCALE
        hi = 255 if bits == 8 else 1023
        code = cp.clip(cp.rint(v * scale + offset), 2, hi)
        code = cp.where(h, code, 0)
        if rfm is not None and m in ("VEL", "SW"):
            code = cp.where(rfm[:, :ng], 1, code)            # range folded: Level II's RF code
        arr = code.astype(cp.uint16 if bits == 16 else cp.uint8)
        host = cp.asnumpy(arr)
        if bits == 16:
            host = host.astype(">u2")
        out[m] = (host, ng, bits, scale, offset)
    return out
