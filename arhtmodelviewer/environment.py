"""The storm environment on ArWen's (gpuwm) grid.

One ``Environment`` holds everything every patch of a run shares: the
stretched vertical coordinate, gpuwm's hydrostatic base state built from the
sounding, the environment wind profile, Coriolis, and the physics settings.
Its ``run_config`` mirrors the April 27 2011 ArWen 1 km run (Morrison
two-moment with hail-like rimed ice, 5th-order horizontal / 3rd-order
vertical advection, positive-definite moisture, monotonic 6th-order filter,
w-damping, implicit Rayleigh damping aloft, epssm 0.5) with WRF's idealized
open lateral boundaries (em_quarter_ss: radiative + emdiv) and a 1.5-order
TKE closure in place of the real-data PBL scheme.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from .sounding import CP, G, KAPPA, P0, Sounding

OMEGA = 7.292e-5


# ------------------------------------------------------------ grid presets
@dataclass(frozen=True)
class GridSpec:
    """Horizontal spacing, levels and time step for one model grid."""
    dx: float                  # m
    nz: int                    # half levels
    dz_bot: float              # lowest layer depth (m)
    dz_top: float              # deepest layer (m)
    dt: float                  # longest time step (s); adaptive dt stays at or below it
    sound: int = 4             # acoustic substeps per step (even)
    diff6: float = 0.08        # WRF diff_6th_factor
    ztop: float = 20000.0

    @property
    def label(self) -> str:
        return f"{self.dx / 1000:g} km" if self.dx >= 1000 else f"{self.dx:g} m"


# No extra 6th-order filter at storm-resolving spacing (WRF em_quarter_ss and CM1
# leave it off; the 5th-order advection is diffusive enough).  With ArWen's
# real-data 0.08 the OUN 1999 supercell lost its mesocyclone ~40 min sooner.
# The 650, 450 and 350 m grids sit between their neighbours: levels, lowest and deepest
# layer interpolated, the time step 6 s per km like every grid.
GRID_PRESETS = {
    "3km": GridSpec(3000.0, 45, 80.0, 700.0, 18.0, 4, 0.12),
    "2km": GridSpec(2000.0, 50, 60.0, 600.0, 12.0, 4, 0.12),
    "1.5km": GridSpec(1500.0, 50, 60.0, 550.0, 9.0, 4, 0.0),
    "1km": GridSpec(1000.0, 50, 50.0, 500.0, 6.0, 4, 0.0),
    "750m": GridSpec(750.0, 56, 40.0, 450.0, 4.5, 6, 0.0),
    "650m": GridSpec(650.0, 60, 36.0, 430.0, 3.9, 6, 0.0),
    "500m": GridSpec(500.0, 64, 30.0, 400.0, 3.0, 6, 0.0),
    "450m": GridSpec(450.0, 68, 28.0, 380.0, 2.7, 6, 0.0),
    "350m": GridSpec(350.0, 74, 24.0, 340.0, 2.1, 6, 0.0),
    "250m": GridSpec(250.0, 80, 20.0, 300.0, 1.5, 6, 0.0),
    # LES spacing (pair with no PBL scheme: the built-in core, or a suite with physics.les): the 250 m
    # grid's progression carried on, 6 s per km
    "200m": GridSpec(200.0, 86, 18.0, 280.0, 1.2, 6, 0.0),
    "150m": GridSpec(150.0, 92, 15.0, 260.0, 0.9, 6, 0.0),
    "100m": GridSpec(100.0, 100, 12.0, 240.0, 0.6, 6, 0.0),
}


def stretched_levels(nz: int, ztop: float, dz_bot: float, dz_top: float) -> np.ndarray:
    """Heights of the nz+1 w levels: layers grow geometrically from dz_bot to
    at most dz_top, then stay uniform, scaled to end exactly at ztop."""
    def total(r, cap):
        dz = np.minimum(dz_bot * r ** np.arange(nz), cap)
        return dz, dz.sum()

    cap = dz_top
    while total(1.25, cap)[1] < ztop:
        cap *= 1.1
    lo, hi = 1.0, 1.25
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if total(mid, cap)[1] < ztop:
            lo = mid
        else:
            hi = mid
    dz, s = total(hi, cap)
    dz = dz * (ztop / s)
    return np.concatenate([[0.0], np.cumsum(dz)])


# --------------------------------------------------------------- physics
@dataclass(frozen=True)
class PhysicsOptions:
    mp_physics: int = 10            # 10 = Morrison 2-moment (ArWen April 27)
    morr_rimed_ice: int = 1         # 1 = hail-like rimed ice
    # 2 = 1.5-order TKE (WRF em_quarter_ss / CM1 sgsmodel=1).  Not 4: without a PBL
    # scheme WRF's vertical_diffusion_2 uses the horizontal Smagorinsky K as the
    # vertical momentum K (xkmv = xkmh, hundreds of m2/s at 1 km), explicit across
    # the 50 m lowest layer -- unstable (K dt / dz^2 > 0.5) under strong cold pools.
    km_opt: int = 2
    c_k: float = 0.10               # TKE mixing constant: CM1's Deardorff c_m = 0.1 (WRF default 0.15 over-mixes 1-km storms)
    mix_isotropic: int = 0          # 1 = one mixing length (dx dy dz)^(1/3) instead of dx horizontally
    surface_drag: bool = True       # bulk drag on the lowest level (isfflx=0, tke_drag_coefficient)
    z0: float = 0.1                 # roughness length (m) for the drag coefficient
    coriolis: bool = True
    latitude: float = 35.0
    epssm: float = 0.5
    w_damping: int = 1
    zdamp: float = 5000.0
    dampcoef: float = 0.2
    # An ArWen physics suite (gpuwm's single-domain physics profiles): its PBL, surface
    # layer, land surface, radiation and microphysics replace the viewer's idealized
    # core (TKE closure, free-slip ground).  None = the built-in core with mp_physics.
    suite: str | None = None
    # LES with a suite: its PBL scheme off and the built-in core's 3-D TKE closure in its place (WRF
    # bl_pbl_physics=0, km_opt=2, diff_opt=2); the suite's surface layer, land and radiation stay.  The
    # built-in core has no PBL scheme anyway.
    les: bool = False
    soil_moisture: float = 0.25    # volumetric (m3/m3), for land-surface suites
    vegetation: float = 50.0        # green vegetation fraction (%)
    landuse: int = 10               # MODIS-IGBP category (10 = grasslands)
    soil_type: int = 6              # STATSGO category (6 = loam)


# Every microphysics scheme ArWen (gpuwm 2.9.2) runs, by WRF number.
MICROPHYSICS = {
    1: "Kessler (warm rain)", 2: "Purdue Lin", 3: "WSM3", 4: "WSM5", 5: "Ferrier-Aligo",
    6: "WSM6", 8: "Thompson", 9: "Milbrandt-Yau 2-moment (best without tables)", 10: "Morrison 2-moment (recommended)",
    16: "WDM6", 18: "NSSL 2-moment (very slow: ~15x; may set off stray cells)", 24: "WSM7 (hail)", 28: "Thompson aerosol-aware",
    50: "P3 (predicted particle properties)", 106: "WSM6R (revised WSM6)",
    900: "TEMPO (double precision: very slow on this GPU)",
}
# What the page offers: the current research-grade two-moment schemes, best first.  The single-moment and
# older schemes (Kessler, Lin, WSM3/5/6/6R/7, WDM6, Ferrier-Aligo) are left out as obsolete for storm-scale
# work.  NSSL is offered last, labelled slow: in this model it set off spurious cells across the domain and
# ran 15x slower (docs/PROGRESS.md, 2026-10-24).  A saved setup with one of the others still runs it.
MICROPHYSICS_MENU = (28, 8, 900, 9, 10, 50, 18)
# The default physics: the classic WRF CONUS suite (Thompson + YSU + MM5 surface layer + Noah + RRTMG).  In
# side-by-side 100-min runs of a classic supercell (2026-10-24) it kept the most realistic cold pools (RFD
# -3.6 to -4.3 K, FFD ~-3 K, as observed in tornadic supercells; the idealized core's RFD reached -8 K), a
# sustained low-level mesocyclone, a clear notch and the strongest hook, the fewest stray cells, and ran
# fastest; the HRRR suite (Thompson-aerosol + MYNN + RUC) looked as good but stopped on a radiation-timing
# check at 90 min.
RECOMMENDED_SUITE = "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"
_SUITE_MP = {28, 8, 900, 9, 10, 50, 18}

# Suite switches the viewer keeps its own values for (its grid presets, idealized flat world).
_SUITE_SKIP = {"moist", "top_lid", "terrain_opt", "diff_6th_factor", "diff_6th_opt", "diff_6th_slopeopt",
               "epssm", "mosaic_lu", "mosaic_soil", "usemonalb", "rdlai2d", "fractional_seaice"}


def physics_suites(menu: bool = False) -> list[dict]:
    """ArWen's physics suites: id, label, maturity and components (``menu``: only those whose microphysics
    is in MICROPHYSICS_MENU, what the page offers)."""
    from gpuwm.physics_compat import SINGLE_DOMAIN_PHYSICS_PROFILES, single_domain_runtime_switches
    from gpuwm.physics_registry import physics_registry
    reg = physics_registry()["templates"]
    out = []
    for pid in SINGLE_DOMAIN_PHYSICS_PROFILES:
        t = reg.get(pid, {})
        sw = single_domain_runtime_switches(pid)
        out.append({"id": pid, "label": t.get("label", pid), "maturity": t.get("maturity", ""),
                    "components": t.get("components", {}), "mp_physics": int(sw.get("mp_physics", 0)),
                    "surface": bool(sw.get("sf_sfclay_physics") or sw.get("bl_pbl_physics")),
                    "radiation": bool(sw.get("ra_lw_physics") or sw.get("ra_sw_physics"))})
    if menu:
        out = [s for s in out if s["mp_physics"] in _SUITE_MP]
        for s in out:
            if s["id"] == RECOMMENDED_SUITE:
                s["label"] += " (best suite)"
    return out


# ----------------------------------------------------------- environment
class Environment:
    """Sounding + grid + physics, and builders for gpuwm states on it."""

    def __init__(self, sounding: Sounding, grid: GridSpec,
                 physics: PhysicsOptions | None = None, motion=(0.0, 0.0)):
        from gpuwm.core.grid import make_base_state, make_vertical_coord

        self.sounding = sounding.cleaned()
        self.grid = grid
        self.physics = physics or PhysicsOptions()
        self.suite = None
        if self.physics.suite:
            from gpuwm.physics_compat import single_domain_runtime_switches
            self.suite = single_domain_runtime_switches(self.physics.suite)
            # the suite's own microphysics and closure; its surface layer does the drag
            self.physics = dataclasses.replace(
                self.physics, mp_physics=int(self.suite.get("mp_physics", self.physics.mp_physics)),
                km_opt=2 if self.physics.les else int(self.suite.get("km_opt", self.physics.km_opt)),
                surface_drag=False if self.suite.get("sf_sfclay_physics") else self.physics.surface_drag)
        self.start_time = None            # datetime (UTC) of model time 0, for radiation (set by the sim)
        self.latlon = None                # f(X, Y) -> (lat, lon) of ground points (set by the sim)
        self.longitude = -97.0
        theta_f, qv_f, u_f, v_f = self.sounding.profile_functions(ztop=grid.ztop + 5000.0)
        self.theta_f, self.qv_f, self.u_f, self.v_f = theta_f, qv_f, u_f, v_f

        # eta levels that land on the stretched heights of the dry base state
        zw = stretched_levels(grid.nz, grid.ztop, grid.dz_bot, grid.dz_top)
        p_surf = float(self.sounding.psfc)
        zf = np.append(np.arange(0.0, grid.ztop, 1.0), grid.ztop)
        zmid = 0.5 * (zf[:-1] + zf[1:])
        pi_s = (p_surf / P0) ** KAPPA
        pi = np.concatenate(([pi_s], pi_s - np.cumsum(G * np.diff(zf) / (CP * theta_f(zmid)))))
        p_of_z = P0 * pi ** (1.0 / KAPPA)
        p_top = p_of_z[-1]
        eta = (np.interp(zw, zf, p_of_z) - p_top) / (p_surf - p_top)
        eta[0], eta[-1] = 1.0, 0.0
        self.coord = make_vertical_coord(grid.nz, eta_levels=eta)
        self.base = make_base_state(self.coord, theta_f, p_surf=p_surf, ztop=grid.ztop)
        phb = np.asarray(self.base.phb, dtype=np.float64)
        self.z_w = phb / G                                   # (nz+1,) w-level heights
        self.z_h = 0.5 * (self.z_w[:-1] + self.z_w[1:])      # (nz,) half-level heights
        # The model grid moves with ``motion`` (CM1's umove/vmove): storms that travel
        # with it stay put on the grid instead of being smeared by advection across it.
        # Model winds are relative to the moving grid; u_ground/v_ground are the sounding's.
        self.motion = (float(motion[0]), float(motion[1]))
        self.u_ground = u_f(self.z_h)
        self.v_ground = v_f(self.z_h)
        self.u_env = self.u_ground - self.motion[0]
        self.v_env = self.v_ground - self.motion[1]
        # the grid's motion history: (start time, ground x, y of the grid origin then, mx, my)
        self._track = [(0.0, 0.0, 0.0, self.motion[0], self.motion[1])]
        self.offset = [0.0, 0.0]          # ground position of the grid origin now (m), set by the engine
        self.clock = 0.0                  # model time now (s), set by the engine
        self.airmass = None               # boundaries.AirMasses of the run (fronts, drylines, outflow)
        self.f = (2.0 * OMEGA * math.sin(math.radians(self.physics.latitude))
                  if self.physics.coriolis else 0.0)
        self.cd = self.drag_coefficient(self.physics.z0) if self.physics.surface_drag else 0.0
        # the viewer applies the drag itself (Engine._surface_drag): on the ground-relative wind, with
        # each column's own roughness when ``roughness`` is set (land cover); ArWen's own takes one
        # coefficient for the whole grid and the grid-relative wind
        self.own_drag = bool(self.cd > 0)
        self.roughness = None             # f(X, Y) -> z0 (m) at ground points, or None: physics.z0 everywhere
        self.terrain = None               # terrain.Terrain: patches get its ground (set by the sim; grid ground-fixed)
        self.env_column = None                                # filled by reference_columns()
        from .sounding import diagnostics
        self.diag = diagnostics(self.sounding)
        self.storm_motion = tuple(self.diag["rm"])            # Bunkers right mover (m/s)

    # --------------------------------------------------------- configs
    def run_config(self, nx: int, ny: int, dt: float | None = None, *, bc: str = "specified"):
        """gpuwm RunConfig for an nx x ny patch.

        ``bc``: "specified" (WRF real-data style specified + relaxation zone,
        driven by the undisturbed environment -- what patches use),
        "open" (WRF idealized radiative) or "periodic" (the reference column).
        """
        from gpuwm.config import RunConfig, validate_run_config

        g, ph = self.grid, self.physics
        cfg = RunConfig(
            nx=int(nx), ny=int(ny), nz=g.nz, dx=g.dx, dy=g.dx, ztop=g.ztop,
            dt=float(dt or g.dt), run_seconds=1.0e9, p_surf=float(self.sounding.psfc),
            time_step_sound=g.sound, epssm=ph.epssm, smdiv=0.1,
            emdiv=0.01 if bc == "open" else 0.0,
            damp_opt=3, zdamp=ph.zdamp, dampcoef=ph.dampcoef, w_damping=ph.w_damping,
            moist=True, mp_physics=ph.mp_physics, morr_rimed_ice=ph.morr_rimed_ice,
            moist_adv_opt=1, h_mom_adv_order=5, h_sca_adv_order=5,
            v_mom_adv_order=3, v_sca_adv_order=3,
            diff_opt=2, km_opt=ph.km_opt, c_s=0.25, c_k=ph.c_k, mix_isotropic=ph.mix_isotropic,
            diff_6th_opt=2, diff_6th_factor=g.diff6, diff_6th_slopeopt=1, diff_6th_thresh=0.1,
            isfflx=0, tke_drag_coefficient=0.0 if self.own_drag else float(self.cd), tke_heat_flux=0.0,
            open_x=bc == "open", open_y=bc == "open",
            specified=bc == "specified", spec_bdy_width=5, spec_zone=1, relax_zone=4,
            output_interval_s=1.0e9, case="arhtmodelviewer",
        )
        if self.suite:
            sw = {k: v for k, v in self.suite.items() if k not in _SUITE_SKIP and hasattr(cfg, k)}
            if g.dx < 3000.0:
                sw["cu_physics"] = 0                   # convection-permitting: no cumulus scheme
                sw.pop("cudt_minutes", None)
            if sw.get("sf_sfclay_physics"):
                sw.update(isfflx=1, tke_drag_coefficient=0.0, tke_heat_flux=0.0)
            if ph.les:
                # no PBL scheme: the TKE closure mixes the column, fed by the surface layer's fluxes.  MYJ's Eta
                # surface layer needs the MYJ PBL's TKE (revised MM5 instead); the HRRR suite's aerosol PBL
                # mixing needs a PBL scheme
                sw.update(bl_pbl_physics=0, km_opt=2, diff_opt=2)
                if sw.get("sf_sfclay_physics") == 2:
                    sw["sf_sfclay_physics"] = 1
                if sw.get("scalar_pblmix"):
                    sw["scalar_pblmix"] = 0
            cfg = dataclasses.replace(cfg, **sw)
        if self.terrain is not None and bc == "specified":
            cfg = dataclasses.replace(cfg, terrain_opt=1)        # 3-D base state (the reference column stays flat)
        return validate_run_config(cfg)

    def surface_temperature(self) -> float:
        return float(self.theta_f(np.array([0.0]))[0] * (float(self.sounding.psfc) / P0) ** KAPPA)

    def needs_driver(self, cfg) -> bool:
        """gpuwm's persistent physics driver is needed by a suite's PBL / surface /
        radiation, by WDM6 and NSSL (they keep state in it), and by Thompson
        aerosol-aware and TEMPO (it initializes their aerosols)."""
        from gpuwm.core.physics_inventory import physics_enabled
        return bool(physics_enabled(cfg) or int(cfg.mp_physics) in (16, 18, 28, 900))

    def attach_physics(self, s, cfg, rect=None) -> None:
        """gpuwm's physics driver for a state: uniform land (the sounding's surface
        temperature, the configured soil and vegetation), radiation at the run's
        time and place."""
        from gpuwm.core.physics import initialize_physics
        from gpuwm.physics_compat import DECLARED_CONSTANT_GLW_WM2
        ph = self.physics
        ny, nx = cfg.ny, cfg.nx
        if rect is not None and self.latlon is not None:
            X, Y = self.ground_xy(rect)
            lat, lon = self.latlon(X, Y)
        else:
            lat, lon = np.full((ny, nx), ph.latitude), np.full((ny, nx), self.longitude)
        t0 = self.surface_temperature()
        kw = dict(landmask=1.0, tsk=t0, soil_temperature=t0 - 2.0, soil_moisture=ph.soil_moisture,
                  ivgtyp=ph.landuse, isltyp=ph.soil_type, vegfra=ph.vegetation, tmn=t0 - 6.0)
        if not cfg.ra_lw_physics:
            kw["glw"] = DECLARED_CONSTANT_GLW_WM2
        if cfg.ra_lw_physics or cfg.ra_sw_physics:
            import datetime as _dt
            start = self.start_time or _dt.datetime(2011, 4, 27, 18, tzinfo=_dt.timezone.utc)
            kw.update(radiation_start_time=start, radiation_latitude=np.asarray(lat, dtype=np.float64),
                      radiation_longitude=np.asarray(lon, dtype=np.float64))
            if cfg.sf_surface_physics == 4:
                kw.update(noahmp_start_time=start, noahmp_latitude=np.asarray(lat, dtype=np.float64),
                          noahmp_longitude=np.asarray(lon, dtype=np.float64))
        drv = initialize_physics(s, cfg, **kw)
        if getattr(s, "physics", None) is None:
            s.physics = drv

    def drag_coefficient(self, z0):
        """Neutral bulk drag coefficient of the lowest level over roughness z0 (m): (k / ln(z1 / z0))^2."""
        z1 = float(self.z_h[0])
        return (0.4 / np.log(np.maximum(z1 / np.maximum(z0, 1.0e-5), 1.5))) ** 2

    # ----------------------------------------------------------- states
    LBC_INTERVAL = 900.0      # s between boundary-table times when air masses move
    LBC_SPAN = 3600.0         # s of boundary tables attached at once (the engine renews them)

    def ground_xy(self, rect, sx: int = 0, sy: int = 0, t: float | None = None):
        """Ground coordinates (m) of a patch's mass points (sx = sy = 0) or of
        its u (sx = 1) / v (sy = 1) faces at time t (the grid may be moving)."""
        t = self.clock if t is None else t
        dx = self.grid.dx
        ox, oy = self.offset_at(t)
        x = (rect.i0 + np.arange(rect.nx + sx) + (0.0 if sx else 0.5)) * dx + ox
        y = (rect.j0 + np.arange(rect.ny + sy) + (0.0 if sy else 0.5)) * dx + oy
        return np.meshgrid(x, y)

    # ----------------------------------------------------------- grid motion
    def offset_at(self, t: float):
        """Ground position (m) of the grid origin at model time t."""
        seg = self._track[0]
        for s in self._track:
            if s[0] <= t:
                seg = s
        ts, x0, y0, mx, my = seg
        return x0 + mx * (t - ts), y0 + my * (t - ts)

    def set_motion(self, t: float, motion) -> tuple:
        """Move the grid with ``motion`` (m/s) from model time t on.  Returns the
        change, which every grid-relative wind must lose (a Galilean shift)."""
        x0, y0 = self.offset_at(t)
        new = (float(motion[0]), float(motion[1]))
        d = (new[0] - self.motion[0], new[1] - self.motion[1])
        self._track = [s for s in self._track if s[0] < t] + [(float(t), x0, y0, new[0], new[1])]
        self._apply_motion(new)
        return d

    def truncate_motion(self, t: float) -> tuple:
        """Forget grid-motion changes after model time t (a rollback); returns the
        change of motion back to the motion at t."""
        keep = [s for s in self._track if s[0] <= t] or self._track[:1]
        self._track = keep
        new = (keep[-1][3], keep[-1][4])
        d = (new[0] - self.motion[0], new[1] - self.motion[1])
        self._apply_motion(new)
        return d

    def _apply_motion(self, new) -> None:
        self.motion = new
        self.u_env = self.u_ground - new[0]
        self.v_env = self.v_ground - new[1]
        self.own_drag = bool(self.cd > 0)
        cache = self.__dict__.get("_dev_cache")
        if cache:
            cache.pop("u_env", None)
            cache.pop("v_env", None)
        self._bsnap = None                    # the boundary tables carry the grid-relative winds

    def _column_php(self, th, qv, mup, xp, base=None):
        """gpuwm's moist balanced column (init_moist_balanced) with dry-mass
        perturbation ``mup``: perturbation geopotential (nz+1, ...).  ``base``: a
        terrain base state of these columns (patch_base), else the flat one."""
        from gpuwm.core import constants as c
        co, b = self.coord, base if base is not None else self.base
        nz = self.grid.nz
        terrain = np.ndim(b.mub) > 0
        mub = xp.asarray(np.asarray(b.mub, dtype=np.float64)) if terrain else float(b.mub)
        c1f, c2f, c1h, c2h = co.c1f, co.c2f, co.c1h, co.c2h
        shape = (-1,) + (1,) * (th.ndim - 1)
        lv = lambda a: xp.asarray(np.asarray(a, dtype=np.float64)).reshape(shape)    # a level array, broadcast
        p = xp.empty_like(th)
        q1 = qv[nz - 1]
        q2 = 1.0 / (1.0 + q1)
        q1 = q1 * q2
        p[nz - 1] = -0.5 * (c1f[nz] * mup + q1 * (c1f[nz] * mub + c2f[nz])) / co.rdnw[nz - 1] / q2
        # downward from the top, p[kk] = p[kk + 1] - t[kk]: every level at once (a reversed running sum; the
        # level-by-level loop it replaces cost ~100 small GPU launches a column set, ~0.5 s a re-placed patch)
        qm = 0.5 * (qv[:-1] + qv[1:])
        q2 = 1.0 / (1.0 + qm)
        q1 = qm * q2
        c1w, c2w = lv(c1f[1:nz]), lv(c2f[1:nz])
        t = (c1w * mup + q1 * (c1w * mub + c2w)) / q2 / lv(co.rdn[1:nz])
        p[:nz - 1] = p[nz - 1] - xp.cumsum(t[::-1], axis=0)[::-1]
        if terrain:
            pb = xp.asarray(np.asarray(b.pb, dtype=np.float64))
            alb = xp.asarray(np.asarray(b.alb, dtype=np.float64))
        else:
            pb = xp.asarray(np.asarray(b.pb, dtype=np.float64)).reshape(shape)
            alb = np.asarray(b.alb, dtype=np.float64)
        alt = (c.RD / c.P0) * th * (1.0 + c.RVOVRD * qv) * ((p + pb) / c.P0) ** (-c.CV / c.CP)
        albv = alb if terrain else xp.asarray(alb).reshape(shape)
        al = alt - albv
        php = xp.zeros((nz + 1,) + th.shape[1:], dtype=th.dtype)
        # upward, php[k + 1] = php[k] - dnw[k] (...): a running sum over the levels
        ch, c2 = lv(c1h[:nz]), lv(c2h[:nz])
        u = lv(co.dnw[:nz]) * (((ch * mub + c2) + ch * mup) * al + (ch * mup) * albv)
        php[1:] = -xp.cumsum(u, axis=0)
        return php

    def balance(self, thp, qv, base=None):
        """Hydrostatic columns for theta' and qv on the eta levels (nz, ...):
        the dry-mass perturbation mu' that keeps the model-top geopotential at
        the environment's -- so pressure above an air mass is undisturbed and a
        cold air mass sits under a surface high -- and the perturbation
        geopotential.  ``base``: the columns' terrain base state (patch_base):
        their top then sits where the flat environment column's does.
        Returns (mup, php)."""
        from .boundaries import _xp
        xp = _xp(thp)
        if base is not None:
            thb = xp.asarray(np.asarray(base.thb, dtype=np.float64))
        else:
            thb = xp.asarray(np.asarray(self.base.thb, dtype=np.float64)).reshape((-1,) + (1,) * (thp.ndim - 1))
        th = thb + thp
        if getattr(self, "_php_top_env", None) is None:
            th0 = np.asarray(self.base.thb, dtype=np.float64)[:, None]
            q0 = self.qv_f(self.z_h)[:, None]
            self._php_top_env = float(self._column_php(th0, q0, np.zeros(1), np)[-1, 0])
        target = self._php_top_env
        if base is not None:
            target = target + float(np.asarray(self.base.phb)[-1]) - xp.asarray(np.asarray(base.phb[-1], np.float64))
        mup = xp.zeros(thp.shape[1:], dtype=xp.float64)
        r0 = self._column_php(th, qv, mup, xp, base)[-1] - target
        d = 100.0
        slope = (self._column_php(th, qv, mup + d, xp, base)[-1] - target - r0) / d
        mup = -r0 / slope
        for _ in range(2):
            php = self._column_php(th, qv, mup, xp, base)
            mup = mup - (php[-1] - target) / slope
        php = self._column_php(th, qv, mup, xp, base)
        return mup, php

    # ----------------------------------------------------------- terrain
    def patch_base(self, X, Y):
        """gpuwm terrain base state of columns at ground points (X, Y) (2-D)."""
        from gpuwm.core.grid import make_base_state
        h = self.terrain.height(X, Y)
        return make_base_state(self.coord, self.theta_f, p_surf=float(self.sounding.psfc), ztop=self.grid.ztop,
                               terrain_z=h)

    def face_heights(self, X, Y):
        """Heights (nz, ...) of the half levels at ground points over the terrain, above the model's zero: the
        sigma levels squeezed over the ground (for the winds at faces)."""
        h = self.terrain.height(X, Y)
        zt = self.grid.ztop
        return h[None] + self.z_h.reshape((-1,) + (1,) * h.ndim) * (1.0 - h[None] / zt)

    def _env_at(self, prof, z):
        """A level profile of the flat environment (on z_h) at heights z (any shape)."""
        return np.interp(np.asarray(z, np.float64), self.z_h, np.asarray(prof, np.float64))

    def terrain_columns(self, X, Y, t: float):
        """The environment (+ air masses at time t, + large-scale ascent) in columns at ground points over the
        terrain (host, float64): (base, thp, qv, mup, php).  The atmosphere is the sounding's at each height; the
        air masses' depths are above the column's own ground (their profiles on the sigma levels)."""
        from gpuwm.core.constants import G as G_
        base = self.patch_base(X, Y)
        phb = np.asarray(base.phb, np.float64)
        zh = 0.5 * (phb[:-1] + phb[1:]) / G_
        qv0 = getattr(self, "qv_env_h", None)
        qv = self._env_at(qv0, zh) if qv0 is not None else self.qv_f(zh)
        thp = np.zeros_like(zh)
        if self.airmass:
            m = self.airmass.maps(X, Y, t)
            thp_am, qfac, _, _ = self.airmass.fields_3d(m, self.z_h)
            thp = thp + thp_am
            qv = qv * qfac
            pk = self._pockets(X, Y)
            if pk is not None:
                thp = thp + pk[0]
                qv = qv * pk[1]
        dth = getattr(self, "ls_dth", None)
        if dth is not None and np.any(dth):              # large-scale ascent so far (largescale.py)
            thp = thp + np.asarray(dth)[:, None, None]
            qv = np.maximum(qv + np.asarray(self.ls_dqv)[:, None, None], 1e-7)
        mup, php = self.balance(thp, qv, base)
        return base, thp, qv, mup, php

    def terrain_wind(self, X, Y, t: float, which: str):
        """u or v (nz, ...) of the environment (+ air masses) at ground points (faces) over the terrain."""
        z = self.face_heights(X, Y)
        w = self._env_at(self.u_env if which == "u" else self.v_env, z)
        if self.airmass:
            w = w + self.airmass_wind(X, Y, t, which)
        return w

    TERRAIN_STRIDE = 2           # rim / keep targets of terrain patches every this many cells

    def terrain_targets(self, rect, t: float) -> dict:
        """Rim (and air-mass keep) targets of a patch over the terrain -- or on flat ground with sounding pockets,
        whose columns the air-mass maps cannot describe -- in the synoptic-target layout (forcing._apply_synoptic):
        u, v, theta', qv (nz, cy, cx) at every TERRAIN_STRIDE-th cell centre."""
        import cupy as cp
        s = self.TERRAIN_STRIDE
        cx, cy = max(1, -(-rect.nx // s)), max(1, -(-rect.ny // s))
        ox, oy = self.offset_at(t)
        dx = self.grid.dx
        x = (rect.i0 + (np.arange(cx) + 0.5) * s) * dx + ox
        y = (rect.j0 + (np.arange(cy) + 0.5) * s) * dx + oy
        X, Y = np.meshgrid(x, y)
        if self.terrain is not None:
            _, thp, qv, _, _ = self.terrain_columns(X, Y, t)
            out = {"thp": thp, "qv": qv, "u": self.terrain_wind(X, Y, t, "u"), "v": self.terrain_wind(X, Y, t, "v")}
        else:
            thp, qv, _, _ = self.airmass_columns(X, Y, t)
            out = {"thp": thp, "qv": qv, "u": self.u_env[:, None, None] + self.airmass_wind(X, Y, t, "u"),
                   "v": self.v_env[:, None, None] + self.airmass_wind(X, Y, t, "v")}
        maps = {k: cp.asarray(v, dtype=cp.float32) for k, v in out.items()}
        maps.update(cy=cy, cx=cx, stride=float(s), keep=bool(self.airmass))
        if self.airmass and not getattr(self.airmass, "synoptic", False):
            # the boundaries' keep as on flat ground: kept air masses, the backed winds ahead of them and
            # the frontal zones' hold (forcing.relax_target3d), not only the edge rim; and the sounding pockets'
            # gentle hold (pockets.keep_maps), deeper
            m = self.airmass.maps(X, Y, t)
            pk = getattr(self.airmass, "pockets", None)
            if pk is not None:
                from .pockets import keep_maps
                kt, kw, kh = keep_maps(pk, X, Y)
                for h, w, k in (("H", "nudge", kt), ("Hq", "nudge", kt), ("Hw", "wnudge", kw)):
                    m[h] = np.where(k > m[w], kh, m[h])
                m["nudge"] = np.maximum(m["nudge"], kt)
                m["wnudge"] = np.maximum(m["wnudge"], kw)
            if np.any(m["nudge"] > 0) or np.any(m["wnudge"] > 0):
                g = {k: cp.asarray(m[k], dtype=cp.float32)
                     for k in ("nudge", "wnudge", "H", "Hq", "Hw", "hold", "pin", "pin_lo")}
                pin = (g["pin"], g["pin_lo"]) if np.any(m["pin"] > 0) else (None, None)
                maps["am"] = {"thp": (g["nudge"], g["H"], g["hold"]) + pin,
                              "qv": (g["nudge"], g["Hq"], g["hold"], None, None),
                              "u": (g["wnudge"], g["Hw"], None, None, None),
                              "v": (g["wnudge"], g["Hw"], None, None, None)}
        return maps

    def _lifted(self, s) -> None:
        """A new state in the environment as large-scale ascent has changed it so far (largescale.py): its
        theta' and qv offsets, the column re-balanced (mu', phi')."""
        dth = getattr(self, "ls_dth", None)
        if dth is None or not np.any(dth):
            return
        import cupy as cp
        from gpuwm.core.diagnostics import update_diagnostics
        from gpuwm.core.state import DTYPE
        qv0 = np.asarray(getattr(self, "qv_env_h", self.qv_f(self.z_h)), dtype=np.float64)
        th = cp.asarray(np.asarray(dth, dtype=np.float64))[:, None]
        qv = cp.asarray(np.maximum(qv0 + self.ls_dqv, 1e-7))[:, None]
        mup, php = self.balance(th, qv)
        s.thp[...] = th[:, :, None].astype(DTYPE)
        s.qv[...] = qv[:, :, None].astype(DTYPE)
        s.php[...] = php[:, :, None].astype(DTYPE)
        s.mup[...] = DTYPE(float(mup[0]))
        update_diagnostics(s)

    def airmass_columns(self, X, Y, t: float, xp=np):
        """theta', qv (nz, ...), mu' and phi' of the air masses at ground points (X, Y)."""
        m = self.airmass.maps(X, Y, t)
        if xp is not np:
            m = self.airmass.to_device(m)
        thp, qfac, _, _ = self.airmass.fields_3d(m, self.z_h)
        qv_env = xp.asarray(self.qv_f(self.z_h)).reshape((-1,) + (1,) * (thp.ndim - 1))
        qv = qv_env * qfac
        pk = self._pockets(X, Y, xp)
        if pk is not None:
            thp = thp + pk[0]
            qv = qv * pk[1]
        dth = getattr(self, "ls_dth", None)
        if dth is not None and np.any(dth):              # large-scale ascent so far (largescale.py)
            shape = (-1,) + (1,) * (thp.ndim - 1)
            thp = thp + xp.asarray(dth).reshape(shape)
            qv = xp.maximum(qv + xp.asarray(self.ls_dqv).reshape(shape), 1e-7)
        mup, php = self.balance(thp, qv)
        return thp, qv, mup, php

    def airmass_wind(self, X, Y, t: float, which: str, xp=np):
        """Air-mass wind anomaly (du or dv, (nz, ...)) at ground points (X, Y)."""
        m = self.airmass.maps(X, Y, t)
        if xp is not np:
            m = self.airmass.to_device(m)
        _, _, du, dv = self.airmass.fields_3d(m, self.z_h)
        pk = self._pockets(X, Y, xp)
        if pk is not None:
            du, dv = du + pk[2], dv + pk[3]
        return du if which == "u" else dv

    def _pockets(self, X, Y, xp=np, levels=None):
        """The sounding pockets' (theta' added, vapour factor, du, dv) at ground points, or None."""
        pk = getattr(self.airmass, "pockets", None) if self.airmass else None
        return pk.fields(X, Y, xp, levels) if pk is not None else None

    def new_state(self, cfg, rect=None):
        """A gpuwm state for ``cfg``: the environment, plus the air masses of the
        run's boundaries when ``rect`` places it on the world (built with
        gpuwm's own balanced-column construction).  A specified-boundary config
        gets the matching lateral boundary forcing."""
        import cupy as cp
        from gpuwm.core.state import DTYPE

        am = self.airmass if rect is not None else None
        terrain = self.terrain is not None and rect is not None
        if terrain:
            from gpuwm.core.diagnostics import update_diagnostics
            from gpuwm.core.state import init_at_rest
            X, Y = self.ground_xy(rect)
            base, thp, qv, mup, php = self.terrain_columns(X, Y, self.clock)
            s = init_at_rest(cfg, self.coord, base)
            s.thp[...] = cp.asarray(thp, dtype=DTYPE)
            s.qv[...] = cp.asarray(qv, dtype=DTYPE)
            s.php[...] = cp.asarray(php, dtype=DTYPE)
            s.mup[...] = cp.asarray(mup, dtype=DTYPE)
            del base, thp, qv, php
            update_diagnostics(s)
            for name, sx, sy in (("u", 1, 0), ("v", 0, 1)):
                Xf, Yf = self.ground_xy(rect, sx, sy)
                getattr(s, name)[...] = cp.asarray(self.terrain_wind(Xf, Yf, self.clock, name), dtype=DTYPE)
        elif am:
            from gpuwm.core.diagnostics import update_diagnostics
            from gpuwm.core.state import init_at_rest
            s = init_at_rest(cfg, self.coord, self.base)
            X, Y = self.ground_xy(rect)
            thp, qv, mup, php = self.airmass_columns(X, Y, self.clock, xp=cp)
            s.thp[...] = thp.astype(DTYPE)
            s.qv[...] = qv.astype(DTYPE)
            s.php[...] = php.astype(DTYPE)
            s.mup[...] = mup.astype(DTYPE)
            del thp, qv, php
            update_diagnostics(s)
            for name, sx, sy, env in (("u", 1, 0, self.u_env), ("v", 0, 1, self.v_env)):
                Xf, Yf = self.ground_xy(rect, sx, sy)
                d = self.airmass_wind(Xf, Yf, self.clock, name, xp=cp)
                getattr(s, name)[...] = (cp.asarray(env)[:, None, None] + d).astype(DTYPE)
        else:
            from gpuwm.core.moist import init_moist_balanced
            s = init_moist_balanced(cfg, self.coord, self.base, self.qv_f, None)
            s.u[...] = cp.asarray(self.u_env, dtype=DTYPE)[:, None, None]
            s.v[...] = cp.asarray(self.v_env, dtype=DTYPE)[:, None, None]
            self._lifted(s)
        tk = getattr(self, "tke_profile", None)          # the spun-up environment's turbulence (Engine.spin_up)
        if tk is not None and getattr(s, "tke", None) is not None and s.tke.shape[0] == len(tk):
            s.tke[...] = cp.asarray(tk, dtype=s.tke.dtype)[:, None, None]
        if self.f != 0.0:
            s.set_map_coriolis(f=np.full((cfg.ny, cfg.nx), self.f),
                               e=np.zeros((cfg.ny, cfg.nx)))
        if self.needs_driver(cfg):
            self.attach_physics(s, cfg, rect)
            tpl = getattr(self, "driver_template", None)
            if tpl and rect is not None and getattr(s, "physics", None) is not None:
                # the land surface the suite's reference column has had so far (Engine._follow_reference)
                from .patch import driver_lookup
                for path, a in tpl.items():
                    d = driver_lookup(s, path)
                    if d is not None and d.dtype == a.dtype and d.shape[:-2] == a.shape[:-2]:
                        d[...] = a
        if cfg.specified:
            s.atmo_lbc_end = self.attach_boundaries(s, cfg, rect if (am or terrain) else None)
        return s

    def attach_boundaries(self, state, cfg, rect=None, t0: float | None = None) -> float:
        """(Re)attach lateral boundary forcing to a patch state, keeping its
        model time; returns the time the tables run out (inf if never)."""
        from gpuwm.ingest.lateral_bc import attach_lateral_boundaries
        elapsed = float(getattr(state, "elapsed_seconds", 0.0) or 0.0)
        lbc = self.lateral_boundaries(cfg.nx, cfg.ny, rect=rect, t0=t0)
        attach_lateral_boundaries(state, lbc)
        state.elapsed_seconds = elapsed
        end = lbc.intervals[-1].end_seconds
        return math.inf if end >= 1.0e8 else end

    def _boundary_snapshot(self):
        """Coupled boundary fields of one undisturbed column (host, float64)."""
        if getattr(self, "_bsnap", None) is None:
            from gpuwm.ingest.lateral_bc import domain_boundary_snapshot
            n = 16
            cfg = self.run_config(n, n, bc="periodic")
            small = self.new_state(cfg)
            snap = {}
            for name, a in domain_boundary_snapshot(small).items():
                a = np.asarray(a, dtype=np.float64)
                if a.ndim == 2:
                    a = a[None]
                sy, sx = a.shape[-2] - n, a.shape[-1] - n
                snap[name] = (a[:, n // 2, n // 2].copy(), sy, sx)
            # the boundary scalar inventory of a specified patch
            pcfg = self.run_config(32, 32, bc="specified")
            from gpuwm.boundary_fields import external_scalar_fields
            want = set(snap) - {"qv"} | set(external_scalar_fields(pcfg))
            missing = want - set(snap)
            if missing:
                raise RuntimeError(f"no environment boundary values for {sorted(missing)}")
            self._bsnap = {k: v for k, v in snap.items() if k in want}
            del small
        return self._bsnap

    def lateral_boundaries(self, nx: int, ny: int, width: int = 5, rect=None, t0: float | None = None):
        """WRF specified/relaxation forcing.  Without air masses (or without a
        place on the world) it is the environment, constant in time; with them
        it is the moving air masses along the patch edges, linear in time
        between tables every LBC_INTERVAL s from ``t0`` (default: now) for LBC_SPAN s."""
        from gpuwm.ingest.lateral_bc import (LateralBoundaries,
                                             build_lateral_interval_from_sides)
        if self.terrain is not None and rect is not None and not self.airmass:
            sides = self.airmass_sides(rect, width, self.clock)     # each column over its own ground
            interval = build_lateral_interval_from_sides(sides, sides, start_seconds=0.0, end_seconds=1.0e9)
            return LateralBoundaries((interval,), width, 1, 4)
        if not (self.airmass and rect is not None):
            sides = {"west": {}, "east": {}, "south": {}, "north": {}}
            for name, (col, sy, sx) in self._boundary_snapshot().items():
                nk = col.shape[0]
                c = col[:, None, None]
                we = np.ascontiguousarray(np.broadcast_to(c, (nk, ny + sy, width)))
                sn = np.ascontiguousarray(np.broadcast_to(c, (nk, width, nx + sx)))
                sides["west"][name] = we
                sides["east"][name] = we
                sides["south"][name] = sn
                sides["north"][name] = sn
            interval = build_lateral_interval_from_sides(sides, sides, start_seconds=0.0,
                                                         end_seconds=1.0e9)
            return LateralBoundaries((interval,), width, 1, 4)
        t0 = self.clock if t0 is None else float(t0)
        n = max(1, int(round(self.LBC_SPAN / self.LBC_INTERVAL)))
        times = [t0 + k * self.LBC_INTERVAL for k in range(n + 1)]
        frames = [self.airmass_sides(rect, width, t) for t in times]
        intervals = tuple(build_lateral_interval_from_sides(
            frames[k], frames[k + 1], start_seconds=times[k], end_seconds=times[k + 1]) for k in range(n))
        return LateralBoundaries(intervals, width, 1, 4)

    def airmass_sides(self, rect, width: int, t: float) -> dict:
        """W/E/S/N boundary strips (coupled, like domain_boundary_snapshot +
        extract_lateral_side) of the environment + air masses at time t."""
        from gpuwm.ingest.lateral_bc import extract_lateral_side
        NX, NY = rect.nx, rect.ny
        w = width
        regions = {"west": (0, w, 0, NY), "east": (NX - w, NX, 0, NY),
                   "south": (0, NX, 0, w), "north": (0, NX, NY - w, NY)}
        return {side: extract_lateral_side(self._coupled_region(rect, *reg, t), side, w)
                for side, reg in regions.items()}

    def _coupled_region(self, rect, ia: int, ib: int, ja: int, jb: int, t: float) -> dict:
        """Coupled boundary fields (gpuwm.ingest.lateral_bc._coupled_device_fields)
        of mass cells [ja:jb, ia:ib] of a patch at ``rect`` with the air masses
        at time t, with u faces ia..ib and v faces ja..jb."""
        from .patch import Rect
        snap = self._boundary_snapshot()
        co = self.coord
        NX, NY = rect.nx, rect.ny
        sub = Rect(rect.i0 + ia, rect.j0 + ja, ib - ia, jb - ja)
        X, Y = self.ground_xy(sub, t=t)
        if self.terrain is not None:
            base, thp, qv, mup, php = self.terrain_columns(X, Y, t)
            mu = np.asarray(base.mub, np.float64) + mup
            thb = np.asarray(base.thb, np.float64)
        else:
            thp, qv, mup, php = self.airmass_columns(X, Y, t)
            mu = float(self.base.mub) + mup                           # (nj, ni)
            thb = np.asarray(self.base.thb, dtype=np.float64)[:, None, None]
        c1h, c2h = co.c1h[:, None, None], co.c2h[:, None, None]
        c1f, c2f = co.c1f[:, None, None], co.c2f[:, None, None]
        chm = c1h * mu[None] + c2h
        chf = c1f * mu[None] + c2f
        # mass at faces: mean of the two cells, the boundary cell's own at the patch edges
        mux = np.empty((mu.shape[0], mu.shape[1] + 1))
        mux[:, 1:-1] = 0.5 * (mu[:, 1:] + mu[:, :-1])
        mux[:, 0] = mu[:, 0]
        mux[:, -1] = mu[:, -1]
        muy = np.empty((mu.shape[0] + 1, mu.shape[1]))
        muy[1:-1] = 0.5 * (mu[1:] + mu[:-1])
        muy[0] = mu[0]
        muy[-1] = mu[-1]
        Xu, Yu = self.ground_xy(sub, 1, 0, t=t)
        Xv, Yv = self.ground_xy(sub, 0, 1, t=t)
        if self.terrain is not None:
            u = self.terrain_wind(Xu, Yu, t, "u")
            v = self.terrain_wind(Xv, Yv, t, "v")
        else:
            u = self.u_env[:, None, None] + self.airmass_wind(Xu, Yu, t, "u")
            v = self.v_env[:, None, None] + self.airmass_wind(Xv, Yv, t, "v")
        out = {
            "u": (c1h * mux[None] + c2h) * u,
            "v": (c1h * muy[None] + c2h) * v,
            "theta": chm * (thb + thp - 300.0),
            "phi": chf * php,
            "mu": mup[None],
            "qv": chm * qv,
        }
        result = {}
        for name, (col, sy, sx) in snap.items():
            if name in out:
                result[name] = out[name]
            else:                       # anything else (hydrometeors...): the environment's
                shape = (jb - ja + sy, ib - ia + sx)
                result[name] = np.broadcast_to(col[:, None, None], (col.shape[0],) + shape)
        return result

    def describe(self) -> dict:
        g = self.grid
        return {
            "dx": g.dx, "nz": g.nz, "ztop": g.ztop, "dt_max": g.dt,
            "z_w": [round(float(z), 1) for z in self.z_w],
            "lowest_level_m": round(float(self.z_h[0]), 1),
            "cd": round(self.cd, 5), "f": self.f,
            "physics": dataclasses.asdict(self.physics),
            "microphysics": MICROPHYSICS.get(int(self.physics.mp_physics), str(self.physics.mp_physics)),
        }
