"""Soundings: parsing (SHARPpy %RAW%, CM1/WRF input_sounding, Wyoming text),
the analytic Weisman-Klemp sounding, the model profile functions and the
usual severe-weather parameters (CAPE/CIN, SRH, shear, Bunkers motion, STP,
SCP).

Thermodynamic constants are WRF's (Rd = 287, Cp = 7/2 Rd, Rv = 461.6,
g = 9.81, p0 = 1000 hPa), the ones ArWen's dycore uses, so a sounding means
the same thing to the parser, the base state and the diagnostics.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

RD = 287.0
RV = 461.6
CP = 7.0 * RD / 2.0
G = 9.81
P0 = 1.0e5
EPS = RD / RV
KAPPA = RD / CP
LV = 2.501e6
KT = 0.514444


def es_water(T):
    """Saturation vapour pressure over water (Pa), Bolton (1980)."""
    T = np.asarray(T, dtype=np.float64)
    return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


def qv_sat(p, T):
    e = np.minimum(es_water(T), 0.5 * np.asarray(p, dtype=np.float64))
    return EPS * e / (p - e)


def es_ice(T):
    """Saturation vapour pressure over ice (Pa), Murphy & Koop (2005)."""
    T = np.asarray(T, dtype=np.float64)
    return np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)


def qv_sat_ice(p, T):
    e = np.minimum(es_ice(T), 0.5 * np.asarray(p, dtype=np.float64))
    return EPS * e / (p - e)


def dewpoint_from_qv(p, qv):
    e = np.maximum(np.asarray(qv) * p / (EPS + np.asarray(qv)), 1e-3)
    lnr = np.log(e / 611.2)
    return 273.15 + 243.5 * lnr / (17.67 - lnr)


def theta_from(p, T):
    return T * (P0 / p) ** KAPPA


def thetae_bolton(p, T, qv):
    """Equivalent potential temperature (Bolton 1980 eq. 43)."""
    r = np.maximum(qv, 1e-10)
    e = r * p / (EPS + r)
    tl = 2840.0 / (3.5 * np.log(T) - np.log(e / 100.0) - 4.805) + 55.0
    return (T * (P0 / p) ** (0.2854 * (1.0 - 0.28 * r))
            * np.exp((3376.0 / tl - 2.54) * r * (1.0 + 0.81 * r)))


@dataclass
class Sounding:
    """A column on height (m MSL) with p (Pa), T (K), qv (kg/kg), u, v (m/s)."""
    z: np.ndarray
    p: np.ndarray
    T: np.ndarray
    qv: np.ndarray
    u: np.ndarray
    v: np.ndarray
    title: str = "sounding"
    source: str = ""

    # ------------------------------------------------------------ basics
    @property
    def zsfc(self) -> float:
        return float(self.z[0])

    @property
    def psfc(self) -> float:
        return float(self.p[0])

    @property
    def theta(self) -> np.ndarray:
        return theta_from(self.p, self.T)

    @property
    def Td(self) -> np.ndarray:
        return dewpoint_from_qv(self.p, self.qv)

    def agl(self) -> np.ndarray:
        return self.z - self.z[0]

    def cleaned(self) -> "Sounding":
        """Sorted, de-duplicated, finite, supersaturation-free copy."""
        z, idx = np.unique(self.z, return_index=True)
        keep = np.isfinite(self.p[idx]) & np.isfinite(self.T[idx]) & np.isfinite(self.qv[idx])
        idx = idx[keep]
        z = self.z[idx]
        p, T = self.p[idx], self.T[idx]
        qv = np.clip(self.qv[idx], 1e-7, None)
        qv = np.minimum(qv, qv_sat(p, T))
        u = np.nan_to_num(self.u[idx])
        v = np.nan_to_num(self.v[idx])
        return Sounding(z, p, T, qv, u, v, self.title, self.source)

    # ---------------------------------------------- model profile functions
    def profile_functions(self, ztop: float = 30000.0):
        """theta(z), qv(z), u(z), v(z) on height AGL, extended to ``ztop``.

        Above the sounding's top the temperature is held isothermal (theta
        grows as exp(g dz / (cp T)), the WK82 stratosphere), vapour decays
        and the wind is held constant.
        """
        s = self.cleaned()
        zagl = s.agl()
        th = s.theta
        ztop_s = float(zagl[-1])
        t_top = float(s.T[-1])
        th_top = float(th[-1])

        def theta(z):
            z = np.asarray(z, dtype=np.float64)
            out = np.interp(z, zagl, th)
            above = z > ztop_s
            if np.any(above):
                out = np.where(above, th_top * np.exp(G * (z - ztop_s) / (CP * t_top)), out)
            return out

        qtop = float(s.qv[-1])

        def qv(z):
            z = np.asarray(z, dtype=np.float64)
            out = np.interp(z, zagl, s.qv)
            above = z > ztop_s
            if np.any(above):
                out = np.where(above, np.maximum(qtop * np.exp(-(z - ztop_s) / 2000.0), 1e-7), out)
            return out

        def uf(z):
            return np.interp(np.asarray(z, dtype=np.float64), zagl, s.u)

        def vf(z):
            return np.interp(np.asarray(z, dtype=np.float64), zagl, s.v)

        return theta, qv, uf, vf

    # ---------------------------------------------------------- transforms
    def mixed(self, depth_hpa: float = 100.0, heat_k: float = 0.0) -> "Sounding":
        """Afternoon boundary layer: the lowest ``depth_hpa`` mixed to constant
        theta (its pressure-weighted mean, plus ``heat_k`` of surface heating)
        and constant mixing ratio; heights re-integrated hydrostatically.
        The layer joins the air above it; nothing is cooled."""
        s = self.cleaned()
        ptop = s.p[0] - depth_hpa * 100.0
        sel = s.p >= ptop
        if sel.sum() < 2:
            return s
        th = s.theta
        w = np.gradient(-s.p)[sel]
        th_ml = float(np.sum(th[sel] * w) / np.sum(w)) + heat_k
        q_ml = float(np.sum(s.qv[sel] * w) / np.sum(w))
        th_new = np.where(sel, th_ml, th)
        T = th_new * (s.p / P0) ** KAPPA
        qv = np.where(sel, q_ml, s.qv)
        qv = np.minimum(qv, qv_sat(s.p, T))
        z = _hydrostatic_heights(s.p, T, qv, s.z[0])
        title = f"{s.title} (mixed {depth_hpa:.0f} hPa" + (f", +{heat_k:.1f} K" if heat_k else "") + ")"
        return Sounding(z, s.p, T, qv, s.u, s.v, title, s.source).cleaned()

    def surface_heated(self, flux: float = 300.0, hours: float = 3.0, bowen: float = 0.6) -> "Sounding":
        """The afternoon's heating: ``hours`` of ``flux`` W/m2 of surface sensible heat (and flux / ``bowen``
        of latent heat) put into the boundary layer, which grows by encroachment -- mixed up to the height h
        where the heat that raises the layer to theta(h) equals what the ground gave, its mixing ratio the
        layer's mean plus the evaporated water.  A morning or evening cap is worn down from below as it is on a
        sunny afternoon; a steady run (steady environment) has no surface heating of its own, so a capped
        sounding stayed capped all run and its storms barely lifted their inflow.  Heights re-integrated."""
        s = self.cleaned()
        e_h = max(float(flux), 0.0) * max(float(hours), 0.0) * 3600.0
        if e_h <= 0.0:
            return s
        agl = s.agl()
        top = min(float(agl[-1]), 6000.0)
        zf = np.arange(0.0, top, 10.0)
        p = np.exp(np.interp(zf, agl, np.log(s.p)))
        T = np.interp(zf, agl, s.T)
        qv = np.exp(np.interp(zf, agl, np.log(s.qv)))
        th = theta_from(p, T)
        rho = p / (RD * T * (1.0 + 0.608 * qv))
        mass = rho * 10.0                                   # kg/m2 per 10 m layer
        thmax = np.maximum.accumulate(th)
        need = CP * np.cumsum(mass * 0.0)                   # placeholder shape
        # heat to raise everything below level k to thmax[k]
        need = CP * (thmax * np.cumsum(mass) - np.cumsum(mass * th))
        k = int(np.searchsorted(need, e_h))
        if k >= len(zf) - 1:
            k = len(zf) - 2
        h = float(zf[k])
        th_ml = float(thmax[k])
        m_ml = float(np.sum(mass[:k + 1]))
        q_ml = float(np.sum(mass[:k + 1] * qv[:k + 1]) / m_ml)
        if bowen and bowen > 0.0:
            q_ml += e_h / float(bowen) / (2.5e6 * m_ml)
        # the mixed layer on the original levels below h, plus its top, with the sounding above unchanged
        za = np.union1d(agl[agl > h + 1.0], np.append(np.arange(0.0, h, 50.0), h))
        pa = np.exp(np.interp(za, agl, np.log(s.p)))
        tha = np.where(za <= h, th_ml, theta_from(pa, np.interp(za, agl, s.T)))
        qa = np.where(za <= h, q_ml, np.exp(np.interp(za, agl, np.log(s.qv))))
        Ta = tha * (pa / P0) ** KAPPA
        qa = np.minimum(qa, qv_sat(pa, Ta))
        z = _hydrostatic_heights(pa, Ta, qa, s.z[0])
        u, v = np.interp(za, agl, s.u), np.interp(za, agl, s.v)
        title = f"{s.title} ({hours:g} h of {flux:g} W/m2 surface heating)"
        return Sounding(z, pa, Ta, qa, u, v, title, s.source).cleaned()

    def moistened(self, dtd: float, depth_hpa: float = 100.0) -> "Sounding":
        """More low-level moisture: the dewpoint of the lowest ``depth_hpa`` raised by ``dtd`` (C),
        in full up to 70 % of the depth and tapering to nothing at its top, never past
        saturation.  Moister inflow: a lower cloud base, more CAPE and less evaporation below
        cloud, so weaker cold pools.  Temperatures and heights are kept."""
        s = self.cleaned()
        frac = (s.p[0] - s.p) / (depth_hpa * 100.0)
        w = np.clip((1.0 - frac) / 0.3, 0.0, 1.0)
        e = es_water(s.Td + dtd * w)
        qv = np.where(w > 0.0, np.maximum(s.qv, EPS * e / (s.p - e)), s.qv)
        title = f"{s.title} (dewpoint +{dtd:g} C in the lowest {depth_hpa:g} hPa)"
        return Sounding(s.z, s.p, s.T, qv, s.u, s.v, title, s.source).cleaned()

    def capped(self, cin: float, base_hpa: float = 100.0) -> "Sounding":
        """A cap: an elevated mixed layer based ``base_hpa`` above the surface, just warm enough
        that the mixed-layer parcel meets ``cin`` J/kg of CIN.  Theta jumps by A over the
        lowest 10 hPa (the inversion), then climbs only 1 K/km (a steep, near dry-adiabatic
        lapse rate) until it meets the sounding's own profile, so a stronger cap is also a
        deeper one.  Storms then go up only where the boundary layer is lifted hard (a storm
        start, a boundary, a strong gust front) instead of off every outflow.  Mixing ratios
        are kept (the warm layer is dry, like a real EML); heights re-integrated.  A sounding
        already capped that much comes back unchanged."""
        s = self.cleaned()
        target = -abs(float(cin))
        if target >= 0.0 or _ml_parcel(*_fine(s)[:4])["cin"] <= target:
            return s
        agl = s.agl()
        pb = s.p[0] - base_hpa * 100.0
        zb = float(np.interp(-pb, -s.p, agl))
        za = np.union1d(agl, np.arange(50.0, min(zb + 6000.0, float(agl[-1])), 50.0))
        p = np.exp(np.interp(za, agl, np.log(s.p)))
        T = np.interp(za, agl, s.T)
        qv = np.exp(np.interp(za, agl, np.log(s.qv)))
        u = np.interp(za, agl, s.u)
        v = np.interp(za, agl, s.v)
        th = theta_from(p, T)
        thb = float(np.interp(zb, za, th))
        zi = max(float(np.interp(-(pb - 1000.0), -p, za)) - zb, 50.0)     # the inversion's depth

        def warmed(a: float) -> "Sounding":
            dz = za - zb
            cap = thb + a * np.clip(dz / zi, 0.0, 1.0) + 0.001 * np.maximum(dz - zi, 0.0)
            T2 = np.where(dz > 0.0, np.maximum(th, cap), th) * (p / P0) ** KAPPA
            z = _hydrostatic_heights(p, T2, qv, s.z[0])
            return Sounding(z, p, T2, qv, u, v, s.title, s.source)

        def cin_of(a: float) -> float:
            return _ml_parcel(*_fine(warmed(a))[:4])["cin"]

        lo, hi = 0.0, 1.0
        while cin_of(hi) > target and hi < 20.0:
            lo, hi = hi, hi * 2.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if cin_of(mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.01:
                break
        out = warmed(hi)
        out.title = f"{s.title} (capped: MLCIN {target:g} J/kg, +{hi:.1f} K at {pb / 100.0:.0f} hPa)"
        return out.cleaned()

    def ice_capped(self) -> "Sounding":
        """Water vapour below freezing limited to ice saturation.  Model soundings
        through cirrus or anvil debris are supersaturated over ice aloft (up to
        180 % on 27 April 2011); spread over the whole model that air nucleates ice
        and snows out everywhere at once, burying the storms in domain-wide echo.
        Storms still make their own anvils.  Heights are kept (the vapour removed
        aloft moves them by centimetres; re-integrating would shift the hodograph);
        the title says so only past 105 % (Morrison nucleates ice from 108 %)."""
        s = self.cleaned()
        qsi = qv_sat_ice(s.p, s.T)
        cold = s.T < 273.15
        if not (cold & (s.qv > qsi)).any():
            return s
        title = s.title + (" (dried to ice saturation aloft)" if (cold & (s.qv > 1.05 * qsi)).any() else "")
        qv = np.where(cold, np.minimum(s.qv, qsi), s.qv)
        return Sounding(s.z, s.p, s.T, qv, s.u, s.v, title, s.source)

    def with_low_level_jet(self, speed: float, height: float = 750.0, depth: float = 1500.0) -> "Sounding":
        """A low-level jet added to the wind: ``speed`` (m/s) along the wind's own direction at
        ``height`` (m AGL), zero at the ground (the surface wind is kept), full at the jet height
        and fading to zero ``depth`` above it.  Strengthens the near-ground shear and helicity,
        like the evening jet that the 00Z OUN sounding of 3 May 1999 predates.  Levels are filled
        in every 50 m through the jet so its shape survives the model's interpolation."""
        s = self.cleaned()
        agl = s.agl()
        top = height + depth
        za = np.union1d(agl, np.arange(50.0, top + 1.0, 50.0))
        p = np.exp(np.interp(za, agl, np.log(s.p)))
        T = np.interp(za, agl, s.T)
        qv = np.exp(np.interp(za, agl, np.log(s.qv)))
        u = np.interp(za, agl, s.u)
        v = np.interp(za, agl, s.v)
        # along the wind at the jet height (else the 0-1 km mean wind, else from the south)
        uj, vj = float(np.interp(height, za, u)), float(np.interp(height, za, v))
        if math.hypot(uj, vj) < 1.0:
            uj, vj = float(u[za <= 1000.0].mean()), float(v[za <= 1000.0].mean())
        if math.hypot(uj, vj) < 1.0:
            uj, vj = 0.0, 1.0
        m = math.hypot(uj, vj)
        shape = np.where(za <= height, np.sin(0.5 * np.pi * za / height),
                         np.where(za < top, np.cos(0.5 * np.pi * (za - height) / depth), 0.0))
        u = u + speed * shape * uj / m
        v = v + speed * shape * vj / m
        title = f"{s.title} (low-level jet +{speed:g} m/s at {height:g} m)"
        return Sounding(za + s.z[0], p, T, qv, u, v, title, s.source).cleaned()

    def mid_moistened(self, rh_pct: float, bottom_hpa: float = 750.0, top_hpa: float = 450.0) -> "Sounding":
        """Moister mid levels: relative humidity raised to at least ``rh_pct`` between ``bottom_hpa``
        and ``top_hpa``, tapering to nothing 50 hPa beyond each edge; moister air is kept.  Dry
        mid-level air is what rain and hail evaporate into on the way down: it drives the downdraft
        CAPE and the cold pools that undercut a storm's inflow (May 20 2019: 45 % at 700-500 hPa,
        DCAPE 1280 J/kg; 27 April 2011: 76 %, 625).  Temperatures and heights are kept."""
        s = self.cleaned()
        ph = s.p / 100.0
        w = np.clip((bottom_hpa - ph) / 50.0 + 1.0, 0.0, 1.0) * np.clip((ph - top_hpa) / 50.0 + 1.0, 0.0, 1.0)
        qv = s.qv + w * np.maximum(rh_pct / 100.0 * qv_sat(s.p, s.T) - s.qv, 0.0)
        title = f"{s.title} (RH ≥ {rh_pct:g} % at {bottom_hpa:g}-{top_hpa:g} hPa)"
        return Sounding(s.z, s.p, s.T, qv, s.u, s.v, title, s.source).cleaned()

    def with_winds_aloft(self, speed: float, base: float = 5000.0, full: float = 9000.0) -> "Sounding":
        """Stronger winds aloft: ``speed`` (m/s) added along the mean wind direction between ``base`` and
        ``full`` m AGL, nothing below ``base``, rising (sin-shaped) to full at ``full``, held for 3 km
        and fading out over the 4 km above.  Upper-level storm-relative wind sets the supercell type:
        weak (< ~18 m/s at 9-11 km) keeps the rain wrapped around the updraft (HP: cold, wet rear-flank
        outflow); stronger carries it downstream (classic).  May 20 2019: 11 m/s; 27 April 2011: 20."""
        s = self.cleaned()
        agl = s.agl()
        lay = (agl >= base) & (agl <= full)
        um, vm = (float(s.u[lay].mean()), float(s.v[lay].mean())) if lay.any() else (0.0, 0.0)
        m = math.hypot(um, vm)
        if m < 1.0:
            um, vm, m = 1.0, 0.0, 1.0           # no wind to go by: from the west
        top = full + 3000.0
        shape = np.where(agl <= base, 0.0, np.where(agl < full, np.sin(0.5 * np.pi * (agl - base) / (full - base)),
                         np.where(agl <= top, 1.0, np.where(agl < top + 4000.0, np.cos(0.5 * np.pi * (agl - top) / 4000.0), 0.0))))
        title = f"{s.title} (winds aloft +{speed:g} m/s above {base / 1000:g} km)"
        return Sounding(s.z, s.p, s.T, s.qv, s.u + speed * shape * um / m, s.v + speed * shape * vm / m,
                        title, s.source)

    def with_wind_scale(self, factor: float) -> "Sounding":
        return Sounding(self.z, self.p, self.T, self.qv, self.u * factor, self.v * factor,
                        f"{self.title} (winds ×{factor:g})", self.source)

    def to_sharppy(self) -> str:
        wdir, wspd = uv_to_dirspd(self.u, self.v)
        lines = ["%TITLE%", " " + self.title, "", "   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD",
                 "-------------------------------------------------------------------", "%RAW%"]
        for p, z, T, Td, d, s in zip(self.p, self.z, self.T, self.Td, wdir, wspd):
            lines.append("%9.2f, %9.2f, %9.2f, %9.2f, %9.2f, %9.2f"
                         % (p / 100.0, z, T - 273.15, Td - 273.15, d, s / KT))
        lines.append("%END%")
        return "\n".join(lines) + "\n"

    def to_json(self, step: int = 1) -> dict:
        wdir, wspd = uv_to_dirspd(self.u, self.v)
        return {
            "title": self.title,
            "p": (self.p[::step] / 100.0).round(2).tolist(),
            "z": self.z[::step].round(1).tolist(),
            "T": (self.T[::step] - 273.15).round(2).tolist(),
            "Td": (self.Td[::step] - 273.15).round(2).tolist(),
            "u": self.u[::step].round(2).tolist(),
            "v": self.v[::step].round(2).tolist(),
        }


def uv_to_dirspd(u, v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    spd = np.hypot(u, v)
    d = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return d, spd


def dirspd_to_uv(d, spd):
    d = np.radians(np.asarray(d, dtype=np.float64))
    spd = np.asarray(spd, dtype=np.float64)
    return -spd * np.sin(d), -spd * np.cos(d)


# ------------------------------------------------------------------ parsers
def _hydrostatic_heights(p, T, qv, z0):
    """Heights (m) from the hypsometric equation, starting at z0."""
    tv = T * (1.0 + qv / EPS) / (1.0 + qv)
    z = np.empty_like(p)
    z[0] = z0
    for k in range(1, len(p)):
        tvm = 0.5 * (tv[k] + tv[k - 1])
        z[k] = z[k - 1] + RD * tvm / G * math.log(p[k - 1] / p[k])
    return z


def parse_sharppy(text: str, title: str | None = None) -> Sounding:
    """SHARPpy / SPC %RAW% text: pres, hght, temp, dwpt, wdir, wspd(kt)."""
    m = re.search(r"%RAW%(.*?)(%END%|$)", text, re.S)
    body = m.group(1) if m else text
    rows = []
    for line in body.splitlines():
        parts = [s.strip() for s in line.replace("\t", ",").split(",")]
        if len(parts) < 6:
            parts = line.split()
        if len(parts) < 6:
            continue
        try:
            vals = [float(x) for x in parts[:6]]
        except ValueError:
            continue
        rows.append(vals)
    if len(rows) < 3:
        raise ValueError("no %RAW% rows (pres, hght, temp, dwpt, wdir, wspd) found")
    a = np.array(rows, dtype=np.float64)
    a = a[np.argsort(-a[:, 0])]
    good = (a[:, 0] > 0) & (a[:, 2] > -200) & (a[:, 1] > -1000)
    a = a[good]
    p = a[:, 0] * 100.0
    T = a[:, 2] + 273.15
    td = np.where(a[:, 3] > -200, a[:, 3], a[:, 2] - 30.0) + 273.15
    qv = EPS * es_water(td) / (p - es_water(td))
    wd = np.where(a[:, 4] > -1000, a[:, 4], np.nan)
    ws = np.where(a[:, 5] > -1000, a[:, 5], np.nan) * KT
    u, v = dirspd_to_uv(wd, ws)
    okw = np.isfinite(u)
    if not okw.all():
        u = np.interp(a[:, 1], a[okw, 1], u[okw])
        v = np.interp(a[:, 1], a[okw, 1], v[okw])
    z = a[:, 1]
    if title is None:
        tm = re.search(r"%TITLE%\s*\n\s*(.+)", text)
        title = tm.group(1).strip() if tm else "SHARPpy sounding"
    return Sounding(z, p, T, qv, u, v, title, "sharppy").cleaned()


def parse_input_sounding(text: str, title: str = "input_sounding") -> Sounding:
    """CM1 / WRF idealized input_sounding: 'psfc(hPa) thsfc(K) qvsfc(g/kg)'
    then 'z(m AGL) theta(K) qv(g/kg) u v' per line."""
    lines = [ln.split() for ln in text.strip().splitlines() if ln.strip()]
    psfc, thsfc, qsfc = (float(x) for x in lines[0][:3])
    rows = [[float(x) for x in ln[:5]] for ln in lines[1:] if len(ln) >= 5]
    a = np.array(rows, dtype=np.float64)
    z = np.concatenate([[0.0], a[:, 0]])
    th = np.concatenate([[thsfc], a[:, 1]])
    qv = np.concatenate([[qsfc], a[:, 2]]) / 1000.0
    u = np.concatenate([[a[0, 3]], a[:, 3]])
    v = np.concatenate([[a[0, 4]], a[:, 4]])
    order = np.argsort(z)
    z, th, qv, u, v = z[order], th[order], qv[order], u[order], v[order]
    z, idx = np.unique(z, return_index=True)
    th, qv, u, v = th[idx], qv[idx], u[idx], v[idx]
    # hydrostatic Exner integration with theta_v
    thv = th * (1.0 + qv / EPS) / (1.0 + qv)
    pi = np.empty_like(z)
    pi[0] = (psfc * 100.0 / P0) ** KAPPA
    for k in range(1, len(z)):
        pi[k] = pi[k - 1] - G * (z[k] - z[k - 1]) / (CP * 0.5 * (thv[k] + thv[k - 1]))
    p = P0 * pi ** (1.0 / KAPPA)
    T = th * pi
    return Sounding(z, p, T, qv, u, v, title, "input_sounding").cleaned()


def parse_wyoming(text: str, title: str = "Wyoming sounding") -> Sounding:
    """University of Wyoming TEXT:LIST (PRES HGHT TEMP DWPT RELH MIXR DRCT SKNT ...)."""
    lines = text.splitlines()
    hdr = None
    for i, ln in enumerate(lines):
        if ln.split()[:4] == ["PRES", "HGHT", "TEMP", "DWPT"]:
            hdr = i
            break
    if hdr is None:
        raise ValueError("no PRES HGHT TEMP DWPT header")
    cols = lines[hdr].split()
    rows = []
    for ln in lines[hdr + 1:]:
        if not ln.strip() or ln.strip().startswith("-"):
            continue
        if re.match(r"\s*[A-Za-z]", ln):
            break
        vals = []
        for c in range(len(cols)):
            s = ln[7 * c:7 * c + 7].strip()
            vals.append(float(s) if s else np.nan)
        rows.append(vals)
    a = np.array(rows, dtype=np.float64)
    ci = {c: i for i, c in enumerate(cols)}
    good = np.isfinite(a[:, ci["PRES"]]) & np.isfinite(a[:, ci["HGHT"]]) & np.isfinite(a[:, ci["TEMP"]])
    a = a[good]
    p = a[:, ci["PRES"]] * 100.0
    T = a[:, ci["TEMP"]] + 273.15
    td = a[:, ci["DWPT"]]
    td = np.where(np.isfinite(td), td, a[:, ci["TEMP"]] - 30.0) + 273.15
    qv = EPS * es_water(td) / (p - es_water(td))
    d = a[:, ci.get("DRCT", 0)]
    s = a[:, ci.get("SKNT", 0)] * KT
    u, v = dirspd_to_uv(d, s)
    z = a[:, ci["HGHT"]]
    okw = np.isfinite(u)
    if okw.sum() >= 2 and not okw.all():
        u = np.interp(z, z[okw], u[okw])
        v = np.interp(z, z[okw], v[okw])
    return Sounding(z, p, T, qv, np.nan_to_num(u), np.nan_to_num(v), title, "wyoming").cleaned()


# column lists of a JSON sounding (e.g. {"level": [...], "hght": [...], ...}) and their other names
JSON_COLUMNS = {"pres": ("level", "pres", "pressure", "p"), "hght": ("hght", "height", "z", "hgt"),
                "temp": ("temp", "temperature", "t"), "dwpt": ("dwpt", "dewpoint", "td", "dwpc"),
                "wdir": ("wdir", "drct", "wind_direction"), "wspd": ("wspd", "sknt", "wind_speed")}


def json_columns(d) -> dict | None:
    """The SHARPpy columns of a JSON sounding (lists of numbers or numeric strings,
    pressure hPa, height m, T and Td C, wind from deg and kt), or None."""
    if not isinstance(d, dict):
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    cols = {}
    for name, keys in JSON_COLUMNS.items():
        col = next((low[k] for k in keys if isinstance(low.get(k), list)), None)
        if col is None:
            return None
        cols[name] = col
    n = len(cols["pres"])
    if n < 3 or any(len(c) != n for c in cols.values()):
        return None
    return cols


def parse_json(text: str, title: str | None = None) -> Sounding:
    import json
    d = json.loads(text)
    cols = json_columns(d)
    if cols is None:
        raise ValueError("JSON without level, hght, temp, dwpt, wdir and wspd lists")
    rows = "\n".join(", ".join(str(cols[c][i]) for c in ("pres", "hght", "temp", "dwpt", "wdir", "wspd"))
                     for i in range(len(cols["pres"])))
    return parse_sharppy("%RAW%\n" + rows + "\n%END%", title or str(d.get("title") or "JSON sounding"))


def parse_file(path) -> Sounding:
    """A sounding file of any supported kind; a generic title becomes the file name."""
    from pathlib import Path
    path = Path(path)
    s = parse_any(path.read_text(errors="replace"))
    if s.title in ("JSON sounding", "input_sounding", "SHARPpy sounding", "Wyoming sounding"):
        s.title = path.stem
    return s


def parse_any(text: str, title: str | None = None) -> Sounding:
    if not text.strip():
        raise ValueError("the sounding text is empty")
    if text.lstrip().startswith("{"):
        return parse_json(text, title)
    if "%RAW%" in text:
        return parse_sharppy(text, title)
    if "PRES" in text and "HGHT" in text and "DWPT" in text:
        return parse_wyoming(text, title or "Wyoming sounding")
    first = text.strip().splitlines()[0].split()
    if len(first) == 3:
        return parse_input_sounding(text, title or "input_sounding")
    # comma-separated SHARPpy-like rows without markers
    return parse_sharppy(text, title)


# ------------------------------------------------------------- analytic
def weisman_klemp(theta_sfc=300.0, qv0=14.0, theta_tr=343.0, z_tr=12000.0, t_tr=213.0,
                  shear_ms=35.0, shear_depth=6000.0, hodograph="quarter", psfc=1.0e5,
                  title=None) -> Sounding:
    """Weisman & Klemp (1982) sounding with a quarter-circle, half-circle or
    straight hodograph (``shear_ms`` over ``shear_depth``)."""
    z = np.arange(0.0, 20001.0, 100.0)
    th = np.where(z <= z_tr, theta_sfc + (theta_tr - theta_sfc) * (z / z_tr) ** 1.25,
                  theta_tr * np.exp(G * (z - z_tr) / (CP * t_tr)))
    rh = np.where(z <= z_tr, 1.0 - 0.75 * (z / z_tr) ** 1.25, 0.25)
    qv = np.zeros_like(z)
    for _ in range(10):
        thv = th * (1.0 + qv / EPS) / (1.0 + qv)
        pi = np.empty_like(z)
        pi[0] = (psfc / P0) ** KAPPA
        pi[1:] = pi[0] - np.cumsum(G * np.diff(z) / (CP * 0.5 * (thv[1:] + thv[:-1])))
        p = P0 * pi ** (1.0 / KAPPA)
        T = th * pi
        qv = np.minimum(rh * qv_sat(p, T), qv0 / 1000.0)
    us = shear_ms
    if hodograph == "straight":
        u = us * np.clip(z / shear_depth, 0, 1) - us / 2.0
        v = np.zeros_like(z)
    elif hodograph == "half":
        ang = np.pi * np.clip(z / shear_depth, 0, 1)
        r = us / 2.0
        u = -r * np.cos(ang)
        v = r * np.sin(ang)
    else:  # quarter circle (2 km radius turning) + straight above, WK82/WRF quarter_ss style
        r = 10.0 * us / 35.0
        zq = 2000.0
        ang = 0.5 * np.pi * np.clip(z / zq, 0, 1)
        u = r - r * np.cos(ang)
        v = r * np.sin(ang)
        upper = z > zq
        frac = np.clip((z - zq) / (shear_depth - zq), 0, 1)
        u = np.where(upper, r + (us - r) * frac, u)
        v = np.where(upper, r, v)
    return Sounding(z, p, T, qv, u, v,
                    title or f"Weisman-Klemp {theta_sfc:.0f} K, {qv0:.0f} g/kg, {shear_ms:.0f} m/s {hodograph}",
                    "analytic").cleaned()


# --------------------------------------------------------------- diagnostics
def _fine(s: Sounding, dz: float = 25.0, top: float = 16000.0):
    zagl = s.agl()
    zt = min(top, float(zagl[-1]))
    zz = np.arange(0.0, zt + 0.1, dz)
    p = np.exp(np.interp(zz, zagl, np.log(s.p)))
    T = np.interp(zz, zagl, s.T)
    qv = np.exp(np.interp(zz, zagl, np.log(np.maximum(s.qv, 1e-8))))
    u = np.interp(zz, zagl, s.u)
    v = np.interp(zz, zagl, s.v)
    return zz, p, T, qv, u, v


def _moist_T(thetae_target, p, t_guess):
    """Temperature at pressure p on the saturated pseudo-adiabat with theta_e."""
    t = np.array(t_guess, dtype=np.float64)
    for _ in range(25):
        f0 = thetae_bolton(p, t, qv_sat(p, t)) - thetae_target
        f1 = thetae_bolton(p, t + 0.05, qv_sat(p, t + 0.05)) - thetae_target
        step = f0 * 0.05 / np.where(np.abs(f1 - f0) < 1e-9, 1e-9, f1 - f0)
        t = t - np.clip(step, -10, 10)
        if np.all(np.abs(step) < 1e-3):
            break
    return t


def lift_parcel(zz, p, T, qv, k0: int, p0=None, t0=None, q0=None):
    """CAPE/CIN (J/kg, virtual-temperature corrected), LCL/LFC/EL heights (m AGL)."""
    p0 = p[k0] if p0 is None else p0
    t0 = T[k0] if t0 is None else t0
    q0 = qv[k0] if q0 is None else q0
    th0 = theta_from(p0, t0)
    # LCL (Bolton 1980)
    e = q0 * p0 / (EPS + q0)
    tlcl = 2840.0 / (3.5 * np.log(t0) - np.log(e / 100.0) - 4.805) + 55.0
    plcl = P0 * (tlcl / th0) ** (1.0 / KAPPA)
    the = thetae_bolton(p0, t0, q0)
    pz = p[k0:]
    tz = np.where(pz >= plcl, th0 * (pz / P0) ** KAPPA, np.nan)
    sat = pz < plcl
    if sat.any():
        guess = np.where(sat, np.nan_to_num(tz, nan=tlcl), tlcl)
        tm = _moist_T(the, pz[sat], np.full(sat.sum(), tlcl) - 0.0 * guess[sat])
        tz[sat] = tm
    qp = np.where(pz >= plcl, q0, qv_sat(pz, tz))
    tvp = tz * (1.0 + qp / EPS) / (1.0 + qp)
    tve = T[k0:] * (1.0 + qv[k0:] / EPS) / (1.0 + qv[k0:])
    b = G * (tvp - tve) / tve
    zl = zz[k0:]
    dz = np.gradient(zl)
    pos = b > 0
    # EL: last level above the LCL with positive buoyancy.  LFC: the lowest base of a
    # positive layer from which a parcel at rest rises freely to the EL, its buoyant
    # energy never running out on the way; a shallow positive layer under a deeper
    # negative one is not it (that negative layer is CIN, not hidden above the LFC)
    zlcl = float(np.interp(-plcl, -pz, zl)) if plcl < pz[0] else float(zl[0])
    above_lcl = zl >= zlcl - 1.0
    cand = np.where(pos & above_lcl)[0]
    if cand.size == 0:
        return dict(cape=0.0, cin=float(np.sum(np.minimum(b, 0) * dz)), lcl=zlcl, lfc=np.nan, el=np.nan)
    kel = cand[-1]
    bases = cand[np.r_[True, np.diff(cand) > 1]]
    energy = np.concatenate(([0.0], np.cumsum(b * dz)))      # energy[k]: integral below level k
    klfc = next(int(k) for k in bases if np.min(energy[k + 1:kel + 2] - energy[k]) >= 0.0)
    cape = float(np.sum(np.where(pos, b, 0.0)[klfc:kel + 1] * dz[klfc:kel + 1]))
    cin = float(np.sum(np.minimum(b[:klfc], 0.0) * dz[:klfc]))
    return dict(cape=cape, cin=cin, lcl=zlcl, lfc=float(zl[klfc]), el=float(zl[kel]))


def dcape(zz, p, T, qv) -> float:
    """Downdraft CAPE (J/kg): the lowest-theta_e air of the lowest 400 hPa, cooled to saturation by
    evaporation and sinking moist-adiabatically to the ground; how cold a storm's outflow can get."""
    low = np.where(p >= p[0] - 40000.0)[0]
    the = thetae_bolton(p[low], T[low], qv[low])
    k = int(low[np.argmin(the)])
    if k < 1:
        return 0.0                                  # its driest air is at the ground (a cold, dry airmass): nothing to sink
    tw = _moist_T(np.array([the.min()]), p[k:k + 1], T[k:k + 1])[0]
    tp = _moist_T(np.full(k + 1, thetae_bolton(p[k], tw, qv_sat(p[k], tw))), p[:k + 1], np.full(k + 1, tw))
    tve = T[:k + 1] * (1.0 + qv[:k + 1] / EPS) / (1.0 + qv[:k + 1])
    qp = qv_sat(p[:k + 1], tp)
    b = G * (tve - tp * (1.0 + qp / EPS) / (1.0 + qp)) / tve
    return float(np.sum(np.maximum(b, 0.0) * np.gradient(zz[:k + 1])))


def bunkers(zz, u, v):
    sel6 = zz <= 6000.0
    um, vm = u[sel6].mean(), v[sel6].mean()
    lo = zz <= 500.0
    hi = (zz >= 5500.0) & (zz <= 6000.0)
    su = u[hi].mean() - u[lo].mean()
    sv = v[hi].mean() - v[lo].mean()
    mag = max(math.hypot(su, sv), 1e-6)
    d = 7.5
    rm = (um + d * sv / mag, vm - d * su / mag)
    lm = (um - d * sv / mag, vm + d * su / mag)
    return rm, lm, (um, vm)


def srh(zz, u, v, cu, cv, depth):
    sel = zz <= depth
    uu = u[sel] - cu
    vv = v[sel] - cv
    return float(np.sum(uu[1:] * vv[:-1] - uu[:-1] * vv[1:]))


def _ml_parcel(zz, p, T, qv) -> dict:
    """The mixed-layer parcel: the lowest 100 hPa's mean theta and qv, lifted from the surface."""
    ml = p >= p[0] - 10000.0
    th_ml = theta_from(p[ml], T[ml]).mean()
    q_ml = qv[ml].mean()
    return lift_parcel(zz, p, T, qv, 0, p0=p[0], t0=th_ml * (p[0] / P0) ** KAPPA, q0=q_ml)


def parcel_trace(s: Sounding, top_hpa: float = 100.0) -> dict:
    """The mixed-layer parcel's path for a skew-T (the parcel of MLCAPE): pressure (hPa) and
    temperature (C), dry-adiabatic to its LCL, then moist pseudo-adiabatic."""
    zz, p, T, qv, _, _ = _fine(s)
    ml = p >= p[0] - 10000.0
    th = float(theta_from(p[ml], T[ml]).mean())
    q0 = float(qv[ml].mean())
    p0 = float(p[0])
    t0 = th * (p0 / P0) ** KAPPA
    e = q0 * p0 / (EPS + q0)
    tlcl = 2840.0 / (3.5 * np.log(t0) - np.log(e / 100.0) - 4.805) + 55.0
    plcl = P0 * (tlcl / th) ** (1.0 / KAPPA)
    pz = p[p >= top_hpa * 100.0][::4]
    tz = th * (pz / P0) ** KAPPA
    sat = pz < plcl
    if sat.any():
        tz[sat] = _moist_T(thetae_bolton(p0, t0, q0), pz[sat], np.full(int(sat.sum()), tlcl))
    return {"p": (pz / 100.0).round(1).tolist(), "T": (tz - 273.15).round(2).tolist(),
            "lcl_hpa": round(float(plcl) / 100.0, 1)}


def diagnostics(s: Sounding, south: bool = False) -> dict:
    """``south``: a southern-hemisphere run, where the cyclonic supercell (the tornadic one) is Bunkers' LEFT
    mover and turns clockwise: "rm" is then that mover and the helicities are about it with the sign turned
    (cyclonic positive, as SHARPpy's southern-hemisphere mode reads them), so STP, SCP and everything that rates a
    storm by them read as in the north."""
    zz, p, T, qv, u, v = _fine(s)
    out = {}
    sb = lift_parcel(zz, p, T, qv, 0)
    mlp = _ml_parcel(zz, p, T, qv)
    # most unstable in the lowest 300 hPa
    low = np.where(p >= p[0] - 30000.0)[0]
    the = thetae_bolton(p[low], T[low], qv[low])
    kmu = int(low[np.argmax(the)])
    mu = lift_parcel(zz, p, T, qv, kmu)
    rm, lm, mean6 = bunkers(zz, u, v)
    if south:
        rm, lm = lm, rm
    sg = -1.0 if south else 1.0
    srh05 = sg * srh(zz, u, v, rm[0], rm[1], 500.0)
    srh1 = sg * srh(zz, u, v, rm[0], rm[1], 1000.0)
    srh3 = sg * srh(zz, u, v, rm[0], rm[1], 3000.0)
    anv = (zz >= 9000.0) & (zz <= 11000.0)
    sr_anvil = float(math.hypot(u[anv].mean() - rm[0], v[anv].mean() - rm[1])) if anv.any() else float("nan")
    def shear(top):
        k = int(np.argmin(np.abs(zz - top)))
        return float(math.hypot(u[k] - u[0], v[k] - v[0]))
    sh1, sh6 = shear(1000.0), shear(6000.0)
    pw = float(np.sum(qv[:-1] * -np.diff(p)) / G)
    lcl = mlp["lcl"]
    stp = ((mlp["cape"] / 1500.0) * float(np.clip((2000.0 - lcl) / 1000.0, 0, 1))
           * (srh1 / 150.0) * float(np.clip(sh6 / 20.0, 0, 1.5) if sh6 >= 12.5 else 0.0)
           * float(np.clip((200.0 + mlp["cin"]) / 150.0, 0, 1)))
    scp = (mu["cape"] / 1000.0) * (srh3 / 50.0) * float(np.clip(sh6 / 20.0, 0, 1) if sh6 >= 10 else 0.0)
    k3 = int(np.argmin(np.abs(zz - 3000.0)))
    lapse = (T[0] - T[k3]) / 3.0 if zz[-1] > 3000 else float("nan")
    k5 = int(np.argmin(np.abs(p - 50000.0)))
    k7 = int(np.argmin(np.abs(p - 70000.0)))
    lapse75 = (T[k7] - T[k5]) / ((zz[k5] - zz[k7]) / 1000.0)
    out.update(
        sbcape=sb["cape"], sbcin=sb["cin"], mlcape=mlp["cape"], mlcin=mlp["cin"],
        mucape=mu["cape"], mucin=mu["cin"], lcl=lcl, lfc=mlp["lfc"], el=mu["el"],
        srh05=srh05, srh1=srh1, srh3=srh3, shear1=sh1, shear6=sh6, sr_anvil=sr_anvil,
        dcape=dcape(zz, p, T, qv),
        rm=[float(rm[0]), float(rm[1])], lm=[float(lm[0]), float(lm[1])],
        mean6=[float(mean6[0]), float(mean6[1])], pw_mm=pw, stp=float(max(stp, 0.0)),
        scp=float(scp), lapse03=float(lapse), lapse75=float(lapse75),
        t2m=float(T[0] - 273.15), td2m=float(dewpoint_from_qv(p[0], qv[0]) - 273.15),
        psfc=float(p[0] / 100.0),
    )
    for k, val in list(out.items()):
        if isinstance(val, float) and not math.isfinite(val):
            out[k] = None
    return out
