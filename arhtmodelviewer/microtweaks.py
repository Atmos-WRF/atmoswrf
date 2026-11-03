"""The viewer's tweak to gpuwm's Thompson microphysics: rain evaporation scaled by RAIN_EVAP (the cold-pool fix).

Thompson's rain evaporation (WRF's prv_rev) cooled the storms' outflow too much in the viewer's runs: HP storms
sat in -5..-8 K cold pools within 10 km, and the rear-flank air under tornadoes was colder than observed (Markowski
et al. 2002: -1..-3 K under tornadic RFDs), which starves low-level stretching.  Part of it is the storms' rain area
(heavy-rain cores 2-3x too wide), part the missing surface fluxes that warm a real cold pool back up; scaling the
evaporation is the lever left.  gpuwm passes every CUDA unit through gpuwm.physics_params.edit_kernel_source before
compiling it, and its headers through kernels._extra_header_text: both are wrapped here to multiply the evaporation
rate, after WRF's cap to what the air can take, by RAIN_EVAP -- in all four Thompson variants (mp 8 classic and
fused column, mp 28 aerosol-aware, the WRF 3.9 fork).  The installed runtime is not touched; compiled kernels are
cached by their source, so a scaled build never mixes with a plain one.  Process-wide: set before the first step.
The environment variable ARHT_RAIN_EVAP overrides it (tests)."""
from __future__ import annotations

import logging
import os
import re

log = logging.getLogger(__name__)

RAIN_EVAP = 0.6          # x Thompson's rain evaporation rate (1 = WRF's own)

_RULES = (
    # aerosol-aware (thompson_aerosol_sat.cu), classic (thompson_classic.cu), the 3.9 fork (thompson_fork.cu)
    re.compile(r"(prv_rev\s*=\s*(?:fmin|dmin_)\(\s*\(double\)\s*rate_max\s*,\s*prv_rev\s*\*\s*\(double\)\s*orho\s*\))\s*;"),
    # mp 8 per-level kernels (thompson.cu)
    re.compile(r"(evaporation_rate\s*=\s*fmin\(\s*\(double\)\s*maximum_rate\s*,\s*evaporation_rate\s*\*\s*\(double\)\s*orho\s*\))\s*;"),
    # the fused column (thompson_column_generated.cuh)
    re.compile(r"(prv_rev\[0\+\(k-\(kts\)\)\*\(1\)\]\s*=\s*tc_min\(\(\(double\)\(rate_max\)\),\(prv_rev\[0\+\(k-\(kts\)\)\*\(1\)\]\*orho\)\))\s*;"),
)
_DONE = {"installed": False, "scale": 1.0, "edits": {}}


def scale() -> float:
    try:
        return float(os.environ.get("ARHT_RAIN_EVAP", RAIN_EVAP))
    except ValueError:
        return RAIN_EVAP


def _edit(name: str, text: str) -> str:
    s = _DONE["scale"]
    if abs(s - 1.0) < 1e-6 or "thompson" not in name and "prv_rev" not in text:
        return text
    n = 0
    for rule in _RULES:
        text, k = rule.subn(lambda m: f"{m.group(1)} * (double){s:.6f};", text)
        n += k
    if n:
        _DONE["edits"][name] = _DONE["edits"].get(name, 0) + n
        log.info("rain evaporation x%.2f in %s (%d sites)", s, name, n)
    return text


def install() -> None:
    """Wrap gpuwm's source hooks (once per process)."""
    if _DONE["installed"]:
        return
    _DONE["installed"] = True
    _DONE["scale"] = scale()
    if abs(_DONE["scale"] - 1.0) < 1e-6:
        return
    try:
        import gpuwm.physics_params as pp
        import gpuwm.core.kernels as kk
    except Exception:
        log.exception("rain evaporation tweak: gpuwm not importable")
        return
    orig_edit = pp.edit_kernel_source

    def edit_kernel_source(module, text):
        return _edit(module, orig_edit(module, text))
    pp.edit_kernel_source = edit_kernel_source
    orig_hdr = kk._extra_header_text

    def _extra_header_text(name, kernel_dir=kk._KDIR):
        return _edit(name + " (headers)", orig_hdr(name, kernel_dir))
    kk._extra_header_text = _extra_header_text


def describe() -> str:
    s = _DONE["scale"]
    return "" if abs(s - 1.0) < 1e-6 else f"Thompson rain evaporation x{s:.2f} (the cold-pool fix)"
