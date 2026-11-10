"""Atmos-WRF: an OutbreakLab-style storm model and radar generator
running on ArWen's GPU WRF dynamical core (gpuwm).

ArWen (gpuwm, Apache-2.0, FahrenheitResearch) does the atmosphere: WRF ARW
split-explicit dynamics, 5th/3rd-order advection with positive-definite
moisture, monotonic 6th-order filter, TKE/Smagorinsky mixing and Morrison
two-moment microphysics.  This package adds the viewer around it:
storm-following patches that make the domain unlimited, a steady
environment, storm initiation, simulated WSR-88D radars and NEXRAD Level II.
"""
import os
import warnings
from pathlib import Path

__version__ = "2.0.0"

PROJECT = Path(__file__).resolve().parents[1]

# Keep every cache on the project drive (D:) instead of the user profile.
_CACHE = PROJECT / "runtime" / "cache"
os.environ.setdefault("CUPY_CACHE_DIR", str(_CACHE / "cupy"))
os.environ.setdefault("CUDA_CACHE_PATH", str(_CACHE / "cuda"))
os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(2 * 1024 ** 3))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
# gpuwm keeps lookup tables, built bridges and caches under ~/.gpuwm; give it a
# home on D: (and the tables we ship) so nothing lands in C:\Users.
_TABLES = PROJECT / "runtime" / "tables"
os.environ.setdefault("GPUWM_THOMPSON_TABLE_ROOT", str(_TABLES / "thompson"))
os.environ.setdefault("GPUWM_THOMPSON_FORK_TABLE_ROOT", str(_TABLES / "thompson-wrf39-noaa"))
os.environ.setdefault("GPUWM_TEMPO_TABLE_ROOT", str(_TABLES / "tempo-v3"))
os.environ.setdefault("GPUWM_GENERATED_TABLE_ROOT", str(_TABLES / "generated"))
_HOME = PROJECT / "runtime" / "home"
_HOME.mkdir(parents=True, exist_ok=True)
# the user's own home (their soundings and color tables), kept for child processes too
os.environ.setdefault("ARHTMODELVIEWER_USER_HOME", os.environ.get("USERPROFILE") or os.environ.get("HOME") or str(Path.home()))
USER_HOME = Path(os.environ["ARHTMODELVIEWER_USER_HOME"])
os.environ["USERPROFILE"] = os.environ["HOME"] = str(_HOME)
# CuPy finds the pip-installed CUDA 12 toolkit (split into nvidia/* packages)
# on its own; it still warns that there is no single CUDA root, which is
# expected for that layout.
warnings.filterwarnings("ignore", message="CUDA path could not be detected")


def _backend_argument():
    """``--backend cpu`` / ``--backend=cpu`` on the command line (python -m arhtmodelviewer ...)."""
    import sys
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--backend" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--backend="):
            return a.split("=", 1)[1]
    return None


# NVIDIA GPU (CUDA) or CPU: decided here, before anything imports CuPy (backend.py)
from . import backend  # noqa: E402

backend.activate(_backend_argument())
