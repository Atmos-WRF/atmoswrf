"""Which processor runs the model: an NVIDIA GPU (CUDA, through CuPy) or the CPU.

Chosen once per process, before ArWen or CuPy is imported, from ``ARHTMODELVIEWER_BACKEND``
(``python -m arhtmodelviewer serve --backend ...`` sets it):

- ``cuda``: CuPy and NVIDIA's CUDA, exactly as before the CPU backend existed;
- ``cpu``: the processor (:mod:`arhtmodelviewer.cpu`): ``import cupy`` finds the viewer's CPU
  stand-in, which runs ArWen's own CUDA kernels compiled for the CPU;
- ``auto`` (default): ``cuda`` when CuPy is installed and Windows has an NVIDIA driver, ``cpu``
  otherwise.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ENV = "ARHTMODELVIEWER_BACKEND"
CHOICES = ("auto", "cuda", "cpu")
NAME: str | None = None          # the active backend once activate() ran


def _nvidia_driver() -> bool:
    if os.name == "nt":
        root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        return (root / "System32" / "nvcuda.dll").exists()
    return Path("/proc/driver/nvidia/version").exists() or Path("/dev/nvidiactl").exists()


def resolve(requested: str | None = None) -> str:
    want = (requested or os.environ.get(ENV) or "auto").strip().lower()
    if want not in CHOICES:
        raise ValueError(f"{ENV} must be one of {', '.join(CHOICES)} (not {want!r})")
    if want != "auto":
        return want
    if importlib.util.find_spec("cupy") is not None and _nvidia_driver():
        return "cuda"
    return "cpu"


def activate(requested: str | None = None) -> str:
    """Pick the backend for this process (idempotent; the first call decides)."""
    global NAME
    if NAME is not None:
        return NAME
    name = resolve(requested)
    if name == "cpu":
        if "cupy" in sys.modules and not getattr(sys.modules["cupy"], "__cpu_backend__", False):
            raise RuntimeError("the CPU backend has to be chosen before CuPy is imported")
        from . import cpu
        shim = str(cpu.SHIM)
        if shim not in sys.path:
            sys.path.insert(0, shim)
    NAME = name
    os.environ[ENV] = name                 # child processes (the server's engine) follow
    return name


def is_cpu() -> bool:
    return (NAME or activate()) == "cpu"


def describe() -> str:
    """What runs the model, for the page and the logs."""
    if is_cpu():
        from .cpu import cpu_name, threads
        return f"CPU ({cpu_name()}, {threads()} threads)"
    try:
        import cupy as cp
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        return (name.decode() if isinstance(name, bytes) else str(name)) + " (CUDA)"
    except Exception:                       # noqa: BLE001 - no card to ask
        return "NVIDIA GPU (CUDA)"
