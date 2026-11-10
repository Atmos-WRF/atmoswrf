"""Which processor runs the model: a GPU through CuPy (NVIDIA's CUDA, or AMD's ROCm on Linux / WSL2) or the CPU.

Chosen once per process, before ArWen or CuPy is imported, from ``ARHTMODELVIEWER_BACKEND``
(``python -m arhtmodelviewer serve --backend ...`` sets it):

- ``cuda``: CuPy and NVIDIA's CUDA, exactly as before the CPU backend existed;
- ``cpu``: the processor (:mod:`arhtmodelviewer.cpu`): ``import cupy`` finds the viewer's CPU
  stand-in, which runs ArWen's own CUDA kernels compiled for the CPU;
- ``auto`` (default): ``cuda`` when CuPy is installed and Windows has an NVIDIA driver, ``cpu``
  otherwise.

AMD GPUs: CuPy's ROCm build (Linux, or WSL2 on a Radeon that AMD's WSL driver supports) runs the same raw CUDA
kernels through HIP; ``cuda`` is then that build (``cp.cuda.runtime.is_hip``), and hip_compat() first rewrites
the few NVRTC-only options and 32-lane warp votes HIP's compiler takes otherwise.  There is no ROCm build of CuPy
for Windows: an AMD card there runs the model on the CPU (docs/AMD.md).
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


def _rocm() -> bool:
    """A ROCm GPU (Linux / WSL2): AMD's kernel driver, or the WSL2 GPU bridge with ROCm installed."""
    if os.name == "nt":
        return False
    return Path("/dev/kfd").exists() or (Path("/dev/dxg").exists() and Path("/opt/rocm").exists())


def amd_on_windows() -> bool:
    """Windows with an AMD GPU (its HIP runtime ships with the driver) and no NVIDIA one."""
    if os.name != "nt" or _nvidia_driver():
        return False
    root = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    try:
        return any(root.glob("amdhip64*.dll"))
    except OSError:
        return False


def resolve(requested: str | None = None) -> str:
    want = (requested or os.environ.get(ENV) or "auto").strip().lower()
    if want not in CHOICES:
        raise ValueError(f"{ENV} must be one of {', '.join(CHOICES)} (not {want!r})")
    if want != "auto":
        return want
    if importlib.util.find_spec("cupy") is not None and (_nvidia_driver() or _rocm()):
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
    if name == "cuda" and _rocm():
        hip_compat()
    return name


# NVRTC options hiprtc does not take, and what to pass instead (None: drop it)
_HIP_OPTIONS = {"--fmad=false": "-ffp-contract=off", "-fmad=false": "-ffp-contract=off",
                "--fmad=true": None, "-use_fast_math": "-ffast-math", "--use_fast_math": "-ffast-math"}
_HIP_PATCHED = False


def hip_source(code: str) -> str:
    """CUDA kernel source made right for HIP's wave64 GPUs: warp votes and shuffles with a 32-lane mask become
    HIP's own whole-wavefront ones (the same answer on wave32 RDNA cards, all lanes on wave64 CDNA ones)."""
    import re
    code = re.sub(r"__ballot_sync\(\s*[^,()]+,\s*", "__ballot(", code)
    code = re.sub(r"__any_sync\(\s*[^,()]+,\s*", "__any(", code)
    code = re.sub(r"__all_sync\(\s*[^,()]+,\s*", "__all(", code)
    code = re.sub(r"__shfl_sync\(\s*[^,()]+,\s*", "__shfl(", code)
    code = re.sub(r"__shfl_down_sync\(\s*[^,()]+,\s*", "__shfl_down(", code)
    code = re.sub(r"__shfl_up_sync\(\s*[^,()]+,\s*", "__shfl_up(", code)
    code = re.sub(r"__shfl_xor_sync\(\s*[^,()]+,\s*", "__shfl_xor(", code)
    code = re.sub(r"\bunsigned int (\w+) = __ballot\(", r"unsigned long long \1 = __ballot(", code)
    return code


def hip_options(options) -> tuple:
    out = []
    for o in options or ():
        if o in _HIP_OPTIONS:
            if _HIP_OPTIONS[o]:
                out.append(_HIP_OPTIONS[o])
        else:
            out.append(o)
    return tuple(out)


def hip_compat() -> bool:
    """With CuPy's ROCm build: RawModule / RawKernel take CUDA source and NVRTC options as on NVIDIA (hip_source,
    hip_options).  Nothing on CUDA.  True when installed."""
    global _HIP_PATCHED
    if _HIP_PATCHED:
        return True
    try:
        import cupy as cp
        if not getattr(cp.cuda.runtime, "is_hip", False):
            return False
    except Exception:                       # noqa: BLE001 - no CuPy, no GPU: nothing to patch
        return False
    def fixed(args, kw, opt_at):
        args = list(args)
        if args and isinstance(args[0], str):
            args[0] = hip_source(args[0])
        elif isinstance(kw.get("code"), str):
            kw["code"] = hip_source(kw["code"])
        if opt_at is not None and len(args) > opt_at:
            args[opt_at] = hip_options(args[opt_at])
        if "options" in kw:
            kw["options"] = hip_options(kw["options"])
        return args, kw

    base_module, base_kernel = cp.RawModule, cp.RawKernel

    class RawModule(base_module):           # (CuPy's are extension types: subclassed, not patched)
        def __init__(self, *args, **kw):
            args, kw = fixed(args, kw, None)
            super().__init__(*args, **kw)

    class RawKernel(base_kernel):
        def __init__(self, *args, **kw):
            args, kw = fixed(args, kw, 2)
            super().__init__(*args, **kw)

    cp.RawModule, cp.RawKernel = RawModule, RawKernel
    _HIP_PATCHED = True
    return True


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
        api = "ROCm" if getattr(cp.cuda.runtime, "is_hip", False) else "CUDA"
        return (name.decode() if isinstance(name, bytes) else str(name)) + f" ({api})"
    except Exception:                       # noqa: BLE001 - no card to ask
        return "NVIDIA GPU (CUDA)"


def note() -> str | None:
    """Why the model runs where it does, when that is not what the user may expect (shown in the Log)."""
    if is_cpu() and amd_on_windows():
        return ("an AMD GPU is installed, but CuPy has no ROCm build for Windows, so the model runs on the CPU; "
                "for the GPU, run it under WSL2 or Linux with ROCm and CuPy for ROCm (docs/AMD.md)")
    return None
