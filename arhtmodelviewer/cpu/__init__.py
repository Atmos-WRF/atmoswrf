"""The CPU backend: ArWen (gpuwm) and the viewer run on the processor instead of an NVIDIA GPU.

Nothing in ArWen changes.  ``shim/`` holds a stand-in for the parts of CuPy that ArWen and the
viewer use (``cupy``, ``cupyx``, ``cupy_backends``): arrays are NumPy arrays with CuPy's few
extra attributes (:mod:`.arrays`), and CUDA source handed to ``cp.RawModule`` & co. -- ArWen's
own kernels, unchanged -- is compiled as C++ for the CPU by clang (the ``ziglang`` wheel) and run
on a thread pool (:mod:`.kernels`, ``runtime/``).  :func:`arhtmodelviewer.backend.activate` puts
``shim/`` in front of the import path, so ``import cupy`` finds it; with the CUDA backend it is
never loaded.
"""
from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM = HERE / "shim"
RUNTIME = HERE / "runtime"
PROJECT = HERE.parents[1]
CACHE = Path(os.environ.get("ARHTMODELVIEWER_CPU_KERNEL_CACHE") or PROJECT / "runtime" / "cache" / "cpu-kernels")


def threads() -> int:
    """Worker threads for kernels: every logical CPU unless ARHTMODELVIEWER_CPU_THREADS says otherwise."""
    env = os.environ.get("ARHTMODELVIEWER_CPU_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, os.cpu_count() or 1)


def compiler_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("ziglang") is not None


def ram() -> tuple[int, int]:
    """(available, total) bytes of RAM."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys), int(st.ullTotalPhys)
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info.get("MemAvailable", info.get("MemFree", 0)), info.get("MemTotal", 0)
    except OSError:
        return 4 << 30, 8 << 30


def cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "CPU"
