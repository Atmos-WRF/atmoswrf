"""CuPy on the CPU -- Atmos-WRF's stand-in (arhtmodelviewer/cpu), not the real CuPy.

On the import path only when the viewer runs with the CPU backend.  Arrays are NumPy arrays
(:class:`arhtmodelviewer.cpu.arrays.ndarray`); every NumPy function is re-exported returning them;
CUDA kernels compile for the CPU (:mod:`arhtmodelviewer.cpu.kernels`).
"""
from __future__ import annotations

import numpy as _np

from arhtmodelviewer.cpu import arrays as _arrays
from arhtmodelviewer.cpu.kernels import ElementwiseKernel, RawKernel, RawModule, ReductionKernel  # noqa: F401

__version__ = "14.2.0"
__cpu_backend__ = True

ndarray = _arrays.ndarray
ufunc = _np.ufunc

from numpy import (bool_, byte, complex64, complex128, csingle, cdouble, double, dtype, e, euler_gamma,  # noqa: E402,F401
                   finfo, float16, float32, float64, floating, generic, iinfo, inexact, inf, int8, int16,
                   int32, int64, integer, intc, intp, longlong, nan, newaxis, number, pi, signedinteger,
                   single, ubyte, uint8, uint16, uint32, uint64, uintc, uintp, ulonglong, unsignedinteger,
                   can_cast, result_type, promote_types, ndim, isscalar, issubdtype, shape, size)
from numpy import bool as bool  # noqa: E402,F401,A001

from . import cuda  # noqa: E402,F401
from . import _core  # noqa: E402,F401


def asarray(a, dtype=None, order=None, *, blocking=False):
    """Device copy of host data; an array that already is one comes back as it is."""
    if isinstance(a, ndarray):
        if dtype is None or _np.dtype(dtype) == a.dtype:
            if order in (None, "K", "A") or (order == "C" and a.flags.c_contiguous) or (
                    order == "F" and a.flags.f_contiguous):
                return a
        return a.astype(dtype or a.dtype, order=order or "K", copy=False).view(ndarray)
    return _np.array(a, dtype=dtype, order=order or "K", copy=True).view(ndarray)


def array(obj, dtype=None, copy=True, order="K", subok=False, ndmin=0, *, blocking=False):
    return _np.array(obj, dtype=dtype, copy=True if copy is None else copy, order=order,
                     ndmin=ndmin).view(ndarray)


def asanyarray(a, dtype=None, order=None):
    return asarray(a, dtype, order)


def ascontiguousarray(a, dtype=None):
    if isinstance(a, ndarray) and a.flags.c_contiguous and (dtype is None or _np.dtype(dtype) == a.dtype):
        return a
    return _np.ascontiguousarray(a, dtype=dtype).view(ndarray)


def asfortranarray(a, dtype=None):
    return _np.asfortranarray(a, dtype=dtype).view(ndarray)


def asnumpy(a, stream=None, order="C", out=None, *, blocking=True):
    """Host copy (CuPy copies device -> host; NumPy input comes back as it is)."""
    if isinstance(a, ndarray):
        return a.get(order=order, out=out)
    if out is not None:
        out[...] = a
        return out
    return _np.asarray(a, order=order)


def get_array_module(*args):
    import sys
    for a in args:
        if isinstance(a, ndarray):
            return sys.modules[__name__]
    return _np


def fuse(*args, **kwargs):
    """cupy.fuse: a no-op on the CPU (NumPy runs the Python function as written)."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda f: f


def is_available():
    return True


def get_default_memory_pool():
    return cuda.memory._DEFAULT_POOL


def get_default_pinned_memory_pool():
    return cuda.memory._PINNED_POOL


def show_config(*, _full=False):
    from arhtmodelviewer import backend
    print("CuPy stand-in on the CPU (Atmos-WRF):", backend.describe())


def __getattr__(name):
    """Every other NumPy name, returning stand-in arrays."""
    if name.startswith("__"):
        raise AttributeError(name)
    if name in ("random", "fft", "linalg", "testing", "polynomial"):
        import importlib
        sub = getattr(_np, name)
        val = _arrays.proxy_module(f"cupy.{name}", sub if name != "testing" else importlib.import_module("numpy.testing"))
        if name == "random":
            val.RandomState = _arrays.RandomState         # CuPy's dtype keyword on the float samplers
    else:
        try:
            obj = getattr(_np, name)
        except AttributeError:
            raise AttributeError(f"module 'cupy' (the viewer's CPU stand-in) has no attribute {name!r}") from None
        if isinstance(obj, type) or not callable(obj):
            val = obj
        elif isinstance(obj, _np.ufunc):
            val = _arrays.ufunc(obj)
        else:
            val = _arrays.wrapped(obj)
    globals()[name] = val
    return val
