"""cupyx on the CPU (Atmos-WRF's stand-in): SciPy through NumPy, pinned arrays are host arrays."""
import numpy as _np

from arhtmodelviewer.cpu import arrays as _arrays

from . import scipy  # noqa: F401


def empty_pinned(shape, dtype=float, order="C"):
    return _np.empty(shape, dtype=dtype, order=order)


def zeros_pinned(shape, dtype=float, order="C"):
    return _np.zeros(shape, dtype=dtype, order=order)


def empty_like_pinned(a, dtype=None, order="K", subok=None, shape=None):
    return _np.empty_like(_np.asarray(a), dtype=dtype, order=order, shape=shape)


def zeros_like_pinned(a, dtype=None, order="K", subok=None, shape=None):
    return _np.zeros_like(_np.asarray(a), dtype=dtype, order=order, shape=shape)


def scatter_add(a, slices, value):
    _np.add.at(a, slices, value)


def scatter_max(a, slices, value):
    _np.maximum.at(a, slices, value)


def scatter_min(a, slices, value):
    _np.minimum.at(a, slices, value)


def rsqrt(x, out=None):
    return _arrays.wrap(_np.divide(1.0, _np.sqrt(x), out=out))


errstate = _np.errstate
