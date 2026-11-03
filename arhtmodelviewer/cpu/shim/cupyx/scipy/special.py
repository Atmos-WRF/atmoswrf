"""cupyx.scipy.special on the CPU: scipy.special returning the stand-in's arrays."""
import numpy as _np
import scipy.special as _sp

from arhtmodelviewer.cpu import arrays as _arrays


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    attr = getattr(_sp, name)
    if isinstance(attr, _np.ufunc):
        val = _arrays.ufunc(attr)
    elif callable(attr) and not isinstance(attr, type):
        val = _arrays.wrapped(attr)
    else:
        val = attr
    globals()[name] = val
    return val
