"""cupyx.scipy.ndimage on the CPU: scipy.ndimage returning the stand-in's arrays."""
import scipy.ndimage as _nd

from arhtmodelviewer.cpu import arrays as _arrays


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    attr = getattr(_nd, name)
    val = _arrays.wrapped(attr) if callable(attr) and not isinstance(attr, type) else attr
    globals()[name] = val
    return val
