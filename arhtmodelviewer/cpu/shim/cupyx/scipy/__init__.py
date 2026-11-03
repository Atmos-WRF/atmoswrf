"""cupyx.scipy on the CPU: SciPy's modules returning the stand-in's arrays."""
import importlib as _importlib

from arhtmodelviewer.cpu import arrays as _arrays


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    mod = _importlib.import_module(f"scipy.{name}")
    val = _arrays.proxy_module(f"cupyx.scipy.{name}", mod)
    globals()[name] = val
    return val
