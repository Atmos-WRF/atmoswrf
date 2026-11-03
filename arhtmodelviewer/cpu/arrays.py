"""CuPy's array on the CPU: a NumPy array with the few things CuPy adds and ArWen uses.

- ``.get()`` copies to a plain NumPy array (CuPy's device -> host), ``.set()`` copies back;
- ``.data.ptr`` is the address of the first element, ``.device`` the one "device";
- reductions and integer indexing give 0-d arrays, as CuPy's do, not NumPy scalars, so
  ``x.max().get()`` and ``x[0].item()`` work the same on both backends.

:func:`wrap` turns what a NumPy function returned into this type; the ``cupy`` stand-in wraps
every NumPy function it re-exports with it.
"""
from __future__ import annotations

import numpy as _np


class MemoryPointer:
    """cupy.cuda.MemoryPointer for an array (``ptr`` is all ArWen reads) or a raw address."""

    def __init__(self, mem=None, offset=0, *, ptr=None, owner=None):
        if ptr is None:
            base = getattr(mem, "ptr", 0) if mem is not None else 0
            ptr = int(base) + int(offset)
        self.ptr = int(ptr)
        self.mem = mem
        self.device_id = 0
        self._owner = owner

    def __int__(self):
        return self.ptr

    def __add__(self, offset):
        return MemoryPointer(ptr=self.ptr + int(offset), owner=self._owner)


def _zero_d(r):
    return _np.asarray(r).view(ndarray)


class ndarray(_np.ndarray):
    """cupy.ndarray on the CPU."""

    # ---- CuPy's host <-> device API
    def get(self, stream=None, order="C", out=None, blocking=True):
        if out is not None:
            out[...] = self
            return out
        return _np.array(self, order=order, copy=True, subok=False)

    def set(self, arr, stream=None):
        self[...] = arr

    @property
    def data(self):
        return MemoryPointer(ptr=self.__array_interface__["data"][0], owner=self)

    @property
    def device(self):
        from cupy.cuda import Device
        return Device(0)

    @property
    def __cuda_array_interface__(self):
        a = self.__array_interface__
        return {"shape": a["shape"], "typestr": a["typestr"], "data": a["data"],
                "strides": a["strides"], "version": 3}

    def toDlpack(self):
        return self.__dlpack__()

    def reduced_view(self, dtype=None):
        return self

    # ---- 0-d results like CuPy
    def __getitem__(self, key):
        r = super().__getitem__(key)
        return r if isinstance(r, _np.ndarray) else _zero_d(r)

    def _r(name):
        base = getattr(_np.ndarray, name)

        def method(self, *args, **kwargs):
            r = base(self, *args, **kwargs)
            if isinstance(r, _np.ndarray):
                return r if type(r) is ndarray else r.view(ndarray)
            return _zero_d(r)
        method.__name__ = name
        method.__doc__ = base.__doc__
        return method

    for _name in ("sum", "mean", "max", "min", "any", "all", "prod", "argmax", "argmin", "std", "var",
                  "ptp", "dot", "trace", "cumsum", "cumprod", "round", "clip", "nonzero"):
        if hasattr(_np.ndarray, _name):
            locals()[_name] = _r(_name)
    del _r, _name

    def nonzero(self):
        return tuple(r.view(ndarray) for r in _np.ndarray.nonzero(self))


def wrap(r):
    """A NumPy function's result as the stand-in's types (arrays, 0-d for scalars, tuples/lists of them)."""
    t = type(r)
    if t is ndarray:
        return r
    if t is _np.ndarray:
        return r.view(ndarray)
    if isinstance(r, _np.ndarray):
        return r.view(ndarray)
    if isinstance(r, _np.generic):
        return _zero_d(r)
    if t is tuple:
        return tuple(wrap(x) for x in r)
    if t is list:
        return [wrap(x) for x in r]
    return r


def wrapped(func):
    """``func`` returning stand-in arrays."""
    def call(*args, **kwargs):
        return wrap(func(*args, **kwargs))
    call.__name__ = getattr(func, "__name__", "function")
    call.__qualname__ = call.__name__
    call.__doc__ = getattr(func, "__doc__", None)
    call.__wrapped__ = func
    return call


class ufunc:
    """A NumPy ufunc returning stand-in arrays (``cp.maximum``, ``cp.add.reduce`` ...)."""

    def __init__(self, uf):
        self._uf = uf
        self.__name__ = uf.__name__
        self.__doc__ = uf.__doc__

    def __call__(self, *args, **kwargs):
        return wrap(self._uf(*args, **kwargs))

    def __getattr__(self, name):
        attr = getattr(self._uf, name)
        return wrapped(attr) if callable(attr) else attr

    def __repr__(self):
        return f"<ufunc '{self.__name__}' (CPU)>"


class RandomState:
    """``cupy.random.RandomState``: NumPy's, with CuPy's ``dtype`` keyword on its float samplers
    (``standard_normal(shape, dtype=cp.float32)``), returning stand-in arrays.  The numbers are
    NumPy's sequence, not CuPy's (the GPU draws different ones for the same seed)."""

    _FLOAT = ("standard_normal", "random_sample", "random", "normal", "uniform", "rand", "randn")

    def __init__(self, seed=None, method=None):
        self._rs = _np.random.RandomState(seed)

    def __getattr__(self, name):
        attr = getattr(self._rs, name)
        if not callable(attr):
            return attr
        if name not in self._FLOAT:
            return wrapped(attr)

        def call(*args, dtype=None, **kwargs):
            r = attr(*args, **kwargs)
            if dtype is not None:
                r = _np.asarray(r).astype(dtype, copy=False)
            return wrap(r)
        call.__name__ = name
        return call


class proxy_module:
    """A NumPy/SciPy module seen through :func:`wrap` (``cp.random``, ``cupyx.scipy.ndimage`` ...)."""

    def __init__(self, name, module):
        self.__name__ = name
        self._module = module

    def __getattr__(self, name):
        attr = getattr(self._module, name)
        if isinstance(attr, type) or not callable(attr):
            val = attr
        elif isinstance(attr, _np.ufunc):
            val = ufunc(attr)
        else:
            val = wrapped(attr)
        setattr(self, name, val)
        return val
