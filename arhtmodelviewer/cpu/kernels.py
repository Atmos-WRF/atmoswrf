"""CUDA kernels on the CPU: CuPy's kernel API (RawModule, RawKernel, the driver Module ArWen's
direct-NVRTC route loads, ElementwiseKernel) over C++ compiled from the unchanged CUDA source.

A module's source gets ``runtime/cuda_cpu.h`` in front (CUDA's device language as C++) and one
``CUDACPU_EXPORT(kernel, fibers)`` line per ``__global__`` function at the end, and clang (the
``ziglang`` wheel) builds it into a DLL, cached by content in ``runtime/cache/cpu-kernels``.  A
launch packs the arguments the way ``cuLaunchKernel`` takes them (a pointer to each argument's
bytes) and the DLL runs the grid's blocks on the shared thread pool (``cudacpu_rt.dll``).

Arithmetic: always ``-ffp-contract=off`` (no fused multiply-adds; NVRTC fuses where it likes, so
GPU and CPU agree to rounding, not bit for bit) and flush-to-zero exactly where NVRTC would
(CuPy's RawModule / ElementwiseKernel add ``-ftz=true``; the direct route and ``--ftz=false``
keep subnormals).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import CACHE, PROJECT, RUNTIME, threads

log = logging.getLogger(__name__)

IMAGE_MAGIC = b"CUDACPU-IMAGE-1\n"
#: bump when the CUDA -> C++ translation or the kernel scan changes (part of every cache key)
TRANSLATOR = "3"
_HEADER = RUNTIME / "cuda_cpu.h"
_RT_SOURCE = RUNTIME / "cudacpu_rt.cpp"
_LOCK = threading.RLock()
_COMPILES = threading.local()


class CompileError(RuntimeError):
    pass


# ------------------------------------------------------------------ the compiler
def _zig_env() -> dict:
    env = dict(os.environ)
    zc = str(PROJECT / "runtime" / "cache" / "zig")
    env.setdefault("ZIG_GLOBAL_CACHE_DIR", zc)
    env.setdefault("ZIG_LOCAL_CACHE_DIR", zc)
    return env


def _march() -> str:
    return os.environ.get("ARHTMODELVIEWER_CPU_MARCH", "native")


_COMPILER_ID = None
#: what NVRTC defines for device code (cuda_cpu.h defines the same after the C++ headers)
_CUDA_PRELUDE = "#define __CUDACC__ 1\n#define __CUDACC_RTC__ 1\n#define __CUDA_ARCH__ 860\n"


def _compiler_id() -> str:
    """zig's version plus this file's and the header's contents: part of every cache key."""
    global _COMPILER_ID
    if _COMPILER_ID is None:
        try:
            import ziglang  # noqa: F401
        except ImportError as exc:
            raise CompileError("the CPU backend needs the 'ziglang' package (a C++ compiler): "
                               "runtime\\venv\\Scripts\\python.exe -m pip install ziglang==0.16.0") from exc
        out = subprocess.run([sys.executable, "-m", "ziglang", "version"], capture_output=True, text=True,
                             env=_zig_env())
        h = hashlib.sha256()
        h.update(out.stdout.strip().encode())
        h.update(_HEADER.read_bytes())
        h.update(TRANSLATOR.encode())
        h.update(_march().encode())
        _COMPILER_ID = h.hexdigest()[:16]
    return _COMPILER_ID


def _clang(args: list[str], what: str, timeout: float = 1800.0) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "ziglang", "c++"] + args
    r = subprocess.run(cmd, capture_output=True, text=True, env=_zig_env(), timeout=timeout)
    if r.returncode != 0:
        msg = "\n".join(l for l in r.stderr.splitlines() if "error" in l)[:4000] or r.stderr[-4000:]
        raise CompileError(f"compiling {what} for the CPU failed:\n{msg}")
    return r


def _build_dll(source: str, key: str, what: str, extra: list[str]) -> Path:
    """Compile ``source`` (complete C++) into ``CACHE/<key>.dll`` unless it is there already.
    Another process building the same key is waited for."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dll = CACHE / f"{key}.dll"
    if dll.exists():
        return dll
    lock = CACHE / f"{key}.lock"
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if dll.exists():
                return dll
            try:
                if time.time() - lock.stat().st_mtime > 1800:
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.25)
    try:
        if dll.exists():
            return dll
        with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
            src = Path(tmp) / "module.cpp"
            src.write_text(source, encoding="utf-8")
            out = Path(tmp) / "module.dll"
            started = time.time()
            _clang(["-std=c++17", "-O2", f"-march={_march()}", "-ffp-contract=off", "-fno-math-errno",
                    "-w", "-shared", f"-I{RUNTIME}"] + extra + [str(src), "-o", str(out)], what)
            os.replace(out, dll)
            log.info("compiled %s for the CPU in %.1f s", what, time.time() - started)
            n = getattr(_COMPILES, "n", 0)
            _COMPILES.n = n + 1
        return dll
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
        if time.time() - t0 > 600:
            log.warning("waited %.0f s for %s", time.time() - t0, what)


# ------------------------------------------------------------------ the thread pool
_RT = None
_PARALLEL = None


def runtime():
    """The process's thread-pool DLL (built once per machine)."""
    global _RT, _PARALLEL
    with _LOCK:
        if _RT is None:
            src = _RT_SOURCE.read_text(encoding="utf-8")
            key = "rt-" + hashlib.sha256((_compiler_id() + src).encode()).hexdigest()[:24]
            path = _build_dll(src, key, "the CPU thread pool", [])
            lib = ctypes.CDLL(str(path))
            lib.cc_rt_init.argtypes = [ctypes.c_int]
            lib.cc_rt_init.restype = ctypes.c_int
            lib.cc_rt_parallel.argtypes = [ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
            lib.cc_rt_parallel.restype = None
            lib.cc_rt_init(threads())
            _PARALLEL = ctypes.cast(lib.cc_rt_parallel, ctypes.c_void_p).value
            _RT = lib
        return _RT


# ------------------------------------------------------------------ CUDA source -> C++
_ASM_OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_ASM_RN = re.compile(r'asm\s*(?:volatile\s*)?\(\s*"(add|sub|mul|div)\.rn\.f(32|64)\s+%0\s*,\s*%1\s*,\s*%2\s*;"\s*'
                     r':\s*"=[fd]"\s*\(\s*(\w+)\s*\)\s*:\s*"[fd]"\s*\(\s*([^()]+?)\s*\)\s*,\s*"[fd]"\s*\(\s*([^()]+?)\s*\)\s*\)')
_EXTERN_SHARED = re.compile(r"extern\s+__shared__\s+(?:__align__\s*\(\s*\d+\s*\)\s+)?"
                            r"((?:unsigned\s+|signed\s+|const\s+)*[A-Za-z_]\w*)\s+(?:__align__\s*\(\s*\d+\s*\)\s+)?"
                            r"([A-Za-z_]\w*)\s*\[\s*\]\s*;")
_GLOBAL_FN = re.compile(r"__global__\s+(?:__launch_bounds__\s*\([^()]*\)\s*)?(?:static\s+|inline\s+)*void\s+"
                        r"(?:__launch_bounds__\s*\([^()]*\)\s*)?([A-Za-z_]\w*)\s*\(")
_SYNC = re.compile(r"\b(__syncthreads|__syncthreads_count|__syncthreads_or|__syncthreads_and|__syncwarp|__shfl\w*_sync|__ballot_sync|__any_sync|__all_sync)\b")
_FN_DEF = re.compile(r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{")
#: kernels whose warp exchange is an optimisation with an exact per-lane fallback (ArWen's
#: positive-definite advection evaluates a warp edge's neighbour itself), so lanes may run alone
_LANE_SAFE = {"pd_flux_limit", "pd_limit_fluxes"}


def _cpp_source(code: str) -> str:
    """The CUDA source as C++ for cuda_cpu.h (inline PTX and extern shared rewritten)."""
    code = _ASM_RN.sub(lambda m: f"{m.group(3)} = ({m.group(4)}) {_ASM_OPS[m.group(1)]} ({m.group(5)})", code)
    if re.search(r"\basm\s*(?:volatile\s*)?\(", code):
        bad = re.search(r"\basm\s*(?:volatile\s*)?\([^;]*;", code).group(0)
        raise CompileError(f"inline PTX the CPU backend does not translate: {bad[:200]}")
    code = _EXTERN_SHARED.sub(lambda m: f"{m.group(1)}* {m.group(2)} = ({m.group(1)}*)__cc_dyn_smem();", code)
    return code


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _bodies(text: str) -> dict[str, list[str]]:
    """name -> bodies of every function defined at file scope (brace matched)."""
    out: dict[str, list[str]] = {}
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            if depth == 0:
                head = text[max(0, i - 4000):i + 1]
                m = None
                for m in _FN_DEF.finditer(head):
                    pass
                start = i
                d = 0
                j = i
                while j < n:
                    if text[j] == "{":
                        d += 1
                    elif text[j] == "}":
                        d -= 1
                        if d == 0:
                            break
                    j += 1
                if m is not None and m.end() == len(head):
                    out.setdefault(m.group(1), []).append(text[start:j + 1])
                elif m is None or m.end() != len(head):
                    # a namespace / extern "C" block / struct: look inside it
                    inner = _bodies(text[start + 1:j])
                    for k, v in inner.items():
                        out.setdefault(k, []).extend(v)
                i = j + 1
                continue
        i += 1
    return out


def _kernels(code: str, options: tuple) -> dict[str, bool]:
    """``__global__`` functions of the module (after preprocessing) -> whether a launch has to
    run its block's threads as fibers (it, or a function it calls, waits for the block)."""
    defines = [o for o in options if o.startswith("-D")]
    includes = [o for o in options if o.startswith("-I")]
    r = subprocess.run([sys.executable, "-m", "ziglang", "c++", "-std=c++17", "-E", "-P", "-x", "c++",
                        "-D__launch_bounds__(...)=", "-D__align__(n)=", "-w",
                        *defines, *includes, "-"],
                       input=_CUDA_PRELUDE + code, capture_output=True, text=True, env=_zig_env())
    pre = r.stdout if r.returncode == 0 and r.stdout.strip() else _strip_comments(code)
    names = []
    for m in _GLOBAL_FN.finditer(pre):
        head = pre[max(0, m.start() - 300):m.start()]
        if re.search(r"template\s*<[^;{}]*>\s*(?:static\s+)?(?:extern\s+\"C\"\s+)?$", head):
            continue
        if m.group(1) not in names:
            names.append(m.group(1))
    if not names:
        return {}
    if not _SYNC.search(pre):
        return {k: False for k in names}
    bodies = _bodies(pre)
    syncing = {f for f, bs in bodies.items() if any(_SYNC.search(b) for b in bs)}
    changed = True
    while changed:
        changed = False
        for f, bs in bodies.items():
            if f in syncing:
                continue
            calls = re.compile(r"\b(" + "|".join(map(re.escape, syncing)) + r")\s*(?:<[^;{}()]*>\s*)?\(")
            if any(calls.search(b) for b in bs):
                syncing.add(f)
                changed = True
    out = {}
    for k in names:
        if k in _LANE_SAFE:
            out[k] = False
        elif k not in bodies:
            out[k] = True                       # not found: be safe
        else:
            out[k] = k in syncing
    return out


def _compile_options(options) -> tuple[list[str], bool | None]:
    """clang flags from NVRTC options, and the -ftz setting they ask for (None: not given)."""
    flags: list[str] = []
    ftz = None
    opts = list(options or ())
    i = 0
    while i < len(opts):
        o = str(opts[i])
        if o in ("-D", "--define-macro", "-I", "--include-path") and i + 1 < len(opts):
            flags.append(("-D" if o in ("-D", "--define-macro") else "-I") + str(opts[i + 1]))
            i += 2
            continue
        if o.startswith("--define-macro="):
            flags.append("-D" + o.split("=", 1)[1])
        elif o.startswith("-D") or o.startswith("-I"):
            flags.append(o)
        elif o.startswith("--include-path="):
            flags.append("-I" + o.split("=", 1)[1])
        elif re.fullmatch(r"-{1,2}ftz=(true|false)", o):
            ftz = o.endswith("true")
        elif o in ("--use_fast_math", "-use_fast_math"):
            ftz = True if ftz is None else ftz
        i += 1
    return flags, ftz


class _Image:
    """A compiled module: the DLL, its launch functions and which need fibers."""

    def __init__(self, code: str, options=(), what: str = "a kernel module"):
        flags, _ = _compile_options(options)
        cpp = _cpp_source(code)
        key_src = "\n".join([_compiler_id(), " ".join(flags), cpp])
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:32]
        meta = CACHE / f"{key}.json"
        kern = None
        if meta.exists() and (CACHE / f"{key}.dll").exists():
            try:
                kern = json.loads(meta.read_text())
            except ValueError:
                kern = None
        if kern is None:
            kern = _kernels(code, tuple(flags))
        exports = "\n".join(f"CUDACPU_EXPORT({k}, {'true' if fib else 'false'})" for k, fib in kern.items())
        full = '#include "cuda_cpu.h"\n#line 1 "kernel.cu"\n' + cpp + "\n\n" + exports + "\n"
        path = _build_dll(full, key, what, flags)
        if not meta.exists():
            meta.write_text(json.dumps(kern))
        runtime()
        self.lib = ctypes.CDLL(str(path))
        self.kernels = kern
        self.path = path
        self._functions: dict[str, ctypes._CFuncPtr] = {}

    def launcher(self, name: str):
        f = self._functions.get(name)
        if f is None:
            if name not in self.kernels:
                raise AttributeError(f"no __global__ function {name!r} in the module "
                                     f"(has {sorted(self.kernels)[:20]})")
            f = self.lib["__cc_launch_" + name]
            f.argtypes = [ctypes.c_void_p]
            f.restype = ctypes.c_int
            self._functions[name] = f
        return f


# ------------------------------------------------------------------ launching
class _Launch(ctypes.Structure):
    _fields_ = [("grid", ctypes.c_uint * 3), ("block", ctypes.c_uint * 3), ("args", ctypes.c_void_p),
                ("shmem", ctypes.c_ulonglong), ("parallel", ctypes.c_void_p), ("ftz", ctypes.c_int)]


def _dims(d) -> tuple[int, int, int]:
    if isinstance(d, (int, np.integer)):
        return int(d), 1, 1
    d = tuple(int(x) for x in d)
    return (d + (1, 1, 1))[:3]


def _pointer(a) -> int:
    return a.__array_interface__["data"][0]


def pack_args(args) -> tuple[ctypes.Array, list]:
    """(void* array, keep-alive list) for cuLaunchKernel-style arguments, CuPy's conversions:
    arrays pass their data pointer, NumPy scalars and 0-d host (NumPy, not stand-in) arrays their
    bytes -- a structured one is a C struct passed by value, as CuPy does (the radar's SweepArgs) --
    Python int/float/bool as int64/float64/bool."""
    n = len(args)
    slots = ctypes.create_string_buffer(16 * max(n, 1))
    ptrs = (ctypes.c_void_p * max(n, 1))()
    base = ctypes.addressof(slots)
    keep = [slots]
    for k, a in enumerate(args):
        off = 16 * k
        by_value = isinstance(a, np.generic) or (type(a) is np.ndarray and a.ndim == 0)
        if isinstance(a, np.ndarray) and not by_value:
            struct.pack_into("<Q", slots, off, _pointer(a))
        elif by_value:
            b = a.tobytes()
            if len(b) <= 16:
                slots[off:off + len(b)] = b
            else:
                big = ctypes.create_string_buffer(b, len(b))
                keep.append(big)
                ptrs[k] = ctypes.addressof(big)
                continue
        elif isinstance(a, bool):
            struct.pack_into("<?", slots, off, a)
        elif isinstance(a, int):
            struct.pack_into("<Q" if a >= 1 << 63 else "<q", slots, off, a)
        elif isinstance(a, float):
            struct.pack_into("<d", slots, off, a)
        elif isinstance(a, complex):
            struct.pack_into("<dd", slots, off, a.real, a.imag)
        elif a is None:
            pass
        elif hasattr(a, "ptr"):
            struct.pack_into("<Q", slots, off, int(a.ptr))
        else:
            raise TypeError(f"kernel argument {k}: unsupported type {type(a).__name__}")
        ptrs[k] = base + off
    keep.append(ptrs)
    return ptrs, keep


_FUNC_IDS: dict[int, "Function"] = {}
DEFAULT_ATTRIBUTES = {"max_threads_per_block": 1024, "shared_size_bytes": 0, "const_size_bytes": 0,
                      "local_size_bytes": 0, "num_regs": 32, "ptx_version": 86, "binary_version": 86,
                      "cache_mode_ca": 0, "max_dynamic_shared_size_bytes": 48 * 1024,
                      "preferred_shared_memory_carveout": -1}


class Function:
    """One kernel: ``f(grid, block, args, shared_mem=0, stream=None)`` like CuPy's Function."""

    def __init__(self, image: _Image, name: str, ftz: bool):
        self._image = image
        self._fn = image.launcher(name)
        self.name = self.kernel_name = name
        self.ftz = ftz
        self.fibers = image.kernels.get(name, False)
        self.attributes = dict(DEFAULT_ATTRIBUTES)
        self.ptr = id(self)
        _FUNC_IDS[self.ptr] = self

    def __call__(self, grid, block, args, shared_mem=0, stream=None, enable_cooperative_groups=False):
        g, b = _dims(grid), _dims(block)
        ptrs, keep = pack_args(tuple(args))
        L = _Launch((ctypes.c_uint * 3)(*g), (ctypes.c_uint * 3)(*b), ctypes.addressof(ptrs),
                    int(shared_mem or 0), _PARALLEL, 1 if self.ftz else 0)
        rc = self._fn(ctypes.byref(L))
        del keep
        if rc:
            raise RuntimeError(f"CPU kernel {self.name} waits for its block (__syncthreads or a warp "
                               "vote) but was built to run its threads one by one")

    def __getattr__(self, name):
        attrs = self.__dict__.get("attributes", {})
        if name in attrs:
            return attrs[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in DEFAULT_ATTRIBUTES and "attributes" in self.__dict__:
            self.attributes[name] = value
        else:
            object.__setattr__(self, name, value)


# ------------------------------------------------------------------ CuPy's module classes
class RawModule:
    """cupy.RawModule: compiled on first use.  CuPy appends -ftz=true to NVRTC's options; so does
    this, unless the options or ArWen's subnormal marker keep subnormals."""

    def __init__(self, code=None, *, path=None, options=(), backend="nvrtc", translate_cucomplex=False,
                 enable_cooperative_groups=False, name_expressions=None, jitify=False):
        if name_expressions:
            raise NotImplementedError("the CPU backend does not instantiate templated kernels (name_expressions)")
        if code is None and path is not None:
            code = Path(path).read_text(encoding="utf-8")
        self.code = code
        self.options = tuple(options or ())
        self.backend = backend
        self.name_expressions = name_expressions
        self.file_path = path
        _, ftz = _compile_options(self.options)
        if ftz is None:
            ftz = "GPUWM_RETAIN_SUBNORMALS" not in (code or "")
        self._ftz = ftz
        self._image = None
        self._functions: dict[str, Function] = {}

    def compile(self, log_stream=None):
        if self._image is None:
            self._image = _Image(self.code, self.options, what=_describe(self.code))
        return self

    @property
    def module(self):
        return self.compile()

    def get_function(self, name):
        f = self._functions.get(name)
        if f is None:
            self.compile()
            f = self._functions[name] = Function(self._image, name, self._ftz)
        return f

    def get_global(self, name):
        raise NotImplementedError("the CPU backend has no device globals (get_global)")


class RawKernel:
    """cupy.RawKernel: one kernel of one source."""

    def __init__(self, code, name, options=(), backend="nvrtc", translate_cucomplex=False,
                 enable_cooperative_groups=False, jitify=False):
        self.code, self.name, self.options = code, name, tuple(options or ())
        self._module = RawModule(code=code, options=self.options)

    @property
    def kernel(self):
        return self._module.get_function(self.name)

    def compile(self, log_stream=None):
        self._module.compile()

    def __call__(self, grid, block, args, **kwargs):
        return self.kernel(grid, block, args, **kwargs)

    @property
    def attributes(self):
        return self.kernel.attributes

    def __getattr__(self, name):
        if name in DEFAULT_ATTRIBUTES:
            return self.kernel.attributes[name]
        raise AttributeError(name)


def make_image(source: str, options=()) -> bytes:
    """What ``compile_using_nvrtc`` returns on the CPU: the source and options, compiled when a
    Module loads it.  NVRTC's own default keeps subnormals (no -ftz unless asked)."""
    head = json.dumps({"options": [str(o) for o in (options or ())]})
    return IMAGE_MAGIC + head.encode() + b"\n" + source.encode("utf-8")


class Module:
    """cupy.cuda.function.Module: ``load(image)`` then ``get_function(name)``."""

    def __init__(self):
        self._image = None
        self._ftz = False
        self._functions: dict[str, Function] = {}

    def load(self, image):
        if isinstance(image, str):
            image = image.encode()
        if not bytes(image).startswith(IMAGE_MAGIC):
            raise ValueError("the CPU backend can only load images from its own compile_using_nvrtc")
        head, _, src = bytes(image)[len(IMAGE_MAGIC):].partition(b"\n")
        options = json.loads(head)["options"]
        _, ftz = _compile_options(options)
        source = src.decode("utf-8")
        self._ftz = bool(ftz)
        self._image = _Image(source, options, what=_describe(source))

    def load_file(self, path):
        self.load(Path(path).read_bytes())

    def get_function(self, name):
        f = self._functions.get(name)
        if f is None:
            f = self._functions[name] = Function(self._image, name, self._ftz)
        return f

    def get_global_var(self, name):
        raise NotImplementedError("the CPU backend has no device globals")

    get_global = get_global_var


def _describe(code: str) -> str:
    names = _GLOBAL_FN.findall(code or "")
    return f"kernels {', '.join(names[:3])}{', ...' if len(names) > 3 else ''}" if names else "a kernel module"


# ------------------------------------------------------------------ ElementwiseKernel
_CTYPES = {"float32": "float", "float64": "double", "int32": "int", "int64": "long long",
           "uint32": "unsigned int", "uint64": "unsigned long long", "int16": "short", "uint16": "unsigned short",
           "int8": "signed char", "uint8": "unsigned char", "bool": "bool", "complex64": "float2",
           "complex128": "double2"}
_PARAM = re.compile(r"^\s*(raw\s+)?(const\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*$")
_MAXP = 32


class _RawD(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p), ("shape", ctypes.c_longlong * 8), ("strides", ctypes.c_longlong * 8),
                ("n", ctypes.c_longlong), ("ndim", ctypes.c_int), ("contiguous", ctypes.c_int)]


class _EwCtx(ctypes.Structure):
    _fields_ = [("n", ctypes.c_longlong), ("shape", ctypes.c_longlong * 8), ("ndim", ctypes.c_int),
                ("fast", ctypes.c_int), ("ftz", ctypes.c_int), ("pad", ctypes.c_int),
                ("ptr", ctypes.c_void_p * _MAXP), ("st", (ctypes.c_longlong * 8) * _MAXP),
                ("scal", ctypes.c_void_p * _MAXP), ("raw", _RawD * _MAXP)]


_EW_PRELUDE = r"""
struct __cc_rawd { char* p; long long shape[8]; long long strides[8]; long long n; int ndim; int contiguous; };
template <class T> struct __cc_rawv {
    const __cc_rawd* d;
    __forceinline__ T& operator[](long long i) const {
        if (d->contiguous) return ((T*)d->p)[i];
        long long off = 0;
        for (int k = d->ndim - 1; k >= 0; --k) { long long q = i % d->shape[k]; i /= d->shape[k]; off += q * d->strides[k]; }
        return *(T*)(d->p + off);
    }
    __forceinline__ long long size() const { return d->n; }
    __forceinline__ T* data() const { return (T*)d->p; }
};
struct __cc_ewctx {
    long long n; long long shape[8]; int ndim; int fast; int ftz; int pad;
    char* ptr[32]; long long st[32][8]; void* scal[32]; __cc_rawd raw[32];
};
"""


def _parse_params(text: str):
    out = []
    for part in [p for p in (text or "").split(",") if p.strip()]:
        m = _PARAM.match(part)
        if not m:
            raise ValueError(f"cannot read kernel parameter {part!r}")
        raw, _const, typ, name = m.groups()
        out.append((typ, name, bool(raw)))
    return out


def _is_dtype(name: str) -> bool:
    if name in _CTYPES:
        return True
    try:
        np.dtype(name)
        return name not in ("T", "U", "S")
    except TypeError:
        return False


class ElementwiseKernel:
    """cupy.ElementwiseKernel: ``operation`` once per element ``i`` of the broadcast shape (or of
    ``size``), inputs by value, outputs by reference, ``raw`` arrays indexed by hand."""

    def __init__(self, in_params, out_params, operation, name="kernel", reduce_dims=True, preamble="",
                 no_return=False, return_tuple=False, loop_prep="", after_loop="", options=(), **kwargs):
        self.in_params = _parse_params(in_params)
        self.out_params = _parse_params(out_params)
        self.nin, self.nout = len(self.in_params), len(self.out_params)
        self.nargs = self.nin + self.nout
        if self.nargs > _MAXP:
            raise ValueError("too many kernel parameters for the CPU backend")
        self.operation, self.name, self.preamble = operation, name, preamble
        self.loop_prep, self.after_loop = loop_prep, after_loop
        self.no_return, self.return_tuple = no_return, return_tuple
        self.options = tuple(options or ())
        _, ftz = _compile_options(self.options)
        self.ftz = True if ftz is None else ftz
        self._dlls: dict[tuple, tuple] = {}

    def _resolve(self, args):
        types: dict[str, np.dtype] = {}
        params = self.in_params + self.out_params
        for (typ, _, _), a in zip(params, args):
            if not _is_dtype(typ) and isinstance(a, (np.ndarray, np.generic)) and typ not in types:
                types[typ] = np.dtype(a.dtype)
        for (typ, _, _), a in zip(params, args):
            if not _is_dtype(typ) and typ not in types and a is not None:
                types[typ] = np.result_type(a)
        return types

    def __call__(self, *args, size=None, block_size=None, stream=None):
        if len(args) not in (self.nin, self.nargs):
            raise TypeError(f"{self.name}: takes {self.nin} or {self.nargs} arguments ({len(args)} given)")
        from .arrays import ndarray as cpu_ndarray
        args = list(args) + [None] * (self.nargs - len(args))
        params = self.in_params + self.out_params
        types = self._resolve(args)

        def dtype_of(typ):
            return np.dtype(typ) if _is_dtype(typ) else types[typ]

        shapes = [a.shape for (t, n, raw), a in zip(params, args) if not raw and isinstance(a, np.ndarray)]
        shape = tuple(np.broadcast_shapes(*shapes)) if shapes else (() if size is None else (int(size),))
        for k in range(self.nin, self.nargs):
            if args[k] is None:
                typ, name, raw = params[k]
                if raw and size is None:
                    raise ValueError(f"{self.name}: raw output {name} has to be given")
                args[k] = np.empty(shape, dtype=dtype_of(typ)).view(cpu_ndarray)
        n = int(size) if size is not None else int(np.prod(shape, dtype=np.int64))
        ndim = len(shape)
        if ndim > 8:
            raise ValueError("the CPU ElementwiseKernel handles up to 8 dimensions")
        ctx = _EwCtx()
        ctx.n = n
        ctx.ndim = ndim
        for d, s in enumerate(shape):
            ctx.shape[d] = s
        ctx.ftz = 1 if self.ftz else 0
        keep = []
        fast = True
        sig = []
        for k, ((typ, name, raw), a) in enumerate(zip(params, args)):
            dt = dtype_of(typ)
            if raw:
                if not isinstance(a, np.ndarray):
                    raise TypeError(f"{self.name}: raw argument {name} must be an array")
                if a.dtype != dt:
                    raise TypeError(f"{self.name}: {name} is {a.dtype}, the kernel wants {dt}")
                r = ctx.raw[k]
                r.p = _pointer(a)
                r.n = a.size
                r.ndim = a.ndim
                r.contiguous = 1 if a.flags.c_contiguous else 0
                for d in range(a.ndim):
                    r.shape[d] = a.shape[d]
                    r.strides[d] = a.strides[d]
                sig.append(("raw", dt.str))
            elif isinstance(a, np.ndarray):
                if k >= self.nin and a.shape != shape:
                    raise ValueError(f"{self.name}: output {name} has shape {a.shape}, not {shape}")
                if a.dtype != dt:
                    if k >= self.nin:
                        raise TypeError(f"{self.name}: output {name} is {a.dtype}, the kernel wants {dt}")
                    a = a.astype(dt)
                    keep.append(a)
                b = np.broadcast_to(a, shape) if a.shape != shape else a
                ctx.ptr[k] = _pointer(b)
                for d in range(ndim):
                    ctx.st[k][d] = b.strides[d]
                if not (a.shape == shape and a.flags.c_contiguous):
                    fast = False
                sig.append(("arr", dt.str))
            else:
                v = np.asarray(a, dtype=dt)
                keep.append(v)
                ctx.scal[k] = _pointer(v)
                sig.append(("val", dt.str))
        ctx.fast = 1 if fast else 0
        key = (tuple(sig), ndim)
        entry = self._dlls.get(key)
        if entry is None:
            entry = self._dlls[key] = self._build(params, sig, types)
        direct, chunk = entry
        if n > 0:
            if n >= 32768:
                runtime().cc_rt_parallel(n, chunk, ctypes.byref(ctx), max(4096, n // (8 * threads())))
            else:
                direct(ctypes.byref(ctx), 0, n, 0)
        del keep
        outs = args[self.nin:]
        if self.no_return:
            return None
        if self.nout == 1 and not self.return_tuple:
            return outs[0]
        return tuple(outs)

    def _build(self, params, sig, types):
        lines = ['#include "cuda_cpu.h"', _EW_PRELUDE, self.preamble or ""]
        for tv, dt in types.items():
            lines.append(f"typedef {_CTYPES[np.dtype(dt).name]} {tv};")
        decl_fast, decl_slow, once = [], [], []
        for k, ((typ, name, raw), (kind, dts)) in enumerate(zip(params, sig)):
            ct = _CTYPES[np.dtype(dts).name]
            out = k >= self.nin
            if kind == "raw":
                once.append(f"const __cc_rawv<{ct}> {name}{{&__cc_c->raw[{k}]}};")
            elif kind == "val":
                once.append(f"const {ct} {name} = *({ct}*)__cc_c->scal[{k}];")
            else:
                ref = f"{ct}&" if out else f"const {ct}"
                decl_fast.append(f"{ref} {name} = (({ct}*)__cc_c->ptr[{k}])[i];")
                decl_slow.append(f"{ref} {name} = *({ct}*)(__cc_c->ptr[{k}] + __cc_off(__cc_c, {k}, i));")
        body = self.operation
        lines.append("""
static __forceinline__ long long __cc_off(const __cc_ewctx* __cc_c, int __cc_k, long long __cc_i) {
    long long __cc_o = 0;
    for (int __cc_d = __cc_c->ndim - 1; __cc_d >= 0; --__cc_d) { long long __cc_q = __cc_i % __cc_c->shape[__cc_d]; __cc_i /= __cc_c->shape[__cc_d]; __cc_o += __cc_q * __cc_c->st[__cc_k][__cc_d]; }
    return __cc_o;
}""")
        lines.append(f"""
static void __cc_ew_body(__cc_ewctx* __cc_c, long long __cc_b, long long __cc_e) {{
    const __cc_index _ind{{__cc_c->n}};
    {' '.join(once)}
    {self.loop_prep or ''}
    if (__cc_c->fast) {{
        for (long long i = __cc_b; i < __cc_e; ++i) {{ {' '.join(decl_fast)} {{ {body}; }} }}
    }} else {{
        for (long long i = __cc_b; i < __cc_e; ++i) {{ {' '.join(decl_slow)} {{ {body}; }} }}
    }}
    {self.after_loop or ''}
}}
extern "C" __declspec(dllexport) void __cc_ew_chunk(void* vc, long long b, long long e, int) {{
    __cc_ewctx* __cc_c = (__cc_ewctx*)vc;
    const unsigned csr = _mm_getcsr();
    _mm_setcsr(__cc_c->ftz ? (csr | 0x8040u) : (csr & ~0x8040u));
    __cc_ew_body(__cc_c, b, e);
    _mm_setcsr(csr);
}}
extern "C" __declspec(dllexport) void __cc_ew_chunk_direct(void* vc, long long b, long long e, int w) {{
    __cc_ew_chunk(vc, b, e, w);
}}
""")
        src = "\n".join(lines)
        flags, _ = _compile_options(self.options)
        key = "ew-" + hashlib.sha256(("\n".join([_compiler_id(), " ".join(flags), src])).encode()).hexdigest()[:28]
        path = _build_dll(src, key, f"elementwise kernel {self.name}", flags)
        runtime()
        lib = ctypes.CDLL(str(path))
        direct = lib["__cc_ew_chunk_direct"]
        direct.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
        direct.restype = None
        chunk = ctypes.cast(lib["__cc_ew_chunk"], ctypes.c_void_p).value
        self._libs = getattr(self, "_libs", []) + [lib]
        return direct, chunk


class ReductionKernel:
    """cupy.ReductionKernel, evaluated with NumPy (map, then reduce over the axes)."""

    def __init__(self, in_params, out_params, map_expr, reduce_expr, post_map_expr, identity, name="reduce_kernel",
                 reduce_type=None, reduce_dims=True, preamble="", options=()):
        raise NotImplementedError(f"ReductionKernel {name}: not available on the CPU backend yet")
