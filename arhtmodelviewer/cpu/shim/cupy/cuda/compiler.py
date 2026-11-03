"""cupy.cuda.compiler on the CPU.  ``compile_using_nvrtc`` (ArWen's direct-NVRTC route) returns an
image that :class:`cupy.cuda.function.Module` compiles for the processor when it loads it."""
from __future__ import annotations

from arhtmodelviewer.cpu.kernels import CompileError, make_image


class CompileException(CompileError):
    pass


def compile_using_nvrtc(source, options=(), arch=None, filename="kern.cu", name_expressions=None,
                        log_stream=None, cache_in_memory=False, jitify=False, method="ptx"):
    if name_expressions:
        raise NotImplementedError("the CPU backend does not instantiate templated kernels (name_expressions)")
    return make_image(source, options), None


def _get_arch() -> str:
    return "86"


def _compile_with_cache_cuda(source, options, arch=None, cache_dir=None, extra_source=None, backend="nvrtc",
                             *args, **kwargs):
    from .function import Module
    m = Module()
    m.load(make_image(source, options))
    return m


class _NVRTCProgram:
    def __init__(self, src, name="default_program", headers=(), include_names=(), name_expressions=None,
                 method="ptx"):
        self.src, self.name = src, name

    def compile(self, options=(), log_stream=None):
        return make_image(self.src, options), None
