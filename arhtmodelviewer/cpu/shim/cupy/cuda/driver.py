"""cupy.cuda.driver on the CPU: kernel attributes for ArWen's direct-NVRTC kernels."""
from __future__ import annotations

CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
_NAMES = {0: "max_threads_per_block", 1: "shared_size_bytes", 2: "const_size_bytes", 3: "local_size_bytes",
          4: "num_regs", 5: "ptx_version", 6: "binary_version", 7: "cache_mode_ca",
          8: "max_dynamic_shared_size_bytes", 9: "preferred_shared_memory_carveout"}


class CUDADriverError(RuntimeError):
    def __init__(self, status=0):
        self.status = status
        super().__init__(f"CUDA_ERROR {status} (CPU backend)")


def funcGetAttribute(attribute: int, func: int) -> int:
    from arhtmodelviewer.cpu.kernels import _FUNC_IDS, DEFAULT_ATTRIBUTES
    f = _FUNC_IDS.get(int(func))
    name = _NAMES.get(int(attribute))
    attrs = f.attributes if f is not None else DEFAULT_ATTRIBUTES
    return int(attrs.get(name, 0))


def funcSetAttribute(func: int, attribute: int, value: int) -> None:
    from arhtmodelviewer.cpu.kernels import _FUNC_IDS
    f = _FUNC_IDS.get(int(func))
    if f is not None and int(attribute) in _NAMES:
        f.attributes[_NAMES[int(attribute)]] = int(value)


def get_build_version() -> int:
    return 12090


def driverGetVersion() -> int:
    return 12090


def ctxGetCurrent() -> int:
    return 1


def deviceGetAttribute(attrib: int, device: int = 0) -> int:
    return 0
