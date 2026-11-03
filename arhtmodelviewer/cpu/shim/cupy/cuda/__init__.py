"""cupy.cuda on the CPU: one "device" (the processor), streams and events that are already done,
pinned memory that is ordinary memory."""
from __future__ import annotations

import contextlib
import threading
import time

from arhtmodelviewer.cpu.arrays import MemoryPointer  # noqa: F401

from . import compiler, driver, function, memory, nvrtc, runtime  # noqa: F401
from .memory import (Memory, MemoryPool, OutOfMemoryError, PinnedMemoryPool, UnownedMemory,  # noqa: F401
                     PinnedMemoryPointer, alloc_pinned_memory)


def is_available() -> bool:
    return True


def get_device_id() -> int:
    return 0


def get_cuda_path():
    return None


def get_nvcc_path():
    return None


class Device:
    """The processor as CuPy's Device 0."""

    def __init__(self, device=None):
        self.id = int(device) if device is not None else 0

    def __int__(self):
        return self.id

    __index__ = __int__

    def __eq__(self, other):
        return isinstance(other, Device) and other.id == self.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"<CUDA Device {self.id} (CPU)>"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def use(self):
        pass

    def synchronize(self):
        pass

    @property
    def compute_capability(self) -> str:
        return "86"

    @property
    def attributes(self) -> dict:
        from arhtmodelviewer.cpu import threads
        return {"MultiProcessorCount": threads(), "MaxThreadsPerMultiProcessor": 2048,
                "MaxThreadsPerBlock": 1024, "WarpSize": 32, "MaxSharedMemoryPerBlock": 48 * 1024,
                "MaxSharedMemoryPerBlockOptin": 48 * 1024, "ComputeCapabilityMajor": 8,
                "ComputeCapabilityMinor": 6, "MaxGridDimX": 2 ** 31 - 1, "MaxGridDimY": 65535,
                "MaxGridDimZ": 65535, "MaxBlockDimX": 1024, "MaxBlockDimY": 1024, "MaxBlockDimZ": 64,
                "ClockRate": 3_000_000, "L2CacheSize": 32 * 1024 * 1024, "Integrated": 0,
                "ConcurrentManagedAccess": 1}

    @property
    def mem_info(self):
        return runtime.memGetInfo()

    @property
    def pci_bus_id(self) -> str:
        return runtime.deviceGetPCIBusId(self.id)

    @property
    def cublas_handle(self):
        raise NotImplementedError("no cuBLAS on the CPU backend")


class Event:
    def __init__(self, block=False, disable_timing=False, interprocess=False):
        self.t = None
        self.ptr = id(self)

    def record(self, stream=None):
        self.t = time.perf_counter()

    def synchronize(self):
        pass

    @property
    def done(self) -> bool:
        return True


def get_elapsed_time(start_event: Event, end_event: Event) -> float:
    if start_event.t is None or end_event.t is None:
        return 0.0
    return (end_event.t - start_event.t) * 1000.0


_CURRENT = threading.local()


class Stream:
    """A stream on the CPU: work is done when it returns."""
    null: "Stream"
    ptds: "Stream"

    def __init__(self, null=False, non_blocking=False, ptds=False, priority=None):
        self.ptr = 0 if (null or ptds) else id(self)
        self.device_id = 0

    def synchronize(self):
        pass

    @property
    def done(self) -> bool:
        return True

    def record(self, event=None):
        event = event or Event()
        event.record(self)
        return event

    def wait_event(self, event):
        pass

    def is_capturing(self) -> bool:
        return False                          # no CUDA graphs on the CPU

    def add_callback(self, callback, arg):
        callback(self, 0, arg)

    def launch_host_func(self, callback, arg):
        callback(arg)

    def use(self):
        _CURRENT.stream = self
        return self

    def __enter__(self):
        stack = _CURRENT.__dict__.setdefault("stack", [])
        stack.append(getattr(_CURRENT, "stream", None))
        _CURRENT.stream = self
        return self

    def __exit__(self, *exc):
        _CURRENT.stream = _CURRENT.stack.pop()
        return False

    def __eq__(self, other):
        return isinstance(other, Stream) and other.ptr == self.ptr

    def __hash__(self):
        return hash(self.ptr)


Stream.null = Stream(null=True)
Stream.ptds = Stream(ptds=True)
ExternalStream = Stream


def get_current_stream(device_id=None) -> Stream:
    return getattr(_CURRENT, "stream", None) or Stream.null


def get_allocator():
    return memory._DEFAULT_POOL.malloc


def set_allocator(allocator=None):
    pass


def set_pinned_memory_allocator(allocator=None):
    pass


@contextlib.contextmanager
def using_allocator(allocator=None):
    yield


class profiler:  # noqa: N801 - cupy.cuda.profiler
    @staticmethod
    def start():
        pass

    @staticmethod
    def stop():
        pass
