"""cupy.cuda.memory on the CPU: NumPy owns the memory, so pools are bookkeeping that frees nothing."""
from __future__ import annotations

import ctypes

from arhtmodelviewer.cpu.arrays import MemoryPointer  # noqa: F401


class OutOfMemoryError(MemoryError):
    def __init__(self, size=0, total=0, limit=0):
        super().__init__(f"out of memory allocating {size} bytes")


class BaseMemory:
    def __init__(self, size=0):
        self.size = int(size)
        self._buf = (ctypes.c_char * max(self.size, 1))()
        self.ptr = ctypes.addressof(self._buf)
        self.device_id = 0


Memory = BaseMemory


class UnownedMemory:
    def __init__(self, ptr, size, owner, device_id=None):
        self.ptr, self.size, self.owner, self.device_id = int(ptr), int(size), owner, 0


class MemoryPool:
    def __init__(self, allocator=None):
        self._limit = 0

    def malloc(self, size):
        return MemoryPointer(BaseMemory(size), 0)

    def used_bytes(self) -> int:
        return 0

    def free_bytes(self) -> int:
        return 0

    def total_bytes(self) -> int:
        return 0

    def n_free_blocks(self) -> int:
        return 0

    def free_all_blocks(self, stream=None):
        pass

    def free_all_free(self):
        pass

    def set_limit(self, size=None, fraction=None):
        self._limit = int(size or 0)

    def get_limit(self) -> int:
        return self._limit


class PinnedMemoryPointer:
    """Host memory with a buffer interface (np.frombuffer works on it) and a ``ptr``."""

    def __init__(self, size):
        self._buf = bytearray(max(int(size), 1))
        self.size = int(size)
        self.ptr = ctypes.addressof((ctypes.c_char * len(self._buf)).from_buffer(self._buf))

    def __buffer__(self, flags):
        return memoryview(self._buf)[:self.size]

    def __len__(self):
        return self.size


class PinnedMemoryPool:
    def __init__(self, allocator=None):
        pass

    def malloc(self, size):
        return PinnedMemoryPointer(size)

    def free_all_blocks(self):
        pass

    def used_bytes(self) -> int:
        return 0

    def n_free_blocks(self) -> int:
        return 0

    def total_bytes(self) -> int:
        return 0


def alloc_pinned_memory(size):
    return PinnedMemoryPointer(size)


def alloc(size):
    return MemoryPointer(BaseMemory(size), 0)


def malloc_managed(size):
    return alloc(size)


_DEFAULT_POOL = MemoryPool()
_PINNED_POOL = PinnedMemoryPool()
