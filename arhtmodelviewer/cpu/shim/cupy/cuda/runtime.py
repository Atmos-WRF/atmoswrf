"""cupy.cuda.runtime on the CPU: the processor as device 0, host RAM as its memory."""
from __future__ import annotations

is_hip = False
cudaLimitStackSize = 0
cudaLimitPrintfFifoSize = 1
cudaLimitMallocHeapSize = 2
cudaLimitDevRuntimeSyncDepth = 3
cudaLimitDevRuntimePendingLaunchCount = 4
cudaLimitMaxL2FetchGranularity = 5
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4
_LIMITS = {cudaLimitStackSize: 1024, cudaLimitPrintfFifoSize: 1 << 20, cudaLimitMallocHeapSize: 8 << 20}


class CUDARuntimeError(RuntimeError):
    def __init__(self, status=0):
        self.status = status
        super().__init__(f"cudaError {status} (CPU backend)")


def _ram() -> tuple[int, int]:
    from arhtmodelviewer.cpu import ram
    return ram()


def cpu_name() -> str:
    from arhtmodelviewer.cpu import cpu_name as name
    return name()


def getDevice() -> int:
    return 0


def setDevice(device: int) -> None:
    pass


def getDeviceCount() -> int:
    return 1


def getDeviceProperties(device: int = 0) -> dict:
    from arhtmodelviewer.cpu import threads
    free, total = _ram()
    return {"name": f"CPU: {cpu_name()}".encode(), "totalGlobalMem": total, "multiProcessorCount": threads(),
            "maxThreadsPerMultiProcessor": 2048, "maxThreadsPerBlock": 1024, "warpSize": 32,
            "major": 8, "minor": 6, "sharedMemPerBlock": 48 * 1024, "sharedMemPerMultiprocessor": 100 * 1024,
            "regsPerBlock": 65536, "clockRate": 3_000_000, "memoryClockRate": 3_200_000,
            "memoryBusWidth": 128, "l2CacheSize": 32 * 1024 * 1024, "integrated": 1, "pciBusID": 0,
            "pciDeviceID": 0, "pciDomainID": 0, "maxGridSize": (2 ** 31 - 1, 65535, 65535),
            "maxThreadsDim": (1024, 1024, 64), "uuid": b"\0" * 16, "concurrentManagedAccess": 1,
            "ECCEnabled": 0, "tccDriver": 0}


def memGetInfo() -> tuple[int, int]:
    return _ram()


def deviceSynchronize() -> None:
    pass


def streamSynchronize(stream=0) -> None:
    pass


def deviceGetLimit(limit: int) -> int:
    return _LIMITS.get(limit, 0)


def deviceSetLimit(limit: int, value: int) -> None:
    _LIMITS[limit] = int(value)


def deviceGetPCIBusId(device: int = 0) -> str:
    return "0000:00:00.0"


def runtimeGetVersion() -> int:
    return 12090


def driverGetVersion() -> int:
    return 12090


def getLastError() -> int:
    return 0


def deviceCanAccessPeer(a, b) -> int:
    return 0
