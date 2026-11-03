"""Setup.cmd's check of the CPU backend: compile a CUDA kernel for the processor and run it.

    set ARHTMODELVIEWER_BACKEND=cpu
    runtime\\venv\\Scripts\\python.exe -m arhtmodelviewer.cpu.selftest
"""
import sys

import arhtmodelviewer
from arhtmodelviewer import backend

SRC = r"""
extern "C" __global__ void twice(float* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 2.0f * a[i] + 1.0f;
}
"""


def main() -> int:
    if not backend.is_cpu():
        print("the CPU backend is not active (set ARHTMODELVIEWER_BACKEND=cpu)")
        return 1
    import cupy as cp
    n = 100000
    a = cp.arange(n, dtype=cp.float32)
    k = cp.RawModule(code=SRC).get_function("twice")
    k(((n + 255) // 256,), (256,), (a, cp.int32(n)))
    want = 2.0 * (n - 1) * n / 2.0 + n
    got = float(a.sum(dtype=cp.float64))
    if abs(got - want) > 1e-6 * want:
        print(f"the test kernel gave {got}, not {want}")
        return 1
    print("CPU OK:", backend.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
