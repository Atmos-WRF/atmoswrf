# Atmos-WRF on an AMD graphics card

The model's GPU code is CUDA kernels run through CuPy. CuPy also has a build for AMD's ROCm (HIP), and Atmos-WRF
can use it: it finds ROCm, and `arhtmodelviewer/backend.py` (`hip_compat`) rewrites the few things HIP's compiler
takes differently. Those are NVRTC-only options such as `--fmad=false`, and warp votes and shuffles written for
32 lanes (`__ballot_sync` and others).

**Untested.** The ROCm path has not been run on an AMD card yet; please report what works and what doesn't.

## Windows

CuPy has **no ROCm build for Windows**, so on a Windows PC with only an AMD card, Atmos-WRF runs the model on the
CPU (Setup.cmd and the run's Log say so). That works, but it is slow: use 2–3 km grids.

## WSL2 (Windows) or Linux, with ROCm

What you need:

- A Radeon that ROCm supports. On WSL2 that is RX 7000 / 9000 series and some Radeon PRO cards, with AMD's
  "Adrenalin for WSL" driver; on Linux, see AMD's ROCm compatibility list.
- ROCm 6.x installed, following AMD's guide for your system. Check that `rocminfo` lists your GPU.
- Python 3.11–3.13, and the build tools CuPy needs (see CuPy's "Installing CuPy from source" for ROCm).

Then, in the Atmos-WRF folder, inside WSL or Linux:

```bash
python3 -m venv ~/atmoswrf-venv
. ~/atmoswrf-venv/bin/activate
pip install "gpuwm==2.9.2" eccodes==2.50.0
CUPY_INSTALL_USE_HIP=1 ROCM_HOME=/opt/rocm pip install cupy   # builds CuPy for your ROCm (10-30 min)
python -m arhtmodelviewer serve 8830 --lan
```

Open `http://localhost:8830` in Windows' browser; WSL2 forwards the port. The page's header shows the card with
"(ROCm)" when the GPU is in use.

What may not work yet:

- **Physics suites with the MYNN PBL or RUC land surface.** Their kernels use warp votes. These are rewritten for
  HIP and give the same answer on RDNA cards (wave32), but are untested.
- **The CPU fallback, if ROCm is not found.** Check that `/dev/kfd` exists (Linux), or `/dev/dxg` plus `/opt/rocm`
  (WSL2).
- **Memory use.** It is the same as on NVIDIA; the GPU memory budget is read through CuPy.

To force a backend, run `python -m arhtmodelviewer serve --backend cuda` (the CuPy build that is installed: ROCm here)
or `--backend cpu`.
