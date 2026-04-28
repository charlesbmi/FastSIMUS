# CUDA self-hosted runner

The `cuda-smoke` GitHub Actions workflow runs on a self-hosted runner registered with the labels `self-hosted` and
`cuda`. It executes the CuPy backend tests plus a small (1K + 10K scatterer) scaling benchmark to catch regressions in
the v25c kernel path on every CUDA-relevant PR.

## One-time runner setup

1. **Provision the host** -- any Linux box with a recent NVIDIA driver (`nvidia-smi` succeeds) and CUDA 12.x. The 4090
   dev box is the reference target.

1. **Register the runner** -- in the GitHub repo settings, navigate to *Actions -> Runners -> New self-hosted runner*.
   Follow the registration steps and add the labels `self-hosted` and `cuda` when prompted.

1. **Verify GPU access** -- run `nvidia-smi` from the runner's working directory; the workflow's *Verify CuPy + GPU*
   step further confirms CuPy can enumerate the device.

1. **CuPy install path** -- the workflow runs `uv sync --group bench --frozen`, which pulls `cupy-cuda12x` from PyPI. No
   system-level CuPy needed.

## What the workflow does

- Runs `tests/backend/test_cupy.py` and `tests/test_simus.py -k cupy` on every PR that touches the CUDA paths (kernel
  source, wrapper, dispatch, fixtures, deps).
- Runs a 1K and 10K scatterer subset of the scaling benchmark to confirm the kernel cache + launch path is healthy. Full
  sweeps (1K-1M) and Nsight Compute profiling stay manual on the dev box.

## Out of scope (explicit)

- The Mac MLX `benchmark` workflow on `macos-latest` continues unchanged.
- No autotune or multi-architecture coverage; the smoke is sm_89-only.
- No artifact upload of `.benchmarks/` JSON; the smoke is pass/fail only. The conference figure is generated manually on
  the dev box.
