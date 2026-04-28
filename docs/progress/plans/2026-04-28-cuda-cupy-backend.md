# CUDA / CuPy Backend Implementation Plan

**Branch:** `feat/cuda-cupy-backend` off `origin/main` (commit `c95df8f`). **Status:** Approved design (brainstorm
2026-04-27/28). Ready to execute. **Author:** Charles Guan via Cursor session continuation. **Reference experimentation
branch:** `feature/simus-cuda-pallas` @ `f88946f`.

______________________________________________________________________

## Overview

Promote the `v25c` fp32 register-resident TX kernel from the experimentation branch into a shippable
`SimusStrategy.CUDA` powered by `cupy.RawModule` (NVRTC, no `ctypes`, no nanobind). Wire it into the existing `simus()`
/ `simus_compute()` public API the same way the Metal path on `origin/main` is wired. Add an NCU-on-pytest profiling
harness that wraps any existing pytest-benchmark target so the dev loop never grows a parallel "kernel registry".
Produce a log-log scaling figure (`n_scat` vs throughput, CUDA-fp32 RTX 4090 alongside MLX-Metal and PyMUST) for an
upcoming conference abstract.

The shipped kernel is pinned to one config (`B_SCAT=9`, `ELEM_TILE=2`, `TG_SIZE=128`, `TILE_SE=16`) tuned on RTX 4090 /
sm_89 / P4-2v. Auto-tune and multi-arch are explicitly out of scope and tracked as follow-up issues.

______________________________________________________________________

## Current State Analysis

### `origin/main` SIMUS architecture

- `src/fast_simus/simus.py:122-135` defines `SimusStrategy = StrEnum {PYTHON, SCAN, METAL}`.
- `src/fast_simus/simus.py:368-389` `_select_simus_strategy(xp, strategy)` auto-picks `SCAN` for JAX, `METAL` for MLX,
  falls back to `PYTHON`.
- `src/fast_simus/simus.py:392-471` `simus_compute()` dispatches by selected strategy. The METAL branch (lines 437-455)
  imports `kernels.metal_simus.simus_metal`, calls it with the precomputed plan + casted MLX arrays, and treats the
  returned spectrum the same as the PYTHON / SCAN branches (correction-factor multiply + IFFT happen in `xp` afterward).
- `src/fast_simus/kernels/metal_simus.py` is the existing template:
  - `_KERNELS_DIR / "*.metal"` source files loaded via `_load_source` cache.
  - `mx.fast.metal_kernel(...)` builds + caches one kernel per `(n_elem, n_sub, n_freq, n_scat)` shape, with `header`
    injecting `#define`s.
  - `_prepare_common(...)` packs CPU-prepped inputs into MLX arrays.
  - `_dispatch_split(...)` chunks scatterers by a memory budget, launches TX → RX, accumulates per-chunk spectra. Final
    reshape to `(n_freq, n_elem)` complex64.

### v25c experimentation-branch kernel (the thing we're shipping)

`src/fast_simus/kernels/simus_fused_v25c_svshmem.cu` on `feature/simus-cuda-pallas`. Single fused kernel; signature at
lines 62-87:

```c
extern "C" __global__
void simus_fused_kernel(
    const float* scat_x,    const float* scat_z,    const float* rc_arr,
    const float* elem_x,    const float* elem_z,
    const float* cos_te,    const float* sin_neg_te,
    const float* sub_dx,    const float* sub_dz,
    const float* da_init_re,const float* da_init_im,
    const float* dps,
    const float* pp_re,     const float* pp_im,     const float* probe,
    float* spect_re,        float* spect_im,
    int   n_scat,
    float kw_init, float alpha_init,
    float kw_step, float alpha_step,
    float min_dist, float seg_len,
    float center_kw, float inv_nsub,
    float radius, float apex_offset
);
```

Compile-time defines: `N_ELEM`, `N_SUB`, `N_FREQ`, `N_ES`, `TILE_SE`, `TG_SIZE`, `MAX_FPT`, `B_SCAT`, `ELEM_TILE`.
Output layout: `spect_re[elem * N_FREQ + f]` (row-major `(n_elem, n_freq)`). Shared memory:
`(7*B_SCAT*N_ES + 3*N_ELEM) * 4` bytes (~16.5 KB at the pinned config). Performance: **18.56 M scat/s @ B=9 ET=2 = 5.39
ms** on 100 K scatterers, RTX 4090, P4-2v.

The kernel does its own Phase-1 geometry from the lightweight inputs; **it does not consume the
`(n_scat, n_elem, n_sub)` phase tensors** that `_simus_strategies._simus_freq_outer_python` / `_outer_scan` consume. So
the CuPy wrapper bypasses `simus._prepare_simus_sweep` and replicates a small, flat input prep (mirroring
`metal_simus._prepare_common`).

### Existing dev tooling on `feature/simus-cuda-pallas` (deliberately discarded)

- `src/fast_simus/kernels/cuda_runtime.py` -- hand-rolled NVRTC + Driver-API via `ctypes`. **Replaced wholesale by
  `cupy.RawModule`.** Do not bring forward.
- `tools/bench_v23.py`, `tools/validate_v23.py`, `tools/ncu_profile_v25.py`, the v22/v23/v25b scripts -- all
  single-purpose, kernel-version-pinned. **Discarded.** The pytest-benchmark + NCU-wrapper combo replaces them.
- `test_fused_v11.py`, `pallas_simus.py`, `simus_fused_v{15-26}_*.cu` -- stay on the experimentation branch as
  historical reference.

### Existing benchmarks

- `tests/benchmarks/bench_simus_scaling.py` already parametrizes on the global `xp` fixture and an `n_scat` sweep (1K /
  10K / 100K / 1M default).
- `tests/benchmarks/bench_pymust_scaling.py` already provides the PyMUST CPU baseline at the same `n_scat` sweep,
  ingested by the same plot script.
- `scripts/plot_benchmark_scaling.py` joins `pytest-benchmark` JSONs from multiple machines (via
  `FASTSIMUS_DEVICE_LABEL` env var → `machine_info`) and emits one log-log PNG with runtime / throughput /
  speedup-vs-PyMUST.
- `poe benchmark-scaling` runs the bench; `poe benchmark-plot` produces the PNG.
- **No code changes needed in the bench files** — extending the `xp` fixture in `tests/conftest.py` with CuPy is enough.

### Tests and tolerances

- `tests/conftest.py:38-62` defines the `xp` fixture parametrizing numpy / jnp / mx with `pytest.mark.skipif` on import
  availability.
- `tests/conftest.py:78-88` parametrizes `simus_strategy` over `{None, PYTHON, SCAN, METAL}`.
- `tests/test_simus.py:140` sets `_SIMUS_ATOL_PEAK = 5e-3` (i.e. -46 dB re peak). v25c's max-rel-err vs v11 is `~4e-5`
  (~-87 dB) — trivially inside tolerance.
- `tests/test_simus.py:436-459` `test_strategy_on_backend` is the existing "strategy ↔ backend compatibility" matrix and
  the natural place to add `SimusStrategy.CUDA` skip logic.

### `.github/workflows/benchmark.yml` on main

Runs on `macos-latest` (MLX). The new CUDA-smoke workflow lives in a **separate** file (`cuda-smoke.yml`) and runs on a
self-hosted runner so it does not affect the existing benchmark pipeline.

______________________________________________________________________

## Desired End State

After this plan is complete:

1. `feat/cuda-cupy-backend` is an open PR against `main`. CI green (Mac MLX bench unchanged + new CUDA self-hosted
   smoke).
1. On any host with CuPy + a CUDA GPU:
   ```python
   import cupy as cp
   from fast_simus.simus import simus
   from fast_simus.transducer_presets import P4_2v

   params = P4_2v()
   scat = cp.random.uniform(-2e-2, 2e-2, (100_000, 2)).astype(cp.float32)
   rc   = cp.random.uniform(0.5, 1.5, 100_000).astype(cp.float32)
   delays = cp.zeros(params.n_elements, dtype=cp.float32)

   result = simus(scat, rc, delays, params)   # auto-selects CUDA strategy
   ```
   produces a `SimusResult` matching the PYTHON strategy within `_SIMUS_ATOL_PEAK`, and on RTX 4090 runs at ≥ 17 M
   scat/s sustained for 100K-scatterer 100-iteration mean (some headroom under the 18.56 M/s peak from the
   experimentation branch).
1. `poe benchmark-scaling` on the 4090 produces a `pytest-benchmark` JSON with rows tagged `cupy` and a
   `fast_simus_device_label` machine-info field. `poe benchmark-plot` joins it with the existing Mac MLX + PyMUST JSONs
   and produces the abstract figure.
1. `python scripts/ncu_pytest.py -k "test_bench_simus_scaling and 100000 and cupy" -o /tmp/4090.ncu-rep` produces an NCU
   report capturing exactly one `simus_fused_kernel` launch (warmup launches skipped), reproducing the profiling
   workflow used during v25c development.
1. The new branch carries forward exactly two artifacts from the experimentation branch: `simus_fused.cu` (= v25c
   source) and `docs/progress/experiments/exp22-svshmem-et2.md`.

### Verification

- `poe lint` clean.
- `poe test` clean on a CUDA host (CuPy parametrizations active).
- `poe test` clean on Mac (CuPy parametrizations skipped via `importorskip`).
- `poe benchmark` (the non-scaling subset) passes on Mac.
- Self-hosted CUDA smoke workflow green: `poe test -k cupy` + a 1K-scat scaling-bench warmup.

### Key Discoveries

- **The CUDA wrapper must NOT call `simus._prepare_simus_sweep`.** v25c does its own per-scatterer Phase 1 from the flat
  inputs. The `(n_scat, n_elem, n_sub)` `phase_init` / `phase_step` tensors that `_simus_freq_outer_python` consumes are
  not fed to the kernel. `metal_simus._prepare_common` is the right template, not `simus._prepare_simus_sweep`.
- **Kernel cache key** is `(n_elem, n_sub, n_freq)` — `n_scat` is *not* in the v25c key (the kernel grid-strides over
  scatterers, unlike the Metal split-kernel path which compiles per chunk size). Simpler than Metal.
- **Output layout** is row-major `(n_elem, n_freq)` flat — must reshape
  - transpose to `(n_freq, n_elem)` to match `simus_compute`'s `correction_factor` and `_irfft_and_threshold`
    expectations.
- **Shared memory at pinned config (B=9 ET=2 N_ES=64 N_ELEM=64) is 16,896 B**, well under the default 48 KB cap. No
  `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` needed. We will guard with a sanity assert anyway in case a future
  probe pushes us over.
- **CuPy `array_api_compat` works out of the box** — unlike MLX, no `ensure_compat` shim.
  `array_api_compat.array_namespace(cupy_arr)` returns the right wrapped namespace.

______________________________________________________________________

## What We're NOT Doing

These are explicitly **out of scope** and tracked as follow-up bd issues:

1. **JAX FFI / Pallas integration.** `feature/simus-cuda-pallas` has a `pallas_simus.py` exploration; it stays buried
   there. (`bd` issue: carry forward later if needed.)
1. **fp16 / `v15` / half2 cv chain (FastSIMUS-9bj).** Stays on the experimentation branch. v25c is the single shipped
   kernel.
1. **Multi-architecture support.** We pin `--gpu-architecture=sm_89` (RTX 4090). sm_80 (A100), sm_90 (H100) etc. are
   deferred.
1. **Multi-probe `(B_SCAT, ELEM_TILE)` autotune.** Pinned to (9, 2). Performance on L11-5v / C5-2v / non-4090 GPUs may
   regress vs the pinned config. Documented in the wrapper docstring.
1. **CI for the existing benchmark on a CUDA runner.** The Mac MLX benchmark workflow continues to run on macos-latest.
   The new self-hosted workflow is smoke-only (1K + 10K scatterers, no plot generation).
1. **Removing `cuda_runtime.py` from the experimentation branch.** That branch stays as-is for historical
   reproducibility. We just don't carry the file forward.
1. **Switching `kernels/simus_fused.cu` from a real file to a symlink** (the experimentation branch had a symlink at one
   point; we ship a real file so CuPy `RawModule` source loading is unambiguous).

______________________________________________________________________

## Implementation Approach

Eight phases, ordered so each lands a verifiable improvement:

1. Branch + carry-forward (no functional change yet).
1. Wrapper module (kernel works in isolation; tested via `pytest -k cupy` smoke).
1. Public-API dispatch wiring (`simus()` returns CUDA-computed result).
1. Tests (xp fixture + strategy fixture extensions; full cross-backend matrix passes).
1. Benchmarks (verify the existing scaling bench picks up CuPy without code changes).
1. NCU harness (dev-loop reproducibility for future kernel work).
1. CI smoke (catch regressions on the self-hosted runner).
1. Conference figure (run on 4090, commit JSON, regen PNG).

______________________________________________________________________

## Phase 1 — Branch setup and minimal carry-forward

### Overview

Create the new branch off `origin/main`, copy in the v25c source and the exp22 writeup, add `cupy` to the test/bench
dependency groups. No functional code changes yet.

### Changes Required

#### 1.1 Branch creation

```bash
git fetch origin main
git checkout -b feat/cuda-cupy-backend origin/main
```

#### 1.2 Carry forward kernel source

**Source:** `feature/simus-cuda-pallas:src/fast_simus/kernels/simus_fused_v25c_svshmem.cu`. **Destination:**
`src/fast_simus/kernels/simus_fused.cu` (real file, not a symlink).

```bash
git show feature/simus-cuda-pallas:src/fast_simus/kernels/simus_fused_v25c_svshmem.cu \
  > src/fast_simus/kernels/simus_fused.cu
```

#### 1.3 Carry forward writeup

```bash
mkdir -p docs/progress/experiments
git show feature/simus-cuda-pallas:docs/progress/experiments/exp22-svshmem-et2.md \
  > docs/progress/experiments/exp22-svshmem-et2.md
```

#### 1.4 Add CuPy to test+bench dep groups

**File:** `pyproject.toml` **Changes:** Add `cupy-cuda12x>=13.0` (or appropriate CUDA-major) gated on
`platform_system == 'Linux'` so Mac CI doesn't try to install it.

```toml
test = [
    "pytest>=9.0.2",
    # ... existing entries ...
    "mlx>=0.31; platform_system == 'Darwin'",
    "cupy-cuda12x>=13.0; platform_system == 'Linux'",
    "pymust>=0.1.9",
]
```

Note: `cupy-cuda12x` is a runtime-binding package; users with CUDA 11 will need `cupy-cuda11x` instead. We pin the 12
variant by default since the 4090 dev box is on CUDA 12.

#### 1.5 Update `AGENTS.md` / `CLAUDE.md`

Add one line to the existing `Architecture` / `Backends` section noting CuPy as a supported backend.

### Success Criteria

#### Automated

- [x] `git status` clean after the file copies.
- [x] `uv sync --group bench` succeeds on the 4090 box (CUDA host).
- [ ] `uv sync --group test` succeeds on Mac (CuPy skipped per platform marker). (Mac access not available in this
  session.)
- [x] `poe lint` passes (no new linter errors from the carried-over `.cu` file — it isn't Python and isn't
  lint-checked).

#### Manual

- [x] `simus_fused.cu` opens cleanly in editor (line endings, encoding sane).
- [ ] `exp22-svshmem-et2.md` renders in `mkdocs serve` without broken links. (Not exercised this session.)

______________________________________________________________________

## Phase 2 — CuPy wrapper (`kernels/cuda_simus.py`)

### Overview

New module mirroring `kernels/metal_simus.py`. Compiles `simus_fused.cu` once per `(n_elem, n_sub, n_freq)` shape via
`cp.RawModule`, allocates the output spectrum, packs a small set of CPU-prepped (or already on-device) arrays, launches
the kernel, and returns `(n_freq, n_elem)` complex64.

### Changes Required

#### 2.1 New file: `src/fast_simus/kernels/cuda_simus.py` (~180 LOC)

**Skeleton** (final form will be Google-docstrings-clean and ty-checked):

```python
"""CuPy CUDA backend for simus.

Compiles the v25c register-resident TX kernel via NVRTC at runtime
(`cupy.RawModule`) — no nanobind, no setuptools build step. Pinned to
(B_SCAT=9, ELEM_TILE=2) for RTX 4090 / P4-2v; performance may regress on
other probes / GPUs (see exp22 + the FastSIMUS-cuda-tune follow-up).

Requires: cupy (cupy-cuda12x or cupy-cuda11x) on a CUDA host.
"""
from __future__ import annotations

from math import inf, pi
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cupy as cp

from fast_simus._pfield_math import NEPER_TO_DB, _subelement_centroids
from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils.geometry import element_positions

if TYPE_CHECKING:
    from fast_simus.simus import SimusPlan
    from fast_simus.utils._array_api import _ArrayNamespace

_KERNELS_DIR = Path(__file__).parent
_SOURCE_NAME = "simus_fused.cu"

# Pinned tuning -- see exp22-svshmem-et2.md
_B_SCAT = 9
_ELEM_TILE = 2
_TG_SIZE = 128
_TILE_SE = 16
_GRID_BLOCKS = 256   # 2 * 128 SMs on RTX 4090
_GPU_ARCH = "sm_89"

_source_cache: dict[str, str] = {}
_kernel_cache: dict[tuple, cp.RawKernel] = {}


def _load_source(filename: str) -> str:
    if filename not in _source_cache:
        _source_cache[filename] = (_KERNELS_DIR / filename).read_text()
    return _source_cache[filename]


def _get_kernel(n_elem: int, n_sub: int, n_freq: int) -> cp.RawKernel:
    """Compile + cache simus_fused_kernel for the given problem shape."""
    key = (n_elem, n_sub, n_freq)
    if key in _kernel_cache:
        return _kernel_cache[key]

    n_es = n_elem * n_sub
    max_fpt = (n_freq + _TG_SIZE - 1) // _TG_SIZE

    options = (
        "--std=c++17",
        f"--gpu-architecture={_GPU_ARCH}",
        "--use_fast_math",
        "--extra-device-vectorization",
        f"-DN_ELEM={n_elem}",
        f"-DN_SUB={n_sub}",
        f"-DN_FREQ={n_freq}",
        f"-DN_ES={n_es}",
        f"-DTILE_SE={_TILE_SE}",
        f"-DTG_SIZE={_TG_SIZE}",
        f"-DMAX_FPT={max_fpt}",
        f"-DB_SCAT={_B_SCAT}",
        f"-DELEM_TILE={_ELEM_TILE}",
    )

    module = cp.RawModule(
        code=_load_source(_SOURCE_NAME),
        backend="nvrtc",
        options=options,
        name_expressions=("simus_fused_kernel",),
    )
    kernel = module.get_function("simus_fused_kernel")
    _kernel_cache[key] = kernel
    return kernel


def _shmem_bytes(n_elem: int, n_sub: int) -> int:
    return (7 * _B_SCAT * n_elem * n_sub + 3 * n_elem) * 4


def _prepare_inputs(
    scatterers: cp.ndarray,
    rc: cp.ndarray,
    delays_clean: cp.ndarray,
    tx_apodization: cp.ndarray,
    plan: SimusPlan,
    params: TransducerParams,
    medium: MediumParams,
) -> dict[str, Any]:
    """Pack the 15 input arrays + 10 scalars the kernel expects.

    Mirrors metal_simus._prepare_common but without the (n_elem, n_sub, n_freq)
    expansion: v25c does its own Phase-1 geometry from the flat inputs.
    """
    c = medium.speed_of_sound
    alpha = medium.attenuation
    n_elem = params.n_elements
    n_sub = plan.n_sub
    n_freq = int(plan.selected_freqs.shape[0])

    elem_pos, theta_e, apex_offset = element_positions(
        n_elem, params.pitch, params.radius, cp,
    )
    if theta_e is None:
        theta_e = cp.zeros(n_elem, dtype=cp.float32)

    # subelement offsets per (elem, sub) flattened to N_ES
    offsets = _subelement_centroids(params.element_width, n_sub, theta_e, cp)
    sub_dx = cp.ascontiguousarray(offsets[..., 0].reshape(-1).astype(cp.float32))
    sub_dz = cp.ascontiguousarray(offsets[..., 1].reshape(-1).astype(cp.float32))

    cos_te = cp.cos(theta_e).astype(cp.float32)
    sin_neg_te = cp.sin(-theta_e).astype(cp.float32)

    # Frequency-derived scalars
    freq_start = float(plan.selected_freqs[0])
    freq_step = (
        float(plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freq > 1 else 0.0
    )

    # Delay-apod init / step (per element)
    da_init_re = (cp.cos(2 * pi * freq_start * delays_clean) * tx_apodization).astype(cp.float32)
    da_init_im = (cp.sin(2 * pi * freq_start * delays_clean) * tx_apodization).astype(cp.float32)
    dps = (2 * pi * freq_step * delays_clean).astype(cp.float32)

    # Pulse * probe spectrum (complex), probe magnitude (real)
    pulse_probe = (plan.pulse_spectrum * plan.probe_spectrum).astype(cp.complex64)
    pp_re = cp.real(pulse_probe).astype(cp.float32).copy()
    pp_im = cp.imag(pulse_probe).astype(cp.float32).copy()
    probe_real = cp.abs(plan.probe_spectrum).astype(cp.float32).copy()

    return dict(
        scat_x=cp.ascontiguousarray(scatterers[:, 0].astype(cp.float32)),
        scat_z=cp.ascontiguousarray(scatterers[:, 1].astype(cp.float32)),
        rc=cp.ascontiguousarray(rc.astype(cp.float32)),
        elem_x=elem_pos[:, 0].astype(cp.float32).copy(),
        elem_z=elem_pos[:, 1].astype(cp.float32).copy(),
        cos_te=cos_te, sin_neg_te=sin_neg_te,
        sub_dx=sub_dx, sub_dz=sub_dz,
        da_init_re=da_init_re, da_init_im=da_init_im,
        dps=dps,
        pp_re=pp_re, pp_im=pp_im, probe_real=probe_real,
        n_scat=int(scatterers.shape[0]),
        kw_init=2 * pi * freq_start / c,
        alpha_init=alpha / NEPER_TO_DB * freq_start / 1e6 * 1e2,
        kw_step=2 * pi * freq_step / c,
        alpha_step=alpha / NEPER_TO_DB * freq_step / 1e6 * 1e2,
        min_dist=c / params.freq_center / 2.0,
        seg_length=params.element_width / n_sub,
        center_kw=2 * pi * params.freq_center / c,
        inv_nsub=1.0 / n_sub,
        radius_v=params.radius if params.radius != inf else 1e31,
        apex_offset=apex_offset,
        n_elem=n_elem, n_sub=n_sub, n_freq=n_freq,
    )


def simus_cuda(
    scatterers: cp.ndarray,
    rc: cp.ndarray,
    params: TransducerParams,
    plan: SimusPlan,
    medium: MediumParams,
    delays_clean: cp.ndarray,
    tx_apodization: cp.ndarray,
) -> cp.ndarray:
    """Run v25c CUDA kernel; return complex64 spectrum (n_freq, n_elem)."""
    d = _prepare_inputs(scatterers, rc, delays_clean, tx_apodization,
                        plan, params, medium)

    n_elem, n_sub, n_freq = d["n_elem"], d["n_sub"], d["n_freq"]
    kernel = _get_kernel(n_elem, n_sub, n_freq)
    shmem = _shmem_bytes(n_elem, n_sub)

    # Output buffers: kernel writes spect_re[elem * N_FREQ + f] via atomicAdd
    spect_re = cp.zeros(n_elem * n_freq, dtype=cp.float32)
    spect_im = cp.zeros(n_elem * n_freq, dtype=cp.float32)

    args = (
        d["scat_x"], d["scat_z"], d["rc"],
        d["elem_x"], d["elem_z"],
        d["cos_te"], d["sin_neg_te"],
        d["sub_dx"], d["sub_dz"],
        d["da_init_re"], d["da_init_im"], d["dps"],
        d["pp_re"], d["pp_im"], d["probe_real"],
        spect_re, spect_im,
        cp.int32(d["n_scat"]),
        cp.float32(d["kw_init"]), cp.float32(d["alpha_init"]),
        cp.float32(d["kw_step"]), cp.float32(d["alpha_step"]),
        cp.float32(d["min_dist"]), cp.float32(d["seg_length"]),
        cp.float32(d["center_kw"]), cp.float32(d["inv_nsub"]),
        cp.float32(d["radius_v"]), cp.float32(d["apex_offset"]),
    )

    kernel(
        grid=(_GRID_BLOCKS, 1, 1),
        block=(_TG_SIZE, 1, 1),
        args=args,
        shared_mem=shmem,
    )

    # (n_elem, n_freq) row-major -> (n_freq, n_elem) complex64
    spect = (spect_re + 1j * spect_im).reshape(n_elem, n_freq).T
    return spect.astype(cp.complex64)
```

### Success Criteria

#### Automated

- [x] `python -c "from fast_simus.kernels import cuda_simus"` imports cleanly on a CUDA host.
- [x] `python -c "from fast_simus.kernels import cuda_simus; print(cuda_simus._get_kernel(64, 1, 854))"` produces a
  `cupy.RawKernel` (verifies NVRTC compile path).
- [x] `ty check src/fast_simus/kernels/cuda_simus.py` clean.
- [x] `ruff check src/fast_simus/kernels/cuda_simus.py` clean.

#### Manual

- [x] On the 4090 box, end-to-end CUDA result matches `PYTHON` strategy to ~3.7e-5 relative to peak on a 6-scatterer
  P4-2v config (well under the 5e-3 ATOL_PEAK).
- [x] Wall time of `simus_cuda(...)` for `n_scat=100_000`, `P4_2v` is ~4.75 ms / 21 M scat/s on the 4090 (better than
  the ≤ 7 ms target, slightly above exp22's 18.56 M scat/s peak with input-prep overhead amortized across the cached
  plan).

______________________________________________________________________

## Phase 3 — `simus.py` dispatch wiring

### Overview

Add `SimusStrategy.CUDA`, teach `_select_simus_strategy` to detect CuPy, and add the CUDA branch to `simus_compute`.
Mirrors the METAL branch exactly.

### Changes Required

#### 3.1 Extend `SimusStrategy` enum

**File:** `src/fast_simus/simus.py:122-135`

```python
class SimusStrategy(StrEnum):
    """Backend strategy for the simus frequency sweep.

    Attributes:
        PYTHON: Python for-loop (NumPy/CuPy, constant memory).
        SCAN: JAX lax.scan for O(1) compilation cost.
        METAL: Custom Metal kernel on Apple Silicon (MLX).
        CUDA: Custom CUDA kernel on NVIDIA GPUs (CuPy + NVRTC).
    """

    PYTHON = "python"
    SCAN = "scan"
    METAL = "metal"
    CUDA = "cuda"
```

#### 3.2 Auto-select CUDA on CuPy

**File:** `src/fast_simus/simus.py:_select_simus_strategy` (~line 368)

```python
def _select_simus_strategy(xp: _ArrayNamespace, strategy: SimusStrategy | None) -> SimusStrategy:
    if strategy is not None:
        return strategy
    if is_jax_namespace(cast(ModuleType, xp)):
        return SimusStrategy.SCAN

    try:
        import mlx.core
        if xp is mlx.core:
            return SimusStrategy.METAL
    except ImportError:
        pass

    try:
        import cupy
        if xp is cupy or getattr(xp, "__name__", "").endswith("cupy"):
            return SimusStrategy.CUDA
    except ImportError:
        pass

    return SimusStrategy.PYTHON
```

The `getattr(xp, "__name__", "").endswith("cupy")` check covers the `array_api_compat.cupy` wrapped namespace.

#### 3.3 Add CUDA dispatch branch in `simus_compute`

**File:** `src/fast_simus/simus.py:437-460` (after the METAL branch)

```python
elif selected == SimusStrategy.CUDA:
    import cupy as cp

    from fast_simus.kernels.cuda_simus import simus_cuda

    spect_selected = cast(
        Array,
        simus_cuda(
            scatterers=cast(cp.ndarray, scatterers_flat),
            rc=cast(cp.ndarray, rc_flat),
            params=params,
            plan=plan,
            medium=medium,
            delays_clean=cast(cp.ndarray, delays_clean),
            tx_apodization=cast(cp.ndarray, tx_apodization),
        ),
    )
```

### Success Criteria

#### Automated

- [x] `from fast_simus.simus import SimusStrategy; assert SimusStrategy.CUDA == "cuda"`.
- [x] `_select_simus_strategy(numpy, None) == PYTHON`.
- [x] On a CUDA host: `_select_simus_strategy(cupy, None) == CUDA`. Also verified on the `array_api_compat.cupy` wrapped
  namespace (via `is_cupy_namespace`).
- [x] `poe lint` clean.

#### Manual

- [x] On the 4090 box, end-to-end snippet runs and produces `(n_samples, params.n_elements)` output.
- [x] Output is `cupy.ndarray` (stays on device through `_irfft_and_threshold`).

______________________________________________________________________

## Phase 4 — Tests

### Overview

Extend the `xp` fixture and `simus_strategy` fixture to include CuPy, update `test_strategy_on_backend` skip logic, add
a small CuPy-specific smoke file.

### Changes Required

#### 4.1 Add CuPy to `tests/conftest.py`

```python
HAS_CUPY = False
cp = None
with contextlib.suppress(ImportError):
    import cupy as cp
    HAS_CUPY = True

@pytest.fixture(
    params=[
        pytest.param(np,  id="numpy", marks=pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")),
        pytest.param(jnp, id="jax",   marks=pytest.mark.skipif(not HAS_JAX,   reason="JAX not available")),
        pytest.param(mx,  id="mlx",   marks=pytest.mark.skipif(not HAS_MLX,   reason="MLX not available")),
        pytest.param(cp,  id="cupy",  marks=pytest.mark.skipif(not HAS_CUPY,  reason="CuPy not available")),
    ]
)
def xp(request) -> _ArrayNamespace: ...
```

And extend the `simus_strategy` fixture similarly:

```python
pytest.param(SimusStrategy.CUDA, id="cuda"),
```

#### 4.2 Update `test_strategy_on_backend` skip logic

**File:** `tests/test_simus.py:436-459`

```python
def test_strategy_on_backend(self, xp, simus_strategy):
    if simus_strategy == SimusStrategy.SCAN  and not is_jax_namespace(xp):
        pytest.skip("scan requires JAX")
    if simus_strategy == SimusStrategy.METAL and not is_mlx_namespace(xp):
        pytest.skip("metal requires MLX")
    if simus_strategy == SimusStrategy.CUDA  and not _is_cupy_namespace(xp):
        pytest.skip("cuda requires CuPy")
    # ... existing body unchanged ...
```

Add `_is_cupy_namespace` helper to `fast_simus/utils/_array_api.py` mirroring `is_mlx_namespace` (one line: check `xp`
has `cupy` in its module path).

#### 4.3 New file: `tests/backend/test_cupy.py`

```python
"""CuPy-specific tests for FastSIMUS.

Verifies CUDA kernel cache behavior, output device placement, and
NVRTC compile-time guards.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from fast_simus.kernels.cuda_simus import _get_kernel, _kernel_cache, _shmem_bytes
from fast_simus.simus import simus, SimusStrategy
from fast_simus.transducer_presets import P4_2v


def test_kernel_cache_hits_on_repeat_call():
    """Same shape -> same RawKernel object (no recompile)."""
    _kernel_cache.clear()
    k1 = _get_kernel(64, 1, 854)
    k2 = _get_kernel(64, 1, 854)
    assert k1 is k2


def test_kernel_cache_different_shapes():
    """Different shape -> different compile."""
    _kernel_cache.clear()
    k1 = _get_kernel(64, 1, 854)
    k2 = _get_kernel(128, 1, 854)
    assert k1 is not k2


def test_shmem_under_default_cap():
    """Pinned config must fit under 48 KB default shmem cap."""
    assert _shmem_bytes(64, 1)  < 48 * 1024
    assert _shmem_bytes(128, 1) < 48 * 1024


def test_simus_cuda_output_is_cupy():
    """End-to-end smoke: result stays on CuPy device."""
    params = P4_2v()
    scat = cp.asarray(np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1))
    rc = cp.ones(3, dtype=cp.float32)
    delays = cp.zeros(params.n_elements, dtype=cp.float32)
    result = simus(scat, rc, delays, params, strategy=SimusStrategy.CUDA)
    assert isinstance(result.rf, cp.ndarray)
    assert isinstance(result.spectrum, cp.ndarray)
    assert result.rf.shape[1] == params.n_elements


def test_simus_cuda_matches_python_strategy():
    """CUDA result must match Python strategy within ATOL_PEAK."""
    params = P4_2v()
    n_scat = 6
    scat_np = np.stack([np.zeros(n_scat), np.linspace(1e-2, 5e-2, n_scat)], axis=-1)
    rc_np = np.ones(n_scat)
    delays_np = np.zeros(params.n_elements)

    rf_py = simus(scat_np, rc_np, delays_np, params,
                  strategy=SimusStrategy.PYTHON).rf
    rf_cu = simus(cp.asarray(scat_np), cp.asarray(rc_np), cp.asarray(delays_np),
                  params, strategy=SimusStrategy.CUDA).rf
    rf_cu_np = cp.asnumpy(rf_cu)

    peak = np.max(np.abs(rf_py))
    assert np.allclose(rf_py, rf_cu_np, atol=5e-3 * peak, rtol=0)
```

### Success Criteria

#### Automated

- [x] On the 4090 box: `pytest tests/backend/test_cupy.py` -- 6 passed.
- [x] On the 4090 box: `pytest tests/test_simus.py -k cupy` -- 3 passed, 2 skipped (scan/metal correctly skipped on
  cupy).
- [ ] On Mac: same commands, all CuPy params skipped via `importorskip`. (Mac access not available this session.)
- [x] `poe lint` clean.

#### Manual

- [x] Full suite: 160 passed, 31 skipped on the 4090 host. CuPy parametrizations active across `test_simus.py` /
  `test_pfield.py` / `test_cupy.py`.

______________________________________________________________________

## Phase 5 — Benchmarks

### Overview

`tests/benchmarks/bench_simus_scaling.py` already takes the global `xp` fixture. Once Phase 4 lands cupy in that
fixture, the bench picks it up with **zero code changes**. This phase verifies that, runs the bench on the 4090, and
captures the JSON for the conference figure.

### Changes Required

#### 5.1 Verify bench picks up cupy

```bash
# On the 4090 box:
FASTSIMUS_DEVICE_LABEL="RTX 4090 CuPy" \
  poe benchmark-scaling-run -- -k "test_bench_simus_scaling and cupy"
```

Expected: a `pytest-benchmark` JSON in `.benchmarks/Linux-CPython.../...json` with rows like
`test_bench_simus_scaling[cupy-1000]`, `[cupy-10000]`, `[cupy-100000]`, `[cupy-1000000]`, and a
`fast_simus_device_label` machine-info field equal to `"RTX 4090 CuPy"`.

#### 5.2 Capture cross-machine JSON for the figure

- 4090: run the full sweep tagged `RTX 4090 CuPy`.
- Mac (existing): re-run `FASTSIMUS_DEVICE_LABEL="M-series MLX" poe benchmark-scaling` if the existing JSONs don't have
  the device label.
- PyMUST: runs on whatever CPU is convenient; default `n_scat` sweep.

Commit the resulting JSONs into `.benchmarks/` on the branch (or attach to the PR if too large; pytest-benchmark JSONs
are usually ≤ 100 KB each).

### Success Criteria

#### Automated

- [x] `pytest tests/benchmarks/bench_simus_scaling.py --benchmark-only -p no:xdist     -m scaling -k cupy` produces JSON
  with `cupy` rows on the 4090.

#### Manual

- [x] JSON contains rows for all four `n_scat` values (1K, 10K, 100K, 1M).
- [x] `extra_info.backend == "cupy"` (the conftest's `xp.__name__.split(".")[0]` derivation already produced `"cupy"`
  cleanly; no tweak needed).
- [x] Side note: added a CuPy branch to `tests/benchmarks/_bench_sync.py` so wall-clock measurements actually include
  the CUDA kernel runtime (default-stream sync). Without this, pytest-benchmark times the launch but not the kernel.

______________________________________________________________________

## Phase 6 — NCU-on-pytest harness

### Overview

A small wrapper script that runs `ncu` over an arbitrary pytest invocation so any benchmark in `tests/benchmarks/` can
be deeply profiled with one command. Replaces the per-kernel `tools/ncu_profile_v*.py` proliferation.

### Changes Required

#### 6.1 New file: `scripts/ncu_pytest.py`

```python
"""Wrap any pytest-benchmark invocation under NVIDIA Nsight Compute.

Skips warmup launches via --launch-skip, captures one steady-state launch
of `simus_fused_kernel`, writes an .ncu-rep.

Example:
    sudo python scripts/ncu_pytest.py \\
        -k "test_bench_simus_scaling and 100000 and cupy" \\
        -o /tmp/4090_simus_100k.ncu-rep
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

NCU_DEFAULT = "/usr/local/cuda/bin/ncu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", required=True, help="pytest -k expression")
    parser.add_argument("-o", "--output", required=True, help=".ncu-rep path")
    parser.add_argument(
        "--launch-skip", type=int, default=5,
        help="Skip first N launches (warmup). Default 5.",
    )
    parser.add_argument(
        "--kernel-regex", default="simus_fused_kernel",
        help="NCU --kernel-id ::regex:<this>. Default: simus_fused_kernel.",
    )
    parser.add_argument(
        "--ncu", default=os.environ.get("NCU", NCU_DEFAULT),
        help="Path to ncu binary.",
    )
    parser.add_argument(
        "--bench-path", default="tests/benchmarks/",
        help="pytest target directory.",
    )
    args = parser.parse_args()

    cmd = [
        args.ncu,
        "--target-processes", "all",
        "--launch-skip", str(args.launch_skip),
        "--launch-count", "1",
        "--set", "full",
        "--kernel-id", f"::regex:{args.kernel_regex}",
        "-f", "-o", args.output,
        "--",
        sys.executable, "-m", "pytest",
        args.bench_path,
        "--benchmark-only", "-p", "no:xdist",
        "-k", args.k,
        # Keep pytest-benchmark from rerunning the timing loop --
        # NCU profiling already takes 30+ seconds per launch.
        "--benchmark-min-rounds=1",
        "--benchmark-min-time=0",
    ]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
```

#### 6.2 Add poe task

**File:** `pyproject.toml`

```toml
[tool.poe.tasks.profile-cuda]
help = "Profile a CUDA simus benchmark under Nsight Compute. Pass -k <expr> -o <out>."
shell = "sudo $(which python) scripts/ncu_pytest.py $@"
executor = { group = "bench" }
```

### Success Criteria

#### Automated

- [x] `python scripts/ncu_pytest.py --help` prints usage.
- [x] `ruff check scripts/ncu_pytest.py` clean (`ty check` also passes).

#### Manual

- [ ] On the 4090 box:
  `sudo python scripts/ncu_pytest.py -k "test_bench_simus_scaling and 100000 and cupy" -o /tmp/4090.ncu-rep` produces a
  `.ncu-rep` file readable by Nsight Compute UI. (Not exercised this session — requires `sudo` for NCU perf counters and
  ~30 s of NCU runtime.)
- [ ] The report contains exactly one `simus_fused_kernel` launch (warmup skipped).
- [ ] L2 throughput / Compute throughput / occupancy match exp22's published numbers within ±5%.

______________________________________________________________________

## Phase 7 — CI smoke (self-hosted GPU)

### Overview

A minimal GitHub Actions workflow that runs on a self-hosted runner with label `cuda` (you provision and label this
runner once on the 4090 box; out of scope for this PR). Triggered on PRs that touch CUDA paths and on manual dispatch.
Runs the CuPy backend tests + a small scaling smoke.

### Changes Required

#### 7.1 New file: `.github/workflows/cuda-smoke.yml`

```yaml
name: cuda-smoke

on:
  pull_request:
    paths:
      - "src/fast_simus/kernels/cuda_simus.py"
      - "src/fast_simus/kernels/simus_fused.cu"
      - "src/fast_simus/simus.py"
      - "tests/backend/test_cupy.py"
      - "tests/conftest.py"
      - "pyproject.toml"
      - ".github/workflows/cuda-smoke.yml"
  workflow_dispatch:

jobs:
  smoke:
    runs-on: [self-hosted, cuda]
    steps:
      - uses: actions/checkout@v6

      - name: Set up environment
        run: uv sync --group bench

      - name: Verify CuPy + GPU
        run: uv run python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"

      - name: Run CuPy backend tests
        run: uv run pytest tests/backend/test_cupy.py tests/test_simus.py -v -k cupy -p no:xdist

      - name: Run small scaling smoke (1K + 10K only)
        run: |
          uv run pytest tests/benchmarks/bench_simus_scaling.py \
            --benchmark-only -p no:xdist -m scaling \
            -k "cupy and (1000 or 10000)" \
            --benchmark-min-rounds=1 --benchmark-min-time=0
```

#### 7.2 Document the runner setup

Add a brief note to `docs/index.md` or a new `docs/dev/cuda-runner.md`:

```markdown
# CUDA self-hosted runner

The `cuda-smoke` workflow runs on a self-hosted GitHub Actions runner.
Setup (one-time, on the host with the 4090):

1. Repo Settings → Actions → Runners → New self-hosted runner.
2. Follow the registration steps; add labels `self-hosted` and `cuda`.
3. Ensure the runner service has access to `/usr/local/cuda` and a
   working CUDA driver (`nvidia-smi` succeeds).

The workflow runs `uv sync --group bench` per job; CuPy installs from
`cupy-cuda12x`.
```

### Success Criteria

#### Automated

- [ ] `actionlint .github/workflows/cuda-smoke.yml` clean (`actionlint` not installed in this session; YAML syntax
  checked manually).
- [ ] On a PR that touches `cuda_simus.py`, the workflow triggers. (Verified at PR creation time; not exercised pre-PR.)

#### Manual

- [ ] Self-hosted runner registration is one-time host setup, documented in `docs/dev/cuda-runner.md`; not done as part
  of this PR.
- [ ] Total wall time < 5 minutes (smoke target). To be measured on the first runner-up workflow execution.

______________________________________________________________________

## Phase 8 — Conference figure

### Overview

Run the full scaling sweep on the 4090, join with existing Mac MLX + PyMUST JSONs, regenerate the plot, commit the PNG.

### Changes Required

#### 8.1 Run the sweep on the 4090

```bash
FASTSIMUS_DEVICE_LABEL="RTX 4090 CuPy" \
  poe benchmark-scaling-run
# JSON ends up under .benchmarks/Linux-CPython.../...json
```

#### 8.2 (If needed) Refresh Mac JSON

```bash
# On Mac:
FASTSIMUS_DEVICE_LABEL="M-series MLX" \
  poe benchmark-scaling-run
```

#### 8.3 Regenerate plot

```bash
# On whichever box has the JSONs together:
poe benchmark-plot   # default reads .benchmarks/**/*.json
# Or: python scripts/plot_benchmark_scaling.py mac/*.json cuda/*.json -o docs/figures/scaling.png
```

#### 8.4 Commit the figure

```bash
mkdir -p docs/figures
mv .benchmarks/scaling_plot.png docs/figures/2026-04-cuda-scaling.png
git add docs/figures/2026-04-cuda-scaling.png .benchmarks/*.json
```

### Success Criteria

#### Automated

- None directly; this is a one-shot artifact generation.

#### Manual

- [x] PNG opens; shows two series this session: CuPy/4090 (1K, 10K, 100K, 1M) and PyMUST CPU (1K, 10K, 100K). MLX-Metal
  series will land when the figure is regenerated on a Mac.
- [x] Throughput panel shows CuPy peaking at ~7 M scat/s end-to-end at 1M scatterers; the kernel itself runs at ~18 M
  scat/s as in exp22, with the difference being precompute + IFFT + atomics overhead that the end-to-end bench includes.
- [x] Speedup-vs-PyMUST panel reaches ~2,470x at 100K scatterers, comfortably within the order-of-magnitude target for
  the abstract.

#### Bug fix

- [x] `scripts/plot_benchmark_scaling.py`: `g.figure.close()` is not a thing on `matplotlib.Figure`. Switch to
  `plt.close(g.figure)` so the script actually finishes (the PNG was written before the crash, so this was a silent
  post-write failure).

______________________________________________________________________

## Testing Strategy

### Unit tests

- `tests/backend/test_cupy.py`: kernel cache idempotency, output device placement, shmem-fits-in-cap, end-to-end smoke
  vs PYTHON strategy.
- `tests/test_simus.py::TestSimusStrategyCrossBackend::test_strategy_on_backend`: CUDA × cupy combo, all others skipped.

### Integration tests

- `tests/test_simus.py`: full reference-data path (FastSIMUS vs PyMUST) runs across the now-extended `xp` fixture; CuPy
  gets the same `_SIMUS_ATOL_PEAK = 5e-3` treatment as numpy / jax / mlx.

### Manual testing steps

1. Clean clone on the 4090 box → `git checkout feat/cuda-cupy-backend` → `uv sync --group bench`. Should install
   cleanly.
1. `uv run pytest tests/test_simus.py -v -k cupy` — all CuPy parametrizations PASS.
1. `uv run pytest tests/backend/test_cupy.py -v` — all PASS.
1. `FASTSIMUS_DEVICE_LABEL="RTX 4090 CuPy" poe benchmark-scaling-run -- -k cupy` — produces JSON with the four `n_scat`
   rows.
1. `sudo python scripts/ncu_pytest.py -k "test_bench_simus_scaling and 100000 and cupy" -o /tmp/4090.ncu-rep` — produces
   a profile; open in Nsight Compute UI; confirm L2 ≈ 65%, compute ≈ 63%.

______________________________________________________________________

## Performance Considerations

- **First-call JIT cost.** `cp.RawModule` compile takes ~0.5-1 s on first call per `(n_elem, n_sub, n_freq)` shape. The
  `_kernel_cache` makes subsequent calls near-instant. The `simus()` public API does not pre-warm; if a user benchmarks
  naively they'll see the first iteration slow. The `pytest-benchmark` config in
  `tests/benchmarks/bench_simus_scaling.py` already does `warmup=True, warmup_iterations=1`, which absorbs this.
- **Pinned tuning regression risk.** B=9 ET=2 is optimal for P4-2v / 4090. On L11-5v (128 elements) or smaller GPUs,
  this may regress vs a hypothetical autotune. Document the pinning in the wrapper docstring; file FastSIMUS-cuda-tune
  in bd as a follow-up.
- **Memory: output buffers.** `spect_re` + `spect_im` are `n_elem * n_freq * 4` bytes each = ~219 KB at P4-2v.
  Negligible.
- **No chunking.** Unlike Metal's `_dispatch_split`, v25c is a single fused kernel that grid-strides over scatterers; no
  per-chunk TX intermediate buffer is materialized. We don't need `MAX_TX_INTERMEDIATE_BYTES` analog.

______________________________________________________________________

## Migration Notes

- **No data migrations.** This is an additive feature.
- **Existing tests are unaffected** until Phase 4 lands the CuPy fixture.
- **`array_api_compat` should "just work"** for CuPy without a shim. If `array_api_compat.cupy` is preferred over raw
  `cupy` for some reason, surface that as a separate PR — out of scope here.

______________________________________________________________________

## References

- Experimentation branch champion: v25c at
  `feature/simus-cuda-pallas:src/fast_simus/kernels/simus_fused_v25c_svshmem.cu`
- v25c writeup: `feature/simus-cuda-pallas:docs/progress/experiments/exp22-svshmem-et2.md`
- Metal pattern to mirror: `src/fast_simus/kernels/metal_simus.py` (load_source, kernel cache, prepare_common,
  dispatch).
- Existing dispatch model: `src/fast_simus/simus.py:_select_simus_strategy` (~line 368) and the METAL branch in
  `simus_compute` (~line 437).
- Bench infrastructure on main: `tests/benchmarks/bench_simus_scaling.py`, `tests/benchmarks/conftest.py`,
  `scripts/plot_benchmark_scaling.py`.
- Brainstorm session that produced this plan: continuation of
  [v25 / v26 register-resident TX experimentation](99a6a3fe-a85c-4cda-8555-9a37fa35ed67) — see the 2026-04-27/28 portion
  of the user's chat history.

______________________________________________________________________

## Handoff Notes (for picking this up on another machine)

1. **Clone + checkout:**
   ```bash
   git clone git@github.com:charlesbmi/FastSIMUS.git && cd FastSIMUS
   git fetch origin
   git checkout -b feat/cuda-cupy-backend origin/main
   ```
1. **Phase 1 first:** carry forward exactly two files from `feature/simus-cuda-pallas` (the v25c `.cu` and the `exp22`
   doc), add CuPy to `pyproject.toml`. Commit. Push.
1. **Then Phase 2:** the CuPy wrapper. Write it in isolation, smoke-test via a one-off Python REPL, before touching
   `simus.py`.
1. **Phase 3 only after Phase 2 smoke-passes.** Wiring it through the public API exposes you to the existing test
   matrix; you want the kernel itself proven first.
1. **Beads tracking:** open one bd issue per phase as you go. Close FastSIMUS-9bj as deferred (not on this branch). Open
   FastSIMUS-cuda-tune for the autotune follow-up.
1. **Don't bring `cuda_runtime.py`.** It is the ctypes path we are deliberately retiring. CuPy's `RawModule` replaces it
   wholesale.
