# CUDA SIMUS Fast-Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move CuPy SIMUS benchmark throughput from ~7 M scat/s to roughly 15-17 M scat/s end-to-end by removing dead
Array API sweep preparation from CUDA and Metal dispatch, then leave kernel-only 20 M scat/s work as an optional
follow-up.

**Architecture:** `simus_precompute()` already runs outside `pytest-benchmark`'s timed region. The current loss is
inside `simus_compute()`: it unconditionally builds `_prepare_simus_sweep()` before dispatch even though CUDA and Metal
ignore that dict and prepare their own flat backend inputs. The primary change is to choose the strategy first and build
the Python/SCAN sweep only for strategies that consume it.

**Tech Stack:** Python 3.12, CuPy + NVRTC, Array API, pytest, pytest-benchmark, Nsight Compute (optional follow-up).

______________________________________________________________________

## Overview

The current CuPy scaling figure reports ~6-7 M scat/s end-to-end because the timed `simus_compute()` path includes
unnecessary `_prepare_simus_sweep()` work before the CUDA branch. Direct timing shows that skipping that sweep moves the
current end-to-end CUDA path to ~15.2 M scat/s at 100K scatterers and ~17.2 M scat/s at 1M scatterers, with the IFFT and
thresholding contributing only ~0.55 ms.

This plan preserves the public API, numerical outputs, and the benchmark contract. It only changes dispatch ordering in
`simus_compute()` and adds regression tests/benchmark verification so future backend-specific strategies do not pay for
Python/SCAN-only preparation.

## Current State Analysis

### Dispatch Path

`src/fast_simus/simus.py` currently flattens inputs, immediately computes `sweep = _prepare_simus_sweep(...)`, then
selects the strategy. Only `PYTHON` and `SCAN` use `**sweep`; `METAL` and `CUDA` dispatch with raw flattened arrays,
`plan`, `params`, `medium`, `delays_clean`, and `tx_apodization`.

Current structure:

```python
# src/fast_simus/simus.py
scatterers_flat = ...
rc_flat = ...

sweep = _prepare_simus_sweep(...)

selected = _select_simus_strategy(xp, strategy)

if selected == SimusStrategy.METAL:
    spect_selected = simus_metal(...)
elif selected == SimusStrategy.CUDA:
    spect_selected = simus_cuda(...)
elif selected == SimusStrategy.SCAN:
    spect_selected = _simus_freq_outer_scan(rc=rc_flat, xp=xp, **sweep)
else:
    spect_selected = _simus_freq_outer_python(rc=rc_flat, xp=xp, **sweep)
```

### Benchmark Path

`tests/benchmarks/bench_simus_scaling.py` already does the right thing for static precompute: `simus_precompute(...)`
runs before `benchmark(run)`. The timed function calls `compute(scatterers, rc, delays)`, where `compute` is a closure
over the precomputed `plan` and `params`.

The benchmark also synchronizes CuPy with `cp.cuda.Stream.null.synchronize()` in `tests/benchmarks/_bench_sync.py`, so
the measured runtime includes the actual CUDA work and is not just kernel launch latency.

### Measured Split on RTX 4090

Measured after the initial CUDA backend implementation:

| `n_scat` | current `simus_compute()` | CUDA fast path without sweep |     `simus_cuda()` only |
| -------: | ------------------------: | ---------------------------: | ----------------------: |
|     100K |    14.7 ms = 6.8 M scat/s |       6.6 ms = 15.2 M scat/s |  6.5 ms = 15.4 M scat/s |
|       1M |   132.1 ms = 7.6 M scat/s |      58.2 ms = 17.2 M scat/s | 58.0 ms = 17.2 M scat/s |

`_irfft_and_threshold()` plus correction factor is only ~0.55 ms for the benchmark shape, so the first-order loss is not
IFFT and not `simus_precompute()`.

## Desired End State

After this plan is complete:

1. `simus_compute(..., strategy=SimusStrategy.CUDA)` does not call `_prepare_simus_sweep()`.
1. CuPy auto strategy (`strategy=None` with CuPy inputs) also avoids `_prepare_simus_sweep()`.
1. `METAL` avoids `_prepare_simus_sweep()` too, matching the same backend-specific pattern as CUDA.
1. `PYTHON` and `SCAN` still call `_prepare_simus_sweep()` and produce unchanged results.
1. CUDA benchmark throughput on the 4090 reaches approximately:
   - 100K scatterers: at least 14 M scat/s end-to-end.
   - 1M scatterers: at least 15 M scat/s end-to-end, expected ~17 M scat/s.
1. Optional follow-up: profile/tune the kernel itself toward a stable 20 M scat/s without changing the public API.

### Key Discoveries

- `simus_precompute()` is already outside the timed benchmark in `tests/benchmarks/bench_simus_scaling.py`; do not move
  it or rewrite it for CUDA as part of this plan.
- `_prepare_simus_sweep()` is required by `_simus_freq_outer_python` and `_simus_freq_outer_scan` only.
- CUDA and Metal both have backend-local preparation:
  - CUDA: `src/fast_simus/kernels/cuda_simus.py::_prepare_inputs`
  - Metal: `src/fast_simus/kernels/metal_simus.py::_prepare_common`
- The MLX/Metal setup is the architectural hint: it already ignores `sweep`; the current issue is just that `sweep` is
  still constructed before the Metal/CUDA branch gets a chance to skip it.
- The optional 20 M scat/s work should start from NCU evidence. Exp22 says the current v25c kernel is register-bound
  (`Block Limit Registers = 2`, ~254 regs/thread) with 520 B/thread local memory and ~65% L2 throughput.

## What We're NOT Doing

- Not writing a CUDA kernel for `simus_precompute()`. It is not in the timed region for scaling benchmarks.
- Not rewriting `_prepare_simus_sweep()` in CuPy Array API. CUDA and Metal do not consume the sweep tensors.
- Not changing the public `simus()` / `simus_compute()` signatures.
- Not changing numerical tolerances or reference behavior.
- Not changing the v25c CUDA kernel in the primary phase.
- Not making the optional 20 M scat/s kernel work block the 15-17 M scat/s fast-path win.
- Not adding broad caching of arbitrary user inputs in `simus_compute()`. Any caching beyond the existing kernel cache
  is optional follow-up and must be justified by profiling.

## Implementation Approach

Use TDD:

1. Add a regression test that makes `_prepare_simus_sweep()` raise and verifies CUDA still computes valid output.
1. Refactor `simus_compute()` to select the strategy before building `sweep`.
1. Keep the post-dispatch correction factor and `_irfft_and_threshold()` shared across all strategies.
1. Verify CUDA benchmarks improve and update the scaling figure.
1. Optionally run NCU and plan/attempt kernel-only tuning toward 20 M scat/s.

## File Structure

- Modify `src/fast_simus/simus.py`
  - Move `selected = _select_simus_strategy(xp, strategy)` before `_prepare_simus_sweep()`.
  - Gate `_prepare_simus_sweep()` behind the `PYTHON`/`SCAN` branch.
- Modify `tests/backend/test_cupy.py`
  - Add a CuPy regression test proving CUDA dispatch does not call `_prepare_simus_sweep()`.
- Modify `tests/test_simus.py`
  - Optionally add a strategy-level spy test proving Python/SCAN still call sweep. Keep this backend-agnostic.
- Modify `docs/figures/2026-04-cuda-scaling.png`
  - Regenerate after the benchmark.
- Optionally modify `docs/progress/plans/2026-04-28-cuda-cupy-backend.md`
  - Add a short note clarifying that the first figure was pre-fast-path and included dead sweep prep.
- Optional follow-up files:
  - `docs/progress/experiments/YYYY-MM-DD-cuda-20m-followup.md`
  - Potential new kernel variant only after NCU evidence, not in the primary fast-path patch.

## Phase 1: Regression Tests for CUDA Fast Path

### Overview

Write tests that fail on the current branch because CUDA dispatch still builds `_prepare_simus_sweep()` before reaching
`simus_cuda()`.

### Changes Required

#### 1. CuPy Dispatch Regression

**File:** `tests/backend/test_cupy.py`

**Add this test:**

```python
def test_simus_cuda_does_not_prepare_python_sweep(monkeypatch):
    """CUDA dispatch must skip _prepare_simus_sweep; v25c prepares flat inputs itself."""
    import fast_simus.simus as simus_mod

    params = P4_2v()
    n_scat = 9
    scat_np = np.stack([np.zeros(n_scat), np.linspace(1e-2, 5e-2, n_scat)], axis=-1).astype(np.float32)
    rc_np = np.ones(n_scat, dtype=np.float32)
    delays_np = np.zeros(params.n_elements, dtype=np.float32)

    scatterers = cp.asarray(scat_np)
    rc = cp.asarray(rc_np)
    delays = cp.asarray(delays_np)
    plan = simus_mod.simus_precompute(scatterers, rc, delays, params)

    def fail_prepare_sweep(*args, **kwargs):
        raise AssertionError("CUDA dispatch should not build _prepare_simus_sweep")

    monkeypatch.setattr(simus_mod, "_prepare_simus_sweep", fail_prepare_sweep)

    result = simus_mod.simus_compute(
        scatterers,
        rc,
        delays,
        plan,
        params,
        strategy=SimusStrategy.CUDA,
    )

    assert isinstance(result.rf, cp.ndarray)
    assert result.rf.shape[1] == params.n_elements
    assert bool(cp.all(cp.isfinite(result.rf)))
```

**Why `n_scat` follows `cuda_simus._B_SCAT`:** It is exactly one scatterer batch, so the test focuses on dispatch ordering
and avoids exercising the padding path in `cuda_simus._prepare_inputs`.

#### 2. Python Strategy Still Uses Sweep

**File:** `tests/test_simus.py`

**Add this test near `TestSimusStrategy`:**

```python
def test_python_strategy_prepares_sweep(monkeypatch):
    """Python strategy still builds the shared Array API sweep tensors."""
    import fast_simus.simus as simus_mod

    params = P4_2v()
    scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
    rc = np.ones(3)
    delays = np.zeros(params.n_elements)

    scatterers_arr = xp.asarray(scatterers)
    rc_arr = xp.asarray(rc)
    delays_arr = xp.asarray(delays)
    plan = simus_precompute(scatterers_arr, rc_arr, delays_arr, params)

    original_prepare = simus_mod._prepare_simus_sweep
    calls = {"count": 0}

    def spy_prepare_sweep(*args, **kwargs):
        calls["count"] += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(simus_mod, "_prepare_simus_sweep", spy_prepare_sweep)

    result = simus_compute(
        scatterers_arr,
        rc_arr,
        delays_arr,
        plan,
        params,
        strategy=SimusStrategy.PYTHON,
    )

    assert calls["count"] == 1
    assert np.max(np.abs(np.asarray(result.rf))) > 0
```

### Success Criteria

#### Automated Verification

- [x] CUDA no-sweep test fails before implementation:
  `uv run pytest tests/backend/test_cupy.py::test_simus_cuda_does_not_prepare_python_sweep -q -p no:xdist -p no:testmon`
  Expected failure: `AssertionError: CUDA dispatch should not build _prepare_simus_sweep`.
- [x] Python spy test passes before implementation:
  `uv run pytest tests/test_simus.py::TestSimusStrategy::test_python_strategy_prepares_sweep -q -p no:xdist -p no:testmon`
  Expected: 1 passed.

#### Manual Verification

- [x] Confirm the new CUDA test fails for the intended reason, not due to CuPy import, NVRTC compile, or numerical
  output.

**Implementation Note:** After completing this phase and all automated verification passes/fails as expected, pause here
for manual confirmation before proceeding to Phase 2.

______________________________________________________________________

## Phase 2: Refactor `simus_compute()` Strategy Dispatch

### Overview

Select the strategy before preparing the sweep. Dispatch CUDA and Metal directly. Build `_prepare_simus_sweep()` only
for `PYTHON` and `SCAN`.

### Changes Required

#### 1. Move Strategy Selection Before Sweep

**File:** `src/fast_simus/simus.py`

**Replace the middle of `simus_compute()` from the flattening section through strategy dispatch with:**

```python
    # Flatten scatterers for the frequency sweep
    n_scat = scatterers.shape[0] if scatterers.ndim >= 2 else 1
    scatterers_flat = xp.reshape(scatterers, (n_scat, 2)) if scatterers.ndim > 2 else scatterers
    rc_flat = xp.reshape(rc, (n_scat,)) if rc.ndim > 1 else rc

    selected = _select_simus_strategy(xp, strategy)

    if selected == SimusStrategy.METAL:
        import mlx.core as mx

        from fast_simus.kernels.metal_simus import simus_metal

        spect_selected = cast(
            Array,
            simus_metal(
                scatterers=cast(mx.array, scatterers_flat),
                rc=cast(mx.array, rc_flat),
                params=params,
                plan=plan,
                medium=medium,
                delays_clean=cast(mx.array, delays_clean),
                tx_apodization=cast(mx.array, tx_apodization),
            ),
        )
    elif selected == SimusStrategy.CUDA:
        from fast_simus.kernels.cuda_simus import simus_cuda

        spect_selected = cast(
            Array,
            simus_cuda(
                scatterers=scatterers_flat,
                rc=rc_flat,
                params=params,
                plan=plan,
                medium=medium,
                delays_clean=delays_clean,
                tx_apodization=tx_apodization,
            ),
        )
    else:
        sweep = _prepare_simus_sweep(
            scatterers_flat,
            delays_clean,
            tx_apodization,
            plan,
            params,
            medium,
            full_frequency_directivity=full_frequency_directivity,
            xp=xp,
        )
        if selected == SimusStrategy.SCAN:
            from fast_simus._simus_strategies import _simus_freq_outer_scan

            spect_selected = _simus_freq_outer_scan(rc=rc_flat, xp=xp, **sweep)
        else:
            from fast_simus._simus_strategies import _simus_freq_outer_python

            spect_selected = _simus_freq_outer_python(rc=rc_flat, xp=xp, **sweep)
```

Keep the existing shared tail unchanged:

```python
    # Apply correction factor
    spect_selected = spect_selected * xp.asarray(plan.correction_factor)

    rf, full_spectrum = _irfft_and_threshold(spect_selected, plan, params.n_elements, xp)

    return SimusResult(rf=rf, spectrum=full_spectrum)
```

#### 2. Preserve `full_frequency_directivity` Behavior

No extra code is needed. Before this refactor, CUDA and Metal already ignored `full_frequency_directivity` because they
ignored `sweep`; the refactor removes dead computation but does not change CUDA/Metal output.

If an implementer is tempted to add an error for `full_frequency_directivity=True` on CUDA/Metal, do not do it in this
plan. That would be a behavior/API change and belongs in a separate compatibility discussion.

### Success Criteria

#### Automated Verification

- [x] CUDA no-sweep test passes:
  `uv run pytest tests/backend/test_cupy.py::test_simus_cuda_does_not_prepare_python_sweep -q -p no:xdist -p no:testmon`
- [x] Python spy test still passes:
  `uv run pytest tests/test_simus.py::TestSimusStrategy::test_python_strategy_prepares_sweep -q -p no:xdist -p no:testmon`
- [x] CuPy backend tests pass: `uv run pytest tests/backend/test_cupy.py -q -p no:xdist -p no:testmon`
- [x] CuPy strategy matrix passes/skips correctly:
  `uv run pytest tests/test_simus.py -q -p no:xdist -p no:testmon -k cupy`
- [x] Full non-benchmark suite passes:
  `uv run pytest tests/ --ignore=tests/benchmarks -n 4 -p no:testmon -p no:benchmark`
- [x] Lint/type checks pass: `uv run --group lint ruff check src tests scripts` `uv run --group lint ty check`

#### Manual Verification

- [x] Inspect `src/fast_simus/simus.py` and confirm `_prepare_simus_sweep()` appears only inside the non-CUDA/non-Metal
  branch in `simus_compute()`.
- [x] Confirm CUDA and Metal branches still share correction factor and `_irfft_and_threshold()` with Python/SCAN.

**Implementation Note:** After completing this phase and all automated verification passes, pause here for manual
confirmation before proceeding to Phase 3.

______________________________________________________________________

## Phase 3: Benchmark the Fast Path and Regenerate the Figure

### Overview

Verify the performance claim with pytest-benchmark JSON, not ad hoc timings. Regenerate the scaling figure so the
plotted CuPy line reflects the fast-path dispatch ordering.

### Changes Required

#### 1. Run Targeted CuPy Scaling Benchmark

**Command:**

```bash
FASTSIMUS_DEVICE_LABEL="RTX 4090 CuPy fast-path" \
  uv run pytest tests/benchmarks/bench_simus_scaling.py \
    --benchmark-only --benchmark-autosave -p no:xdist -m scaling \
    -k cupy --n-scat=100000,1000000
```

**Expected benchmark rows:**

- `test_bench_simus_scaling[cupy-100000]`
- `test_bench_simus_scaling[cupy-1000000]`

**Expected performance envelope on RTX 4090:**

- 100K: mean \<= 7.2 ms, throughput >= 13.9 M scat/s.
- 1M: mean \<= 67 ms, throughput >= 14.9 M scat/s.

The target is intentionally slightly looser than the observed fast-path split (~6.6 ms at 100K, ~58.2 ms at 1M) to avoid
failing on normal run-to-run variance.

#### 2. Run Full CuPy Sweep for Plot

**Command:**

```bash
FASTSIMUS_DEVICE_LABEL="RTX 4090 CuPy fast-path" \
  uv run pytest tests/benchmarks/bench_simus_scaling.py \
    --benchmark-only --benchmark-autosave -p no:xdist -m scaling \
    -k cupy
```

#### 3. Regenerate Figure

**File:** `docs/figures/2026-04-cuda-scaling.png`

**Command:**

```bash
uv run python scripts/plot_benchmark_scaling.py \
  .benchmarks/Linux-CPython-3.12-64bit/*.json \
  -o docs/figures/2026-04-cuda-scaling.png
```

If the plot includes both pre-fast-path and fast-path CuPy JSONs, keep the labels distinct via `FASTSIMUS_DEVICE_LABEL`.
The current plotting script treats machine labels as series, so both lines may appear. For the conference figure, prefer
using only the latest fast-path JSON plus the PyMUST/MLX baselines by passing exact JSON file paths instead of a
wildcard.

#### 4. Optional Plan Doc Note

**File:** `docs/progress/plans/2026-04-28-cuda-cupy-backend.md`

Add a short note under Phase 8 explaining that the first committed figure captured pre-fast-path end-to-end throughput
because CUDA dispatch still paid `_prepare_simus_sweep()` cost.

Use this wording:

```markdown
> Follow-up note: the first CuPy scaling figure included dead `_prepare_simus_sweep()` work in
> `simus_compute()`. The CUDA kernel itself was already near the exp22 target; a later fast-path
> dispatch refactor should move end-to-end benchmark throughput from ~7 M scat/s to ~15-17 M scat/s.
```

### Success Criteria

#### Automated Verification

- [x] Targeted CuPy benchmark completes and writes JSON under `.benchmarks/Linux-CPython-3.12-64bit/`.
- [ ] 100K row throughput is >= 13.9 M scat/s.
- [x] 1M row throughput is >= 14.9 M scat/s.
- [x] `scripts/plot_benchmark_scaling.py` exits 0 and writes `docs/figures/2026-04-cuda-scaling.png`.
- [x] `uv run --group lint ruff check src tests scripts` passes.
- [x] `uv run --group lint ty check` passes.
- [x] `uv run --group lint mdformat README.md docs --wrap 120` passes or leaves only expected formatting changes that
  are committed with the docs update.

Observed 2026-04-28: targeted run wrote
`.benchmarks/Linux-CPython-3.12-64bit/0003_74696bdbaff2cfaefc4c857a1f5c951b57ce18ed_20260428_210200_uncommited-changes.json`.
100K measured 8.4616 ms = 11.82 M scat/s, below the 13.9 M scat/s gate. 1M measured 60.1277 ms = 16.63 M scat/s, above
the 14.9 M scat/s gate. A 100K-only rerun measured 8.4823 ms = 11.79 M scat/s, confirming the 100K miss. User guidance:
prioritize the 1M scale because it is the larger time sink; do not block Phase 3 on the 100K miss unless an easy cleanup
falls out naturally. Full CuPy sweep wrote
`.benchmarks/Linux-CPython-3.12-64bit/0005_74696bdbaff2cfaefc4c857a1f5c951b57ce18ed_20260428_212045_uncommited-changes.json`.
The regenerated figure used that JSON plus the PyMUST CPU baseline
`.benchmarks/Linux-CPython-3.12-64bit/0002_60a71e35b185e0eb00ed75ce6f27a9ec7362dd9b_20260428_195358.json`.
After the Phase 4 `B_SCAT=10` constant adjustment, the refreshed CuPy sweep wrote
`.benchmarks/Linux-CPython-3.12-64bit/0006_329560fd18e690bc806f269e7cb0dca1f76a85a6_20260428_234933_uncommited-changes.json`.
The 100K row improved to 8.1315 ms = 12.30 M scat/s, and the 1M row improved to 59.6752 ms = 16.76 M scat/s.

#### Manual Verification

- [ ] Open `docs/figures/2026-04-cuda-scaling.png` and confirm the fast-path CuPy throughput line is around 15-17 M
  scat/s at 100K/1M.
- [x] Confirm the figure legend makes it clear whether it shows pre-fast-path CuPy, fast-path CuPy, or both.
- [x] Confirm speedup-vs-PyMUST still renders when PyMUST rows are present.

**Implementation Note:** After completing this phase and all automated verification passes, pause here for manual
confirmation before proceeding to the optional Phase 4.

______________________________________________________________________

## Phase 4 (Optional): Kernel-Only Path Toward 20 M scat/s

### Overview

Only start this phase after Phase 3 demonstrates ~15-17 M scat/s end-to-end. This phase is profiling-driven: use Nsight
Compute to decide whether the remaining gap to 20 M/s is in wrapper overhead, output allocation/zeroing, atomics,
register pressure, or the known local-memory spill path.

### Changes Required

#### 1. Capture NCU Baseline on Fast Path

**Command:**

```bash
sudo $(which python) scripts/ncu_pytest.py \
  -k "cupy and 100000 and not 1000000" \
  -o /tmp/4090_fastpath_100k.ncu-rep \
  --launch-skip 5 --launch-count 1 \
  --bench-path tests/benchmarks/bench_simus_scaling.py
```

**Record these metrics in a new experiment doc.**

**File:** `docs/progress/experiments/2026-04-cuda-fastpath-ncu.md`

Create with this structure, replacing the example metric names with actual measured values before committing:

```markdown
# CUDA fast-path NCU baseline

## Setup

- GPU: RTX 4090
- Kernel: `simus_fused_kernel` v25c (`B_SCAT=9`, `ELEM_TILE=2`, `TG_SIZE=128`, `TILE_SE=16`)
- Benchmark: `test_bench_simus_scaling[cupy-100000]`

## Fast-path benchmark

Record the `test_bench_simus_scaling[cupy-100000]` mean runtime and throughput from the pytest-benchmark JSON.

## NCU metrics

Record these values from the `.ncu-rep` report:

- Compute (SM) Throughput
- L2 Cache Throughput
- Achieved Occupancy
- Block Limit Registers
- Local Memory / Thread
- Eligible Warps / Scheduler

## Interpretation

- Name the current limiter using the NCU evidence above.
- Name the one next experiment selected from Experiment A, B, or C.
```

Do not commit the experiment doc until it contains concrete measured values and an interpretation based on those values.

#### 2. Choose One Experiment Based on NCU Evidence

Use exactly one of these experiments for the next commit; do not combine them.

**Experiment A: Wrapper allocation/cache split**

When to choose:

- NCU kernel duration is near exp22 (~5.4 ms at 100K) but Python/CuPy wrapper timing remains materially higher.

Implementation sketch:

- Add a private `CudaSimusPrepared` `NamedTuple` in `src/fast_simus/kernels/cuda_simus.py` containing static arrays:
  `elem_x`, `elem_z`, `cos_te`, `sin_neg_te`, `sub_dx`, `sub_dz`, `da_init_re`, `da_init_im`, `dps`, `pp_re`, `pp_im`,
  `probe_real`, scalar fields, and `kernel`.
- Add `_prepare_static_inputs(...) -> CudaSimusPrepared`.
- Keep `simus_cuda(...)` public/private API unchanged by calling `_prepare_static_inputs(...)` internally first.
- Add a separate benchmark-only helper later only if needed; do not expose a public cache in this first experiment.

**Experiment B: Output buffer reuse inside benchmark helper**

When to choose:

- NCU shows kernel time includes large zero-fill or allocation overhead around `spect_re`/`spect_im`, or CuPy profiler
  shows allocation dominates small/medium sizes.

Implementation sketch:

- Do not change public `simus_cuda()` yet.
- Add an internal `_launch_simus_cuda(..., spect_re, spect_im) -> cp.ndarray` that accepts caller-provided zeroed
  buffers.
- Keep `simus_cuda()` allocating fresh buffers for API safety.
- Add a benchmark-only path only after confirming the reusable-buffer path preserves correctness.

**Experiment C: Kernel tuning from exp22 follow-up**

When to choose:

- NCU shows the kernel itself is the dominant limiter and still has v25c-style register/local memory pressure.

Implementation sketch:

- Start from exp22's recommended high-leverage direction: fp16/half2 `cv` chain or another way to reduce local memory
  traffic/register pressure.
- Create a new kernel source variant only if the experiment is substantial: `src/fast_simus/kernels/simus_fused_v27.cu`
- Add wrapper selection behind a private module constant while experimenting; do not make a user-facing strategy.
- Validate against `SimusStrategy.PYTHON` within `_SIMUS_ATOL_PEAK`.

### Success Criteria

#### Automated Verification

- [x] NCU command produces a `.ncu-rep` file.
- [x] Experiment doc contains concrete measured values from pytest-benchmark and NCU before commit.
- [x] Any kernel variant passes CUDA backend tests:
  `uv run pytest tests/backend/test_cupy.py -q -p no:xdist -p no:testmon`
- [ ] Any public dispatch changes pass full suite:
  `uv run pytest tests/ --ignore=tests/benchmarks -n 4 -p no:testmon -p no:benchmark`
- [x] Any benchmark improvement is measured with pytest-benchmark JSON, not hand timing.

#### Manual Verification

- [x] Compare NCU metrics to exp22 (`Compute (SM)`, `L2`, occupancy, local memory/thread).
- [x] Decide whether the next bottleneck is wrapper overhead or kernel execution based on NCU, not intuition.
- [ ] If a 20 M/s experiment is committed, record before/after rows in the experiment doc.

______________________________________________________________________

## Testing Strategy

### Unit Tests

- `tests/backend/test_cupy.py::test_simus_cuda_does_not_prepare_python_sweep`
  - Proves CUDA dispatch skips `_prepare_simus_sweep()`.
- `tests/test_simus.py::TestSimusStrategy::test_python_strategy_prepares_sweep`
  - Proves the refactor did not remove sweep prep from the Python strategy.
- Existing `tests/backend/test_cupy.py::test_simus_cuda_matches_python_strategy`
  - Ensures numerical equivalence to Python strategy remains within `5e-3 * peak`.
- Existing `tests/test_simus.py::TestSimusStrategyCrossBackend::test_strategy_on_backend`
  - Exercises auto/python/scan/metal/cuda compatibility matrix.

### Integration Tests

- Full non-benchmark suite:

```bash
uv run pytest tests/ --ignore=tests/benchmarks -n 4 -p no:testmon -p no:benchmark
```

- CuPy-specific integration:

```bash
uv run pytest tests/backend/test_cupy.py tests/test_simus.py -v -k cupy -p no:xdist -p no:testmon
```

### Benchmark Verification

- Targeted performance gate:

```bash
FASTSIMUS_DEVICE_LABEL="RTX 4090 CuPy fast-path" \
  uv run pytest tests/benchmarks/bench_simus_scaling.py \
    --benchmark-only --benchmark-autosave -p no:xdist -m scaling \
    -k cupy --n-scat=100000,1000000
```

- Full plot update:

```bash
FASTSIMUS_DEVICE_LABEL="RTX 4090 CuPy fast-path" \
  uv run pytest tests/benchmarks/bench_simus_scaling.py \
    --benchmark-only --benchmark-autosave -p no:xdist -m scaling \
    -k cupy
```

### Manual Testing Steps

1. Open the regenerated `docs/figures/2026-04-cuda-scaling.png`.
1. Confirm the CuPy fast-path line is around 15-17 M scat/s at 100K/1M.
1. Confirm the plot is labeled clearly enough for the conference abstract.
1. If optional Phase 4 is attempted, open the NCU report in Nsight Compute and record the key metrics in the experiment
   doc before committing.

## Performance Considerations

- The primary win comes from not materializing Python/SCAN phase tensors for CUDA/Metal. It is a dispatch-order fix.
- IFFT is not the main bottleneck for the P4-2v benchmark shape; it measured ~0.55 ms.
- `simus_precompute()` is not in the timed scaling benchmark. Moving it to CUDA would not improve the current plotted
  throughput.
- CUDA and Metal currently ignore `full_frequency_directivity` in practice. This plan preserves that behavior.
- The optional 20 M/s work should be treated as a kernel/wrapper profiling task. Exp22 indicates the kernel is
  register-bound and local-memory-heavy; without fresh NCU data, choosing fp16/half2 or buffer reuse would be guessing.

## Migration Notes

- No data migrations.
- No public API changes.
- Existing saved benchmark JSONs remain valid historical artifacts but should be relabeled or replaced in the conference
  figure to avoid mixing pre-fast-path and fast-path CuPy lines unintentionally.
- If this branch already has a committed figure, regenerate it after Phase 3 and commit the updated PNG in the same PR.

## References

- Existing dispatch: `src/fast_simus/simus.py`
- CUDA wrapper: `src/fast_simus/kernels/cuda_simus.py`
- Metal wrapper pattern: `src/fast_simus/kernels/metal_simus.py`
- Scaling benchmark: `tests/benchmarks/bench_simus_scaling.py`
- Benchmark sync helper: `tests/benchmarks/_bench_sync.py`
- v25c performance writeup: `docs/progress/experiments/exp22-svshmem-et2.md`
- Current CUDA backend implementation plan: `docs/progress/plans/2026-04-28-cuda-cupy-backend.md`
