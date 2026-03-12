# FastSIMUS MLX & Pfield Research & Learnings

Summary of research and implementation learnings across FastSIMUS branches (feature/mlx, feature/pfield, feature/benchmark, etc.).

## MLX Array API Compatibility

**Problem:** MLX is not fully Array API compliant. `array_api_compat` does not yet support MLX (tracking: https://github.com/data-apis/array-api-compat/issues/162).

**Solution:** Dedicated `src/fast_simus/backends/mlx.py` module that patches the MLX namespace when `array_namespace()` detects MLX arrays.

### Patches in `backends/mlx.py`

| Gap | Fix |
|-----|-----|
| `asin`, `acos`, `atan2`, `bool` | Alias to `arcsin`, `arccos`, `arctan2`, `bool_` |
| `isdtype` | Custom `_make_isdtype()` using `issubdtype` |
| `astype` | Wrapper around `x.astype(dtype)` |
| `asarray` | Wrapper with `_fastsimus_wrapped` sentinel to prevent re-wrapping |
| `device()` | Patch `array_api_compat` helpers to return `xp.default_device()` for MLX unified memory |

### Protocol Trimming

`_ArrayNamespace` and `Array` protocols were trimmed to avoid MLX-incompatible members:
- Removed: `complex128`, `empty`, `empty_like`, `acos`, `vecdot`, `nonzero`, `__pos__`, `__index__`
- Not used in codebase; MLX compatibility is the driver.

### Boolean Indexing

MLX does not support boolean indexing. `_FrequencyPlan` had `freq_mask` removed; replaced with slice indexing so `pfield_precompute` / `pfield_compute` work with MLX.

## Type Checking (Beartype / jaxtyping)

**Problem:** MLX arrays fail beartype/jaxtyping validation because MLX is not fully Array API compliant (`Array` protocol).

**Solution:** `FASTSIMUS_SKIP_TYPECHECK` env var switches to a no-op typechecker in `pfield.py`, `spectrum.py`, `geometry.py`. Use when benchmarking or profiling MLX:
- `poe benchmark` task sets `FASTSIMUS_SKIP_TYPECHECK=1`
- Profile scripts set it when `--which` is `mlx`, `mlx-compile`, or `all`

## MLX Compile (mx.compile)

**Problem:** Python frequency loop in `_pfield_core_with_plan` prevents effective graph fusion under `mx.compile`.

**Solution:** For MLX, use `mx.vmap` over the frequency dimension instead of a Python loop:
- `_pfield_freq_loop_vmap`: precompute `phase_decays` as `phase_decay_init * (phase_decay_step ** exponents)`, then `mx.vmap(step_fn, in_axes=(0,0,0,0))` over `(phase_decays, pulse_spect, probe_spect, wavenumbers)`
- Result: ~3.3x speedup (e.g. grid32: ~32 ms eager -> ~9.6 ms compiled; grid64: ~49 ms -> ~37 ms)

**Test:** `tests/backend/test_mlx.py` verifies `mx.compile(lambda pos, dl: pfield_compute(pos, dl, plan, params))` produces valid output.

## Benchmark & Profile Setup

- Benchmark task in `pyproject.toml`: `env = { FASTSIMUS_SKIP_TYPECHECK = "1" }` for MLX runs
- Profile script (`scripts/profile_pfield.py`): `--which mlx`, `--which mlx-compile`, `--which all` for MLX variants

## PyMUST Validation

FastSIMUS pfield is validated against PyMUST reference (rtol=1e-4). `tests/test_pfield.py` uses `array_api_strict` for NumPy correctness. MLX/JAX tests use different fixtures and do not run PyMUST comparison.

## References

- Plan: `thoughts/shared/plans/2026-03-03-mlx-compat-refactor.md`
- Array API compat MLX issue: https://github.com/data-apis/array-api-compat/issues/162
- MLX docs: https://ml-explore.github.io/mlx/build/html/
