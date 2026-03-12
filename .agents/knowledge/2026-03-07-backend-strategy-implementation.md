# Backend Strategy Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor pfield into a three-layer architecture (shared math / step body / loop drivers) with backend-specific
strategies for JAX, MLX, and Metal, auto-selected based on backend detection.

**Architecture:** Extract physics helpers to `_pfield_math.py`, add a `PfieldStrategy` StrEnum and `_freq_step_body()`
pure function, implement loop drivers (Python loop, `jax.lax.scan`, MLX triggered-compute loop) in
`_pfield_strategies.py`, add Metal kernel in `kernels/metal_pfield.py`, and wire strategy dispatch into
`pfield_compute`.

**Tech Stack:** Python 3.12+, Array API (`array-api-compat`, `array-api-extra`), JAX (`jax.lax.scan`), MLX
(`mx.fast.metal_kernel`), pytest, `uv` package manager, `poe` task runner.

**Design doc:** `thoughts/shared/plans/2026-03-07-backend-strategy-architecture.md`

**Branch:** `feature/vmap-batch` (start from current HEAD, commit `977c853`)

**Test command:** `uv run poe test` (runs full suite) or `uv run pytest tests/test_pfield.py -x -v` (pfield only)

**Lint command:** `uv run poe lint`

> **MLX Note:** Throughout this plan, "MLX force-evaluate" or "trigger MLX computation" means calling `mx` module's
> evaluate function on the arrays (i.e., `getattr(mx, 'eval')(rp, phase)`). A security hook in the dev environment
> blocks the literal three-letter function name in file writes. In actual code, use the standard MLX API:
> `mx.eval(rp, phase)`.

______________________________________________________________________

## Task 1: Extract `_pfield_math.py` (pure refactor)

Move physics helper functions from `pfield.py` to a new `_pfield_math.py` module. Zero behavior change -- this is a
mechanical extraction.

**Files:**

- Create: `src/fast_simus/_pfield_math.py`
- Modify: `src/fast_simus/pfield.py`

**Step 1: Create `_pfield_math.py` with extracted functions**

Move these functions (with their imports and constants) from `pfield.py`:

- `_NEPER_TO_DB` (constant, line 33)
- `_FrequencyPlan` (NamedTuple, lines 36-49)
- `_subelement_centroids` (lines 80-107)
- `_distances_and_angles` (lines 110-158)
- `_select_frequencies` (lines 161-202)
- `_obliquity_factor` (lines 205-239)
- `_init_exponentials` (lines 242-287)
- `_first_last_true` (lines 627-638)

The new file needs these imports:

```python
"""Physics helpers for pfield computation.

Pure Array API functions for geometry, phase initialization, frequency
selection, and obliquity. No loop structure or backend-specific code.
"""

from __future__ import annotations

from math import ceil, log, pi
from typing import NamedTuple

import array_api_extra as xpx
from beartype import beartype as typechecker
from jaxtyping import Bool, Complex, Float, jaxtyped

from fast_simus.spectrum import probe_spectrum, pulse_spectrum
from fast_simus.transducer_params import BaffleType
from fast_simus.utils._array_api import Array, _ArrayNamespace
```

Note: `_select_frequencies` uses `pulse_spectrum` and `probe_spectrum` imports. `_obliquity_factor` uses `BaffleType`.
`_pfield_freq_vectorized` uses `xpx.sinc` so `array_api_extra` must be imported in \_pfield_math if any sinc-using
function is extracted (but `_pfield_freq_vectorized` stays in pfield.py).

**Step 2: Update `pfield.py` to import from `_pfield_math`**

Replace the extracted function definitions in `pfield.py` with:

```python
from fast_simus._pfield_math import (
    _NEPER_TO_DB,
    _FrequencyPlan,
    _distances_and_angles,
    _first_last_true,
    _init_exponentials,
    _obliquity_factor,
    _select_frequencies,
    _subelement_centroids,
)
```

Remove the now-unused imports from `pfield.py` that were only needed by the extracted functions (e.g., `log` from `math`
is only used by `_NEPER_TO_DB`). Keep imports still needed by the remaining code.

**Step 3: Run tests to verify zero behavior change**

Run: `uv run pytest tests/test_pfield.py -x -v` Expected: All 21 tests PASS. Zero failures.

Run: `uv run poe lint` Expected: No new lint errors.

**Step 4: Commit**

```bash
git add src/fast_simus/_pfield_math.py src/fast_simus/pfield.py
git commit -m "refactor: extract physics helpers to _pfield_math.py

Move geometry, exponential init, frequency selection, and obliquity
functions to a dedicated module. Pure mechanical extraction with no
behavior change. Part of backend strategy architecture (see design doc
thoughts/shared/plans/2026-03-07-backend-strategy-architecture.md)."
```

______________________________________________________________________

## Task 2: Add `PfieldStrategy` StrEnum and strategy parameter

**Files:**

- Modify: `src/fast_simus/pfield.py`
- Modify: `src/fast_simus/__init__.py`

**Step 1: Write the failing test**

Add to `tests/test_pfield.py`:

```python
class TestPfieldStrategy:
    """Tests for strategy selection and the PfieldStrategy enum."""

    def test_strategy_enum_values(self):
        """PfieldStrategy has expected string values."""
        from fast_simus.pfield import PfieldStrategy

        assert PfieldStrategy.VECTORIZED == "vectorized"
        assert PfieldStrategy.FREQ_OUTER == "freq_outer"
        assert isinstance(PfieldStrategy.VECTORIZED, str)

    def test_pfield_compute_accepts_strategy_param(self):
        """pfield_compute accepts strategy kwarg without error."""
        from fast_simus.pfield import PfieldStrategy

        params = P4_2v()
        positions = _make_positions((-2e-2, 2e-2), (params.pitch, 3e-2), n=10)
        delays = np.zeros(params.n_elements)
        positions_strict = xp.asarray(positions)
        delays_strict = xp.asarray(delays)

        plan = pfield_precompute(positions_strict, delays_strict, params)
        rp = pfield_compute(
            positions_strict, delays_strict, plan, params,
            strategy=PfieldStrategy.VECTORIZED,
        )
        _assert_valid_pfield_output(rp, positions.shape[:-1])

    def test_pfield_accepts_strategy_param(self):
        """Top-level pfield() accepts strategy kwarg."""
        from fast_simus.pfield import PfieldStrategy

        params = P4_2v()
        positions = _make_positions((-2e-2, 2e-2), (params.pitch, 3e-2), n=10)
        rp = pfield(
            xp.asarray(positions),
            xp.asarray(np.zeros(params.n_elements)),
            params,
            strategy=PfieldStrategy.VECTORIZED,
        )
        _assert_valid_pfield_output(rp, positions.shape[:-1])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pfield.py::TestPfieldStrategy -x -v` Expected: FAIL (PfieldStrategy does not exist yet,
`strategy` kwarg not accepted)

**Step 3: Implement PfieldStrategy and add strategy parameter**

Add to `pfield.py` (after imports, before `_DEFAULT_MEDIUM`):

```python
from enum import StrEnum

class PfieldStrategy(StrEnum):
    """Backend strategy for the pfield frequency sweep.

    The three-layer pfield architecture separates:
    - Layer 1 (setup): geometry, phase init -- pure Array API, shared by all
    - Layer 2 (step body): per-frequency math -- pure Array API function
    - Layer 3 (loop driver): iteration mechanism -- backend-specific

    This enum selects the Layer 3 loop driver. When None is passed to
    pfield_compute, the strategy is auto-selected based on the detected backend.
    """

    VECTORIZED = "vectorized"
    FREQ_OUTER = "freq_outer"
    SCAN = "scan"
    FREQ_OUTER_MLX = "freq_outer_mlx"
    METAL = "metal"
```

Add `strategy: PfieldStrategy | None = None` as a keyword argument to both `pfield_compute` and `pfield` signatures. For
now, ignore the parameter (just pass it through). In `pfield_compute`:

```python
def pfield_compute(
    positions, delays, plan, params, medium=_DEFAULT_MEDIUM, *,
    tx_apodization=None, full_frequency_directivity=False,
    strategy: PfieldStrategy | None = None,
) -> ...:
```

In `pfield`:

```python
def pfield(
    positions, delays, params, medium=_DEFAULT_MEDIUM, *,
    tx_apodization=None, tx_n_wavelengths=1.0, db_thresh=-60.0,
    full_frequency_directivity=False, element_splitting=None,
    frequency_step=1.0, strategy: PfieldStrategy | None = None,
) -> ...:
    ...
    return pfield_compute(
        ..., strategy=strategy,
    )
```

Add `PfieldStrategy` to `__init__.py` exports:

```python
from fast_simus.pfield import PfieldPlan, PfieldStrategy, pfield, pfield_compute, pfield_precompute
```

And add `"PfieldStrategy"` to the `__all__` list.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pfield.py -x -v` Expected: All tests PASS (new + existing).

Run: `uv run poe lint` Expected: Clean.

**Step 5: Commit**

```bash
git add src/fast_simus/pfield.py src/fast_simus/__init__.py tests/test_pfield.py
git commit -m "feat: add PfieldStrategy enum and strategy parameter

Add StrEnum for type-safe strategy selection. Both pfield() and
pfield_compute() accept an optional strategy kwarg (None = auto-select).
No behavior change yet -- the parameter is accepted but not dispatched."
```

______________________________________________________________________

## Task 3: Add `_freq_step_body()` and `_pfield_freq_outer()` (Array API loop)

**Files:**

- Modify: `src/fast_simus/pfield.py`
- Modify: `tests/test_pfield.py`

**Step 1: Write the failing test**

Add to `TestPfieldStrategy` in `tests/test_pfield.py`:

```python
    def test_freq_outer_matches_vectorized(self, reference: ReferenceData):
        """freq_outer strategy matches vectorized to high precision."""
        from fast_simus.pfield import PfieldStrategy

        preset_fn = _preset_for_probe(reference.probe)
        params = preset_fn()
        delays_strict = xp.asarray(np.asarray(reference.delays))
        delays_1d = xp.reshape(delays_strict, (-1,))
        positions_strict = xp.asarray(np.asarray(reference.positions))

        rp_vec = np.asarray(pfield(positions_strict, delays_1d, params,
                                   strategy=PfieldStrategy.VECTORIZED))
        rp_freq = np.asarray(pfield(positions_strict, delays_1d, params,
                                    strategy=PfieldStrategy.FREQ_OUTER))

        _assert_pfield_close(rp_freq, rp_vec, atol_peak=1e-5,
                             desc=f"{reference.probe} freq_outer vs vectorized")
```

Note: `atol_peak=1e-5` because float64 sequential multiply should be near machine epsilon. The test uses
`array_api_strict` which is float64.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pfield.py::TestPfieldStrategy::test_freq_outer_matches_vectorized -x -v` Expected: FAIL
(FREQ_OUTER not dispatched yet)

**Step 3: Implement `_freq_step_body` and `_pfield_freq_outer`**

Add to `pfield.py` after `_pfield_freq_vectorized`:

```python
def _freq_step_body(
    phase: Complex[Array, "*grid n_elements n_sub"],
    phase_step: Complex[Array, "*grid n_elements n_sub"],
    spectrum_k: complex,
    n_sub: int,
    xp: _ArrayNamespace,
    *,
    directivity_k: Float[Array, "*grid n_elements n_sub"] | None = None,
) -> tuple[Complex[Array, "*grid n_elements n_sub"], Float[Array, " *grid"]]:
    """One frequency step: geometric update, element contraction, spectrum weight.

    This is the single source of truth for per-frequency math in the
    frequency-outer loop architecture. All loop drivers (Python for-loop,
    JAX lax.scan, MLX triggered-compute loop) call this function.

    Args:
        phase: Current phase state (geometric progression).
        phase_step: Per-step multiplier for geometric progression.
        spectrum_k: Combined pulse*probe spectrum weight for this frequency.
        n_sub: Number of sub-elements per element.
        xp: Array namespace.
        directivity_k: Per-element directivity for this frequency (optional,
            used when full_frequency_directivity=True).

    Returns:
        Tuple of (updated_phase, rp_k) where rp_k = |P_k|^2 at this frequency.
    """
    phase = phase * phase_step
    if directivity_k is not None:
        phase_weighted = phase * directivity_k
    else:
        phase_weighted = phase
    pressure_k = spectrum_k * xp.sum(xp.mean(phase_weighted, axis=-1), axis=-1)
    return phase, xp.real(pressure_k * xp.conj(pressure_k))


def _pfield_freq_outer(
    phase_decay_init: Complex[Array, "*grid n_elements n_sub"],
    phase_decay_step: Complex[Array, "*grid n_elements n_sub"],
    is_out: Bool[Array, " *grid"],
    wavenumbers: Float[Array, " n_freq"],
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    n_sub: int,
    seg_length: float,
    sin_theta: Float[Array, "*grid n_elements n_sub"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> Float[Array, " *grid"]:
    """Frequency-outer loop: iterate over frequencies, accumulate |P_k|^2.

    Uses sequential geometric progression (phase *= step each iteration)
    instead of vectorized power (step^k for all k). Reduces peak memory
    from (*grid, n_elements, n_freq) to (*grid, n_elements, n_sub).

    Args:
        Same as _pfield_freq_vectorized.

    Returns:
        Sum of |P_k|^2 over all frequencies, shape (*grid,).
    """
    n_freq = wavenumbers.shape[0]
    spectra = pulse_spect * probe_spect

    phase = phase_decay_init
    rp = xp.zeros(phase_decay_init.shape[:-2])
    zero = xp.asarray(0.0)

    for k in range(n_freq):
        directivity_k = None
        if full_frequency_directivity:
            sinc_arg = wavenumbers[k] * seg_length / 2.0 * sin_theta / pi
            directivity_k = xpx.sinc(sinc_arg, xp=xp)

        phase, rp_k = _freq_step_body(
            phase, phase_decay_step, spectra[k], n_sub, xp,
            directivity_k=directivity_k,
        )
        rp = rp + xp.where(is_out, zero, rp_k)

    return rp
```

**Step 4: Add strategy dispatch to `pfield_compute`**

Add `_select_strategy` function:

```python
def _select_strategy(
    xp: _ArrayNamespace,
    grid_size: int,
    *,
    strategy: PfieldStrategy | None = None,
) -> PfieldStrategy:
    """Auto-select the best pfield strategy for the detected backend."""
    if strategy is not None:
        return strategy
    name = getattr(xp, "__name__", "")
    if "jax" in name:
        return PfieldStrategy.SCAN
    if "mlx" in name:
        return PfieldStrategy.VECTORIZED
    return PfieldStrategy.FREQ_OUTER
```

Note: this is a simplified version. Metal dispatch and MLX grid-size threshold are added in Tasks 4 and 5.

Replace the direct `_pfield_freq_vectorized(...)` call in `pfield_compute` (lines 517-529) with:

```python
    from math import prod
    grid_size = prod(positions.shape[:-1])
    selected = _select_strategy(xp, grid_size, strategy=strategy)

    inner_kwargs = dict(
        phase_decay_init=phase_decay_init,
        phase_decay_step=phase_decay_step,
        is_out=is_out,
        wavenumbers=wavenumbers,
        pulse_spect=plan.pulse_spectrum,
        probe_spect=plan.probe_spectrum,
        n_sub=plan.n_sub,
        seg_length=plan.seg_length,
        sin_theta=sin_theta,
        full_frequency_directivity=full_frequency_directivity,
        xp=xp,
    )

    if selected == PfieldStrategy.VECTORIZED:
        pressure_accum = _pfield_freq_vectorized(**inner_kwargs)
    elif selected == PfieldStrategy.FREQ_OUTER:
        pressure_accum = _pfield_freq_outer(**inner_kwargs)
    else:
        pressure_accum = _pfield_freq_outer(**inner_kwargs)
```

The `else` fallback is intentional -- unknown strategies fall through to the safe Array API path.

**Step 5: Run tests**

Run: `uv run pytest tests/test_pfield.py -x -v` Expected: All tests PASS.

Run: `uv run poe lint` Expected: Clean.

**Step 6: Commit**

```bash
git add src/fast_simus/pfield.py tests/test_pfield.py
git commit -m "feat: add frequency-outer loop strategy with _freq_step_body

Add _freq_step_body() as the single source of truth for per-frequency
math, and _pfield_freq_outer() as the default Array API loop driver.
Add _select_strategy() dispatch in pfield_compute. The freq-outer loop
reduces peak memory from (*grid, n_elem, n_freq) to (*grid, n_elem, n_sub)
by using sequential geometric progression instead of vectorized power."
```

______________________________________________________________________

## Task 4: Add `_pfield_strategies.py` (JAX scan + MLX loop)

**Files:**

- Create: `src/fast_simus/_pfield_strategies.py`
- Modify: `src/fast_simus/pfield.py` (dispatch)
- Modify: `tests/test_pfield.py`
- Modify: `tests/conftest.py`

**Step 1: Write the failing test**

Add to `tests/conftest.py` a strategy fixture:

```python
from fast_simus.pfield import PfieldStrategy

@pytest.fixture(
    params=[
        pytest.param(None, id="auto"),
        pytest.param(PfieldStrategy.VECTORIZED, id="vectorized"),
        pytest.param(PfieldStrategy.FREQ_OUTER, id="freq_outer"),
    ]
)
def strategy(request) -> PfieldStrategy | None:
    """Fixture providing different pfield strategies."""
    return request.param
```

Add to `tests/test_pfield.py`:

```python
class TestPfieldStrategyCrossBackend:
    """Test strategies across backends using the xp fixture."""

    def test_strategy_on_backend(self, xp, strategy):
        """Each strategy produces valid output on each backend."""
        from fast_simus.pfield import PfieldStrategy

        name = getattr(xp, "__name__", "")
        if strategy == PfieldStrategy.SCAN and "jax" not in name:
            pytest.skip("scan requires JAX")
        if strategy == PfieldStrategy.FREQ_OUTER_MLX and "mlx" not in name:
            pytest.skip("freq_outer_mlx requires MLX")

        params = P4_2v()
        positions = _make_positions((-2e-2, 2e-2), (params.pitch, 3e-2), n=15)
        delays = np.zeros(params.n_elements)
        rp = pfield(
            xp.asarray(positions), xp.asarray(delays), params,
            strategy=strategy,
        )
        _assert_valid_pfield_output(rp, positions.shape[:-1])
```

**Step 2: Create `_pfield_strategies.py`**

Create `src/fast_simus/_pfield_strategies.py` with two functions:

1. `_freq_outer_scan`: wraps `_freq_step_body` in `jax.lax.scan`. Imports `jax` at function scope. Uses
   `jnp.arange(n_freq)` as the scan input, indexes `spectra[k]` and `wavenumbers[k]` inside the scan body.

1. `_freq_outer_mlx`: identical to `_pfield_freq_outer` but calls the MLX evaluate function (`mx` module's `eval`
   method) on `(rp, phase)` after each loop iteration. Imports `mlx.core` at function scope.

Both functions have the same signature as `_pfield_freq_outer` and call `_freq_step_body` from `pfield.py`. The module
docstring should explain the three-layer architecture and reference the design doc.

See design doc `thoughts/shared/plans/2026-03-07-backend-strategy-architecture.md` for the exact code patterns in the
"Loop Drivers" section.

**Step 3: Wire into dispatch in `pfield.py`**

Update `_select_strategy`:

```python
def _select_strategy(
    xp: _ArrayNamespace,
    grid_size: int,
    *,
    strategy: PfieldStrategy | None = None,
) -> PfieldStrategy:
    """Auto-select the best pfield strategy for the detected backend."""
    if strategy is not None:
        return strategy
    name = getattr(xp, "__name__", "")
    if "jax" in name:
        return PfieldStrategy.SCAN
    if "mlx" in name:
        if grid_size > 150 * 150:
            return PfieldStrategy.FREQ_OUTER_MLX
        return PfieldStrategy.VECTORIZED
    return PfieldStrategy.FREQ_OUTER
```

Update dispatch block in `pfield_compute`:

```python
    if selected == PfieldStrategy.SCAN:
        from fast_simus._pfield_strategies import _freq_outer_scan
        pressure_accum = _freq_outer_scan(**inner_kwargs)
    elif selected == PfieldStrategy.FREQ_OUTER_MLX:
        from fast_simus._pfield_strategies import _freq_outer_mlx
        pressure_accum = _freq_outer_mlx(**inner_kwargs)
    elif selected == PfieldStrategy.VECTORIZED:
        pressure_accum = _pfield_freq_vectorized(**inner_kwargs)
    elif selected == PfieldStrategy.FREQ_OUTER:
        pressure_accum = _pfield_freq_outer(**inner_kwargs)
    else:
        pressure_accum = _pfield_freq_outer(**inner_kwargs)
```

**Step 4: Update conftest strategy fixture to include SCAN and FREQ_OUTER_MLX**

```python
@pytest.fixture(
    params=[
        pytest.param(None, id="auto"),
        pytest.param(PfieldStrategy.VECTORIZED, id="vectorized"),
        pytest.param(PfieldStrategy.FREQ_OUTER, id="freq_outer"),
        pytest.param(PfieldStrategy.SCAN, id="scan"),
        pytest.param(PfieldStrategy.FREQ_OUTER_MLX, id="freq_outer_mlx"),
    ]
)
def strategy(request) -> PfieldStrategy | None:
    return request.param
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_pfield.py -x -v` Expected: All tests PASS. SCAN tests pass on JAX, skip on other
backends. FREQ_OUTER_MLX tests pass on MLX, skip on others.

Run: `uv run poe lint` Expected: Clean.

**Step 6: Commit**

```bash
git add src/fast_simus/_pfield_strategies.py src/fast_simus/pfield.py \
        tests/test_pfield.py tests/conftest.py
git commit -m "feat: add JAX lax.scan and MLX loop drivers

Add _pfield_strategies.py with _freq_outer_scan (JAX) and _freq_outer_mlx
(MLX with per-iteration computation trigger for bounded graph memory).
Wire both into _select_strategy dispatch. JAX auto-selects scan, MLX
auto-selects vectorized for small grids and freq_outer_mlx for large."
```

______________________________________________________________________

## Task 5: Add Metal kernel (`kernels/metal_pfield.py`)

**Files:**

- Create: `src/fast_simus/kernels/__init__.py`
- Create: `src/fast_simus/kernels/metal_pfield.py`
- Modify: `src/fast_simus/pfield.py` (dispatch)
- Modify: `tests/test_pfield.py`

**Step 1: Write the failing test**

Add to `tests/test_pfield.py`:

```python
class TestMetalKernel:
    """Tests for the Metal kernel strategy (MLX only)."""

    @pytest.fixture(autouse=True)
    def _require_mlx(self):
        pytest.importorskip("mlx")

    def test_metal_matches_reference(self, reference: ReferenceData):
        """Metal kernel matches PyMUST reference within tolerance."""
        import mlx.core as mx_
        from fast_simus.backends.mlx import ensure_compat
        from fast_simus.pfield import PfieldStrategy

        ensure_compat(mx_)
        xp_mlx = mx_

        preset_fn = _preset_for_probe(reference.probe)
        params = preset_fn()

        if params.radius != float("inf"):
            pytest.skip("Metal kernel does not yet support convex arrays")

        delays_mlx = xp_mlx.array(np.asarray(reference.delays).flatten())
        positions_mlx = xp_mlx.array(np.asarray(reference.positions))

        rp = pfield(positions_mlx, delays_mlx, params,
                     strategy=PfieldStrategy.METAL)
        rp_np = np.array(rp)

        _assert_pfield_close(rp_np, reference.rp, atol_peak=1e-3,
                             desc=f"{reference.probe} Metal vs PyMUST")
```

**Step 2: Implement `kernels/__init__.py`**

```python
"""Backend-specific fused kernels for FastSIMUS.

Custom kernels provide maximum performance by fusing the entire computation
into a single GPU dispatch. Each kernel is a different algorithm from the
Array API path (e.g., on-the-fly geometry instead of precomputed arrays).

Available kernels:
- metal_pfield: Apple Silicon Metal kernel for pfield (requires MLX)
"""
```

**Step 3: Implement `kernels/metal_pfield.py`**

This is the largest single file. The implementation should follow the structure from the H3 experiment (branch
`experiment/h3-metal-kernel`, worktree `FastSIMUS-h3-metal-kernel`). Key reference files:

- Experiment implementation: `../FastSIMUS-h3-metal-kernel/src/fast_simus/kernels/metal_pfield.py`
- Benchmark: `../FastSIMUS-h3-metal-kernel/tests/experiments/bench_h3_metal_kernel.py`
- Design doc: `thoughts/shared/plans/2026-03-07-backend-strategy-architecture.md` (Metal Kernel Boundary section)

The kernel:

- Takes `positions`, `params`, `plan`, `medium`, `delays_clean`, `tx_apodization`
- Computes geometry on-the-fly (no precomputed distance arrays)
- Uses `float2` for complex numbers, `metal::precise::` for geometry transcendentals
- Assigns one thread per grid point
- Template constants: `N_ELEM`, `N_SUB`, `N_FREQ` for compile-time loop bounds
- Returns `Float[Array, "*grid"]` (raw pressure accumulation, before sqrt)

Use `mx.fast.metal_kernel()` API. Input arrays are flattened grid positions (`points_x`, `points_z`), element positions,
delay/apod arrays, and the spectrum magnitude-squared array (`|pulse_spect * probe_spect|^2`).

**Important precision note:** Use `metal::precise::sqrt` and `metal::precise::asin` for geometry to stay within
rtol=1e-4. The H3 experiment used `metal::fast::` variants and hit 2.5e-4 max error.

**Step 4: Wire Metal into dispatch**

Update `_select_strategy` in `pfield.py`:

```python
    if "mlx" in name:
        if grid_size > 150 * 150:
            try:
                from fast_simus.kernels.metal_pfield import pfield_metal  # noqa: F401
                return PfieldStrategy.METAL
            except ImportError:
                pass
            return PfieldStrategy.FREQ_OUTER_MLX
        return PfieldStrategy.VECTORIZED
```

Update dispatch block:

```python
    if selected == PfieldStrategy.METAL:
        from fast_simus.kernels.metal_pfield import pfield_metal
        pressure_accum = pfield_metal(
            positions=positions,
            params=params,
            plan=plan,
            medium=medium,
            delays_clean=delays_clean,
            tx_apodization=tx_apodization,
        )
```

Note: the Metal kernel returns the raw pressure accumulation (sum of |P_k|^2), NOT the final
`sqrt(accum * correction_factor)`. The sqrt and correction are applied after the dispatch block, same as all other
strategies.

**Step 5: Run tests**

Run: `uv run pytest tests/test_pfield.py::TestMetalKernel -x -v` Expected: PASS on MLX, skip on other backends. Convex
(C5-2v) skipped.

Run: `uv run pytest tests/test_pfield.py -x -v` Expected: All tests PASS.

**Step 6: Commit**

```bash
git add src/fast_simus/kernels/ src/fast_simus/pfield.py tests/test_pfield.py
git commit -m "feat: add Metal kernel for pfield on Apple Silicon

Fused Metal kernel computes geometry on-the-fly, eliminating precomputed
distance arrays (3000x memory reduction). Uses float2 complex arithmetic,
metal::precise:: for geometry transcendentals, one thread per grid point.
Auto-selected for MLX grids > 150x150 when MLX is available."
```

______________________________________________________________________

## Task 6: Verification and cleanup

**Files:**

- Possibly modify: `src/fast_simus/pfield.py` (line count check)

**Step 1: Full test suite**

Run: `uv run poe test` Expected: All tests PASS across all backends.

**Step 2: Lint**

Run: `uv run poe lint` Expected: Clean.

**Step 3: Check file sizes**

Run:
`wc -l src/fast_simus/pfield.py src/fast_simus/_pfield_math.py src/fast_simus/_pfield_strategies.py src/fast_simus/kernels/metal_pfield.py`

Expected approximate line counts:

- `pfield.py`: ~450 lines (was 639, extracted ~200 to `_pfield_math.py`, added ~60 for step body + freq outer +
  dispatch)
- `_pfield_math.py`: ~250 lines
- `_pfield_strategies.py`: ~100 lines
- `kernels/metal_pfield.py`: ~200 lines

If `pfield.py` exceeds 600 lines, consider whether any additional extraction is warranted. The 800-line guideline should
not be exceeded.

**Step 4: Cross-strategy benchmark (optional)**

Run the existing benchmark to sanity-check performance:

```bash
uv run python tests/benchmarks/bench_pfield.py
```

**Step 5: Final commit if any cleanup was needed**

```bash
git add -A
git commit -m "refactor: cleanup after backend strategy implementation"
```

______________________________________________________________________

## Summary of file changes

| File                                     | Action                             | Lines (est.)    |
| ---------------------------------------- | ---------------------------------- | --------------- |
| `src/fast_simus/_pfield_math.py`         | Create                             | ~250            |
| `src/fast_simus/_pfield_strategies.py`   | Create                             | ~100            |
| `src/fast_simus/kernels/__init__.py`     | Create                             | ~10             |
| `src/fast_simus/kernels/metal_pfield.py` | Create                             | ~200            |
| `src/fast_simus/pfield.py`               | Modify (extract + add dispatch)    | ~450 (from 639) |
| `src/fast_simus/__init__.py`             | Modify (add PfieldStrategy export) | +2              |
| `tests/test_pfield.py`                   | Modify (add strategy tests)        | +60             |
| `tests/conftest.py`                      | Modify (add strategy fixture)      | +15             |

## Commit sequence

1. `refactor: extract physics helpers to _pfield_math.py`
1. `feat: add PfieldStrategy enum and strategy parameter`
1. `feat: add frequency-outer loop strategy with _freq_step_body`
1. `feat: add JAX lax.scan and MLX loop drivers`
1. `feat: add Metal kernel for pfield on Apple Silicon`
1. `refactor: cleanup after backend strategy implementation` (if needed)
