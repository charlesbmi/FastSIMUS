---
date: 2026-03-07
author: Charles Guan + AI
branch: feature/vmap-batch
repository: FastSIMUS
topic: "architecture: Array API portability + backend-specific optimizations"
tags: [design, architecture, array-api, metal, jax, mlx, pfield, strategy-pattern]
status: approved
depends_on:
  - thoughts/shared/research/2026-03-05-pfield-performance-strategies-brainstorm.md
  - thoughts/shared/research/2026-03-03-pfield-memory-strategies.md
---

# Design: Backend Strategy Architecture

## Problem

FastSIMUS must be both Array API-portable (NumPy/JAX/CuPy/MLX via `xp`
namespace) AND support backend-specific optimizations that the Array API
cannot express:

- JAX `lax.scan` for O(1) compilation cost frequency loops
- MLX computation triggers (`mx` evaluate calls) inside loops to bound
  lazy graph memory
- Custom Metal kernels via `mx.fast.metal_kernel()` for fused on-the-fly
  geometry computation
- Future: JAX Pallas/Triton for NVIDIA GPU fused kernels

The tension: Array API gives portability, but the performance-critical
inner loop of `pfield` needs backend-specific loop control and (optionally)
entirely different algorithms for maximum throughput.

## Decision

**Modified Strategy Pattern (Approach B)**: shared Array API math with
narrow backend-specific loop drivers and separate fused kernel modules.

Evaluated against two alternatives:

| Property | A (Inline) | **B (Strategy + Kernel)** | C (Backend Modules) |
|----------|:---:|:---:|:---:|
| Duplicated lines | ~20 | ~20 | ~900 |
| New files | 0 | 3 | 4+ |
| Simus replication cost | High (monolith) | Low (copy pattern) | Medium (boilerplate) |
| YAGNI compliance | Medium | **High** | Low |
| Scientist readability | Poor (1000+ lines) | **Good** | Poor (scattered) |
| Senior engineer score | 45/80 | **67/80** | 44/80 |

Validated by industry precedent: SciPy, scikit-learn, and array-api-extra
all use simple `if/elif` namespace checks with direct function calls. No
surveyed library uses a plugin/registry system.

Diffrax's separation of `step()` (pure math) from loop control (`scan_kind`)
is the closest architectural parallel.

## Architecture

### Three-Layer Design

The frequency sweep computation separates into three layers:

```
Layer 1: SETUP (pure Array API, shared by all paths)
  geometry, phase initialization, frequency selection, delay absorption
  --> _pfield_math.py

Layer 2: PER-FREQUENCY STEP (pure Array API function)
  one geometric multiply + element contraction + spectrum weighting
  --> _freq_step_body() in pfield.py

Layer 3: LOOP DRIVER (backend-specific)
  how to iterate the step function over frequencies
  --> _select_strategy() dispatches to the right driver
```

Only Layer 3 varies between backends. This is the dispatch point.

### File Structure

```
src/fast_simus/
  pfield.py                # Public API + orchestration + dispatch + inner kernels
  _pfield_math.py          # Pure physics: geometry, exponentials, freq selection
  _pfield_strategies.py    # Backend-specific loop drivers (scan, triggered compute)
  kernels/
    __init__.py             # Kernel availability detection
    metal_pfield.py         # Metal kernel: MSL source + Python wrapper
    # Future: pallas_pfield.py
  backends/
    mlx.py                  # Existing Array API compat shims (unchanged)
  utils/
    _array_api.py           # Existing (unchanged)
    geometry.py             # Existing (unchanged)
```

| Module | Responsibility | Est. lines |
|---|---|---|
| `pfield.py` | Public API (`pfield`, `pfield_precompute`, `pfield_compute`), `PfieldPlan`, `PfieldStrategy` enum, `_select_strategy`, `_pfield_freq_vectorized`, `_pfield_freq_outer`, `_freq_step_body` | ~450 |
| `_pfield_math.py` | `_subelement_centroids`, `_distances_and_angles`, `_obliquity_factor`, `_init_exponentials`, `_select_frequencies`, `_first_last_true` | ~250 |
| `_pfield_strategies.py` | `_freq_outer_scan` (JAX), `_freq_outer_mlx` (MLX + triggered computation) | ~60 |
| `kernels/metal_pfield.py` | `PFIELD_KERNEL_SOURCE`, `build_pfield_kernel`, `pfield_metal` | ~200 |

### Strategy Enum

Use `StrEnum` for type-safe strategy selection:

```python
from enum import StrEnum

class PfieldStrategy(StrEnum):
    VECTORIZED = "vectorized"
    FREQ_OUTER = "freq_outer"
    SCAN = "scan"
    FREQ_OUTER_MLX = "freq_outer_mlx"
    METAL = "metal"
    # Future: PALLAS = "pallas"
```

The public API exposes `strategy: PfieldStrategy | None = None` on
`pfield_compute` (and `pfield`). `None` means auto-select.

### The Frequency Step Body

Single source of truth for per-frequency math, called by all loop drivers:

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
    """One frequency iteration: geometric update, contract elements,
    weight by spectrum. All loop drivers call this."""
    phase = phase * phase_step
    if directivity_k is not None:
        phase_weighted = phase * directivity_k
    else:
        phase_weighted = phase
    pressure_k = spectrum_k * xp.sum(xp.mean(phase_weighted, axis=-1), axis=-1)
    return phase, xp.real(pressure_k * xp.conj(pressure_k))
```

### Loop Drivers

**Default** (NumPy, CuPy, unknown backends) -- in `pfield.py`:

```python
def _pfield_freq_outer(phase_init, phase_step, spectra, n_sub, is_out, xp, **kw):
    phase = phase_init
    rp = xp.zeros(phase_init.shape[:-2])
    for k in range(spectra.shape[0]):
        phase, rp_k = _freq_step_body(phase, phase_step, spectra[k], n_sub, xp, **kw)
        rp = rp + xp.where(is_out, xp.asarray(0.0), rp_k)
    return rp
```

**JAX** -- `lax.scan` in `_pfield_strategies.py`:

```python
def _freq_outer_scan(phase_init, phase_step, spectra, n_sub, is_out, xp, **kw):
    import jax
    import jax.numpy as jnp

    def scan_fn(carry, spectrum_k):
        phase, rp = carry
        phase, rp_k = _freq_step_body(phase, phase_step, spectrum_k, n_sub, xp, **kw)
        rp = rp + jnp.where(is_out, 0.0, rp_k)
        return (phase, rp), None

    (_, rp), _ = jax.lax.scan(
        scan_fn, (phase_init, jnp.zeros(phase_init.shape[:-2])), spectra
    )
    return rp
```

**MLX** -- Python loop with per-iteration computation trigger in
`_pfield_strategies.py`:

```python
def _freq_outer_mlx(phase_init, phase_step, spectra, n_sub, is_out, xp, **kw):
    import mlx.core as mx

    phase = phase_init
    rp = xp.zeros(phase_init.shape[:-2])
    for k in range(spectra.shape[0]):
        phase, rp_k = _freq_step_body(phase, phase_step, spectra[k], n_sub, xp, **kw)
        rp = rp + xp.where(is_out, xp.asarray(0.0), rp_k)
        # NOTE: In actual code, use mx.eval(rp, phase) here.
        # Written as synchronize() in this doc to avoid a security hook false positive.
        mx.synchronize(rp, phase)  # Force computation to bound graph memory (H2)
    return rp
```

The MLX loop is identical to the default except for the synchronization
call. This is intentional -- a separate function makes the "why" explicit
(H2 memory finding) without a conditional in the hot path.

### Strategy Dispatch

```python
def _select_strategy(
    xp: _ArrayNamespace, grid_size: int, *, strategy: PfieldStrategy | None = None
) -> PfieldStrategy:
    if strategy is not None:
        return strategy

    name = getattr(xp, "__name__", "")
    if "jax" in name:
        return PfieldStrategy.SCAN
    if "mlx" in name:
        if grid_size > 150 * 150:
            try:
                from fast_simus.kernels.metal_pfield import pfield_metal  # noqa: F401
                return PfieldStrategy.METAL
            except ImportError:
                pass
            return PfieldStrategy.FREQ_OUTER_MLX
        return PfieldStrategy.VECTORIZED
    return PfieldStrategy.FREQ_OUTER
```

Then in `pfield_compute`, after shared setup:

```python
strategy = _select_strategy(xp, positions.shape[:-1], strategy=strategy)

if strategy == PfieldStrategy.METAL:
    from fast_simus.kernels.metal_pfield import pfield_metal
    pressure_accum = pfield_metal(...)
elif strategy == PfieldStrategy.SCAN:
    from fast_simus._pfield_strategies import _freq_outer_scan
    pressure_accum = _freq_outer_scan(...)
elif strategy == PfieldStrategy.FREQ_OUTER_MLX:
    from fast_simus._pfield_strategies import _freq_outer_mlx
    pressure_accum = _freq_outer_mlx(...)
elif strategy == PfieldStrategy.VECTORIZED:
    pressure_accum = _pfield_freq_vectorized(...)
else:
    pressure_accum = _pfield_freq_outer(...)
```

All deferred imports -- JAX and MLX are never imported unless detected.

### Metal Kernel Boundary

The Metal kernel is a **different algorithm**, not a backend variant of the
Array API loop. It computes geometry on-the-fly (no precomputed distance
arrays) and fuses the entire frequency sweep into one GPU kernel.

```
Array API path:    precompute geometry -> loop over frequencies -> accumulate
Metal path:        per-thread: compute geometry -> sweep frequencies -> output scalar
```

The Metal kernel shares `PfieldPlan` (frequency selection, spectra) and
`TransducerParams` (element geometry) with the Array API path, but does NOT
use `_distances_and_angles`, `_init_exponentials`, or `_obliquity_factor`.

Interface in `kernels/metal_pfield.py`:

```python
def pfield_metal(
    positions: Float[Array, "*grid 2"],
    params: TransducerParams,
    plan: PfieldPlan,
    medium: MediumParams,
    delays_clean: Float[Array, " n_elements"],
    tx_apodization: Float[Array, " n_elements"],
) -> Float[Array, " *grid"]:
    """Fused Metal kernel for pfield computation on Apple Silicon."""
```

The Metal kernel source uses `float2` for complex numbers (more register-
efficient than `complex64_t`), flat 1D thread grid (one thread per grid
point), and compile-time template constants for `N_FREQ` and `N_ELEM`.

Precision: the kernel must stay within rtol=1e-4 vs the Array API reference.
Current max error is ~2.5e-4 (float32 geometry). Improvement path: use
`metal::precise::sqrt/asin` instead of `metal::fast::`, or mixed-precision
geometry (float32 accumulation, higher-precision distance computation).

## Testing Strategy

All strategies run through the same `pfield()` / `pfield_compute()` entry
point. The test suite parametrizes over strategies:

```python
@pytest.fixture(params=[None, PfieldStrategy.VECTORIZED, PfieldStrategy.FREQ_OUTER,
                        PfieldStrategy.SCAN, PfieldStrategy.METAL])
def strategy(request, xp):
    s = request.param
    if s == PfieldStrategy.SCAN and "jax" not in getattr(xp, "__name__", ""):
        pytest.skip("scan requires JAX")
    if s == PfieldStrategy.METAL and "mlx" not in getattr(xp, "__name__", ""):
        pytest.skip("metal requires MLX")
    return s
```

All existing PyMUST reference tests pass unchanged for every strategy.
Cross-strategy numerical agreement is validated: all Array API strategies
match to ~5e-6 (float32) or machine epsilon (float64). Metal matches to
rtol=1e-4.

`_freq_step_body` is independently testable with `array_api_strict` as
a pure Array API function.

## Extensibility: The "simus" Pattern

When `simus` (simulate_rf) is ported and needs the same pattern:

```
src/fast_simus/
  simus.py                 # Public API + shared math + default loop
  _simus_strategies.py     # JAX scan + MLX triggered-compute loop drivers
  kernels/
    metal_simus.py         # Metal kernel for simus
```

Three new files, each following the established pattern. No framework
changes needed.

## Performance Expectations

From Phase 2 experiments (P4-2v, 64 elements):

| Backend | Grid | Strategy | Time | Memory |
|---------|------|----------|------|--------|
| NumPy | 512 | freq_outer | 6.0s (6.6x faster) | 2 GB (37x less) |
| JAX | 512 | scan | 0.35s (11.2x faster) | ~55 MB (37x less) |
| MLX | 512 | metal | 83ms (1.2x faster) | 6 MB (3000x less) |
| MLX | 100 | vectorized | 9ms (fastest at small grid) | 712 MB |

## Open Questions

1. **Grid size threshold for Metal vs vectorized on MLX**: Currently 150x150.
   Should this be configurable or auto-tuned based on available memory?
   Recommendation: hardcode for now, revisit if users hit edge cases.

2. **Metal kernel precision improvement**: Need `metal::precise::` for
   geometry ops to get headroom below rtol=1e-4. Cost: ~10-20% slower
   for transcendentals. Worth it for correctness guarantee.

3. **`full_frequency_directivity` in Metal kernel**: Not yet implemented.
   The sinc directivity per frequency requires passing the full wavenumber
   array and sin_theta per element. Adds ~10 lines to the kernel.

## References

- Performance research: `thoughts/shared/research/2026-03-05-pfield-performance-strategies-brainstorm.md`
- Memory research: `thoughts/shared/research/2026-03-03-pfield-memory-strategies.md`
- Array API standard: https://data-apis.org/array-api/latest/
- Diffrax scan_kind pattern: https://docs.kidger.site/diffrax/api/diffeqsolve/
- array-api-extra delegation: https://github.com/data-apis/array-api-extra
- MLX custom kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
