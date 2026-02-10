# Array API Typing & Protocols (FastSIMUS-specific)

## When to Use

- Working in the FastSIMUS codebase
- Writing ultrasound simulation code that needs multi-backend support
- Using jaxtyping with beartype for shape validation in this project

## Overview

FastSIMUS uses the [Array API Standard](https://data-apis.org/array-api/2024.12/) for backend-agnostic numerical code.
This enables the same code to run on NumPy, JAX, CuPy, and other compliant backends.

## Core Pattern: Backend Detection

```python
from array_api_compat import array_namespace

def compute_delays(element_positions, focus_x, focus_z, c=1540.0):
    """Compute transmit delays - works with any Array API backend."""
    xp = array_namespace(element_positions)  # Detect backend from input

    # Use xp.* for ALL numerical operations
    distances = xp.sqrt((element_positions - focus_x)**2 + focus_z**2)
    delays = (xp.max(distances) - distances) / c

    return delays  # Returns same type as input
```

## Type Annotations with jaxtyping

Use `jaxtyping` with `beartype` for runtime shape validation:

```python
from typing import Any
from beartype import beartype as typechecker
from jaxtyping import Float, Num, jaxtyped

# Type alias until Array Protocol is standardized
ArrayAPIObj = Any

@jaxtyped(typechecker=typechecker)
def simus(
    x: Float[ArrayAPIObj, " n_scatterers"],
    z: Float[ArrayAPIObj, " n_scatterers"],
    rc: Float[ArrayAPIObj, " n_scatterers"],
    delays: Float[ArrayAPIObj, " n_elements"],
    params: "TransducerParams",
) -> Float[ArrayAPIObj, "n_samples n_elements"]:
    """Simulate RF signals with runtime shape checking."""
    xp = array_namespace(x)
    # ... implementation
```

### Shape Annotation Patterns

```python
# Fixed dimensions
Float[ArrayAPIObj, "height width"]           # 2D image
Float[ArrayAPIObj, "n_samples n_elements"]   # RF signals

# Variable/batch dimensions
Float[ArrayAPIObj, "*batch n_freq"]          # Any batch dims + frequency
Num[ArrayAPIObj, "..."]                      # Any shape

# Named dimensions (self-documenting)
Float[ArrayAPIObj, " n_scatterers"]           # 1D array of scatterer coords
```

For jaxtyping/ruff compatibility, 1-D annotations should have a leading space, e.g. " x"

## Backend-Specific Optimizations

For operations not in Array API or needing backend-specific speedups:

```python
from array_api_compat import is_jax_array, is_cupy_array, is_numpy_array

def frequency_loop(exp_init, exp_df, n_freq):
    """Frequency loop with backend-specific optimizations."""
    xp = array_namespace(exp_init)

    if is_jax_array(exp_init):
        # Use JAX lax.scan for efficient accumulation
        import jax.lax as lax
        return _jax_scan_loop(exp_init, exp_df, n_freq)

    elif is_cupy_array(exp_init):
        # Use CuPy kernel fusion
        import cupy
        return _cupy_fused_loop(exp_init, exp_df, n_freq)

    else:
        # Pure Array API fallback (works for NumPy, array-api-strict)
        return _array_api_loop(exp_init, exp_df, n_freq)
```

## Helper Functions for Missing Array API Features

Some functions aren't in the standard yet. Implement with backend branches:

```python
def histogram(x, bins, range=None, weights=None, density=False):
    """Array-api compatible histogram."""
    xp = array_namespace(x)

    if is_numpy_array(x):
        import numpy as np
        return np.histogram(x, bins=bins, range=range, weights=weights, density=density)

    elif is_jax_array(x):
        import jax.numpy as jnp
        return jnp.histogram(x, bins=bins, range=range, weights=weights, density=density)

    else:
        # Fallback or warn
        import warnings
        warnings.warn(f"histogram not optimized for {xp.__name__}")
        import numpy as np
        return np.histogram(np.asarray(x), bins=bins, range=range, weights=weights, density=density)
```

## Testing with array-api-strict

Use `array-api-strict` to verify Array API compliance:

```python
import pytest
import numpy
import array_api_strict

@pytest.mark.parametrize("xp", [numpy, array_api_strict])
def test_delays_array_api_compliant(xp):
    """Test works with strict Array API implementation."""
    positions = xp.linspace(-0.01, 0.01, 64)
    delays = compute_delays(positions, focus_x=0.0, focus_z=0.03)

    assert delays.shape == positions.shape
    assert hasattr(delays, '__array_namespace__')  # Is Array API compliant
```

## Custom Assertions

```python
from array_api_extra import isclose

def assert_array_equal(actual, expected, rtol=1e-7, atol=0.0):
    """Array-api compatible assertion for tests."""
    xp = array_namespace(actual, expected)

    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")

    close = isclose(actual, expected, rtol=rtol, atol=atol, xp=xp)
    if not xp.all(close):
        max_diff = float(xp.max(xp.abs(actual - expected)))
        raise AssertionError(f"Arrays differ by up to {max_diff}")
```

## Dependencies

```toml
# pyproject.toml
dependencies = [
    "array-api-compat>=1.4",
    "beartype>=0.18",
    "jaxtyping>=0.2.28",
]

[dependency-groups]
test = [
    "array-api-strict>=2.0",  # For Array API compliance testing
]
```

## Missing function signatures

If missing from Protocol, grab the exact function signature from:
https://data-apis.org/array-api/latest/API_specification/index.html

## Key Rules

1. **Always use `xp = array_namespace(input)`** - never hardcode numpy
1. **Preserve input types** - output arrays should match input backend
1. **Test with array-api-strict** - catches non-compliant operations
1. **Document shapes** - use jaxtyping annotations on all public functions
1. **Branch for optimization** - use `is_jax_array()` etc. for backend-specific code
