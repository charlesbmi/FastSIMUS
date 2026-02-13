"""Pytest configuration for FastSIMUS tests."""

import contextlib

import array_api_strict as xp_strict
import pytest

# Try to import array backends
HAS_NUMPY = False
np = None
with contextlib.suppress(ImportError):
    import numpy as np

    HAS_NUMPY = True

HAS_JAX = False
jnp = None
with contextlib.suppress(ImportError):
    import jax.numpy as jnp

    HAS_JAX = True


@pytest.fixture(
    params=[
        pytest.param(
            xp_strict,
            id="array-api-strict",
        ),
        pytest.param(np, id="numpy", marks=pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")),
        pytest.param(
            jnp,
            id="jax",
            marks=[
                pytest.mark.skipif(not HAS_JAX, reason="JAX not available"),
            ],
        ),
    ]
)
def xp(request):
    """Fixture providing different array API backends.

    Parametrizes tests to run with NumPy, array-api-strict, and JAX.
    """
    return request.param
