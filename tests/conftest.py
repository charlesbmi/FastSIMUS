"""Pytest configuration for FastSIMUS tests."""

import contextlib
from typing import cast

import pytest

from fast_simus.pfield import PfieldStrategy
from fast_simus.utils._array_api import _ArrayNamespace

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

HAS_MLX = False
mx = None
with contextlib.suppress(ImportError):
    import mlx.core as mx

    from fast_simus.backends.mlx import ensure_compat

    ensure_compat(mx)
    HAS_MLX = True


@pytest.fixture(
    params=[
        pytest.param(np, id="numpy", marks=pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")),
        pytest.param(
            jnp,
            id="jax",
            marks=[
                pytest.mark.skipif(not HAS_JAX, reason="JAX not available"),
            ],
        ),
        pytest.param(
            mx,
            id="mlx",
            marks=pytest.mark.skipif(not HAS_MLX, reason="MLX not available"),
        ),
    ]
)
def xp(request) -> _ArrayNamespace:
    """Fixture providing different array API backends.

    Parametrizes tests to run with NumPy, JAX, and MLX.
    Does not include array-api-strict, which can be used in place of a parametrized backend
    for Array API compliance testing.
    """
    return cast(_ArrayNamespace, request.param)


@pytest.fixture(
    params=[
        pytest.param(None, id="auto"),
        pytest.param(PfieldStrategy.VECTORIZED, id="vectorized"),
        pytest.param(PfieldStrategy.SCAN, id="scan"),
        pytest.param(PfieldStrategy.METAL, id="metal"),
    ]
)
def strategy(request) -> PfieldStrategy | None:
    """Fixture providing different pfield strategies."""
    return request.param
