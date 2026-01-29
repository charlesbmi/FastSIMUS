"""Pytest configuration for FastSIMUS tests."""

import contextlib

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
    import jax.numpy as jnp  # type: ignore[import-untyped]

    HAS_JAX = True

HAS_CUPY = False
cp = None
with contextlib.suppress(ImportError):
    import cupy as cp  # type: ignore[import-untyped]

    HAS_CUPY = True

HAS_ARRAY_API_STRICT = False
array_api_strict = None
with contextlib.suppress(ImportError):
    import array_api_strict

    HAS_ARRAY_API_STRICT = True


@pytest.fixture(
    params=[
        pytest.param(np, id="numpy", marks=pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")),
        pytest.param(
            array_api_strict,
            id="array-api-strict",
            marks=pytest.mark.skipif(not HAS_ARRAY_API_STRICT, reason="array-api-strict not available"),
        ),
        pytest.param(
            jnp,
            id="jax",
            marks=[
                pytest.mark.skipif(not HAS_JAX, reason="JAX not available"),
                pytest.mark.cuda,
            ],
        ),
        pytest.param(
            cp,
            id="cupy",
            marks=[
                pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available"),
                pytest.mark.cuda,
            ],
        ),
    ]
)
def xp(request):
    """Fixture providing different array API backends.

    Parametrizes tests to run with NumPy, array-api-strict, JAX, and CuPy.
    GPU backends (JAX, CuPy) are marked with 'cuda' and can be skipped.
    """
    return request.param
