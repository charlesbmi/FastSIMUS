"""Pytest configuration for FastSIMUS tests."""

import contextlib
import sys

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

# PyMUST has SyntaxError on Python 3.14+ due to invalid escape sequences
PYMUST_AVAILABLE = False
if sys.version_info < (3, 14):
    import pymust  # noqa: F401

    PYMUST_AVAILABLE = True


def pytest_collection_modifyitems(config, items):
    """Modify pytest collection to skip PyMUST tests on Python 3.14+."""
    if PYMUST_AVAILABLE:
        return
    skip_pymust = pytest.mark.skip(reason="PyMUST not available")
    for item in items:
        if "requires_pymust" in item.keywords:
            item.add_marker(skip_pymust)


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
