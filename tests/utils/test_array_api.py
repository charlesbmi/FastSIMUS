"""Test the array API utilities."""

import numpy as np

from fast_simus.utils._array_api import (
    Array,
    as_numpy,
    default_namespace,
    is_cupy_namespace,
    is_mlx_namespace,
)
from tests.conftest import HAS_CUPY, HAS_MLX


def test_array_protocol(xp):
    """Test that the Array protocol accurately describes supported array libraries."""
    arr = xp.asarray([[1, 2], [3, 4]])
    assert isinstance(arr, Array)


def test_as_numpy_roundtrip(xp):
    """Host copy matches the array values on every supported backend, including CuPy."""
    np.testing.assert_allclose(as_numpy(xp.asarray([1.5, -2.0])), [1.5, -2.0])


def test_default_namespace_is_usable():
    """The selected default can create arrays and describe itself."""
    xp = default_namespace()
    np.testing.assert_array_equal(as_numpy(xp.asarray([1.0, 2.0])), [1.0, 2.0])
    assert xp.__name__


def test_default_namespace_prefers_gpu_backend():
    """An available GPU backend wins over the CPU fallback."""
    xp = default_namespace()
    if HAS_CUPY:
        assert is_cupy_namespace(xp)
    elif HAS_MLX:
        assert is_mlx_namespace(xp)
