"""Test the array API utilities."""

import numpy as np
import pytest

from fast_simus.utils._array_api import (
    Array,
    as_numpy,
    default_namespace,
    is_cupy_namespace,
    is_mlx_namespace,
    namespace_label,
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
    """default_namespace returns an Array API namespace that can build arrays."""
    xp = default_namespace()
    np.testing.assert_array_equal(as_numpy(xp.asarray([1.0, 2.0])), [1.0, 2.0])


def test_default_namespace_prefers_gpu_backend():
    """CuPy with a CUDA device wins; otherwise MLX if installed; else NumPy."""
    xp = default_namespace()
    if HAS_CUPY:
        assert is_cupy_namespace(xp)
    elif HAS_MLX:
        assert is_mlx_namespace(xp)
    else:
        assert getattr(xp, "__name__", "") == "numpy"


def test_namespace_label_names_the_selected_backend():
    """namespace_label identifies CuPy, MLX, or NumPy for the default backend."""
    xp = default_namespace()
    label = namespace_label(xp)
    if HAS_CUPY:
        assert label.startswith("CuPy")
    elif HAS_MLX:
        assert label == "MLX"
    else:
        assert label == "NumPy"


def test_mlx_concat_alias():
    """Array API concat is available on MLX after array_namespace applies the shim."""
    mx = pytest.importorskip("mlx.core")
    from fast_simus.utils._array_api import array_namespace

    a = mx.array([1.0])
    xp = array_namespace(a)
    out = as_numpy(xp.concat([a, mx.array([2.0])], axis=0))
    np.testing.assert_array_equal(out, [1.0, 2.0])
