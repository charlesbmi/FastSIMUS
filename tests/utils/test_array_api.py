"""Test the array API utilities."""

import numpy as np

from fast_simus.utils._array_api import (
    Array,
    as_numpy,
)


def test_array_protocol(xp):
    """Test that the Array protocol accurately describes supported array libraries."""
    arr = xp.asarray([[1, 2], [3, 4]])
    assert isinstance(arr, Array)


def test_as_numpy_roundtrip(xp):
    """Host copy matches the array values on every supported backend, including CuPy."""
    np.testing.assert_allclose(as_numpy(xp.asarray([1.5, -2.0])), [1.5, -2.0])
