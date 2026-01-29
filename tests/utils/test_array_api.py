"""Test the array API utilities."""

from fast_simus.utils._array_api import Array


def test_array_protocol(xp):
    """Test that the Array protocol accurately describes supported array libraries."""
    arr = xp.asarray([[1, 2], [3, 4]])
    assert isinstance(arr, Array)
