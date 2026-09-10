"""Behavioral tests for signed-dB display mapping."""

import numpy as np
import pytest

from fast_simus.utils._array_api import _ArrayNamespace
from fast_simus.utils.display import signed_db
from tests.conftest import to_numpy


def test_peak_maps_to_plus_minus_dynamic_range(xp: _ArrayNamespace) -> None:
    """The reference peak saturates the colorbar at ±DR."""
    pressure = xp.asarray([-2.0, 0.0, 2.0])
    out = to_numpy(signed_db(pressure, peak=2.0, dynamic_range=40.0))
    np.testing.assert_allclose(out, [-40.0, 0.0, 40.0], atol=1e-5)


def test_floor_amplitude_collapses_to_zero(xp: _ArrayNamespace) -> None:
    """Values at or below -DR of the peak sit at white."""
    peak = 1.0
    dr = 20.0
    floor = peak * 10.0 ** (-dr / 20.0)
    pressure = xp.asarray([floor, -floor, 0.0])
    out = to_numpy(signed_db(pressure, peak=peak, dynamic_range=dr))
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-4)


def test_mid_range_amplitude_is_halfway_on_the_colorbar(xp: _ArrayNamespace) -> None:
    """An amplitude 10 dB below a 20 dB peak uses half the colorbar."""
    peak = 1.0
    dr = 20.0
    mid = peak * 10.0 ** (-0.5)
    pressure = xp.asarray([mid, -mid])
    out = to_numpy(signed_db(pressure, peak=peak, dynamic_range=dr))
    np.testing.assert_allclose(out, [10.0, -10.0], atol=1e-4)


def test_rejects_non_positive_limits(xp: _ArrayNamespace) -> None:
    """Peak and dynamic range must be positive reference values."""
    pressure = xp.asarray([1.0])
    with pytest.raises(ValueError, match="peak"):
        signed_db(pressure, peak=0.0, dynamic_range=40.0)
    with pytest.raises(ValueError, match="dynamic_range"):
        signed_db(pressure, peak=1.0, dynamic_range=0.0)
