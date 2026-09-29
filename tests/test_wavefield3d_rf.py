"""Received RF in the notebook shares physical time with wavefield playback."""

import numpy as np
import pytest

from examples._wavefield3d import simulate


@pytest.mark.parametrize("scene", ["None", "Point"])
def test_received_channels_and_round_trip(scene):
    """Receive channels vanish without scatterers and arrive after a two-leg path."""
    result = simulate(
        dict(scene=scene, count=2, backend="NumPy", transmit="Plane wave", steer=0, focus=18, side=2, smoke=True)
    )
    assert result.rf.shape == (len(result.rf_times), len(result.elements))
    assert np.isfinite(result.rf).all()
    assert result.rf_times[0] == result.times[0] == 0
    if scene == "None":
        np.testing.assert_array_equal(result.rf, 0)
    else:
        arrival = result.rf_times[np.argmax(np.abs(result.rf[:, 0]))]
        assert abs(arrival - 2 * 0.017 / 1540) < 0.5e-6
        assert arrival < result.times[-1]
