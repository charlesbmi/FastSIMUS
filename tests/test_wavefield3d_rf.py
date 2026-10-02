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
        assert abs(arrival - 2 * result.scatterers[0, 2] / result.sound_speed) < 0.5e-6
        assert arrival < result.times[-1]


@pytest.mark.parametrize("depth_mm", [20, 28])
def test_scene_depth_and_scatterers(depth_mm):
    """Scene bounds include every phantom target at either selectable depth."""
    from examples._wavefield3d import phantom

    points, _ = phantom("Phantom", 100, depth_mm=depth_mm)
    assert np.all(points[:, 2] >= 0.002)
    assert np.all(points[:, 2] <= depth_mm / 1000)
    result = simulate(
        dict(
            scene="Point",
            count=2,
            backend="NumPy",
            transmit="Plane wave",
            steer=0,
            focus=12,
            side=2,
            smoke=True,
            depth_mm=depth_mm,
        )
    )
    assert result.slices.axes[2][-1] == depth_mm / 1000
    assert result.times[-1] > 2 * result.scatterers[0, 2] / result.sound_speed
