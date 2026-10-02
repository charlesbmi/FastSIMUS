"""Physical geometry and general delay contracts."""

import numpy as np
import pytest

from fast_simus.aperture import RectangularAperture, matrix_aperture, transform_aperture
from fast_simus.transducer import Transducer, transducer_from_params
from fast_simus.transducer_params import TransducerParams
from fast_simus.tx_delay import focus_delays, plane_wave_delays


def test_matrix_order_and_transform():
    """Matrix channels are x-fastest and proper transforms preserve frames."""
    a = matrix_aperture(shape=(2, 3), pitch=(0.003, 0.004), size=(0.002, 0.002), xp=np, dtype=np.float64)
    np.testing.assert_allclose(np.asarray(a.centers[:2]), [[-0.0015, -0.004, 0], [0.0015, -0.004, 0]])
    r = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    b = transform_aperture(a, r, np.array([0.01, 0.02, 0.03]))
    np.testing.assert_allclose(np.asarray(b.centers), a.centers @ r.T + [0.01, 0.02, 0.03])
    np.testing.assert_allclose(b.normals, np.tile([1, 0, 0], (6, 1)))
    with pytest.raises(ValueError, match="rotation"):
        transform_aperture(a, -np.eye(3), np.zeros(3))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_invalid_sizes(bad):
    """Finite positive dimensions are mandatory."""
    with pytest.raises(ValueError):
        matrix_aperture(shape=(1, 1), pitch=(1.0, 1.0), size=(bad, 1.0), xp=np)


def test_invalid_frames_and_response():
    """Bad frames and acoustic parameters fail at construction."""
    a = matrix_aperture(shape=(1, 1), pitch=(1.0, 1.0), size=(0.001, 0.002), xp=np)
    with pytest.raises(ValueError, match="orthogonal"):
        RectangularAperture(a.centers, a.width_axes, a.width_axes, a.sizes)
    with pytest.raises(ValueError):
        Transducer(a, model="3d", freq_center=np.inf)
    with pytest.raises(ValueError, match="height"):
        transducer_from_params(TransducerParams(freq_center=2e6, pitch=0.001, width=0.0005, n_elements=2), xp=np)


def test_delay_arrivals_and_direction():
    """Focused arrivals coincide; plane delays retain the steering sign."""
    centers = np.array([[-0.002, 0.0, 0.0], [0.003, 0.001, 0.0]])
    target = np.array([0.002, 0.004, 0.02])
    delays = focus_delays(centers, target)
    arrival = delays + np.linalg.norm(centers - target, axis=-1) / 1540
    np.testing.assert_allclose(arrival, arrival[0])
    np.testing.assert_allclose(plane_wave_delays(centers, np.array([1.0, 0.0, 0.0])), [0.0, 0.005 / 1540])
    with pytest.raises(ValueError):
        plane_wave_delays(centers, np.ones(3))
