"""Tests for transmit delay calculations.

These tests validate that our tx_delay implementation matches PyMUST outputs.
"""

import numpy as np
import pytest
from array_api_compat import array_namespace
from pymust import getparam, txdelayCircular, txdelayFocused, txdelayPlane

from fast_simus.transducer_presets import C5_2v, P4_2v
from fast_simus.utils.geometry import element_positions


class TestFocusedDelays:
    """Test focused beam delay calculations."""

    def test_focused_linear_array_matches_pymust(self):
        """Focused delays for linear array should match PyMUST txdelay."""
        from fast_simus.tx_delay import focused

        # Get PyMUST reference (squeeze: PyMUST always returns (1, n) for scalars)
        pymust_param = getparam("P4-2v")
        x0, z0 = 0.02, 0.05  # 2cm lateral, 5cm depth
        pymust_delays = np.squeeze(txdelayFocused(pymust_param, x0, z0))

        # FastSIMUS implementation
        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        focus = np.array([x0, z0])
        fs_delays = focused(elem_pos, params.speed_of_sound, params.radius, focus, apex)

        # Should match within numerical tolerance
        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_focused_virtual_source_matches_pymust(self):
        """Virtual source (negative z0) should match PyMUST."""
        from fast_simus.tx_delay import focused

        pymust_param = getparam("P4-2v")
        x0, z0 = 0.01, -0.03  # Virtual source above transducer
        pymust_delays = np.squeeze(txdelayFocused(pymust_param, x0, z0))

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        focus = np.array([x0, z0])
        fs_delays = focused(elem_pos, params.speed_of_sound, params.radius, focus, apex)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_focused_convex_array_matches_pymust(self):
        """Focused delays for convex array should match PyMUST."""
        from fast_simus.tx_delay import focused

        pymust_param = getparam("C5-2v")
        x0, z0 = 0.0, 0.06  # On-axis focus
        pymust_delays = np.squeeze(txdelayFocused(pymust_param, x0, z0))

        params = C5_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        focus = np.array([x0, z0])
        fs_delays = focused(elem_pos, params.speed_of_sound, params.radius, focus, apex)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_focused_vectorized_matches_pymust(self):
        """Vectorized focal points should match PyMUST."""
        from fast_simus.tx_delay import focused

        pymust_param = getparam("P4-2v")
        x0 = np.array([0.0, 0.01, 0.02])
        z0 = np.array([0.04, 0.05, 0.06])
        pymust_delays = txdelayFocused(pymust_param, x0, z0)  # type: ignore[invalid-argument-type]  # PyMUST also accepts arrays

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        focus = np.stack([x0, z0], axis=-1)  # Shape (3, 2)
        fs_delays = focused(elem_pos, params.speed_of_sound, params.radius, focus, apex)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)


class TestPlaneWaveDelays:
    """Test plane wave delay calculations."""

    def test_plane_wave_linear_matches_pymust(self):
        """Plane wave delays for linear array should match PyMUST."""
        from fast_simus.tx_delay import plane_wave

        pymust_param = getparam("P4-2v")
        tilt = np.pi / 18  # 10 degrees
        pymust_delays = np.squeeze(txdelayPlane(pymust_param, tilt))

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        fs_delays = plane_wave(elem_pos, params.speed_of_sound, params.radius, tilt, apex)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_plane_wave_zero_tilt_matches_pymust(self):
        """Zero tilt should give zero delays."""
        from fast_simus.tx_delay import plane_wave

        pymust_param = getparam("P4-2v")
        tilt = 0.0
        pymust_delays = np.squeeze(txdelayPlane(pymust_param, tilt))

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        fs_delays = plane_wave(elem_pos, params.speed_of_sound, params.radius, tilt, apex)

        np.testing.assert_allclose(fs_delays, pymust_delays, atol=1e-12)
        np.testing.assert_allclose(fs_delays, 0.0, atol=1e-12)

    def test_plane_wave_convex_matches_pymust(self):
        """Plane wave delays for convex array should match PyMUST."""
        from fast_simus.tx_delay import plane_wave

        pymust_param = getparam("C5-2v")
        tilt = np.pi / 18  # 10 degrees
        pymust_delays = np.squeeze(txdelayPlane(pymust_param, tilt))

        params = C5_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        fs_delays = plane_wave(elem_pos, params.speed_of_sound, params.radius, tilt, apex)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_plane_wave_vectorized_matches_pymust(self):
        """Vectorized tilt angles should match PyMUST."""
        from fast_simus.tx_delay import plane_wave

        pymust_param = getparam("P4-2v")
        tilt = np.array([0.0, np.pi / 18, -np.pi / 18])
        pymust_delays = txdelayPlane(pymust_param, tilt)  # type: ignore[invalid-argument-type]  # PyMUST also accepts arrays

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        fs_delays = plane_wave(elem_pos, params.speed_of_sound, params.radius, tilt, apex)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)


class TestCircularWaveDelays:
    """Test circular wave delay calculations."""

    def test_circular_wave_matches_pymust(self):
        """Circular wave delays should match PyMUST."""
        from fast_simus.tx_delay import diverging_wave

        pymust_param = getparam("P4-2v")
        tilt = np.pi / 18  # 10 degrees
        width = np.pi / 6  # 30 degrees
        pymust_delays = np.squeeze(txdelayCircular(pymust_param, tilt, width))

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        angles = np.array([tilt, width])
        L = (params.n_elements - 1) * params.pitch
        fs_delays = diverging_wave(elem_pos, params.speed_of_sound, params.radius, angles, L)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_circular_wave_vectorized_matches_pymust(self):
        """Vectorized circular wave parameters should match PyMUST."""
        from fast_simus.tx_delay import diverging_wave

        pymust_param = getparam("P4-2v")
        tilt = np.array([0.0, np.pi / 18])
        width = np.array([np.pi / 6, np.pi / 4])
        pymust_delays = txdelayCircular(pymust_param, tilt, width)  # type: ignore[]

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        angles = np.stack([tilt, width], axis=-1)  # Shape (2, 2)
        L = (params.n_elements - 1) * params.pitch
        fs_delays = diverging_wave(elem_pos, params.speed_of_sound, params.radius, angles, L)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_circular_wave_convex_raises_error(self):
        """Circular waves not supported for convex arrays."""
        from fast_simus.tx_delay import diverging_wave

        params = C5_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        angles = np.array([0.0, np.pi / 6])
        L = (params.n_elements - 1) * params.pitch
        with pytest.raises(ValueError, match=r"Diverging.*convex"):
            diverging_wave(elem_pos, params.speed_of_sound, params.radius, angles, L)


class TestArrayAPICompliance:
    """Test Array API compatibility."""

    def test_focused_delays_preserves_backend(self):
        """Output should preserve input array backend."""
        import array_api_strict as xp

        from fast_simus.tx_delay import focused

        params = P4_2v()
        xp_np = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp_np)
        elem_pos = xp.asarray(np.stack([x, z], axis=-1))
        focus = xp.asarray([[0.02, 0.05]])

        delays = focused(elem_pos, params.speed_of_sound, params.radius, focus, apex)

        assert hasattr(delays, "__array_namespace__")
        assert xp.asarray(delays).__array_namespace__().__name__ == "array_api_strict"

    def test_plane_wave_delays_preserves_backend(self):
        """Output should preserve input array backend."""
        import array_api_strict as xp

        from fast_simus.tx_delay import plane_wave

        params = P4_2v()
        xp_np = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp_np)
        elem_pos = xp.asarray(np.stack([x, z], axis=-1))
        tilt = xp.asarray([0.0, np.pi / 18])

        delays = plane_wave(elem_pos, params.speed_of_sound, params.radius, tilt, apex)

        assert hasattr(delays, "__array_namespace__")
        assert xp.asarray(delays).__array_namespace__().__name__ == "array_api_strict"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tilt_angle_validation(self):
        """Tilt angles must satisfy |tilt| < pi/2."""
        from fast_simus.tx_delay import plane_wave

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        with pytest.raises(ValueError, match="Tilt"):
            plane_wave(elem_pos, params.speed_of_sound, params.radius, np.pi / 2 + 0.1, apex)

    def test_width_angle_validation(self):
        """Width angles must satisfy 0 < width < pi."""
        from fast_simus.tx_delay import diverging_wave

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        L = (params.n_elements - 1) * params.pitch
        with pytest.raises(ValueError, match="Width"):
            diverging_wave(elem_pos, params.speed_of_sound, params.radius, np.array([0.0, np.pi + 0.1]), L)

        with pytest.raises(ValueError, match="Width"):
            diverging_wave(elem_pos, params.speed_of_sound, params.radius, np.array([0.0, -0.1]), L)

    def test_mismatched_vector_lengths(self):
        """Focus dimensions must be valid."""
        from jaxtyping import TypeCheckError

        from fast_simus.tx_delay import focused

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        # Invalid shape (should be (n, 2) or (2,))
        focus = np.array([0.0, 0.01, 0.05])  # Shape (3,) instead of (3, 2)

        # Jaxtyping catches this before our validation
        with pytest.raises((ValueError, TypeCheckError)):
            focused(elem_pos, params.speed_of_sound, params.radius, focus, apex)
