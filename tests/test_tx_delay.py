"""Tests for transmit delay calculations.

These tests validate that our tx_delay implementation matches PyMUST outputs.
"""

import numpy as np
import pytest
from pymust import getparam, txdelayCircular, txdelayFocused, txdelayPlane

from fast_simus.transducer_presets import C5_2v, P4_2v


class TestFocusedDelays:
    """Test focused beam delay calculations."""

    def test_focused_linear_array_matches_pymust(self):
        """Focused delays for linear array should match PyMUST txdelay."""
        from fast_simus.tx_delay import compute_focused_delays

        # Get PyMUST reference
        pymust_param = getparam("P4-2v")
        x0, z0 = 0.02, 0.05  # 2cm lateral, 5cm depth
        pymust_delays = txdelayFocused(pymust_param, x0, z0)

        # FastSIMUS implementation
        params = P4_2v()
        fs_delays = compute_focused_delays(params, x0_m=x0, z0_m=z0)

        # Should match within numerical tolerance
        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_focused_virtual_source_matches_pymust(self):
        """Virtual source (negative z0) should match PyMUST."""
        from fast_simus.tx_delay import compute_focused_delays

        pymust_param = getparam("P4-2v")
        x0, z0 = 0.01, -0.03  # Virtual source above transducer
        pymust_delays = txdelayFocused(pymust_param, x0, z0)

        params = P4_2v()
        fs_delays = compute_focused_delays(params, x0_m=x0, z0_m=z0)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_focused_convex_array_matches_pymust(self):
        """Focused delays for convex array should match PyMUST."""
        from fast_simus.tx_delay import compute_focused_delays

        pymust_param = getparam("C5-2v")
        x0, z0 = 0.0, 0.06  # On-axis focus
        pymust_delays = txdelayFocused(pymust_param, x0, z0)

        params = C5_2v()
        fs_delays = compute_focused_delays(params, x0_m=x0, z0_m=z0)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_focused_vectorized_matches_pymust(self):
        """Vectorized focal points should match PyMUST."""
        from fast_simus.tx_delay import compute_focused_delays

        pymust_param = getparam("P4-2v")
        x0 = np.array([0.0, 0.01, 0.02])
        z0 = np.array([0.04, 0.05, 0.06])
        pymust_delays = txdelayFocused(pymust_param, x0, z0)

        params = P4_2v()
        fs_delays = compute_focused_delays(params, x0_m=x0, z0_m=z0)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)


class TestPlaneWaveDelays:
    """Test plane wave delay calculations."""

    def test_plane_wave_linear_matches_pymust(self):
        """Plane wave delays for linear array should match PyMUST."""
        from fast_simus.tx_delay import compute_plane_wave_delays

        pymust_param = getparam("P4-2v")
        tilt = np.pi / 18  # 10 degrees
        pymust_delays = txdelayPlane(pymust_param, tilt)

        params = P4_2v()
        fs_delays = compute_plane_wave_delays(params, tilt_rad=tilt)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_plane_wave_zero_tilt_matches_pymust(self):
        """Zero tilt should give zero delays."""
        from fast_simus.tx_delay import compute_plane_wave_delays

        pymust_param = getparam("P4-2v")
        tilt = 0.0
        pymust_delays = txdelayPlane(pymust_param, tilt)

        params = P4_2v()
        fs_delays = compute_plane_wave_delays(params, tilt_rad=tilt)

        np.testing.assert_allclose(fs_delays, pymust_delays, atol=1e-12)
        np.testing.assert_allclose(fs_delays, 0.0, atol=1e-12)

    def test_plane_wave_convex_matches_pymust(self):
        """Plane wave delays for convex array should match PyMUST."""
        from fast_simus.tx_delay import compute_plane_wave_delays

        pymust_param = getparam("C5-2v")
        tilt = np.pi / 18  # 10 degrees
        pymust_delays = txdelayPlane(pymust_param, tilt)

        params = C5_2v()
        fs_delays = compute_plane_wave_delays(params, tilt_rad=tilt)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_plane_wave_vectorized_matches_pymust(self):
        """Vectorized tilt angles should match PyMUST."""
        from fast_simus.tx_delay import compute_plane_wave_delays

        pymust_param = getparam("P4-2v")
        tilt = np.array([0.0, np.pi / 18, -np.pi / 18])
        pymust_delays = txdelayPlane(pymust_param, tilt)

        params = P4_2v()
        fs_delays = compute_plane_wave_delays(params, tilt_rad=tilt)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)


class TestCircularWaveDelays:
    """Test circular wave delay calculations."""

    def test_circular_wave_matches_pymust(self):
        """Circular wave delays should match PyMUST."""
        from fast_simus.tx_delay import compute_circular_wave_delays

        pymust_param = getparam("P4-2v")
        tilt = np.pi / 18  # 10 degrees
        width = np.pi / 6  # 30 degrees
        pymust_delays = txdelayCircular(pymust_param, tilt, width)

        params = P4_2v()
        fs_delays = compute_circular_wave_delays(params, tilt_rad=tilt, width_rad=width)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_circular_wave_vectorized_matches_pymust(self):
        """Vectorized circular wave parameters should match PyMUST."""
        from fast_simus.tx_delay import compute_circular_wave_delays

        pymust_param = getparam("P4-2v")
        tilt = np.array([0.0, np.pi / 18])
        width = np.array([np.pi / 6, np.pi / 4])
        pymust_delays = txdelayCircular(pymust_param, tilt, width)

        params = P4_2v()
        fs_delays = compute_circular_wave_delays(params, tilt_rad=tilt, width_rad=width)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_circular_wave_convex_raises_error(self):
        """Circular waves not supported for convex arrays."""
        from fast_simus.tx_delay import compute_circular_wave_delays

        params = C5_2v()
        with pytest.raises(ValueError, match=r"Circular.*convex"):
            compute_circular_wave_delays(params, tilt_rad=0.0, width_rad=np.pi / 6)


class TestArrayAPICompliance:
    """Test Array API compatibility."""

    def test_focused_delays_preserves_backend(self):
        """Output should preserve input array backend."""
        import array_api_strict as xp

        from fast_simus.tx_delay import compute_focused_delays

        params = P4_2v()
        x0 = xp.asarray([0.02])
        z0 = xp.asarray([0.05])

        delays = compute_focused_delays(params, x0_m=x0, z0_m=z0)

        assert hasattr(delays, "__array_namespace__")
        assert xp.asarray(delays).__array_namespace__().__name__ == "array_api_strict"

    def test_plane_wave_delays_preserves_backend(self):
        """Output should preserve input array backend."""
        import array_api_strict as xp

        from fast_simus.tx_delay import compute_plane_wave_delays

        params = P4_2v()
        tilt = xp.asarray([0.0, np.pi / 18])

        delays = compute_plane_wave_delays(params, tilt_rad=tilt)

        assert hasattr(delays, "__array_namespace__")
        assert xp.asarray(delays).__array_namespace__().__name__ == "array_api_strict"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tilt_angle_validation(self):
        """Tilt angles must satisfy |tilt| < pi/2."""
        from fast_simus.tx_delay import compute_plane_wave_delays

        params = P4_2v()
        with pytest.raises(ValueError, match="Tilt"):
            compute_plane_wave_delays(params, tilt_rad=np.pi / 2 + 0.1)

    def test_width_angle_validation(self):
        """Width angles must satisfy 0 < width < pi."""
        from fast_simus.tx_delay import compute_circular_wave_delays

        params = P4_2v()
        with pytest.raises(ValueError, match="Width"):
            compute_circular_wave_delays(params, tilt_rad=0.0, width_rad=np.pi + 0.1)

        with pytest.raises(ValueError, match="Width"):
            compute_circular_wave_delays(params, tilt_rad=0.0, width_rad=-0.1)

    def test_mismatched_vector_lengths(self):
        """x0 and z0 must have same length."""
        from jaxtyping import TypeCheckError

        from fast_simus.tx_delay import compute_focused_delays

        params = P4_2v()
        x0 = np.array([0.0, 0.01])
        z0 = np.array([0.05])  # Different length

        # Jaxtyping catches this before our validation
        with pytest.raises((ValueError, TypeCheckError)):
            compute_focused_delays(params, x0_m=x0, z0_m=z0)
