"""Tests for transmit delay calculations.

These tests validate that our tx_delay implementation matches PyMUST outputs.
"""

import array_api_strict as xp_strict
import numpy as np
import pytest
from array_api_compat import array_namespace
from jaxtyping import TypeCheckError
from pymust import getparam, txdelayCircular, txdelayFocused, txdelayPlane

from fast_simus.transducer_presets import C5_2v, P4_2v
from fast_simus.tx_delay import diverging_wave, focused, plane_wave
from fast_simus.utils.geometry import element_positions


class TestFocusedDelays:
    """Test focused beam delay calculations."""

    def test_focused_linear_array_matches_pymust(self):
        """Focused delays for linear array should match PyMUST txdelay."""
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
        fs_delays = focused(
            elem_pos, focus, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_focused_virtual_source_matches_pymust(self):
        """Virtual source (negative z0) should match PyMUST."""
        pymust_param = getparam("P4-2v")
        x0, z0 = 0.01, -0.03  # Virtual source above transducer
        pymust_delays = np.squeeze(txdelayFocused(pymust_param, x0, z0))

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        focus = np.array([x0, z0])
        fs_delays = focused(
            elem_pos, focus, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_focused_convex_array_matches_pymust(self):
        """Focused delays for convex array should match PyMUST."""
        pymust_param = getparam("C5-2v")
        x0, z0 = 0.0, 0.06  # On-axis focus
        pymust_delays = np.squeeze(txdelayFocused(pymust_param, x0, z0))

        params = C5_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        focus = np.array([x0, z0])
        fs_delays = focused(
            elem_pos, focus, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_focused_vectorized_matches_pymust(self):
        """Vectorized focal points should match PyMUST."""
        pymust_param = getparam("P4-2v")
        x0 = np.array([0.0, 0.01, 0.02])
        z0 = np.array([0.04, 0.05, 0.06])
        pymust_delays = txdelayFocused(pymust_param, x0, z0)  # type: ignore[invalid-argument-type]  # PyMUST also accepts arrays

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)

        fs_delays = np.stack(
            [
                focused(
                    elem_pos,
                    np.array([x, z]),
                    speed_of_sound=params.speed_of_sound,
                    radius=params.radius,
                    apex_offset=apex,
                )
                for x, z in zip(x0, z0, strict=True)
            ]
        )

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)


class TestPlaneWaveDelays:
    """Test plane wave delay calculations."""

    def test_plane_wave_linear_matches_pymust(self):
        """Plane wave delays for linear array should match PyMUST."""
        pymust_param = getparam("P4-2v")
        tilt = np.deg2rad(10)
        pymust_delays = np.squeeze(txdelayPlane(pymust_param, tilt))

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        fs_delays = plane_wave(
            elem_pos, tilt, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_plane_wave_zero_tilt_matches_pymust(self):
        """Zero tilt should give zero delays."""
        pymust_param = getparam("P4-2v")
        tilt = 0.0
        pymust_delays = np.squeeze(txdelayPlane(pymust_param, tilt))

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        fs_delays = plane_wave(
            elem_pos, tilt, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(fs_delays, pymust_delays, atol=1e-12)
        np.testing.assert_allclose(fs_delays, 0.0, atol=1e-12)

    def test_plane_wave_convex_matches_pymust(self):
        """Plane wave delays for convex array should match PyMUST."""
        pymust_param = getparam("C5-2v")
        tilt = np.deg2rad(10)
        pymust_delays = np.squeeze(txdelayPlane(pymust_param, tilt))

        params = C5_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        fs_delays = plane_wave(
            elem_pos, tilt, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_plane_wave_vectorized_matches_pymust(self):
        """Vectorized tilt angles should match PyMUST."""
        pymust_param = getparam("P4-2v")
        tilt = np.array([0.0, np.pi / 18, -np.pi / 18])
        pymust_delays = txdelayPlane(pymust_param, tilt)  # type: ignore[invalid-argument-type]  # PyMUST also accepts arrays

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)

        fs_delays = np.stack(
            [
                plane_wave(elem_pos, t, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex)
                for t in tilt
            ]
        )

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)


class TestCircularWaveDelays:
    """Test circular wave delay calculations."""

    def test_circular_wave_matches_pymust(self):
        """Circular wave delays should match PyMUST."""
        pymust_param = getparam("P4-2v")
        tilt = np.deg2rad(10)
        width = np.deg2rad(30)
        pymust_delays = np.squeeze(txdelayCircular(pymust_param, tilt, width))

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        L = (params.n_elements - 1) * params.pitch
        fs_delays = diverging_wave(elem_pos, tilt, width, aperture_length=L, speed_of_sound=params.speed_of_sound)

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)

    def test_circular_wave_vectorized_matches_pymust(self):
        """Vectorized circular wave parameters should match PyMUST."""
        pymust_param = getparam("P4-2v")
        tilt = np.array([0.0, np.pi / 18])
        width = np.array([np.pi / 6, np.pi / 4])
        pymust_delays = txdelayCircular(pymust_param, tilt, width)  # type: ignore[]

        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        L = (params.n_elements - 1) * params.pitch

        fs_delays = np.stack(
            [
                diverging_wave(elem_pos, t, w, aperture_length=L, speed_of_sound=params.speed_of_sound)
                for t, w in zip(tilt, width, strict=True)
            ]
        )

        np.testing.assert_allclose(fs_delays, pymust_delays, rtol=1e-4)


class TestArrayAPICompliance:
    """Test Array API compatibility."""

    def test_focused_delays_preserves_backend(self):
        """Output should preserve input array backend."""
        params = P4_2v()
        xp_np = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp_np)
        elem_pos = xp_strict.asarray(np.stack([x, z], axis=-1))
        focus = xp_strict.asarray([0.02, 0.05])

        delays = focused(elem_pos, focus, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex)

        assert hasattr(delays, "__array_namespace__")
        assert delays.__array_namespace__().__name__ == "array_api_strict"

    def test_plane_wave_delays_preserves_backend(self):
        """Output should preserve input array backend."""
        params = P4_2v()
        xp_np = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp_np)
        elem_pos = xp_strict.asarray(np.stack([x, z], axis=-1))
        tilt = 0.0

        delays = plane_wave(
            elem_pos, tilt, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex
        )

        assert hasattr(delays, "__array_namespace__")
        assert delays.__array_namespace__().__name__ == "array_api_strict"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tilt_angle_validation(self):
        """Tilt angles must satisfy |tilt| < pi/2."""
        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        with pytest.raises(ValueError, match="Tilt"):
            plane_wave(
                elem_pos,
                np.pi / 2 + 0.1,
                speed_of_sound=params.speed_of_sound,
                radius=params.radius,
                apex_offset=apex,
            )

    def test_width_angle_validation(self):
        """Width angles must satisfy 0 < width < pi."""
        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        L = (params.n_elements - 1) * params.pitch

        with pytest.raises(ValueError, match="Width"):
            diverging_wave(elem_pos, 0.0, np.pi + 0.1, aperture_length=L, speed_of_sound=params.speed_of_sound)

        with pytest.raises(ValueError, match="Width"):
            diverging_wave(elem_pos, 0.0, -0.1, aperture_length=L, speed_of_sound=params.speed_of_sound)

    def test_mismatched_vector_lengths(self):
        """Focus dimensions must be valid."""
        params = P4_2v()
        xp = array_namespace(np.array([1.0]))
        x, z, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        elem_pos = np.stack([x, z], axis=-1)
        # Invalid shape (should be (dim,) matching element_positions)
        focus = np.array([0.0, 0.01, 0.05])  # Shape (3,) instead of (2,)

        with pytest.raises((ValueError, TypeCheckError)):
            focused(elem_pos, focus, speed_of_sound=params.speed_of_sound, radius=params.radius, apex_offset=apex)
