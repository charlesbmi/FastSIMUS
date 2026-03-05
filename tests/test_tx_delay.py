"""Tests for transmit delay calculations.

These tests validate that our tx_delay implementation matches PyMUST outputs.
"""

from typing import cast

import array_api_compat
import array_api_strict
import numpy as np
import pymust
import pytest
from jaxtyping import TypeCheckError

from fast_simus import MediumParams
from fast_simus.transducer_presets import C5_2v, P4_2v
from fast_simus.tx_delay import diverging_wave, focused, plane_wave
from fast_simus.utils._array_api import _ArrayNamespace
from fast_simus.utils.geometry import element_positions

SPEED_OF_SOUND = MediumParams().speed_of_sound

# Tell type-checker to treat array-api-strict as a _ArrayNamespace instead of ModuleType
xp = cast(_ArrayNamespace, array_api_strict)


class TestFocusedDelays:
    """Test focused beam delay calculations."""

    def test_focused_linear_array_matches_pymust(self):
        """Focused delays for linear array should match PyMUST txdelay."""
        # Get PyMUST reference (squeeze: PyMUST always returns (1, n) for scalars)
        pymust_param = pymust.getparam("P4-2v")
        x0, z0 = 0.02, 0.05  # 2cm lateral, 5cm depth
        pymust_delays = np.squeeze(pymust.txdelayFocused(pymust_param, x0, z0))

        # FastSIMUS implementation
        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        focus = xp.asarray([x0, z0])
        fastsimus_delays = focused(
            elem_pos, focus, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, rtol=1e-4)

    def test_focused_virtual_source_matches_pymust(self):
        """Virtual source (negative z0) should match PyMUST."""
        pymust_param = pymust.getparam("P4-2v")
        x0, z0 = 0.01, -0.03  # Virtual source above transducer
        pymust_delays = np.squeeze(pymust.txdelayFocused(pymust_param, x0, z0))

        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        focus = xp.asarray([x0, z0])
        fastsimus_delays = focused(
            elem_pos, focus, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, rtol=1e-4)

    def test_focused_convex_array_matches_pymust(self):
        """Focused delays for convex array should match PyMUST."""
        pymust_param = pymust.getparam("C5-2v")
        x0, z0 = 0.0, 0.06  # On-axis focus
        pymust_delays = np.squeeze(pymust.txdelayFocused(pymust_param, x0, z0))

        params = C5_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        focus = xp.asarray([x0, z0])
        fastsimus_delays = focused(
            elem_pos, focus, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, rtol=1e-4)

    def test_focused_vectorized_matches_pymust(self):
        """Vectorized focal points should match PyMUST."""
        pymust_param = pymust.getparam("P4-2v")
        x0 = np.array([0.0, 0.01, 0.02])
        z0 = np.array([0.04, 0.05, 0.06])
        pymust_delays = pymust.txdelayFocused(pymust_param, x0, z0)  # type: ignore[invalid-argument-type]  # PyMUST also accepts arrays

        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)

        fastsimus_delays = xp.stack(
            [
                focused(
                    elem_pos,
                    xp.asarray([x, z]),
                    speed_of_sound=SPEED_OF_SOUND,
                    radius=params.radius,
                    apex_offset=apex,
                )
                for x, z in zip(x0, z0, strict=True)
            ]
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, rtol=1e-4)


class TestPlaneWaveDelays:
    """Test plane wave delay calculations."""

    def test_plane_wave_linear_matches_pymust(self):
        """Plane wave delays for linear array should match PyMUST."""
        pymust_param = pymust.getparam("P4-2v")
        tilt = np.deg2rad(10)
        pymust_delays = np.squeeze(pymust.txdelayPlane(pymust_param, tilt))

        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        fastsimus_delays = plane_wave(
            elem_pos, tilt, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, rtol=1e-4)

    def test_plane_wave_zero_tilt_matches_pymust(self):
        """Zero tilt should give zero delays."""
        pymust_param = pymust.getparam("P4-2v")
        tilt = 0.0
        pymust_delays = np.squeeze(pymust.txdelayPlane(pymust_param, tilt))

        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        fastsimus_delays = plane_wave(
            elem_pos, tilt, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, atol=1e-12)
        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), 0.0, atol=1e-12)

    def test_plane_wave_convex_matches_pymust(self):
        """Plane wave delays for convex array should match PyMUST."""
        pymust_param = pymust.getparam("C5-2v")
        tilt = np.deg2rad(10)
        pymust_delays = np.squeeze(pymust.txdelayPlane(pymust_param, tilt))

        params = C5_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        fastsimus_delays = plane_wave(
            elem_pos, tilt, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, rtol=1e-4)

    def test_plane_wave_vectorized_matches_pymust(self):
        """Vectorized tilt angles should match PyMUST."""
        pymust_param = pymust.getparam("P4-2v")
        tilt = np.array([0.0, np.pi / 18, -np.pi / 18])
        pymust_delays = pymust.txdelayPlane(pymust_param, tilt)  # type: ignore[invalid-argument-type]  # PyMUST also accepts arrays

        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)

        fastsimus_delays = xp.stack(
            [
                plane_wave(elem_pos, t, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex)
                for t in tilt
            ]
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, rtol=1e-4)


class TestCircularWaveDelays:
    """Test circular wave delay calculations."""

    def test_circular_wave_matches_pymust(self):
        """Circular wave delays should match PyMUST."""
        pymust_param = pymust.getparam("P4-2v")
        tilt = np.deg2rad(10)
        width = np.deg2rad(30)
        pymust_delays = np.squeeze(pymust.txdelayCircular(pymust_param, tilt, width))

        params = P4_2v()
        elem_pos, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        aperture_length = (params.n_elements - 1) * params.pitch
        fastsimus_delays = diverging_wave(
            elem_pos, tilt, width, aperture_length=aperture_length, speed_of_sound=SPEED_OF_SOUND
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, rtol=1e-4)

    def test_circular_wave_vectorized_matches_pymust(self):
        """Vectorized circular wave parameters should match PyMUST."""
        pymust_param = pymust.getparam("P4-2v")
        tilt = np.array([0.0, np.pi / 18])
        width = np.array([np.pi / 6, np.pi / 4])
        pymust_delays = pymust.txdelayCircular(pymust_param, tilt, width)  # type: ignore[]

        params = P4_2v()
        elem_pos, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        aperture_length = (params.n_elements - 1) * params.pitch

        fastsimus_delays = xp.stack(
            [
                diverging_wave(elem_pos, t, w, aperture_length=aperture_length, speed_of_sound=SPEED_OF_SOUND)
                for t, w in zip(tilt, width, strict=True)
            ]
        )

        np.testing.assert_allclose(cast(np.ndarray, fastsimus_delays), pymust_delays, rtol=1e-4)


class TestArrayAPICompliance:
    """Test Array API compatibility."""

    def test_focused_delays_preserves_backend(self):
        """Output should preserve input array backend."""
        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        focus = xp.asarray([0.02, 0.05])

        delays = focused(elem_pos, focus, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex)

        assert array_api_compat.is_array_api_strict_namespace(array_api_compat.array_namespace(delays))

    def test_plane_wave_delays_preserves_backend(self):
        """Output should preserve input array backend."""
        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        tilt = 0.0

        delays = plane_wave(elem_pos, tilt, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex)

        assert array_api_compat.is_array_api_strict_namespace(array_api_compat.array_namespace(delays))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tilt_angle_validation(self):
        """Tilt angles must satisfy |tilt| < pi/2."""
        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        with pytest.raises(ValueError, match="Tilt"):
            plane_wave(
                elem_pos,
                np.pi / 2 + 0.1,
                speed_of_sound=SPEED_OF_SOUND,
                radius=params.radius,
                apex_offset=apex,
            )

    def test_width_angle_validation(self):
        """Width angles must satisfy 0 < width < pi."""
        params = P4_2v()
        elem_pos, _, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
        aperture_length = (params.n_elements - 1) * params.pitch

        with pytest.raises(ValueError, match="Width"):
            diverging_wave(elem_pos, 0.0, np.pi + 0.1, aperture_length=aperture_length, speed_of_sound=SPEED_OF_SOUND)

        with pytest.raises(ValueError, match="Width"):
            diverging_wave(elem_pos, 0.0, -0.1, aperture_length=aperture_length, speed_of_sound=SPEED_OF_SOUND)

    def test_mismatched_vector_lengths(self):
        """Focus dimensions must be valid."""
        params = P4_2v()
        elem_pos, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        # Invalid shape (should be (dim,) matching element_positions)
        focus = xp.asarray([0.0, 0.01, 0.05])  # Shape (3,) instead of (2,)

        with pytest.raises((ValueError, TypeCheckError)):
            focused(elem_pos, focus, speed_of_sound=SPEED_OF_SOUND, radius=params.radius, apex_offset=apex)
