"""Unit tests for pfield helper functions.

Uses array-api-strict to test Array API compliance.
"""

from math import pi
from typing import cast

import array_api_extra as xpx
import array_api_strict
import pytest
from jaxtyping import Bool, Complex, Float, jaxtyped

from fast_simus.pfield import (
    _distances_and_angles,
    _init_exponentials,
    _obliquity_factor,
    _select_frequencies,
    _subelement_centroids,
)
from fast_simus.transducer_params import BaffleType
from fast_simus.utils._array_api import Array, _ArrayNamespace

# Tell type-checker to treat array-api-strict as a _ArrayNamespace instead of ModuleType
xp = cast(_ArrayNamespace, array_api_strict)


class TestSubelementCentroids:
    """Tests for _subelement_centroids."""

    def test_single_subelement_at_origin(self):
        """n_sub=1, theta=0: single centroid at (0, 0)."""
        element_width = 1.0
        n_sub = 1
        theta_e = xp.asarray([0.0])

        offsets = _subelement_centroids(element_width, n_sub, theta_e, xp)

        expected = xp.asarray([0.0, 0.0])
        assert bool(xp.all(xpx.isclose(offsets[0, 0, ...], expected, atol=1e-10, xp=xp)))

    def test_two_subelements_lateral(self):
        """n_sub=2, theta=0: purely lateral offsets [[-w/4, 0], [+w/4, 0]]."""
        element_width = 1.0
        n_sub = 2
        theta_e = xp.asarray([0.0])

        offsets = _subelement_centroids(element_width, n_sub, theta_e, xp)

        expected = xp.asarray([[[-0.25, 0.0], [0.25, 0.0]]])
        assert bool(xp.all(xpx.isclose(offsets, expected, atol=1e-10, xp=xp)))

    def test_two_subelements_rotated_90deg(self):
        """n_sub=2, theta=pi/2: offsets rotate to axial (dx≈0, dz=±w/4)."""
        element_width = 1.0
        n_sub = 2
        theta_e = xp.asarray([pi / 2])

        offsets = _subelement_centroids(element_width, n_sub, theta_e, xp)

        # Rotation by pi/2: (x, 0) -> (0, -x)
        # So [-0.25, 0] -> [0, 0.25], [0.25, 0] -> [0, -0.25]
        expected_x = xp.asarray([0.0, 0.0])
        expected_z = xp.asarray([0.25, -0.25])
        atol = 1e-5
        assert bool(xp.all(xpx.isclose(offsets[0, :, 0], expected_x, atol=atol, xp=xp)))
        assert bool(xp.all(xpx.isclose(offsets[0, :, 1], expected_z, atol=atol, xp=xp)))

    def test_subelements_centered_invariant(self):
        """Invariant: sum(offsets, axis=1) == 0 (sub-elements centered on element)."""
        element_width = 0.5
        n_sub = 5
        theta_e = xp.asarray([0.0, pi / 6, -pi / 4])

        offsets = _subelement_centroids(element_width, n_sub, theta_e, xp)

        # Sum over sub-elements (axis=1) should be zero
        sum_offsets = xp.sum(offsets, axis=1)
        assert bool(xp.all(xpx.isclose(sum_offsets, 0.0, atol=1e-5, xp=xp)))


class TestDistancesAndAngles:
    """Tests for _distances_and_angles."""

    def test_point_on_axis(self):
        """Point at (0, 1) → distance=1, theta_arr≈0, sin_theta≈0."""
        points = xp.asarray([[0.0, 1.0]])
        subelement_offsets = xp.zeros((1, 1, 2))
        element_pos = xp.asarray([[0.0, 0.0]])
        theta_e = xp.asarray([0.0])
        speed_of_sound = 1540.0
        freq_center = 3e6

        distances, sin_theta, theta_arr = _distances_and_angles(
            points, subelement_offsets, element_pos, theta_e, speed_of_sound, freq_center, xp
        )

        assert float(distances[0, 0, 0]) == pytest.approx(1.0, abs=1e-6)
        assert float(theta_arr[0, 0, 0]) == pytest.approx(0.0, abs=1e-6)
        assert float(sin_theta[0, 0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_point_on_lateral_axis(self):
        """Point at (1, 0) → distance=1, theta_arr≈pi/2, sin_theta≈1."""
        points = xp.asarray([[1.0, 0.0]])
        subelement_offsets = xp.zeros((1, 1, 2))
        element_pos = xp.asarray([[0.0, 0.0]])
        theta_e = xp.asarray([0.0])
        speed_of_sound = 1540.0
        freq_center = 3e6

        distances, sin_theta, theta_arr = _distances_and_angles(
            points, subelement_offsets, element_pos, theta_e, speed_of_sound, freq_center, xp
        )

        assert float(distances[0, 0, 0]) == pytest.approx(1.0, abs=1e-6)
        assert float(theta_arr[0, 0, 0]) == pytest.approx(pi / 2, abs=1e-6)
        assert float(sin_theta[0, 0, 0]) == pytest.approx(1.0, abs=1e-6)

    def test_distance_clipping_at_origin(self):
        """Point at origin → distance is clamped to c/(2*fc), not zero."""
        points = xp.asarray([[0.0, 0.0]])
        subelement_offsets = xp.zeros((1, 1, 2))
        element_pos = xp.asarray([[0.0, 0.0]])
        theta_e = xp.asarray([0.0])
        speed_of_sound = 1540.0
        freq_center = 3e6

        distances, sin_theta, theta_arr = _distances_and_angles(
            points, subelement_offsets, element_pos, theta_e, speed_of_sound, freq_center, xp
        )

        min_distance = speed_of_sound / freq_center / 2.0
        assert float(distances[0, 0, 0]) == pytest.approx(min_distance, abs=1e-10)
        assert bool(xp.all(xp.isfinite(theta_arr))), "Angles should be finite"
        assert bool(xp.all(xp.isfinite(sin_theta))), "Sine of angles should be finite"


class TestSelectFrequencies:
    """Tests for _select_frequencies."""

    def test_frequencies_in_range(self):
        """All selected frequencies are in [0, 2*fc]."""
        fc = 3e6
        bandwidth = 0.6
        tx_n_wavelengths = 1.5
        db_thresh = -60.0
        max_freq_step = 1e5

        plan = _select_frequencies(fc, bandwidth, tx_n_wavelengths, db_thresh, max_freq_step, xp)

        assert bool(xp.all(plan.selected_freqs >= 0))
        assert bool(xp.all(plan.selected_freqs <= 2 * fc))

    def test_frequency_plan_consistent_lengths(self):
        """Use jaxtyped context to verify all arrays have consistent lengths."""
        fc = 3e6
        bandwidth = 0.6
        tx_n_wavelengths = 1.5
        db_thresh = -60.0
        max_freq_step = 1e5

        plan = _select_frequencies(fc, bandwidth, tx_n_wavelengths, db_thresh, max_freq_step, xp)

        with jaxtyped("context"):
            assert isinstance(plan.selected_freqs, Float[Array, " n_sampling"])  # ty: ignore[invalid-argument-type]
            assert isinstance(plan.pulse_spectrum, Complex[Array, " n_sampling"])  # ty: ignore[invalid-argument-type]
            assert isinstance(plan.probe_spectrum, Float[Array, " n_sampling"])  # ty: ignore[invalid-argument-type]
            assert isinstance(plan.freq_mask, Bool[Array, " n_freq"])  # ty: ignore[invalid-argument-type]

    def test_tighter_threshold_fewer_frequencies(self):
        """Tighter db_thresh (-10 dB) selects fewer frequencies than looser (-60 dB)."""
        fc = 3e6
        bandwidth = 0.6
        tx_n_wavelengths = 1.5
        max_freq_step = 1e5

        plan_tight = _select_frequencies(fc, bandwidth, tx_n_wavelengths, -10.0, max_freq_step, xp)
        plan_loose = _select_frequencies(fc, bandwidth, tx_n_wavelengths, -60.0, max_freq_step, xp)

        assert plan_tight.selected_freqs.shape[0] < plan_loose.selected_freqs.shape[0]


class TestObliquityFactor:
    """Tests for _obliquity_factor."""

    def test_output_in_valid_range(self):
        """Output is in [0, 1] across a sweep of angles in (-pi/2, pi/2)."""
        theta_arr = xp.reshape(xp.linspace(-pi / 2 + 0.01, pi / 2 - 0.01, 20), (1, 1, -1))

        for baffle in [BaffleType.SOFT, BaffleType.RIGID, 1.0, 2.5]:
            result = _obliquity_factor(theta_arr, baffle, xp)
            assert bool(xp.all(result >= 0))
            assert bool(xp.all(result <= 1))

    def test_near_zero_at_horizon(self):
        """At |theta| >= pi/2, output is near zero for all baffle types."""
        theta_arr = xp.asarray([[[pi / 2, -pi / 2, pi / 2 + 0.1]]])

        for baffle in [BaffleType.SOFT, BaffleType.RIGID, 1.0]:
            result = _obliquity_factor(theta_arr, baffle, xp)
            # Should be epsilon ≈ 1e-16
            assert bool(xp.all(result < 1e-10))

    def test_rigid_baffle_always_one(self):
        """Rigid baffle: output is 1.0 for all angles strictly inside hemisphere."""
        theta_arr = xp.reshape(xp.linspace(-pi / 2 + 0.1, pi / 2 - 0.1, 10), (1, 1, -1))

        result = _obliquity_factor(theta_arr, BaffleType.RIGID, xp)

        expected = xp.ones_like(result)
        assert bool(xp.all(xpx.isclose(result, expected, atol=1e-10, xp=xp)))


class TestInitExponentials:
    """Tests for _init_exponentials."""

    def test_zero_attenuation_unit_magnitude_step(self):
        """Zero attenuation: |phase_decay_step| == 1 (pure phase rotation)."""
        freq_start = 3e6
        speed_of_sound = 1540.0
        attenuation = 0.0
        distances = xp.asarray([[[1.0, 2.0, 0.5]]])
        obliquity_factor = xp.ones_like(distances)
        freq_step = 1e5

        phase_decay, phase_decay_step = _init_exponentials(
            freq_start, speed_of_sound, attenuation, distances, obliquity_factor, freq_step, xp
        )

        # With zero attenuation, phase_decay_step should have magnitude 1
        magnitude = xp.abs(phase_decay_step)
        expected = xp.ones_like(magnitude)
        assert bool(xp.all(xpx.isclose(magnitude, expected, atol=1e-10, xp=xp)))
        # Initial phase_decay should also be finite
        assert bool(xp.all(xp.isfinite(phase_decay)))

    def test_zero_attenuation_zero_freq_geometric_spreading(self):
        """Zero attenuation, freq_start=0: |phase_decay| = obliquity_factor / sqrt(distance)."""
        freq_start = 0.0
        speed_of_sound = 1540.0
        attenuation = 0.0
        distances = xp.asarray([[[1.0, 4.0, 9.0]]])
        obliquity_factor = xp.asarray([[[1.0, 0.8, 0.6]]])
        freq_step = 1e5

        phase_decay, phase_decay_step = _init_exponentials(
            freq_start, speed_of_sound, attenuation, distances, obliquity_factor, freq_step, xp
        )

        expected_magnitude = obliquity_factor / xp.sqrt(distances)
        actual_magnitude = xp.abs(phase_decay)
        assert bool(xp.all(xpx.isclose(actual_magnitude, expected_magnitude, atol=1e-10, xp=xp)))
        # Step should also be well-defined
        assert bool(xp.all(xp.isfinite(phase_decay_step)))

    def test_phase_stepping_consistency(self):
        """Applying phase_decay_step n=3 times matches computing at freq_start + 3*freq_step."""
        freq_start = 2e6
        speed_of_sound = 1540.0
        attenuation = 0.5
        distances = xp.asarray([[[1.0, 2.0]]])
        obliquity_factor = xp.ones_like(distances)
        freq_step = 1e5
        n_steps = 3

        # Initial exponentials
        phase_decay_init, phase_decay_step = _init_exponentials(
            freq_start, speed_of_sound, attenuation, distances, obliquity_factor, freq_step, xp
        )

        # Step forward n times
        phase_decay_stepped = phase_decay_init
        for _ in range(n_steps):
            phase_decay_stepped = phase_decay_stepped * phase_decay_step

        # Compute directly at new frequency
        freq_new = freq_start + n_steps * freq_step
        phase_decay_direct, _ = _init_exponentials(
            freq_new, speed_of_sound, attenuation, distances, obliquity_factor, freq_step, xp
        )

        assert bool(xp.all(xpx.isclose(phase_decay_stepped, phase_decay_direct, atol=1e-5, xp=xp)))
