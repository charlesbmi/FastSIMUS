"""Unit tests for internal pfield math functions.

Tests cover _subelement_centroids, _distances_and_angles, _obliquity_factor,
_init_exponentials, _freq_step_body, _select_frequencies, and _first_last_true.
"""

from __future__ import annotations

from math import cos, pi, sin, sqrt
from typing import cast

import array_api_strict
import numpy as np

from fast_simus._pfield_math import (
    NEPER_TO_DB,
    _distances_and_angles,
    _first_last_true,
    _init_exponentials,
    _obliquity_factor,
    _select_frequencies,
    _subelement_centroids,
)
from fast_simus._pfield_strategies import _freq_step_body
from fast_simus.transducer_params import BaffleType
from fast_simus.utils._array_api import _ArrayNamespace

xp = cast(_ArrayNamespace, array_api_strict)


# ---------------------------------------------------------------------------
# TestSubelementCentroids
# ---------------------------------------------------------------------------


class TestSubelementCentroids:
    """Tests for _subelement_centroids."""

    def test_nsub1_offset_is_zero(self):
        """Single sub-element should sit at element center (zero offset)."""
        theta_e = xp.asarray([0.0])
        result = _subelement_centroids(1e-3, 1, theta_e, xp)
        assert result.shape == (1, 1, 2)
        np.testing.assert_allclose(np.asarray(result), 0.0, atol=1e-15)

    def test_nsub2_symmetric_offsets(self):
        """Two sub-elements should be symmetric about element center."""
        w = 1e-3
        theta_e = xp.asarray([0.0])
        result = _subelement_centroids(w, 2, theta_e, xp)
        assert result.shape == (1, 2, 2)
        dx = np.asarray(result[0, :, 0])
        np.testing.assert_allclose(dx, [-w / 4.0, w / 4.0], atol=1e-15)
        dz = np.asarray(result[0, :, 1])
        np.testing.assert_allclose(dz, [0.0, 0.0], atol=1e-15)

    def test_nsub3_center_and_outer(self):
        """Three sub-elements: center at 0, outer at +-w/3."""
        w = 1e-3
        theta_e = xp.asarray([0.0])
        result = _subelement_centroids(w, 3, theta_e, xp)
        assert result.shape == (1, 3, 2)
        dx = np.asarray(result[0, :, 0])
        seg = w / 3
        expected_dx = [-seg, 0.0, seg]
        np.testing.assert_allclose(dx, expected_dx, atol=1e-15)

    def test_linear_theta_zero_dz(self):
        """Linear array (theta=0) should have zero axial offsets."""
        theta_e = xp.asarray([0.0, 0.0, 0.0])
        result = _subelement_centroids(1e-3, 4, theta_e, xp)
        dz = np.asarray(result[..., 1])
        np.testing.assert_allclose(dz, 0.0, atol=1e-15)

    def test_nonzero_theta_rotates(self):
        """Non-zero theta should rotate sub-element offsets via cos/sin."""
        w = 1e-3
        angle = pi / 6
        theta_e = xp.asarray([angle])
        result = _subelement_centroids(w, 2, theta_e, xp)
        dx = np.asarray(result[0, :, 0])
        dz = np.asarray(result[0, :, 1])
        seg_offsets = np.array([-w / 4.0, w / 4.0])
        expected_dx = seg_offsets * cos(angle)
        expected_dz = seg_offsets * sin(-angle)
        np.testing.assert_allclose(dx, expected_dx, rtol=1e-12)
        np.testing.assert_allclose(dz, expected_dz, rtol=1e-12)

    def test_multiple_elements(self):
        """Multiple elements should produce correct output shape."""
        theta_e = xp.asarray([0.0, pi / 4])
        result = _subelement_centroids(2e-3, 2, theta_e, xp)
        assert result.shape == (2, 2, 2)


# ---------------------------------------------------------------------------
# TestDistancesAndAngles
# ---------------------------------------------------------------------------


class TestDistancesAndAngles:
    """Tests for _distances_and_angles."""

    def test_single_element_known_distance(self):
        """Hand-computed distance from origin to (3mm, 4mm) should be 5mm."""
        px, pz = 3e-3, 4e-3
        points = xp.asarray([[px, pz]])
        element_pos = xp.asarray([[0.0, 0.0]])
        theta_e = xp.asarray([0.0])
        subelement_offsets = _subelement_centroids(1e-4, 1, theta_e, xp)

        distances, _sin_theta, _theta_arr = _distances_and_angles(
            points, subelement_offsets, element_pos, theta_e, 1540.0, 5e6, xp
        )
        expected_dist = sqrt(px**2 + pz**2)
        np.testing.assert_allclose(np.asarray(distances[0, 0, 0]), expected_dist, rtol=1e-6)

    def test_min_distance_clipping(self):
        """Points very close to element should have distance clamped to lambda/2."""
        c = 1540.0
        fc = 5e6
        min_dist = c / fc / 2.0
        near_x = min_dist * 0.01
        near_z = min_dist * 0.01
        points = xp.asarray([[near_x, near_z]])
        element_pos = xp.asarray([[0.0, 0.0]])
        theta_e = xp.asarray([0.0])
        subelement_offsets = _subelement_centroids(1e-4, 1, theta_e, xp)

        distances, _, _ = _distances_and_angles(points, subelement_offsets, element_pos, theta_e, c, fc, xp)
        assert float(distances[0, 0, 0]) >= min_dist

    def test_angle_directly_above_element(self):
        """Point directly above element center should have near-zero angle."""
        points = xp.asarray([[0.0, 10e-3]])
        element_pos = xp.asarray([[0.0, 0.0]])
        theta_e = xp.asarray([0.0])
        subelement_offsets = _subelement_centroids(1e-4, 1, theta_e, xp)

        _, _, theta_arr = _distances_and_angles(points, subelement_offsets, element_pos, theta_e, 1540.0, 5e6, xp)
        np.testing.assert_allclose(np.asarray(theta_arr[0, 0, 0]), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# TestObliquityFactor
# ---------------------------------------------------------------------------


class TestObliquityFactor:
    """Tests for _obliquity_factor."""

    def test_rigid_returns_one(self):
        """Rigid baffle should return 1.0 for all angles."""
        theta = xp.asarray([[0.3, -0.5]])
        result = _obliquity_factor(theta, BaffleType.RIGID, xp)
        np.testing.assert_allclose(np.asarray(result), 1.0, atol=1e-15)

    def test_soft_returns_cos(self):
        """Soft baffle should return cos(theta)."""
        angles = [0.0, 0.3, 0.8]
        theta = xp.asarray([angles])
        result = _obliquity_factor(theta, BaffleType.SOFT, xp)
        expected = [cos(a) for a in angles]
        np.testing.assert_allclose(np.asarray(result)[0], expected, rtol=1e-12)

    def test_custom_float_baffle(self):
        """Custom float baffle should return cos(theta)/(cos(theta)+baffle)."""
        baffle_val = 1.75
        angles = [0.0, 0.3, 0.8]
        theta = xp.asarray([angles])
        result = _obliquity_factor(theta, baffle_val, xp)
        expected = [cos(a) / (cos(a) + baffle_val) for a in angles]
        np.testing.assert_allclose(np.asarray(result)[0], expected, rtol=1e-12)

    def test_horizon_floor(self):
        """Angles at or beyond pi/2 should be clamped to 1e-16."""
        theta = xp.asarray([[pi / 2, pi * 0.7]])
        result_soft = _obliquity_factor(theta, BaffleType.SOFT, xp)
        np.testing.assert_allclose(np.asarray(result_soft), 1e-16, atol=1e-20)

        result_rigid = _obliquity_factor(theta, BaffleType.RIGID, xp)
        np.testing.assert_allclose(np.asarray(result_rigid), 1e-16, atol=1e-20)


# ---------------------------------------------------------------------------
# TestInitExponentials
# ---------------------------------------------------------------------------


class TestInitExponentials:
    """Tests for _init_exponentials."""

    def test_zero_attenuation_magnitude(self):
        """With zero attenuation, |phase_decay| should equal obliquity/sqrt(distance)."""
        distances = xp.asarray([[0.01, 0.02]])
        obliquity = xp.asarray([[1.0, 1.0]])
        freq_start = 3e6
        c = 1540.0
        freq_step = 1e5

        phase_decay, _ = _init_exponentials(freq_start, c, 0.0, distances, obliquity, freq_step, xp)

        expected_mag = np.array([1.0, 1.0]) / np.sqrt([0.01, 0.02])
        np.testing.assert_allclose(np.abs(np.asarray(phase_decay))[0], expected_mag, rtol=1e-6)

    def test_phase_decay_step_attenuation(self):
        """Step magnitude should be exp(-attenuation_neper * distance)."""
        dist_val = 0.02
        distances = xp.asarray([[dist_val]])
        obliquity = xp.asarray([[1.0]])
        freq_start = 3e6
        freq_step = 1e5
        c = 1540.0
        atten_db = 0.5  # dB/cm/MHz

        _, phase_decay_step = _init_exponentials(freq_start, c, atten_db, distances, obliquity, freq_step, xp)

        atten_neper = atten_db / NEPER_TO_DB * freq_step / 1e6 * 1e2
        expected_mag = np.exp(-atten_neper * dist_val)
        np.testing.assert_allclose(np.abs(np.asarray(phase_decay_step[0, 0])), expected_mag, rtol=1e-10)

    def test_phase_in_0_to_2pi(self):
        """Phase of initial exponential should be wrapped to [0, 2pi)."""
        distances = xp.asarray([[0.05, 0.10]])
        obliquity = xp.asarray([[1.0, 1.0]])
        freq_start = 5e6
        c = 1540.0
        freq_step = 1e5

        phase_decay, _ = _init_exponentials(freq_start, c, 0.0, distances, obliquity, freq_step, xp)
        phases = np.angle(np.asarray(phase_decay))[0]
        phases_mod = phases % (2 * pi)
        assert np.all(phases_mod >= 0)
        assert np.all(phases_mod < 2 * pi + 1e-10)


# ---------------------------------------------------------------------------
# TestFreqStepBody
# ---------------------------------------------------------------------------


class TestFreqStepBody:
    """Tests for _freq_step_body."""

    def test_phase_geometric_progression(self):
        """Output phase should equal input phase * phase_step."""
        phase = xp.asarray([[[1.0 + 0.5j, 0.5 + 1.0j]]])
        phase_step = xp.asarray([[[0.9 + 0.1j, 0.8 + 0.2j]]])
        spectrum_k = xp.asarray(1.0 + 0j)

        new_phase, _ = _freq_step_body(phase, phase_step, spectrum_k, 2, xp)
        expected = np.asarray(phase) * np.asarray(phase_step)
        np.testing.assert_allclose(np.asarray(new_phase), expected, rtol=1e-12)

    def test_rp_k_value(self):
        """rp_k should be |spectrum * sum(mean(phase))|^2."""
        phase = xp.asarray([[[1.0 + 0j, 1.0 + 0j]]])
        phase_step = xp.asarray([[[1.0 + 0j, 1.0 + 0j]]])
        spectrum_k = xp.asarray(2.0 + 0j)

        _, rp_k = _freq_step_body(phase, phase_step, spectrum_k, 2, xp)
        # mean over n_sub: 1.0, sum over n_elements: 1.0, |2.0|^2 = 4.0
        np.testing.assert_allclose(np.asarray(rp_k), 4.0, rtol=1e-12)

    def test_with_directivity(self):
        """Directivity should weight phase before reduction."""
        phase = xp.asarray([[[1.0 + 0j, 1.0 + 0j]]])
        phase_step = xp.asarray([[[1.0 + 0j, 1.0 + 0j]]])
        spectrum_k = xp.asarray(1.0 + 0j)
        directivity = xp.asarray([[[0.5, 0.5]]])

        _, rp_k = _freq_step_body(phase, phase_step, spectrum_k, 2, xp, directivity_k=directivity)
        # phase * 0.5, mean: 0.5, sum: 0.5, |0.5|^2 = 0.25
        np.testing.assert_allclose(np.asarray(rp_k), 0.25, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestSelectFrequencies
# ---------------------------------------------------------------------------


class TestSelectFrequencies:
    """Tests for _select_frequencies."""

    def test_returns_nonempty(self):
        """Selected frequencies should be non-empty."""
        plan = _select_frequencies(5e6, 0.65, 2.5, -40.0, 5e4, xp)
        assert plan.selected_freqs.shape[0] > 0

    def test_freq_step_positive(self):
        """Frequency step should be positive."""
        plan = _select_frequencies(5e6, 0.65, 2.5, -40.0, 5e4, xp)
        assert plan.freq_step > 0

    def test_selected_range_within_bounds(self):
        """Selected frequencies should lie in [0, 2*fc]."""
        fc = 5e6
        plan = _select_frequencies(fc, 0.65, 2.5, -40.0, 5e4, xp)
        freqs = np.asarray(plan.selected_freqs)
        assert np.all(freqs >= 0)
        assert np.all(freqs <= 2 * fc)

    def test_spectra_shapes_match(self):
        """Pulse and probe spectra should have same length as selected_freqs."""
        plan = _select_frequencies(5e6, 0.65, 2.5, -40.0, 5e4, xp)
        n = plan.selected_freqs.shape[0]
        assert plan.pulse_spectrum.shape[0] == n
        assert plan.probe_spectrum.shape[0] == n


# ---------------------------------------------------------------------------
# TestFirstLastTrue
# ---------------------------------------------------------------------------


class TestFirstLastTrue:
    """Tests for _first_last_true."""

    def test_all_true(self):
        """All-True mask should return (0, n-1)."""
        mask = xp.asarray([True, True, True, True])
        first, last = _first_last_true(xp, mask)
        assert first == 0
        assert last == 3

    def test_all_false(self):
        """All-False mask should return (0, 0)."""
        mask = xp.asarray([False, False, False])
        first, last = _first_last_true(xp, mask)
        assert first == 0
        assert last == 0

    def test_single_true(self):
        """Single True at position k should return (k, k)."""
        mask = xp.asarray([False, False, True, False, False])
        first, last = _first_last_true(xp, mask)
        assert first == 2
        assert last == 2

    def test_range_true(self):
        """True from position i to j should return (i, j)."""
        mask = xp.asarray([False, True, True, True, False])
        first, last = _first_last_true(xp, mask)
        assert first == 1
        assert last == 3
