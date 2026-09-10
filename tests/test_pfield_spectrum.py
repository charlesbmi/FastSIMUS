"""Tests for pfield_spectrum (complex pressure spectrum on a grid).

pfield_spectrum exposes the complex field P(x, f) that pfield squares away.
The strongest available check needs no reference implementation: pfield's RMS
output is exactly sqrt(correction_factor * sum_k |P_k|^2), so recomputing the
RMS from the spectrum must reproduce pfield to floating-point tolerance.

Reference parity against PyMUST's pfield SPECT output is also covered here.
"""

import numpy as np
import pymust
import pytest

from fast_simus.medium_params import MediumParams
from fast_simus.pfield import pfield, pfield_spectrum, rms_from_spectrum
from fast_simus.transducer_presets import C5_2v, P4_2v
from fast_simus.tx_delay import focused, plane_wave
from fast_simus.utils._array_api import _ArrayNamespace
from fast_simus.utils.geometry import element_positions
from tests.conftest import to_numpy

GRID_SIZE = 40


def _positions(x_range: tuple[float, float], z_range: tuple[float, float], n: int = GRID_SIZE) -> np.ndarray:
    """Build an (n, n, 2) grid of positions in meters."""
    x_grid, z_grid = np.meshgrid(np.linspace(*x_range, n), np.linspace(*z_range, n))
    return np.stack([x_grid, z_grid], axis=-1)


def _focused_delays(params, xp: _ArrayNamespace, focus_m: tuple[float, float]):
    """Transmit delays focusing at focus_m for the given preset."""
    elem_pos, _theta, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
    return focused(
        elem_pos,
        xp.asarray(list(focus_m)),
        speed_of_sound=MediumParams().speed_of_sound,
        radius=params.radius,
        apex_offset=apex,
    )


class TestRmsIdentity:
    """pfield must equal the frequency-integrated magnitude of pfield_spectrum."""

    def test_focused_matches_pfield(self, xp: _ArrayNamespace) -> None:
        """sqrt(sum |P_k|^2 * correction_factor) reproduces pfield for a focused transmit."""
        params = P4_2v()
        positions = xp.asarray(_positions((-0.02, 0.02), (1e-3, 0.06)))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        spectrum, info = pfield_spectrum(positions, delays, params)
        expected = pfield(positions, delays, params)

        np.testing.assert_allclose(
            to_numpy(rms_from_spectrum(spectrum, info)),
            to_numpy(expected),
            rtol=1e-5,
            atol=0.0,
        )

    def test_plane_wave_matches_pfield(self, xp: _ArrayNamespace) -> None:
        """The identity also holds for a steered plane wave."""
        params = P4_2v()
        positions = xp.asarray(_positions((-0.02, 0.02), (1e-3, 0.06)))
        elem_pos, _theta, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
        delays = plane_wave(
            elem_pos,
            np.deg2rad(15.0),
            speed_of_sound=MediumParams().speed_of_sound,
            radius=params.radius,
            apex_offset=apex,
        )

        spectrum, info = pfield_spectrum(positions, delays, params)
        expected = pfield(positions, delays, params)

        np.testing.assert_allclose(
            to_numpy(rms_from_spectrum(spectrum, info)),
            to_numpy(expected),
            rtol=1e-5,
            atol=0.0,
        )

    def test_convex_array_matches_pfield(self, xp: _ArrayNamespace) -> None:
        """The identity holds for a convex array, which masks an interior region."""
        params = C5_2v()
        positions = xp.asarray(_positions((-0.03, 0.03), (1e-3, 0.08), n=24))
        delays = _focused_delays(params, xp, (0.0, 0.05))

        spectrum, info = pfield_spectrum(positions, delays, params)
        expected = pfield(positions, delays, params)

        np.testing.assert_allclose(
            to_numpy(rms_from_spectrum(spectrum, info)),
            to_numpy(expected),
            rtol=1e-5,
            atol=0.0,
        )

    def test_apodization_matches_pfield(self, xp: _ArrayNamespace) -> None:
        """Transmit apodization is applied identically in both paths."""
        params = P4_2v()
        positions = xp.asarray(_positions((-0.02, 0.02), (1e-3, 0.06), n=24))
        delays = _focused_delays(params, xp, (0.0, 0.03))
        apod = xp.asarray(np.hanning(params.n_elements))

        spectrum, info = pfield_spectrum(positions, delays, params, tx_apodization=apod)
        expected = pfield(positions, delays, params, tx_apodization=apod)

        np.testing.assert_allclose(
            to_numpy(rms_from_spectrum(spectrum, info)),
            to_numpy(expected),
            rtol=1e-5,
            atol=0.0,
        )

    def test_full_frequency_directivity_matches_pfield(self, xp: _ArrayNamespace) -> None:
        """The identity holds with frequency-dependent element directivity."""
        params = P4_2v()
        positions = xp.asarray(_positions((-0.02, 0.02), (1e-3, 0.06), n=24))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        spectrum, info = pfield_spectrum(positions, delays, params, full_frequency_directivity=True)
        expected = pfield(positions, delays, params, full_frequency_directivity=True)

        np.testing.assert_allclose(
            to_numpy(rms_from_spectrum(spectrum, info)),
            to_numpy(expected),
            rtol=1e-5,
            atol=0.0,
        )


class TestShapeAndInfo:
    """Shape and frequency-grid bookkeeping."""

    def test_trailing_axis_is_temporal_frequency(self, xp: _ArrayNamespace) -> None:
        """Spectrum keeps the spatial grid and appends one temporal-frequency axis."""
        params = P4_2v()
        positions = xp.asarray(_positions((-0.02, 0.02), (1e-3, 0.06), n=17))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        spectrum, info = pfield_spectrum(positions, delays, params)

        assert spectrum.shape == (17, 17, info.selected_freqs.shape[0])

    def test_selected_band_fits_inside_full_grid(self, xp: _ArrayNamespace) -> None:
        """The selected band is a contiguous slice of the uniform [0, 2 fc] grid."""
        params = P4_2v()
        positions = xp.asarray(_positions((-0.02, 0.02), (1e-3, 0.06), n=17))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        _spectrum, info = pfield_spectrum(positions, delays, params)
        n_selected = info.selected_freqs.shape[0]

        assert info.freq_idx_start >= 0
        assert info.freq_idx_start + n_selected <= info.n_freq_full
        assert info.freq_step > 0.0

    def test_full_grid_spans_zero_to_twice_center_frequency(self, xp: _ArrayNamespace) -> None:
        """The implied full grid is linspace(0, 2 fc, n_freq_full)."""
        params = P4_2v()
        positions = xp.asarray(_positions((-0.02, 0.02), (1e-3, 0.06), n=17))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        _spectrum, info = pfield_spectrum(positions, delays, params)

        assert info.freq_step * (info.n_freq_full - 1) == pytest.approx(2.0 * params.freq_center, rel=1e-5)

    def test_freq_idx_start_locates_the_band(self, xp: _ArrayNamespace) -> None:
        """freq_idx_start times freq_step is the first selected frequency."""
        params = P4_2v()
        positions = xp.asarray(_positions((-0.02, 0.02), (1e-3, 0.06), n=17))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        _spectrum, info = pfield_spectrum(positions, delays, params)

        first = float(to_numpy(info.selected_freqs)[0])
        assert info.freq_idx_start * info.freq_step == pytest.approx(first, rel=1e-4)


class TestPhysicalProperties:
    """Properties that follow from the physics rather than a reference."""

    def test_field_is_zero_behind_the_array(self, xp: _ArrayNamespace) -> None:
        """Points at negative depth receive nothing."""
        params = P4_2v()
        positions = xp.asarray(_positions((-0.02, 0.02), (-0.02, -1e-3), n=12))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        spectrum, _info = pfield_spectrum(positions, delays, params)

        np.testing.assert_array_equal(to_numpy(xp.abs(spectrum)), 0.0)

    def test_convex_array_interior_is_zero(self, xp: _ArrayNamespace) -> None:
        """Points inside a convex array's radius of curvature receive nothing."""
        params = C5_2v()
        apex_z = params.radius - np.sqrt(params.radius**2 - (params.pitch * (params.n_elements - 1) / 2) ** 2)
        positions = xp.asarray(_positions((-0.005, 0.005), (1e-4, apex_z * 0.5), n=8))
        delays = _focused_delays(params, xp, (0.0, 0.05))

        spectrum, _info = pfield_spectrum(positions, delays, params)

        np.testing.assert_array_equal(to_numpy(xp.abs(spectrum)), 0.0)

    def test_focus_is_the_spectral_energy_peak(self, xp: _ArrayNamespace) -> None:
        """Total spectral energy peaks near the requested focus."""
        params = P4_2v()
        focus_z = 0.03
        x_lin = np.linspace(-0.02, 0.02, 41)
        z_lin = np.linspace(0.01, 0.05, 41)
        x_grid, z_grid = np.meshgrid(x_lin, z_lin)
        positions = xp.asarray(np.stack([x_grid, z_grid], axis=-1))
        delays = _focused_delays(params, xp, (0.0, focus_z))

        spectrum, _info = pfield_spectrum(positions, delays, params)
        energy = to_numpy(xp.sum(xp.real(spectrum * xp.conj(spectrum)), axis=-1))

        iz, ix = np.unravel_index(np.argmax(energy), energy.shape)
        assert abs(x_lin[ix]) < 1e-3
        assert abs(z_lin[iz] - focus_z) < 3e-3


class TestPymustParity:
    """Parity against PyMUST's pfield SPECT / IDX outputs.

    One scaling difference is deliberate. PyMUST multiplies SPECT by CorFac
    linearly (pfield.py:861) while using the same CorFac as the quadratic
    measure for RP, so its SPECT and RP scalings are inconsistent; mkmovie
    normalizes the result and never notices. FastSIMUS returns the raw complex
    pressure and reports correction_factor separately, which is what makes the
    RMS identity above exact. Parity therefore compares against
    ``spectrum * correction_factor``.
    """

    def test_matches_pymust_spect(self) -> None:
        """FastSIMUS spectrum matches PyMUST's SPECT for a focused transmit."""
        params = P4_2v()
        param = pymust.getparam("P4-2v")
        x_grid, z_grid = np.meshgrid(np.linspace(-0.02, 0.02, 20), np.linspace(1e-3, 0.06, 30))
        delays = np.asarray(pymust.txdelay(0.0, 0.03, param))

        _rp, spect_ref, _idx = pymust.pfield(x_grid, [], z_grid, delays, param)

        positions = np.stack([x_grid, z_grid], axis=-1)
        spectrum, info = pfield_spectrum(positions, np.ravel(delays), params)

        ours = np.asarray(spectrum) * info.correction_factor
        ref = np.asarray(spect_ref)
        assert ours.shape == ref.shape

        scale = np.abs(ref).max()
        np.testing.assert_allclose(ours / scale, ref / scale, atol=1e-4)

    def test_matches_pymust_frequency_grid(self) -> None:
        """freq_idx_start and n_freq_full agree with PyMUST's IDX mask."""
        params = P4_2v()
        param = pymust.getparam("P4-2v")
        x_grid, z_grid = np.meshgrid(np.linspace(-0.02, 0.02, 20), np.linspace(1e-3, 0.06, 30))
        delays = np.asarray(pymust.txdelay(0.0, 0.03, param))

        _rp, _spect, idx = pymust.pfield(x_grid, [], z_grid, delays, param)
        idx = np.asarray(idx).astype(bool)

        positions = np.stack([x_grid, z_grid], axis=-1)
        _spectrum, info = pfield_spectrum(positions, np.ravel(delays), params)

        assert info.n_freq_full == idx.size
        assert info.freq_idx_start == int(np.argmax(idx))
        assert info.selected_freqs.shape[0] == int(idx.sum())
