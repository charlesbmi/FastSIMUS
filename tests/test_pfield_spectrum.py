"""Behavioral and PyMUST parity tests for the pressure spectrum."""

from typing import cast

import numpy as np
import pymust
import pytest

from fast_simus.medium_params import MediumParams
from fast_simus.pfield import (
    pfield,
    pfield_precompute,
    pfield_spectrum,
    pfield_spectrum_compute,
    rms_from_spectrum,
)
from fast_simus.transducer_presets import C5_2v, P4_2v
from fast_simus.tx_delay import focused
from fast_simus.utils._array_api import Array, _ArrayNamespace
from fast_simus.utils.geometry import element_positions
from tests.conftest import to_numpy


def _positions(x_range: tuple[float, float], z_range: tuple[float, float], n: int = 24) -> np.ndarray:
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


@pytest.mark.parametrize("full_frequency_directivity", [False, True], ids=["center-directivity", "full-directivity"])
def test_spectrum_reconstructs_rms(xp: _ArrayNamespace, full_frequency_directivity: bool) -> None:
    """Integrating the spectrum reproduces the public RMS field."""
    params = P4_2v()
    positions = xp.asarray(_positions((-0.02, 0.02), (1e-3, 0.06)))
    delays = _focused_delays(params, xp, (0.0, 0.03))
    apodization = xp.asarray(np.hanning(params.n_elements))

    spectrum, info = pfield_spectrum(
        positions,
        delays,
        params,
        tx_apodization=apodization,
        full_frequency_directivity=full_frequency_directivity,
    )
    actual = to_numpy(rms_from_spectrum(spectrum, info))
    expected = to_numpy(
        pfield(
            positions,
            delays,
            params,
            tx_apodization=apodization,
            full_frequency_directivity=full_frequency_directivity,
        )
    )

    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(actual / scale, expected / scale, rtol=1e-4, atol=1e-6)


def test_spectrum_contract(xp: _ArrayNamespace) -> None:
    """Spectrum shape and metadata describe one contiguous frequency band."""
    params = C5_2v()
    positions = xp.asarray(_positions((-0.03, 0.03), (1e-3, 0.08), n=12))
    delays = _focused_delays(params, xp, (0.0, 0.05))

    spectrum, info = pfield_spectrum(positions, delays, params)
    frequencies = to_numpy(info.selected_freqs)

    assert spectrum.shape == (*positions.shape[:-1], frequencies.size)
    assert info.freq_idx_start >= 0
    assert info.freq_idx_start + frequencies.size <= info.n_freq_full
    np.testing.assert_allclose(
        frequencies,
        (info.freq_idx_start + np.arange(frequencies.size)) * info.freq_step,
        rtol=1e-5,
    )
    assert info.freq_step * (info.n_freq_full - 1) == pytest.approx(2.0 * params.freq_center, rel=1e-5)


def test_spectrum_precompute_compute_matches_public_function(xp: _ArrayNamespace) -> None:
    """The split spectrum API preserves the high-level result."""
    params = P4_2v()
    positions = xp.asarray(_positions((-0.01, 0.01), (1e-3, 0.03), n=8))
    delays = _focused_delays(params, xp, (0.0, 0.02))

    expected, _info = pfield_spectrum(positions, delays, params)
    plan = pfield_precompute(positions, delays, params)
    actual = pfield_spectrum_compute(positions, delays, plan, params)

    np.testing.assert_array_equal(to_numpy(actual), to_numpy(expected))
    np.testing.assert_array_equal(
        to_numpy(rms_from_spectrum(actual, plan)),
        to_numpy(rms_from_spectrum(expected, _info)),
    )


def test_matches_pymust_spectrum_and_frequency_grid() -> None:
    """Complex pressure and frequency selection match PyMUST."""
    params = P4_2v()
    param = pymust.getparam("P4-2v")
    x_grid, z_grid = np.meshgrid(np.linspace(-0.02, 0.02, 20), np.linspace(1e-3, 0.06, 30))
    delays = np.asarray(pymust.txdelay(0.0, 0.03, param))
    _rp, spectrum_ref, idx = pymust.pfield(x_grid, np.array([]), z_grid, delays, param)

    positions = np.stack([x_grid, z_grid], axis=-1)
    spectrum, info = pfield_spectrum(cast(Array, positions), cast(Array, np.ravel(delays)), params)
    idx = np.asarray(idx, dtype=bool)
    actual = np.asarray(spectrum) * info.correction_factor
    expected = np.asarray(spectrum_ref)

    assert actual.shape == expected.shape
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(actual / scale, expected / scale, atol=1e-4)
    assert (info.n_freq_full, info.freq_idx_start, info.selected_freqs.shape[0]) == (
        idx.size,
        int(np.argmax(idx)),
        int(idx.sum()),
    )
