"""Behavioral and PyMUST parity tests for time-domain wavefields."""

from typing import cast

import numpy as np
import pymust
import pytest

from fast_simus.medium_params import MediumParams
from fast_simus.pfield import PfieldSpectrumInfo
from fast_simus.transducer_presets import P4_2v
from fast_simus.tx_delay import focused
from fast_simus.utils._array_api import Array, _ArrayNamespace
from fast_simus.utils.geometry import element_positions
from fast_simus.wavefield import spectrum_to_wavefield, wavefield
from tests.conftest import to_numpy


def _grid(
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    nx: int,
    nz: int,
) -> np.ndarray:
    x_grid, z_grid = np.meshgrid(np.linspace(*x_range, nx), np.linspace(*z_range, nz))
    return np.stack([x_grid, z_grid], axis=-1)


def _focused_delays(params, xp: _ArrayNamespace, focus_m):
    elements, _theta, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
    return focused(
        elements,
        xp.asarray(list(focus_m)),
        speed_of_sound=MediumParams().speed_of_sound,
        radius=params.radius,
        apex_offset=apex,
    )


def test_wavefield_shape_and_timing(xp: _ArrayNamespace) -> None:
    """Frequency step changes duration without changing frame spacing."""
    params = P4_2v()
    positions = xp.asarray(_grid((-0.02, 0.02), (1e-3, 0.05), 7, 9))
    delays = _focused_delays(params, xp, (0.0, 0.03))

    coarse = wavefield(positions, delays, params, frequency_step=1.0)
    fine = wavefield(positions, delays, params, frequency_step=0.5)
    coarse_times = to_numpy(coarse.times)
    fine_times = to_numpy(fine.times)

    assert coarse.frames.shape == (9, 7, coarse_times.size)
    assert fine.frames.shape == (9, 7, fine_times.size)
    assert fine_times.size > coarse_times.size
    assert coarse_times[0] == pytest.approx(0.0)
    np.testing.assert_allclose(np.diff(coarse_times), 1.0 / (4.0 * params.freq_center), rtol=2e-3)
    assert np.diff(coarse_times)[0] == pytest.approx(np.diff(fine_times)[0], rel=1e-6)


def test_single_frequency_reconstructs_that_frequency(xp: _ArrayNamespace) -> None:
    """A single populated spectrum bin reconstructs at that frequency."""
    freq_step = 1.0e5
    index = 20
    info = PfieldSpectrumInfo(
        selected_freqs=xp.asarray([index * freq_step]),
        freq_idx_start=index,
        n_freq_full=129,
        freq_step=freq_step,
        correction_factor=1.0,
    )
    spectrum = xp.reshape(xp.asarray([1.0 + 0.0j]), (1, 1))

    result = spectrum_to_wavefield(spectrum, info)
    trace = to_numpy(result.frames)[0]
    times = to_numpy(result.times)
    frequencies = np.fft.rfftfreq(trace.size, d=float(times[1] - times[0]))

    assert frequencies[np.argmax(np.abs(np.fft.rfft(trace)))] == pytest.approx(index * freq_step, rel=0.02)


def test_time_oversampling_preserves_original_ifft_samples(xp: _ArrayNamespace) -> None:
    """Frequency-domain zero padding gives denser samples of the same field."""
    freq_step = 1.0e5
    info = PfieldSpectrumInfo(
        selected_freqs=xp.asarray([2.0e6, 2.1e6, 2.2e6]),
        freq_idx_start=20,
        n_freq_full=65,
        freq_step=freq_step,
        correction_factor=1.0,
    )
    spectrum = xp.reshape(xp.asarray([1.0 + 0.5j, -0.25 + 1.0j, 0.5 - 0.75j]), (1, 3))

    original = spectrum_to_wavefield(spectrum, info)
    dense = spectrum_to_wavefield(spectrum, info, time_oversampling=2)

    assert dense.frames.shape[-1] == 2 * original.frames.shape[-1]
    np.testing.assert_allclose(to_numpy(dense.frames)[..., ::2], to_numpy(original.frames), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(to_numpy(dense.times)[::2], to_numpy(original.times), rtol=1e-7, atol=1e-12)


def test_time_oversampling_requires_positive_integer(xp: _ArrayNamespace) -> None:
    """Invalid zero-padding factors fail before any transform work."""
    info = PfieldSpectrumInfo(
        selected_freqs=xp.asarray([1.0e6]),
        freq_idx_start=1,
        n_freq_full=4,
        freq_step=1.0e6,
        correction_factor=1.0,
    )

    with pytest.raises(ValueError, match="positive integer"):
        spectrum_to_wavefield(xp.asarray([1.0 + 0.0j]), info, time_oversampling=0)


def test_wavefront_travels_at_speed_of_sound(xp: _ArrayNamespace) -> None:
    """Arrival time changes with range according to the medium sound speed."""
    params = P4_2v()
    medium = MediumParams()
    z_near, z_far = 0.02, 0.04
    positions = xp.asarray([[0.0, z_near], [0.0, z_far]])

    result = wavefield(positions, xp.zeros(params.n_elements), params, medium, frequency_step=0.5)
    frames = to_numpy(result.frames)
    times = to_numpy(result.times)
    arrival_gap = times[np.argmax(np.abs(frames[1]))] - times[np.argmax(np.abs(frames[0]))]

    assert arrival_gap == pytest.approx((z_far - z_near) / medium.speed_of_sound, rel=0.1)


def test_matches_pymust_mkmovie() -> None:
    """The normalized time-domain field matches PyMUST's movie."""
    params = P4_2v()
    param = pymust.getparam("P4-2v")
    aperture = cast(float, param.pitch) * (cast(int, param.Nelements) - 1)
    param.movie = np.array([2 * aperture * 100, 2 * aperture * 100, 50])
    delays = pymust.txdelay(0.0, 0.03, param)
    frames_ref, _info, _param = pymust.mkmovie(delays, param)
    expected = np.transpose(np.asarray(frames_ref, dtype=np.float64), (1, 0, 2)) / 255 * 2 - 1

    pixel_size = 1e-2 / 50
    axis = np.arange(pixel_size / 2, 2 * aperture + pixel_size / 2, pixel_size)
    x_grid, z_grid = np.meshgrid(axis - axis.mean(), axis)
    result = wavefield(
        cast(Array, np.stack([x_grid, z_grid], axis=-1)),
        cast(Array, np.ravel(np.asarray(delays))),
        params,
        element_splitting=1,
        frequency_step=0.5,
    )
    actual = np.asarray(result.frames)

    assert actual.shape[:2] == expected.shape[:2]
    n_frames = min(actual.shape[-1] - 1, expected.shape[-1])
    actual = actual[..., 1 : n_frames + 1]
    expected = expected[..., :n_frames]
    actual /= np.max(np.abs(actual))
    expected /= np.max(np.abs(expected))
    assert np.corrcoef(actual.ravel(), expected.ravel())[0, 1] > 0.99
