"""Tests for wavefield (propagating pressure field over time).

wavefield is the FastSIMUS equivalent of MUST's mkmovie: it inverse transforms
the complex spectrum from pfield_spectrum along the temporal-frequency axis to
obtain p(x, z, t).

The frame spacing is dt = 1/(4 fc) regardless of the frequency step, because
the full frequency grid always spans [0, 2 fc]. That makes frames from
different frequency-step choices, and from MUST, directly comparable sample by
sample.
"""

import numpy as np
import pymust
import pytest

from fast_simus.medium_params import MediumParams
from fast_simus.transducer_presets import P4_2v
from fast_simus.tx_delay import focused
from fast_simus.utils._array_api import _ArrayNamespace
from fast_simus.utils.geometry import element_positions
from fast_simus.wavefield import spectrum_to_wavefield, wavefield
from tests.conftest import to_numpy


def _grid(x_range, z_range, nx, nz) -> np.ndarray:
    """Build an (nz, nx, 2) grid of positions in meters."""
    x_grid, z_grid = np.meshgrid(np.linspace(*x_range, nx), np.linspace(*z_range, nz))
    return np.stack([x_grid, z_grid], axis=-1)


def _focused_delays(params, xp: _ArrayNamespace, focus_m):
    """Transmit delays focusing at focus_m."""
    elem_pos, _theta, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
    return focused(
        elem_pos,
        xp.asarray(list(focus_m)),
        speed_of_sound=MediumParams().speed_of_sound,
        radius=params.radius,
        apex_offset=apex,
    )


class TestTimeAxis:
    """Frame timing follows from the [0, 2 fc] frequency grid."""

    def test_frame_spacing_is_quarter_period(self, xp: _ArrayNamespace) -> None:
        """Dt equals 1/(4 fc), matching simus's default sampling frequency."""
        params = P4_2v()
        positions = xp.asarray(_grid((-0.02, 0.02), (1e-3, 0.05), 12, 12))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        result = wavefield(positions, delays, params)
        times = to_numpy(result.times)

        assert times[0] == pytest.approx(0.0, abs=1e-12)
        # Loose rtol: float32 backends lose precision accumulating the time axis.
        np.testing.assert_allclose(np.diff(times), 1.0 / (4.0 * params.freq_center), rtol=2e-3)

    def test_frame_spacing_is_independent_of_frequency_step(self, xp: _ArrayNamespace) -> None:
        """A finer frequency step lengthens the record but keeps dt fixed."""
        params = P4_2v()
        positions = xp.asarray(_grid((-0.02, 0.02), (1e-3, 0.05), 8, 8))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        coarse = wavefield(positions, delays, params, frequency_step=1.0)
        fine = wavefield(positions, delays, params, frequency_step=0.5)

        dt_coarse = float(np.diff(to_numpy(coarse.times))[0])
        dt_fine = float(np.diff(to_numpy(fine.times))[0])

        assert dt_coarse == pytest.approx(dt_fine, rel=1e-6)
        assert fine.times.shape[0] > coarse.times.shape[0]

    def test_shape_is_grid_by_time(self, xp: _ArrayNamespace) -> None:
        """Frames keep the spatial grid and append a time axis."""
        params = P4_2v()
        positions = xp.asarray(_grid((-0.02, 0.02), (1e-3, 0.05), 9, 11))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        result = wavefield(positions, delays, params)

        assert result.frames.shape == (11, 9, result.times.shape[0])


class TestPureTransform:
    """spectrum_to_wavefield on a synthetic spectrum."""

    def test_single_frequency_bin_gives_a_sinusoid(self, xp: _ArrayNamespace) -> None:
        """One populated bin reconstructs a cosine at exactly that frequency."""
        from fast_simus.pfield import PfieldSpectrumInfo

        n_freq_full = 129
        freq_step = 1.0e5
        idx = 20
        info = PfieldSpectrumInfo(
            selected_freqs=xp.asarray([idx * freq_step]),
            freq_idx_start=idx,
            n_freq_full=n_freq_full,
            freq_step=freq_step,
            correction_factor=1.0,
        )
        spectrum = xp.reshape(xp.asarray([1.0 + 0.0j]), (1, 1))

        result = spectrum_to_wavefield(spectrum, info)
        trace = to_numpy(result.frames)[0, :]
        times = to_numpy(result.times)

        # Dominant frequency of the reconstructed trace must be idx * freq_step.
        dt = float(times[1] - times[0])
        freqs = np.fft.rfftfreq(trace.size, d=dt)
        dominant = freqs[np.argmax(np.abs(np.fft.rfft(trace)))]
        assert dominant == pytest.approx(idx * freq_step, rel=0.02)

    def test_amplitude_is_independent_of_frequency_grid(self, xp: _ArrayNamespace) -> None:
        """Refining the frequency step must not change the reconstructed amplitude."""
        params = P4_2v()
        positions = xp.asarray(_grid((-0.01, 0.01), (0.02, 0.04), 6, 6))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        coarse = wavefield(positions, delays, params, frequency_step=1.0)
        fine = wavefield(positions, delays, params, frequency_step=0.5)

        n = min(coarse.times.shape[0], fine.times.shape[0])
        peak_coarse = float(np.max(np.abs(to_numpy(coarse.frames)[..., :n])))
        peak_fine = float(np.max(np.abs(to_numpy(fine.frames)[..., :n])))

        assert peak_coarse == pytest.approx(peak_fine, rel=0.05)


class TestPhysics:
    """Reference-free checks that the wave actually propagates correctly."""

    def test_wave_arrives_at_focus_at_the_geometric_time(self, xp: _ArrayNamespace) -> None:
        """Peak pressure at the focus occurs at delay + range/c."""
        params = P4_2v()
        medium = MediumParams()
        focus = (0.0, 0.03)

        positions = xp.asarray(np.reshape(np.asarray(focus), (1, 2)))
        delays = _focused_delays(params, xp, focus)

        result = wavefield(positions, delays, params, medium, frequency_step=0.5)
        trace = to_numpy(result.frames)[0, :]
        times = to_numpy(result.times)

        elem_pos = to_numpy(element_positions(params.n_elements, params.pitch, params.radius, xp)[0])
        delays_np = to_numpy(delays)
        ranges = np.hypot(focus[0] - elem_pos[:, 0], focus[1] - elem_pos[:, 1])
        expected_arrival = float(np.mean(delays_np + ranges / medium.speed_of_sound))

        observed_arrival = times[int(np.argmax(np.abs(trace)))]
        # Within one transmit pulse period.
        assert abs(observed_arrival - expected_arrival) < 1.0 / params.freq_center

    def test_wavefront_travels_at_the_speed_of_sound(self, xp: _ArrayNamespace) -> None:
        """Two on-axis depths peak one range difference apart in time."""
        params = P4_2v()
        medium = MediumParams()
        z_near, z_far = 0.02, 0.04

        positions = xp.asarray(np.asarray([[0.0, z_near], [0.0, z_far]]))
        # Unfocused, unsteered transmit: a flat wavefront leaves the array at t = 0.
        delays = xp.zeros(params.n_elements)

        result = wavefield(positions, delays, params, medium, frequency_step=0.5)
        frames = to_numpy(result.frames)
        times = to_numpy(result.times)

        t_near = times[int(np.argmax(np.abs(frames[0, :])))]
        t_far = times[int(np.argmax(np.abs(frames[1, :])))]

        expected_gap = (z_far - z_near) / medium.speed_of_sound
        assert (t_far - t_near) == pytest.approx(expected_gap, rel=0.1)

    def test_field_is_zero_behind_the_array(self, xp: _ArrayNamespace) -> None:
        """Negative-depth points stay silent at all times."""
        params = P4_2v()
        positions = xp.asarray(_grid((-0.01, 0.01), (-0.02, -1e-3), 6, 6))
        delays = _focused_delays(params, xp, (0.0, 0.03))

        result = wavefield(positions, delays, params)

        np.testing.assert_array_equal(to_numpy(result.frames), 0.0)


class TestPymustParity:
    """Parity against PyMUST's mkmovie.

    Two deliberate differences are compensated for here.

    Axis order: PyMUST's mkmovie returns frames whose first two axes are
    transposed relative to its own info.Xgrid / info.Zgrid labels
    (mkmovie.py:328 applies an order='F' reshape). FastSIMUS's orientation is
    the physically correct one, so the reference is transposed.

    Time origin: mkmovie.py:342 flips the inverse transform in time, which puts
    its frame k at what the standard convention calls sample k+1. FastSIMUS
    uses irfft(conj(P)), for which frame n is exactly
    Re{sum_k P_k exp(-i w_k n dt)} with no offset. The comparison therefore
    drops our first frame.
    """

    def test_matches_pymust_mkmovie(self) -> None:
        """Frames correlate with PyMUST's mkmovie for a focused transmit."""
        params = P4_2v()
        param = pymust.getparam("P4-2v")

        aperture = param.pitch * (param.Nelements - 1)
        roi_cm = 2 * aperture * 100
        param.movie = np.array([roi_cm, roi_cm, 50])
        txdel = pymust.txdelay(0.0, 0.03, param)

        frames_ref, _info, _param = pymust.mkmovie(txdel, param)
        ref = np.transpose(np.asarray(frames_ref).astype(np.float64), (1, 0, 2)) / 255 * 2 - 1

        pix = 1e-2 / 50
        xi = np.arange(pix / 2, 2 * aperture + pix / 2, pix)
        zi = np.arange(pix / 2, 2 * aperture + pix / 2, pix)
        x_grid, z_grid = np.meshgrid(xi - xi.mean(), zi)
        positions = np.stack([x_grid, z_grid], axis=-1)

        result = wavefield(
            positions,
            np.ravel(np.asarray(txdel)),
            params,
            element_splitting=1,
            frequency_step=0.5,
        )
        ours = np.asarray(result.frames)

        assert ours.shape[:2] == ref.shape[:2]
        n = min(ours.shape[-1] - 1, ref.shape[-1])
        a = ours[..., 1 : n + 1]
        b = ref[..., :n]
        a = a / np.abs(a).max()
        b = b / np.abs(b).max()

        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.99
