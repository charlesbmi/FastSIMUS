"""Canonical frequency and sampling metadata, independent of spatial models."""

from dataclasses import dataclass
from math import ceil, inf, isfinite, log2, pi
from typing import cast

from fast_simus._pfield_math import _first_last_true
from fast_simus.spectrum import probe_spectrum, pulse_spectrum
from fast_simus.utils._array_api import Array, _ArrayNamespace, _ArrayNamespaceWithFFT


@dataclass(frozen=True, eq=False)
class FrequencyGrid:
    """Uniform global grid and retained contiguous band."""

    selected_freqs: Array
    freq_step: float
    n_freq_full: int
    freq_idx_start: int
    is_cw: bool


def frequency_grid(fc, bandwidth, pulse, threshold, max_step, xp, dtype):
    """Select a band while retaining unrounded scalar grid spacing."""
    if pulse != inf and (not isfinite(pulse) or pulse <= 0):
        raise ValueError("Pulse duration must be positive or +inf for CW")
    if not isfinite(threshold) or threshold >= 0 or not isfinite(max_step) or max_step <= 0:
        raise ValueError("Require a negative finite threshold and positive frequency step")
    if pulse == inf:
        grid = FrequencyGrid(xp.asarray([fc], dtype=dtype), fc, 3, 1, True)
    else:
        n = 2 * ceil(fc / max_step) + 1
        step = 2 * fc / (n - 1)
        frequencies = xp.arange(n, dtype=dtype) * step
        magnitude = xp.abs(
            pulse_spectrum(2 * pi * frequencies, fc, pulse) * probe_spectrum(2 * pi * frequencies, fc, bandwidth)
        )
        mask = magnitude > xp.max(magnitude) * 10 ** (threshold / 20)
        first, last = _first_last_true(xp, mask)
        grid = FrequencyGrid(frequencies[first : last + 1], step, n, first, False)
    omega = 2 * pi * grid.selected_freqs
    return (
        grid,
        (xp.ones_like(omega) + 0j if grid.is_cw else pulse_spectrum(omega, fc, pulse)),
        probe_spectrum(omega, fc, bandwidth),
    )


def _two_way_pulse_duration(
    freq_center: float,
    bandwidth: float,
    tx_n_wavelengths: float,
    xp: _ArrayNamespace,
) -> float:
    """Compute the temporal extent of the two-way (pulse-echo) pulse.

    Replicates the pulse duration computation from PyMUST's getpulse(param, 2).
    Uses pulse_spectrum * probe_spectrum^2, IFFTs, and thresholds at 1/1023.

    Args:
        freq_center: Center frequency in Hz.
        bandwidth: Fractional bandwidth (0.75 = 75%).
        tx_n_wavelengths: Number of wavelengths of the TX pulse.
        xp: Array namespace (must have FFT extension).

    Returns:
        Pulse duration in seconds.
    """
    # hasattr instead of isinstance(_ArrayNamespaceWithFFT) because Python 3.12+
    # Protocol isinstance uses getattr_static, which misses lazy sub-module attrs
    # like numpy.fft. See https://docs.python.org/3/whatsnew/3.12.html#typing
    if not hasattr(xp, "fft"):
        msg = "simus requires an array backend with FFT support (e.g. numpy, jax, cupy)"
        raise RuntimeError(msg)
    xp_fft = cast(_ArrayNamespaceWithFFT, xp)

    dt = 1e-9
    df = freq_center / tx_n_wavelengths / 32
    p = ceil(log2(1.0 / dt / 2.0 / df))
    n_fft = 2**p
    omega = 2.0 * pi * xp.linspace(0, 1.0 / dt / 2.0, n_fft)

    # Two-way spectrum: pulse * probe^2
    ps = pulse_spectrum(omega, freq_center, tx_n_wavelengths)
    pr = probe_spectrum(omega, freq_center, bandwidth)
    two_way = ps * pr**2

    pulse = xp_fft.fft.fftshift(xp_fft.fft.irfft(two_way))
    pulse = pulse / xp.max(xp.abs(pulse))

    above = pulse > (1.0 / 1023)
    n = above.shape[0]
    indices = xp.arange(n)
    masked_min = xp.where(above, indices, xp.asarray(n))
    masked_max = xp.where(above, indices, xp.asarray(-1))
    idx1 = int(xp.min(masked_min))
    idx2 = int(xp.max(masked_max))

    if idx1 >= n:
        return tx_n_wavelengths / freq_center

    trim_idx = min(idx1 + 1, 2 * n_fft - 1 - idx2 - 1)
    pulse_trimmed = pulse[-trim_idx : trim_idx - 2 : -1]
    return float(pulse_trimmed.shape[0] * dt)


@dataclass(frozen=True)
class SamplingInfo:
    """Requested and effective causal sample grid."""

    requested_sampling_frequency: float
    n_fft: int
    freq_step: float
    time_origin: float = 0.0

    @property
    def sampling_frequency(self):
        """Effective sampling rate after integer FFT rounding."""
        return self.n_fft * self.freq_step

    def times(self, xp, dtype):
        """Time axis in seconds for the retained causal half."""
        return xp.arange((self.n_fft + 1) // 2, dtype=dtype) / self.sampling_frequency + self.time_origin


class SamplingMetadata:
    """Read-only timing properties shared by echo and acquisition results."""

    _sampling: SamplingInfo

    @property
    def requested_sampling_frequency(self):
        """Requested sample rate in Hz."""
        return self._sampling.requested_sampling_frequency

    @property
    def sampling_frequency(self):
        """Effective sample rate in Hz."""
        return self._sampling.sampling_frequency

    @property
    def time_origin(self):
        """Time origin relative to each independent trigger, in seconds."""
        return self._sampling.time_origin
