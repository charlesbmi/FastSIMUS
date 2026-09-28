"""Canonical frequency and sampling metadata, independent of spatial models."""

from dataclasses import dataclass
from math import ceil, inf, isfinite, pi

from fast_simus._pfield_math import _first_last_true
from fast_simus.spectrum import probe_spectrum, pulse_spectrum
from fast_simus.utils._array_api import Array


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
