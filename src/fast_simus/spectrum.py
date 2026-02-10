"""Spectrum computation for ultrasound pulse and probe response.

Provides factory functions that return callables computing frequency-domain
representations of the transmitted pulse and probe frequency response.

References:
    Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
    Part I: theory & examples. CMPB, 2022;218:106726.
"""

from __future__ import annotations

from collections.abc import Callable
from math import log, pi

import numpy as np

# Epsilon to avoid division by zero in sinc computation
_EPS: float = 1e-16


def mysinc(x: np.ndarray) -> np.ndarray:
    """Compute the unnormalized sinc function: sin(x)/x.

    Uses |x| + eps to avoid division by zero, matching the MUST convention.
    Note: this is NOT numpy.sinc which computes sin(pi*x)/(pi*x).

    Args:
        x: Input array in radians.

    Returns:
        sin(|x|+eps) / (|x|+eps), element-wise.
    """
    abs_x = np.abs(x) + _EPS
    return np.sin(abs_x) / abs_x


def pulse_spectrum_fn(
    freq_center: float,
    tx_n_wavelengths: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a pulse spectrum function for a windowed sine pulse.

    Returns a callable that computes the frequency-domain representation
    of a windowed sine pulse with the given center frequency and number
    of wavelengths.

    Args:
        freq_center: Center frequency in Hz. Must be positive.
        tx_n_wavelengths: Number of wavelengths of the TX pulse.
            Defaults to 1.0.

    Returns:
        Callable taking angular frequency w (rad/s) and returning
        the complex spectrum.
    """
    t_pulse = tx_n_wavelengths / freq_center  # pulse duration (s)
    wc = 2.0 * pi * freq_center  # center angular frequency (rad/s)

    def _pulse_spectrum(w: np.ndarray) -> np.ndarray:
        return 1j * (mysinc(t_pulse * (w - wc) / 2.0) - mysinc(t_pulse * (w + wc) / 2.0))

    return _pulse_spectrum


def probe_spectrum_fn(
    freq_center: float,
    bandwidth: float = 0.75,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a probe frequency response function.

    Returns a callable that computes the one-way probe frequency response
    using a generalized normal window. The bandwidth parameter defines the
    pulse-echo 6 dB fractional bandwidth.

    The returned function is the square root of the pulse-echo response,
    appropriate for one-way (transmit-only or receive-only) use.

    Args:
        freq_center: Center frequency in Hz. Must be positive.
        bandwidth: Fractional bandwidth (0.75 = 75%). Must be in (0, 2.0].
            Defaults to 0.75.

    Returns:
        Callable taking angular frequency w (rad/s) and returning
        the real-valued probe response (one-way).

    References:
        Generalized normal window:
        https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window
    """
    wc = 2.0 * pi * freq_center
    # Convert fractional bandwidth to angular bandwidth
    # PyMUST uses bandwidth in % and divides by 100; we use fraction directly
    w_bw = bandwidth * wc
    # Shape parameter for the generalized normal window
    # 126 = 10^(6dB * 2 / (20/log10(e))) = 10^(12/20*log10(e))
    # Actually: 6dB pulse-echo -> 3dB one-way -> ratio = 10^(6/20) per side
    # PyMUST uses log(126) which is log(2) * 6dB * 20/log(10) related
    p = log(126) / log(2.0 * wc / w_bw)

    # Denominator of the exponent in the generalized normal window
    sigma = w_bw / 2.0 / (log(2) ** (1.0 / p))

    def _probe_spectrum(w: np.ndarray) -> np.ndarray:
        # Pulse-echo (squared) response
        spectrum_sqr = np.exp(-(np.abs(w - wc) / sigma) ** p)
        # One-way response (square root)
        return np.sqrt(spectrum_sqr)

    return _probe_spectrum
