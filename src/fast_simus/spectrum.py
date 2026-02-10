"""Spectrum computation for ultrasound pulse and probe response.

Provides factory functions that return callables computing frequency-domain
representations of the transmitted pulse and probe frequency response.

All functions are Array API compliant and work with NumPy, JAX, CuPy backends.

References:
    Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
    Part I: theory & examples. CMPB, 2022;218:106726.
"""

from __future__ import annotations

from collections.abc import Callable
from math import log, pi

from array_api_compat import array_namespace

from fast_simus.utils._array_api import Array

# Epsilon to avoid division by zero in sinc computation
_EPS: float = 1e-16


def mysinc(x: Array) -> Array:
    """Compute the unnormalized sinc function: sin(x)/x.

    Uses |x| + eps to avoid division by zero, matching the MUST convention.
    Note: this is NOT numpy.sinc which computes sin(pi*x)/(pi*x).

    Args:
        x: Input array in radians.

    Returns:
        sin(|x|+eps) / (|x|+eps), element-wise.
    """
    xp = array_namespace(x)
    abs_x = xp.abs(x) + _EPS
    return xp.sin(abs_x) / abs_x


def pulse_spectrum_fn(
    freq_center: float,
    tx_n_wavelengths: float = 1.0,
) -> Callable[[Array], Array]:
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

    def _pulse_spectrum(w: Array) -> Array:
        return 1j * (mysinc(t_pulse * (w - wc) / 2.0) - mysinc(t_pulse * (w + wc) / 2.0))

    return _pulse_spectrum


def probe_spectrum_fn(
    freq_center: float,
    bandwidth: float = 0.75,
) -> Callable[[Array], Array]:
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
    w_bw = bandwidth * wc
    # Shape parameter for the generalized normal window
    p = log(126) / log(2.0 * wc / w_bw)
    # Denominator of the exponent
    sigma = w_bw / 2.0 / (log(2) ** (1.0 / p))

    def _probe_spectrum(w: Array) -> Array:
        xp = array_namespace(w)
        # Pulse-echo (squared) response
        spectrum_sqr = xp.exp(-((xp.abs(w - wc) / sigma) ** p))
        # One-way response (square root)
        return xp.sqrt(spectrum_sqr)

    return _probe_spectrum
