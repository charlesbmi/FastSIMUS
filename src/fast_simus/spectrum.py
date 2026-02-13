"""Spectrum computation for ultrasound pulse and probe response.

Provides functions to compute frequency-domain representations of the
transmitted pulse and probe frequency response.

All functions are Array API compliant and work with NumPy, JAX, CuPy backends.

References:
    Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
    Part I: theory & examples. CMPB, 2022;218:106726.
"""

from __future__ import annotations

from math import log, pi

import array_api_extra as xpx
from beartype import beartype as typechecker
from jaxtyping import Complex, Float, jaxtyped

from fast_simus.utils._array_api import Array, array_namespace


@jaxtyped(typechecker=typechecker)
def pulse_spectrum(
    angular_freq: Float[Array, " n_freqs"],
    freq_center: float | int,
    tx_n_wavelengths: float | int = 1.0,
) -> Complex[Array, " n_freqs"]:
    """Compute the pulse spectrum for a windowed sine pulse.

    Computes the frequency-domain representation of a windowed sine pulse
    with the given center frequency and number of wavelengths.

    Args:
        angular_freq: Angular frequency in rad/s. Shape (n_freqs,).
        freq_center: Center frequency in Hz. Must be positive.
        tx_n_wavelengths: Number of wavelengths of the TX pulse.

    Returns:
        Complex spectrum at the given angular frequencies. Shape (n_freqs,).
    """
    pulse_duration_s = tx_n_wavelengths / freq_center
    angular_freq_center = 2.0 * pi * freq_center

    xp = array_namespace(angular_freq)
    sinc_arg_lower = pulse_duration_s * (angular_freq - angular_freq_center) / 2.0 / pi
    sinc_arg_upper = pulse_duration_s * (angular_freq + angular_freq_center) / 2.0 / pi
    # array-api-extra does not have type interoperability
    return 1j * (xpx.sinc(sinc_arg_lower, xp=xp) - xpx.sinc(sinc_arg_upper, xp=xp))  # ty: ignore[invalid-argument-type, invalid-return-type]


@jaxtyped(typechecker=typechecker)
def probe_spectrum(
    angular_freq: Float[Array, " n_freqs"],
    freq_center: float | int,
    bandwidth: float | int = 0.75,
) -> Float[Array, " n_freqs"]:
    """Compute the probe frequency response.

    Computes the one-way probe frequency response using a generalized normal
    window. The bandwidth parameter defines the pulse-echo 6 dB fractional
    bandwidth.

    The returned spectrum is the square root of the pulse-echo response,
    appropriate for one-way (transmit-only or receive-only) use.

    Args:
        angular_freq: Angular frequency in rad/s. Shape (n_freqs,).
        freq_center: Center frequency in Hz. Must be positive.
        bandwidth: Fractional bandwidth (0.75 = 75%). Must be in (0, 2.0).

    Returns:
        Real-valued probe response (one-way) at the given angular frequencies.
        Shape (n_freqs,).

    References:
        Generalized normal window:
        https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window

        Reference implementation (log(126) constant verified against):
        https://github.com/creatis-ULTIM/PyMUST/blob/df02b42/src/pymust/utils.py#L141
    """
    # Validate bandwidth
    if not (0.0 < bandwidth < 2.0):
        msg = f"bandwidth must be in (0, 2.0), got {bandwidth!r}"
        raise ValueError(msg)

    angular_freq_center = 2.0 * pi * freq_center
    # Convert fractional bandwidth to angular bandwidth
    angular_bandwidth = bandwidth * angular_freq_center
    # Shape parameter for the generalized normal window
    # The constant 126 comes from the two-way 6 dB bandwidth criterion:
    # For pulse-echo, the total response is the product of TX and RX responses,
    # so the one-way response at -3 dB corresponds to -6 dB two-way.
    # In linear scale: 10^(6/10) ≈ 3.98, but the generalized normal window
    # parameterization uses 126 = 2 * (2^6) to define the bandwidth edges
    # where the two-way response falls to -6 dB.
    shape_param = log(126) / log(2.0 * angular_freq_center / angular_bandwidth)
    # Denominator of the exponent
    sigma = angular_bandwidth / 2.0 / (log(2) ** (1.0 / shape_param))

    xp = array_namespace(angular_freq)
    # Pulse-echo (squared) response
    spectrum_squared = xp.exp(-((xp.abs(angular_freq - angular_freq_center) / sigma) ** shape_param))
    # One-way response (square root)
    return xp.sqrt(spectrum_squared)
