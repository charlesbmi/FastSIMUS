"""Ultrasound RF signal simulation for linear and convex arrays.

Implements the SIMUS algorithm: for each frequency in the transmit bandwidth,
compute forward TX pressure at scatterers, scatter by reflection coefficients,
back-propagate to receive elements (acoustic reciprocity), accumulate complex
RF spectrum, then IFFT to time-domain RF signals.

All functions are Array API compliant and work with NumPy, JAX, CuPy backends.

References:
    Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
    Part I: theory & examples. CMPB, 2022;218:106726.
"""

from __future__ import annotations

from math import ceil, log2, pi
from typing import NamedTuple, cast

import array_api_extra as xpx
from jaxtyping import Complex, Float

from fast_simus._pfield_math import _select_frequencies
from fast_simus._simus_dispatch import _SimusSpectrumRequest, compute_simus_spectrum
from fast_simus.backends._selection import BackendKind
from fast_simus.medium_params import MediumParams
from fast_simus.spectrum import probe_spectrum as _probe_spectrum_fn
from fast_simus.spectrum import pulse_spectrum as _pulse_spectrum_fn
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import (
    Array,
    _ArrayNamespace,
    _ArrayNamespaceWithFFT,
    array_namespace,
)
from fast_simus.utils.geometry import element_positions

_DEFAULT_MEDIUM = MediumParams()


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
    ps = _pulse_spectrum_fn(omega, freq_center, tx_n_wavelengths)
    pr = _probe_spectrum_fn(omega, freq_center, bandwidth)
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


class SimusResult(NamedTuple):
    """Result of simus RF signal simulation.

    Attributes:
        rf: Time-domain RF signals, shape (n_samples, n_elements).
        spectrum: Complex RF spectrum, shape (n_freq_full, n_elements).
    """

    rf: Float[Array, "n_samples n_elements"]
    spectrum: Complex[Array, "n_freq_full n_elements"]


class SimusPlan(NamedTuple):
    """Precomputed plan for simus computation.

    Contains all data-dependent quantities so that ``simus_compute`` has
    static array shapes. The selected band is contiguous on the regular grid
    from zero to twice the original probe's center frequency. Phase frequencies
    follow the integer grid metadata, independent of the stored samples' dtype.

    Attributes:
        selected_freqs: Significant frequency samples in Hz.
        pulse_spectrum: Pulse spectrum at selected frequencies.
        probe_spectrum: Probe response at selected frequencies.
        n_sub: Number of sub-elements per transducer element.
        seg_length: Sub-element length in meters.
        correction_factor: Scaling factor for the integration
            (df * element_width, or element_width when tx_n_wavelengths=inf).
        n_freq_full: Total number of frequency bins (0 to 2*fc).
        freq_idx_start: Index of first selected frequency in full spectrum.
        n_fft: Number of points for the IFFT (from fs, fc, Nf).
    """

    selected_freqs: Float[Array, " n_frequencies"]
    pulse_spectrum: Complex[Array, " n_frequencies"]
    probe_spectrum: Float[Array, " n_frequencies"]
    n_sub: int
    seg_length: float
    correction_factor: float
    n_freq_full: int
    freq_idx_start: int
    n_fft: int


def simus_precompute(
    scatterers: Float[Array, "*batch 2"],
    rc: Float[Array, " *batch"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    fs: float | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    element_splitting: int | None = None,
    frequency_step: float | int = 1.0,
) -> SimusPlan:
    """Precompute static quantities for simus computation.

    Args:
        scatterers: Scatterer positions in meters. Shape ``(*batch, 2)``.
        rc: Reflection coefficients. Shape ``(*batch,)``.
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        params: Transducer parameters.
        medium: Medium parameters.
        fs: Sampling frequency in Hz. Defaults to 4 * fc.
        tx_n_wavelengths: Number of wavelengths in the TX pulse.
        db_thresh: Threshold in dB for frequency component selection.
        element_splitting: Number of sub-elements per element (None = auto).
        frequency_step: Scaling factor for the frequency step.

    Returns:
        SimusPlan with static-shaped arrays and precomputed scalars.
    """
    xp = array_namespace(scatterers, rc, delays)
    speed_of_sound = medium.speed_of_sound
    fc = params.freq_center

    if fs is None:
        fs = 4.0 * fc

    # NaN-clean delays
    delays_clean = xp.where(xp.isnan(delays), xp.asarray(0.0), delays)

    # Element splitting
    if element_splitting is not None:
        n_sub = element_splitting
    else:
        lambda_min = speed_of_sound / (fc * (1.0 + params.bandwidth / 2.0))
        n_sub = ceil(params.element_width / lambda_min)

    seg_length = params.element_width / n_sub

    # Max distance for frequency step (use element centers, matching PyMUST simus)
    element_pos, theta_elements, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(params.n_elements)

    x = scatterers[..., 0]
    z = scatterers[..., 1]
    d2 = (xp.reshape(x, (-1, 1)) - element_pos[:, 0]) ** 2 + (xp.reshape(z, (-1, 1)) - element_pos[:, 1]) ** 2
    max_d = float(xp.max(xp.sqrt(d2)))

    # Two-way pulse length correction (matches MATLAB: getpulse(param,2))
    if tx_n_wavelengths != float("inf"):
        tp = _two_way_pulse_duration(fc, params.bandwidth, tx_n_wavelengths, xp)
        max_d = max_d + tp * speed_of_sound

    # Round-trip frequency step (matches PyMUST simus df formula)
    df = 1.0 / 2.0 / (2.0 * max_d / speed_of_sound + float(xp.max(delays_clean)))
    df = float(frequency_step) * df

    # Full frequency grid
    n_freq_full = int(2 * ceil(fc / df) + 1)

    # Frequency selection using shared helper
    freq_plan = _select_frequencies(fc, params.bandwidth, tx_n_wavelengths, db_thresh, df, xp)
    df_actual = freq_plan.freq_step

    # Find start index of selected frequencies in full spectrum
    freq_idx_start = round(float(freq_plan.selected_freqs[0]) / df_actual) if df_actual > 0 else 0

    # Correction factor
    correction_factor = 1.0 if tx_n_wavelengths == float("inf") else df_actual
    correction_factor = correction_factor * params.element_width

    # IFFT length
    n_fft = ceil(fs / 2.0 / fc * (n_freq_full - 1))

    return SimusPlan(
        selected_freqs=freq_plan.selected_freqs,
        pulse_spectrum=freq_plan.pulse_spectrum,
        probe_spectrum=freq_plan.probe_spectrum,
        n_sub=n_sub,
        seg_length=seg_length,
        correction_factor=correction_factor,
        n_freq_full=n_freq_full,
        freq_idx_start=freq_idx_start,
        n_fft=n_fft,
    )


def _irfft_and_threshold(
    spect_selected: Complex[Array, "n_freq_sel n_elem"],
    plan: SimusPlan,
    n_elements: int,
    xp: _ArrayNamespace,
) -> tuple[Float[Array, "n_samples n_elem"], Complex[Array, "n_freq_full n_elem"]]:
    """Place selected spectrum, IFFT to time domain, apply smooth thresholding."""
    if not hasattr(xp, "fft"):
        msg = "simus requires an array backend with FFT support (e.g. numpy, jax, cupy)"
        raise RuntimeError(msg)
    xp_fft = cast(_ArrayNamespaceWithFFT, xp)

    n_freq_sel = spect_selected.shape[0]
    full_spectrum = xp.zeros((plan.n_freq_full, n_elements), dtype=spect_selected.dtype)
    full_spectrum = xpx.at(full_spectrum)[plan.freq_idx_start : plan.freq_idx_start + n_freq_sel, :].set(  # type: ignore[attr-defined]
        spect_selected
    )

    rf = xp_fft.fft.irfft(xp.conj(full_spectrum), n=plan.n_fft, axis=0)

    n_keep = (plan.n_fft + 1) // 2
    rf = rf[:n_keep, ...]

    # Smooth thresholding of small values (-100 dB)
    rel_thresh = 1e-5
    rf_peak = xp.max(xp.abs(rf))
    rel_rf = xp.abs(rf) / (rf_peak + xp.asarray(1e-30))
    smooth_gate = 0.5 * (1.0 + xp.tanh((rel_rf - rel_thresh) / (rel_thresh / 10.0)))  # type: ignore[attr-defined]

    rf = rf * smooth_gate

    return rf, full_spectrum


def simus_compute(
    scatterers: Float[Array, "*batch 2"],
    rc: Float[Array, " *batch"],
    delays: Float[Array, " n_elements"],
    plan: SimusPlan,
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    full_frequency_directivity: bool = False,
    backend: BackendKind | str | None = None,
) -> SimusResult:
    """Compute RF signals given a precomputed plan.

    Args:
        scatterers: Scatterer positions in meters. Shape ``(*batch, 2)``.
        rc: Reflection coefficients. Shape ``(*batch,)``.
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        plan: Precomputed plan from ``simus_precompute``.
        params: Transducer parameters.
        medium: Medium parameters.
        tx_apodization: Transmit apodization weights. Shape ``(n_elements,)``.
        full_frequency_directivity: If True, compute element directivity at
            every frequency.
        backend: Optional execution request. Omit or pass ``"auto"`` to
            infer from the input arrays; pass a backend name to require a
            custom kernel or force a portable namespace implementation.

    Returns:
        SimusResult with RF signals and complex spectrum.
    """
    namespace_arrays = (
        scatterers,
        rc,
        delays,
        plan.selected_freqs,
        plan.pulse_spectrum,
        plan.probe_spectrum,
    )
    if tx_apodization is not None:
        namespace_arrays += (tx_apodization,)
    xp = array_namespace(*namespace_arrays)

    if tx_apodization is None:
        tx_apodization = xp.ones(params.n_elements)

    nan_mask = xp.isnan(delays)
    tx_apodization = xp.where(nan_mask, xp.asarray(0.0), tx_apodization)
    delays_clean = xp.where(nan_mask, xp.asarray(0.0), delays)

    # Flatten scatterers for the frequency sweep
    n_scat = scatterers.shape[0] if scatterers.ndim >= 2 else 1
    scatterers_flat = xp.reshape(scatterers, (n_scat, 2)) if scatterers.ndim > 2 else scatterers
    rc_flat = xp.reshape(rc, (n_scat,)) if rc.ndim > 1 else rc

    spect_selected = compute_simus_spectrum(
        _SimusSpectrumRequest(
            scatterers=scatterers_flat,
            rc=rc_flat,
            delays_clean=delays_clean,
            tx_apodization=tx_apodization,
            plan=plan,
            params=params,
            medium=medium,
            full_frequency_directivity=full_frequency_directivity,
            xp=xp,
            backend=backend,
        )
    )

    # Apply correction factor
    spect_selected = spect_selected * xp.asarray(plan.correction_factor)

    rf, full_spectrum = _irfft_and_threshold(spect_selected, plan, params.n_elements, xp)

    return SimusResult(rf=rf, spectrum=full_spectrum)


def simus(
    scatterers: Float[Array, "*batch 2"],
    rc: Float[Array, " *batch"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    fs: float | None = None,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    full_frequency_directivity: bool = False,
    element_splitting: int | None = None,
    frequency_step: float | int = 1.0,
    backend: BackendKind | str | None = None,
) -> SimusResult:
    """Simulate ultrasound RF signals for a linear or convex array.

    Computes RF radio-frequency signals generated by an ultrasound uniform
    linear or convex array insonifying a medium of scatterers. Uses the SIMUS
    algorithm: TX forward propagation, scattering, RX back-propagation
    (acoustic reciprocity), and IFFT to time domain.

    Args:
        scatterers: Scatterer positions in meters. Shape ``(*batch, 2)`` where
            ``[..., 0]`` is lateral (x) and ``[..., 1]`` is axial (z).
        rc: Reflection coefficients. Shape ``(*batch,)``. Same size as scatterers
            (excluding last dimension).
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        params: Transducer parameters (geometry, frequency, bandwidth).
        medium: Medium parameters (speed of sound, attenuation).
        fs: Sampling frequency in Hz. Defaults to ``4 * params.freq_center``.
        tx_apodization: Transmit apodization weights. Shape ``(n_elements,)``.
        tx_n_wavelengths: Number of wavelengths in the TX pulse.
        db_thresh: Threshold in dB for frequency component selection.
        full_frequency_directivity: If True, compute element directivity at
            every frequency. If False, use center-frequency-only directivity.
        element_splitting: Number of sub-elements per element (None = auto).
        frequency_step: Scaling factor for the frequency step.
        backend: Optional execution request. Omit or pass ``"auto"`` to
            infer from the input arrays; pass a backend name to require a
            custom kernel or force a portable namespace implementation.

    Returns:
        SimusResult with:
        - rf: Time-domain RF signals, shape (n_samples, n_elements)
        - spectrum: Complex RF spectrum, shape (n_freq_full, n_elements)
    """
    plan = simus_precompute(
        scatterers,
        rc,
        delays,
        params,
        medium,
        fs=fs,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
    )
    return simus_compute(
        scatterers,
        rc,
        delays,
        plan,
        params,
        medium,
        tx_apodization=tx_apodization,
        full_frequency_directivity=full_frequency_directivity,
        backend=backend,
    )
