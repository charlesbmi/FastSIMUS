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

from enum import StrEnum
from math import ceil, inf, pi
from types import ModuleType
from typing import NamedTuple, cast

import array_api_extra as xpx
import numpy as np
from array_api_compat import is_jax_namespace
from jaxtyping import Complex, Float

from fast_simus._pfield_math import (
    _distances_and_angles,
    _init_exponentials,
    _obliquity_factor,
    _select_frequencies,
    _subelement_centroids,
)
from fast_simus.medium_params import MediumParams
from fast_simus.spectrum import probe_spectrum as _probe_spectrum_fn
from fast_simus.spectrum import pulse_spectrum as _pulse_spectrum_fn
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace, array_namespace
from fast_simus.utils.geometry import element_positions

_DEFAULT_MEDIUM = MediumParams()


def _two_way_pulse_duration(
    freq_center: float,
    bandwidth: float,
    tx_n_wavelengths: float,
) -> float:
    """Compute the temporal extent of the two-way (pulse-echo) pulse.

    Replicates the pulse duration computation from PyMUST's getpulse(param, 2).
    Uses pulse_spectrum * probe_spectrum^2, IFFTs, and thresholds at 1/1023.

    Args:
        freq_center: Center frequency in Hz.
        bandwidth: Fractional bandwidth (0.75 = 75%).
        tx_n_wavelengths: Number of wavelengths of the TX pulse.

    Returns:
        Pulse duration in seconds.
    """
    dt = 1e-9
    df = freq_center / tx_n_wavelengths / 32
    p = int(np.ceil(np.log2(1.0 / dt / 2.0 / df)))
    n_fft = 2**p
    f = np.linspace(0, 1.0 / dt / 2.0, n_fft)
    omega = 2.0 * pi * f

    # Two-way spectrum: pulse * probe^2
    omega_arr: Array = np.asarray(omega)  # type: ignore[assignment]
    ps = _pulse_spectrum_fn(omega_arr, freq_center, tx_n_wavelengths)
    pr = _probe_spectrum_fn(omega_arr, freq_center, bandwidth)
    two_way = np.asarray(ps) * np.asarray(pr) ** 2

    pulse = np.fft.fftshift(np.fft.irfft(two_way))
    pulse = pulse / np.max(np.abs(pulse))
    idx = np.where(pulse > (1.0 / 1023))[0]
    if len(idx) == 0:
        return tx_n_wavelengths / freq_center
    idx1 = idx[0]
    idx2 = idx[-1]
    trim_idx = min(idx1 + 1, 2 * n_fft - 1 - idx2 - 1)
    pulse_trimmed = pulse[-trim_idx : trim_idx - 2 : -1]
    return float(len(pulse_trimmed) * dt)


class SimusStrategy(StrEnum):
    """Backend strategy for the simus frequency sweep.

    Attributes:
        PYTHON: Python for-loop (NumPy/CuPy, constant memory).
        SCAN: JAX lax.scan for O(1) compilation cost.
        METAL: Custom Metal kernel on Apple Silicon (MLX).
    """

    PYTHON = "python"
    SCAN = "scan"
    METAL = "metal"


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
    static array shapes.

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
    xp = array_namespace(scatterers, delays)
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
        tp = _two_way_pulse_duration(fc, params.bandwidth, tx_n_wavelengths)
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


def _prepare_simus_sweep(
    scatterers: Float[Array, "*batch 2"],
    delays_clean: Float[Array, " n_elements"],
    tx_apodization: Float[Array, " n_elements"],
    plan: SimusPlan,
    params: TransducerParams,
    medium: MediumParams,
    *,
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> dict:
    """Compute geometry and phase arrays for simus frequency sweep.

    Unlike pfield's _prepare_frequency_sweep, this keeps per-element structure
    (n_scat, n_elem, n_sub) instead of flattening to (n_scat, n_sources).
    Delay+apodization are NOT absorbed into the geometric progression --
    they are kept separate for the TX/RX chain.
    """
    element_pos, theta_elements, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(params.n_elements)

    speed_of_sound = medium.speed_of_sound
    attenuation = medium.attenuation

    subelement_offsets = _subelement_centroids(params.element_width, plan.n_sub, theta_elements, xp)

    x = scatterers[..., 0]
    z = scatterers[..., 1]
    is_out = z < 0
    if params.radius != inf:
        is_out = is_out | ((x**2 + (z + apex_offset) ** 2) <= params.radius**2)

    distances, sin_theta, theta_arr = _distances_and_angles(
        scatterers, subelement_offsets, element_pos, theta_elements, speed_of_sound, params.freq_center, xp
    )

    obliquity_factor = _obliquity_factor(theta_arr, params.baffle, xp)

    freq_start = plan.selected_freqs[0]
    n_freqs = plan.selected_freqs.shape[0]
    freq_step = (plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freqs > 1 else xp.asarray(0.0)

    phase_init, phase_step = _init_exponentials(
        freq_start, speed_of_sound, attenuation, distances, obliquity_factor, freq_step, xp
    )

    if not full_frequency_directivity:
        center_wavenumber = 2.0 * pi * params.freq_center / speed_of_sound
        sinc_arg = xp.asarray(center_wavenumber * plan.seg_length / 2.0) * sin_theta / pi
        phase_init = phase_init * xpx.sinc(sinc_arg, xp=xp)

    # Delay+apodization as separate geometric progressions (not absorbed)
    delay_apod_init = xp.exp(xp.asarray(1j * 2.0 * pi) * freq_start * delays_clean) * tx_apodization
    delay_apod_step = xp.exp(xp.asarray(1j * 2.0 * pi) * freq_step * delays_clean)

    wavenumbers = xp.asarray(2.0 * pi) * plan.selected_freqs / speed_of_sound

    return {
        "phase_init": phase_init,
        "phase_step": phase_step,
        "delay_apod_init": delay_apod_init,
        "delay_apod_step": delay_apod_step,
        "is_out": is_out,
        "wavenumbers": wavenumbers,
        "pulse_spect": plan.pulse_spectrum,
        "probe_spect": plan.probe_spectrum,
        "seg_length": plan.seg_length,
        "sin_theta": sin_theta,
        "full_frequency_directivity": full_frequency_directivity,
    }


def _irfft_and_threshold(
    spect_selected: Complex[Array, "n_freq_sel n_elem"],
    plan: SimusPlan,
    n_elements: int,
    xp: _ArrayNamespace,
) -> tuple[Float[Array, "n_samples n_elem"], Complex[Array, "n_freq_full n_elem"]]:
    """Place selected spectrum, IFFT to time domain, apply smooth thresholding.

    Backend-aware: uses jax.numpy.fft when xp is JAX, else numpy.fft.
    """
    n_freq_sel = spect_selected.shape[0]
    full_spectrum = xp.zeros((plan.n_freq_full, n_elements), dtype=spect_selected.dtype)
    full_spectrum = xpx.at(full_spectrum)[plan.freq_idx_start : plan.freq_idx_start + n_freq_sel, :].set(  # type: ignore[attr-defined]
        spect_selected
    )

    if is_jax_namespace(cast(ModuleType, xp)):
        import jax.numpy as jnp

        rf = jnp.fft.irfft(jnp.conj(full_spectrum), plan.n_fft, axis=0)
    else:
        full_np = np.asarray(full_spectrum)
        rf_np = np.fft.irfft(np.conj(full_np), plan.n_fft, axis=0)
        rf = xp.asarray(rf_np)

    n_keep = (plan.n_fft + 1) // 2
    rf = rf[:n_keep, ...]

    # Smooth thresholding of small values (-100 dB)
    rel_thresh = 1e-5
    rf_peak = xp.max(xp.abs(rf))
    rel_rf = xp.abs(rf) / (rf_peak + xp.asarray(1e-30))
    smooth_gate = 0.5 * (1.0 + xp.tanh((rel_rf - rel_thresh) / (rel_thresh / 10.0)))  # type: ignore[attr-defined]

    rf = rf * smooth_gate

    return rf, full_spectrum


def _select_simus_strategy(xp: _ArrayNamespace, strategy: SimusStrategy | None) -> SimusStrategy:
    """Auto-select simus strategy based on array backend."""
    if strategy is not None:
        return strategy

    if is_jax_namespace(cast(ModuleType, xp)):
        return SimusStrategy.SCAN

    try:
        import mlx.core

        if xp is mlx.core:
            return SimusStrategy.METAL
    except ImportError:
        pass

    return SimusStrategy.PYTHON


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
    strategy: SimusStrategy | None = None,
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
        strategy: Backend strategy for the frequency sweep. If None,
            auto-selects based on the detected array backend.

    Returns:
        SimusResult with RF signals and complex spectrum.
    """
    xp = array_namespace(scatterers, rc, delays)

    if tx_apodization is None:
        tx_apodization = xp.ones(params.n_elements)

    nan_mask = xp.isnan(delays)
    tx_apodization = xp.where(nan_mask, xp.asarray(0.0), tx_apodization)
    delays_clean = xp.where(nan_mask, xp.asarray(0.0), delays)

    # Flatten scatterers for the frequency sweep
    n_scat = scatterers.shape[0] if scatterers.ndim >= 2 else 1
    scatterers_flat = xp.reshape(scatterers, (n_scat, 2)) if scatterers.ndim > 2 else scatterers
    rc_flat = xp.reshape(rc, (n_scat,)) if rc.ndim > 1 else rc

    sweep = _prepare_simus_sweep(
        scatterers_flat,
        delays_clean,
        tx_apodization,
        plan,
        params,
        medium,
        full_frequency_directivity=full_frequency_directivity,
        xp=xp,
    )

    selected = _select_simus_strategy(xp, strategy)

    if selected == SimusStrategy.METAL:
        import mlx.core as mx

        from fast_simus.kernels.metal_simus import simus_metal

        spect_selected = cast(
            Array,
            simus_metal(
                scatterers=cast(mx.array, scatterers_flat),
                rc=cast(mx.array, rc_flat),
                params=params,
                plan=plan,
                medium=medium,
                delays_clean=cast(mx.array, delays_clean),
                tx_apodization=cast(mx.array, tx_apodization),
            ),
        )
    elif selected == SimusStrategy.SCAN:
        from fast_simus._simus_strategies import _simus_freq_outer_scan

        spect_selected = _simus_freq_outer_scan(rc=rc_flat, xp=xp, **sweep)
    else:
        from fast_simus._simus_strategies import _simus_freq_outer_python

        spect_selected = _simus_freq_outer_python(rc=rc_flat, xp=xp, **sweep)

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
    strategy: SimusStrategy | None = None,
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
        strategy: Backend strategy for the frequency sweep. If None,
            auto-selects based on the detected array backend.

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
        strategy=strategy,
    )
