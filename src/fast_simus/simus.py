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

from math import ceil, prod
from typing import NamedTuple, overload

from jaxtyping import Complex, Float

from fast_simus._blocking import legacy_point_count
from fast_simus._compat import _clean_transmit_inputs, _validate_apodization
from fast_simus._echo import echo_spectrum
from fast_simus._frequency import _two_way_pulse_duration
from fast_simus._pfield_math import _select_frequencies
from fast_simus._simus_dispatch import _SimusSpectrumRequest, compute_simus_spectrum, require_portable_backend
from fast_simus._spectral_output import _irfft_and_threshold
from fast_simus.backends._selection import BackendKind
from fast_simus.execution import ExecutionOptions
from fast_simus.medium_params import MediumParams
from fast_simus.plans import EchoPlan, FieldPlan, prepare_echo
from fast_simus.transducer import Transducer
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import (
    Array,
    array_namespace,
)
from fast_simus.utils.geometry import element_positions

_DEFAULT_MEDIUM = MediumParams()


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


@overload
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
    element_splitting: int | tuple[int, int] | None = None,
    frequency_step: float | int = 1.0,
    execution: ExecutionOptions | None = None,
) -> SimusPlan: ...


@overload
def simus_precompute(
    scatterers: Float[Array, "*batch 2"],
    rc: Float[Array, " *batch"],
    delays: Float[Array, " n_elements"],
    params: Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    fs: float | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    element_splitting: int | tuple[int, int] | None = None,
    frequency_step: float | int = 1.0,
    execution: ExecutionOptions | None = None,
) -> EchoPlan: ...


def simus_precompute(
    scatterers: Float[Array, "*batch 2"],
    rc: Float[Array, " *batch"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams | Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    fs: float | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    element_splitting: int | tuple[int, int] | None = None,
    frequency_step: float | int = 1.0,
    execution: ExecutionOptions | None = None,
) -> SimusPlan | EchoPlan:
    """Precompute static quantities for simus computation.

    Args:
        execution: Optional bound on live numerical workspace; retained by new plans.
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
    if isinstance(params, Transducer):
        return prepare_echo(
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
            execution=execution,
        )
    if isinstance(element_splitting, tuple):
        raise ValueError("2D element_splitting must be an integer")
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

    flat = xp.reshape(scatterers, (-1, 2))
    size = flat.shape[0] if execution is None else legacy_point_count(execution, params.n_elements, n_sub)
    max_d = 0.0
    for start in range(0, flat.shape[0], size):
        delta = flat[start : start + size, None, :] - element_pos
        max_d = max(max_d, float(xp.max(xp.sqrt(xp.sum(delta * delta, axis=-1)))))

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


def simus_compute(
    scatterers: Float[Array, "*batch 2"],
    rc: Float[Array, " *batch"],
    delays: Float[Array, " n_elements"],
    plan: SimusPlan | EchoPlan,
    params: TransducerParams | Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    full_frequency_directivity: bool = False,
    backend: BackendKind | str | None = None,
    execution: ExecutionOptions | None = None,
) -> SimusResult:
    """Compute RF signals given a precomputed plan.

    Args:
        execution: Optional bound on live numerical workspace; retained by new plans.
        scatterers: Scatterer positions in meters. Shape ``(*batch, 2)``.
        rc: Reflection coefficients. Shape ``(*batch,)``.
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        plan: Precomputed plan from ``simus_precompute``.
        params: Transducer parameters.
        medium: Medium parameters.
        tx_apodization: Transmit apodization weights. Shape ``(n_elements,)``.
        full_frequency_directivity: If True, compute element directivity at
            every frequency.
        backend: Optional backend name. Auto permits portable fallback; explicit
            native kernels require a supported physical model and execution mode.

    Returns:
        SimusResult with RF signals and complex spectrum.
    """
    if isinstance(params, Transducer):
        if not isinstance(plan, EchoPlan):
            raise ValueError("3D RF requires an EchoPlan")
        if execution is not None and execution != plan.execution:
            raise ValueError("execution differs from plan")
        require_portable_backend(array_namespace(scatterers), backend, "finite 3D apertures")
        spect = echo_spectrum(
            scatterers, rc, delays, plan, params, medium, tx_apodization, full_frequency_directivity, None
        )
        rf, full = _irfft_and_threshold(spect, plan, params.n_elements, array_namespace(scatterers))
        return SimusResult(rf, full)
    if isinstance(plan, FieldPlan):
        raise ValueError("2D RF requires a legacy plan")
    xp = array_namespace(
        scatterers, rc, delays, tx_apodization, plan.selected_freqs, plan.pulse_spectrum, plan.probe_spectrum
    )

    delays_clean, tx_apodization = _clean_transmit_inputs(delays, tx_apodization, params.n_elements, xp)

    # Flatten scatterers for the frequency sweep
    n_scat = prod(scatterers.shape[:-1])
    scatterers_flat = xp.reshape(scatterers, (n_scat, 2))
    rc_flat = xp.reshape(rc, (n_scat,))

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
            execution=execution,
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
    params: TransducerParams | Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    fs: float | None = None,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    full_frequency_directivity: bool = False,
    element_splitting: int | tuple[int, int] | None = None,
    frequency_step: float | int = 1.0,
    backend: BackendKind | str | None = None,
    execution: ExecutionOptions | None = None,
) -> SimusResult:
    """Simulate ultrasound RF signals for a linear or convex array.

    Computes RF radio-frequency signals generated by an ultrasound uniform
    linear or convex array insonifying a medium of scatterers. Uses the SIMUS
    algorithm: TX forward propagation, scattering, RX back-propagation
    (acoustic reciprocity), and IFFT to time domain.

    Args:
        execution: Optional bound on live numerical workspace; retained by new plans.
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
        backend: Optional backend name. Auto permits portable fallback; explicit
            native kernels require a supported physical model and execution mode.

    Returns:
        SimusResult with:
        - rf: Time-domain RF signals, shape (n_samples, n_elements)
        - spectrum: Complex RF spectrum, shape (n_freq_full, n_elements)
    """
    if isinstance(params, Transducer):
        _validate_apodization(tx_apodization, delays)
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
        execution=execution,
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
        execution=execution,
    )
