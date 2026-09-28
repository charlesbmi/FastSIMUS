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
from math import ceil, prod
from types import ModuleType
from typing import NamedTuple, cast, overload

from array_api_compat import is_jax_namespace
from jaxtyping import Complex, Float

from fast_simus._blocking import legacy_point_count
from fast_simus._capabilities import _require_strategy, _unsupported
from fast_simus._compat import _clean_transmit_inputs, _transfer_plan, _validate_apodization
from fast_simus._echo import echo_spectrum
from fast_simus._frequency import _two_way_pulse_duration
from fast_simus._pfield_math import _select_frequencies
from fast_simus._spectral_output import _irfft_and_threshold
from fast_simus._transfer import _prepare_strip_transfer
from fast_simus.execution import ExecutionOptions
from fast_simus.medium_params import MediumParams
from fast_simus.plans import EchoPlan, FieldPlan, prepare_echo
from fast_simus.transducer import Transducer
from fast_simus.transducer_params import BaffleType, TransducerParams
from fast_simus.utils._array_api import (
    Array,
    _ArrayNamespace,
    array_namespace,
)
from fast_simus.utils.geometry import element_positions

_DEFAULT_MEDIUM = MediumParams()


class SimusStrategy(StrEnum):
    """Backend strategy for the simus frequency sweep.

    Attributes:
        PYTHON: Python for-loop (NumPy/CuPy, constant memory).
        SCAN: JAX lax.scan for O(1) compilation cost.
        METAL: Custom Metal kernel on Apple Silicon (MLX).
        CUDA: Custom CUDA kernel on NVIDIA GPUs (CuPy + NVRTC).
    """

    PYTHON = "python"
    SCAN = "scan"
    METAL = "metal"
    CUDA = "cuda"


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
    transfer = _prepare_strip_transfer(
        scatterers,
        delays_clean,
        tx_apodization,
        _transfer_plan(plan, params.freq_center),
        params,
        medium,
        full_frequency_directivity=full_frequency_directivity,
        xp=xp,
    )
    return {
        "phase_init": transfer.phase,
        "phase_step": transfer.phase_step,
        "delay_apod_init": transfer.delay_apod,
        "delay_apod_step": transfer.delay_apod_step,
        "is_out": transfer.is_out,
        "wavenumbers": transfer.wavenumbers,
        "pulse_spect": plan.pulse_spectrum,
        "probe_spect": plan.probe_spectrum,
        "seg_length": plan.seg_length,
        "sin_theta": transfer.sin_theta,
        "full_frequency_directivity": full_frequency_directivity,
    }


def _select_simus_strategy(
    xp: _ArrayNamespace,
    strategy: SimusStrategy | None,
    baffle: BaffleType | float = BaffleType.SOFT,
    full_frequency_directivity: bool = False,
) -> SimusStrategy:
    """Select an execution path that supports the requested physics."""
    if strategy is not None:
        _require_strategy(strategy, xp, baffle, full_frequency_directivity)
        return strategy
    if is_jax_namespace(cast(ModuleType, xp)):
        return SimusStrategy.SCAN
    for native in (SimusStrategy.METAL, SimusStrategy.CUDA):
        if not _unsupported(native, xp, baffle, full_frequency_directivity):
            return native
    return SimusStrategy.PYTHON


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
    strategy: SimusStrategy | None = None,
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
        strategy: Backend strategy for the frequency sweep. If None,
            auto-selects based on the detected array backend.

    Returns:
        SimusResult with RF signals and complex spectrum.
    """
    if isinstance(params, Transducer):
        if not isinstance(plan, EchoPlan):
            raise ValueError("3D RF requires an EchoPlan")
        if execution is not None and execution != plan.execution:
            raise ValueError("execution differs from plan")
        spect = echo_spectrum(
            scatterers, rc, delays, plan, params, medium, tx_apodization, full_frequency_directivity, strategy
        )
        rf, full = _irfft_and_threshold(spect, plan, params.n_elements, array_namespace(scatterers))
        return SimusResult(rf, full)
    if isinstance(plan, FieldPlan):
        raise ValueError("2D RF requires a legacy plan")
    xp = array_namespace(scatterers, rc, delays)

    delays_clean, tx_apodization = _clean_transmit_inputs(delays, tx_apodization, params.n_elements, xp)

    # Flatten scatterers for the frequency sweep
    n_scat = prod(scatterers.shape[:-1])
    scatterers_flat = xp.reshape(scatterers, (n_scat, 2))
    rc_flat = xp.reshape(rc, (n_scat,))

    if execution is not None and strategy in (SimusStrategy.METAL, SimusStrategy.CUDA):
        raise NotImplementedError("Native strategy does not support an execution budget")
    selected = _select_simus_strategy(
        xp, SimusStrategy.PYTHON if execution is not None else strategy, params.baffle, full_frequency_directivity
    )

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
    elif selected == SimusStrategy.CUDA:
        from fast_simus.kernels.cuda_simus import simus_cuda

        spect_selected = cast(
            Array,
            simus_cuda(
                scatterers=scatterers_flat,
                rc=rc_flat,
                params=params,
                plan=plan,
                medium=medium,
                delays_clean=delays_clean,
                tx_apodization=tx_apodization,
            ),
        )
    else:
        from fast_simus._simus_strategies import _simus_freq_outer_python, _simus_freq_outer_scan

        driver = _simus_freq_outer_scan if selected == SimusStrategy.SCAN else _simus_freq_outer_python
        size = n_scat if execution is None else legacy_point_count(execution, params.n_elements, plan.n_sub)
        spect_selected = xp.zeros((plan.selected_freqs.shape[0], params.n_elements), dtype=plan.pulse_spectrum.dtype)
        for start in range(0, n_scat, size):
            sweep = _prepare_simus_sweep(
                scatterers_flat[start : start + size, :],
                delays_clean,
                tx_apodization,
                plan,
                params,
                medium,
                full_frequency_directivity=full_frequency_directivity,
                xp=xp,
            )
            block = driver(rc=rc_flat[start : start + size], xp=xp, **sweep)
            spect_selected = spect_selected + block

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
    strategy: SimusStrategy | None = None,
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
        strategy: Backend strategy for the frequency sweep. If None,
            auto-selects based on the detected array backend.

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
        strategy=strategy,
        execution=execution,
    )
