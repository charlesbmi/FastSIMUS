"""Pressure field computation for ultrasound transducer arrays.

Implements PFIELD algorithm for simulating ultrasound beam patterns from
phased/linear/convex arrays using Fraunhofer (far-field) approximation in
the azimuthal plane and Fresnel (paraxial) approximation in elevation.

All functions are Array API compliant and work with NumPy, JAX, CuPy backends.

References:
    Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
    Part I: theory & examples. CMPB, 2022;218:106726.
"""

from __future__ import annotations

from enum import StrEnum
from math import ceil, prod
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple, cast, overload

from array_api_compat import is_jax_namespace
from beartype import beartype as typechecker
from jaxtyping import Bool, Complex, Float, jaxtyped

from fast_simus._blocking import legacy_point_count
from fast_simus._capabilities import _require_strategy, _unsupported
from fast_simus._compat import _clean_transmit_inputs, _transfer_plan, _validate_apodization
from fast_simus._field import field_spectrum
from fast_simus._pfield_math import (
    _distances_and_angles,
    _select_frequencies,
    _subelement_centroids,
)
from fast_simus._pfield_math import _init_exponentials as _init_exponentials
from fast_simus._pfield_math import _obliquity_factor as _obliquity_factor
from fast_simus._transfer import _prepare_strip_transfer
from fast_simus.execution import ExecutionOptions
from fast_simus.medium_params import MediumParams
from fast_simus.plans import FieldPlan, FieldSpectrumInfo, prepare_field
from fast_simus.transducer import Transducer
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace, array_namespace
from fast_simus.utils.geometry import element_positions

_DEFAULT_MEDIUM = MediumParams()


class PfieldStrategy(StrEnum):
    """Backend strategy for the pfield frequency sweep.

    The three-layer pfield architecture separates:
    - Layer 1 (setup): geometry, phase init -- pure Array API, shared by all
    - Layer 2 (step body): per-frequency math -- pure Array API function
    - Layer 3 (loop driver): iteration mechanism -- backend-specific

    This enum selects the Layer 3 loop driver. When None is passed to
    pfield_compute, the strategy is auto-selected based on the detected backend.
    """

    VECTORIZED = "vectorized"
    SCAN = "scan"
    METAL = "metal"


class PfieldPlan(NamedTuple):
    """Precomputed plan for pfield computation.

    Contains all data-dependent quantities so
    that ``pfield_compute`` has static array shapes and can be JIT-compiled.

    Use ``pfield_precompute`` to construct this; do not build manually.

    Attributes:
        selected_freqs: Significant frequency samples in Hz (uniformly spaced).
        pulse_spectrum: Pulse spectrum at selected frequencies (complex).
        probe_spectrum: Probe response at selected frequencies (real).
        n_sub: Number of sub-elements per transducer element.
        seg_length: Sub-element length in meters (element_width / n_sub).
        correction_factor: Scaling factor for the RMS integration
            (df * element_width, or element_width when tx_n_wavelengths=inf).
        freq_step: Spacing of the full frequency grid in Hz.
        n_freq_full: Number of samples in the full ``[0, 2 * fc]`` grid.
        freq_idx_start: Index of ``selected_freqs[0]`` in the full grid.
    """

    selected_freqs: Float[Array, " n_frequencies"]
    pulse_spectrum: Complex[Array, " n_frequencies"]
    probe_spectrum: Float[Array, " n_frequencies"]
    n_sub: int
    seg_length: float
    correction_factor: float
    freq_step: float
    n_freq_full: int
    freq_idx_start: int


class PfieldSpectrumInfo(NamedTuple):
    """Frequency-grid metadata for a pressure spectrum.

    The selected frequencies form a contiguous slice of the uniform full grid
    ``linspace(0, 2 * freq_center, n_freq_full)``.

    Attributes:
        selected_freqs: Evaluated frequencies in Hz. Shape ``(n_freq_selected,)``.
        freq_idx_start: Index of the first selected frequency in the full grid.
        n_freq_full: Length of the uniform ``[0, 2 * freq_center]`` grid.
        freq_step: Spacing of the full grid in Hz.
        correction_factor: Scale for converting summed spectral energy to RMS.
    """

    selected_freqs: Float[Array, " n_freq_selected"]
    freq_idx_start: int
    n_freq_full: int
    freq_step: float
    correction_factor: float


class _SweepInputs(NamedTuple):
    """Precomputed inputs for the Array API frequency-sweep strategies.

    Source points are flattened: n_sources = n_elements * n_sub.
    The 1/n_sub normalization is absorbed into phase_decay_init.
    """

    phase_decay_init: Complex[Array, " *grid n_sources"]
    phase_decay_step: Complex[Array, " *grid n_sources"]
    is_out: Bool[Array, " *grid"]
    wavenumbers: Float[Array, " n_freq"]
    pulse_spect: Complex[Array, " n_freq"]
    probe_spect: Float[Array, " n_freq"]
    seg_length: float
    sin_theta: Float[Array, " *grid n_sources"]
    full_frequency_directivity: bool


def _prepare_frequency_sweep(
    positions: Float[Array, "*grid_shape dim"],
    delays_clean: Float[Array, " n_elements"],
    tx_apodization: Float[Array, " n_elements"],
    plan: PfieldPlan,
    params: TransducerParams,
    medium: MediumParams,
    *,
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> _SweepInputs:
    """Compute geometry, phases, and obliquity for Array API loop drivers.

    Shared setup for VECTORIZED and SCAN strategies. The Metal kernel
    computes geometry on-the-fly and does not use this function.
    """
    transfer = _prepare_strip_transfer(
        positions,
        delays_clean,
        tx_apodization,
        _transfer_plan(plan, params.freq_center),
        params,
        medium,
        full_frequency_directivity=full_frequency_directivity,
        xp=xp,
    )
    phase_decay_init, phase_decay_step = transfer.phase, transfer.phase_step
    sin_theta = transfer.sin_theta

    # Absorb delay+apodization into the geometric progression so loop
    # drivers don't need a per-frequency multiply for delays.
    phase_decay_init = phase_decay_init * transfer.delay_apod[:, None]
    phase_decay_step = phase_decay_step * transfer.delay_apod_step[:, None]

    # Absorb 1/n_sub normalization and flatten (n_elements, n_sub) -> (n_sources,).
    # After this, sub-elements and elements are equivalent source points
    # and all loop drivers use a single sum(axis=-1).
    n_sub = plan.n_sub
    phase_decay_init = phase_decay_init / n_sub

    def _flatten_sources(arr: Array) -> Array:
        return xp.reshape(arr, (*arr.shape[:-2], arr.shape[-2] * arr.shape[-1]))

    phase_decay_init = _flatten_sources(phase_decay_init)
    phase_decay_step = _flatten_sources(phase_decay_step)
    sin_theta = _flatten_sources(sin_theta)

    return _SweepInputs(
        phase_decay_init=phase_decay_init,
        phase_decay_step=phase_decay_step,
        is_out=transfer.is_out,
        wavenumbers=transfer.wavenumbers,
        pulse_spect=plan.pulse_spectrum,
        probe_spect=plan.probe_spectrum,
        seg_length=plan.seg_length,
        sin_theta=sin_theta,
        full_frequency_directivity=full_frequency_directivity,
    )


def _select_strategy(
    xp: _ArrayNamespace,
    grid_size: int,
    params: TransducerParams,
    full_frequency_directivity: bool,
    *,
    strategy: PfieldStrategy | None = None,
) -> PfieldStrategy:
    """Auto-select the best pfield strategy for the detected backend."""
    if strategy is not None:
        _require_strategy(strategy, xp, params.baffle, full_frequency_directivity)
        return strategy
    if is_jax_namespace(cast(ModuleType, xp)):
        return PfieldStrategy.SCAN
    if not _unsupported("metal", xp, params.baffle, full_frequency_directivity):
        return PfieldStrategy.METAL
    return PfieldStrategy.VECTORIZED


@overload
def pfield_precompute(
    positions: Float[Array, "*grid_shape dim"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    element_splitting: int | tuple[int, int] | None = None,
    frequency_step: float | int = 1.0,
    execution: ExecutionOptions | None = None,
) -> PfieldPlan: ...


@overload
def pfield_precompute(
    positions: Float[Array, "*grid_shape dim"],
    delays: Float[Array, " n_elements"],
    params: Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    element_splitting: int | tuple[int, int] | None = None,
    frequency_step: float | int = 1.0,
    execution: ExecutionOptions | None = None,
) -> FieldPlan: ...


def pfield_precompute(
    positions: Float[Array, "*grid_shape dim"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams | Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    element_splitting: int | tuple[int, int] | None = None,
    frequency_step: float | int = 1.0,
    execution: ExecutionOptions | None = None,
) -> PfieldPlan | FieldPlan:
    """Precompute static quantities for pfield computation.

    Extracts all data-dependent scalars and
    dynamically-shaped arrays so that ``pfield_compute`` has static shapes
    suitable for JAX JIT compilation.

    Args:
        execution: Optional bound on live numerical workspace; retained by new plans.
        positions: Grid positions in meters. Shape ``(*grid_shape, 2)``.
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        params: Transducer parameters.
        medium: Medium parameters.
        tx_n_wavelengths: Number of wavelengths in the TX pulse.
        db_thresh: Threshold in dB for frequency component selection.
        element_splitting: Number of sub-elements per element (None = auto).
        frequency_step: Scaling factor for the frequency step.

    Returns:
        PfieldPlan with static-shaped arrays and precomputed scalars.
    """
    if isinstance(params, Transducer):
        return prepare_field(
            positions,
            delays,
            params,
            medium,
            tx_n_wavelengths=tx_n_wavelengths,
            db_thresh=db_thresh,
            element_splitting=element_splitting,
            frequency_step=frequency_step,
            execution=execution,
        )
    if isinstance(element_splitting, tuple):
        raise ValueError("2D element_splitting must be an integer")
    xp = array_namespace(positions, delays)
    speed_of_sound = medium.speed_of_sound

    if positions.size == 0:
        raise ValueError("Grid has no points")

    # NaN-clean delays (for max-delay calculation)
    delays_clean = xp.where(xp.isnan(delays), xp.asarray(0.0), delays)

    # Element splitting: requires Python ceil on computed float
    if element_splitting is not None:
        n_sub = element_splitting
    else:
        lambda_min = speed_of_sound / (params.freq_center * (1.0 + params.bandwidth / 2.0))
        n_sub = ceil(params.element_width / lambda_min)

    seg_length = params.element_width / n_sub

    # Geometry for max-distance calculation
    element_pos, theta_elements, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(params.n_elements)
    subelement_offsets = _subelement_centroids(params.element_width, n_sub, theta_elements, xp)
    if execution is None:
        distances, _, _ = _distances_and_angles(
            positions, subelement_offsets, element_pos, theta_elements, speed_of_sound, params.freq_center, xp
        )
        maximum = float(xp.max(distances))
    else:
        block_size = legacy_point_count(execution, params.n_elements, n_sub)
        flat = xp.reshape(positions, (-1, 2))
        maximum = 0.0
        for start in range(0, flat.shape[0], block_size):
            distances, _, _ = _distances_and_angles(
                flat[start : start + block_size, :],
                subelement_offsets,
                element_pos,
                theta_elements,
                speed_of_sound,
                params.freq_center,
                xp,
            )
            maximum = max(maximum, float(xp.max(distances)))
    df = 1.0 / (maximum / speed_of_sound + float(xp.max(delays_clean)))
    df = float(frequency_step) * df

    # Frequency selection: uses boolean masking -> dynamic n_frequencies
    freq_plan = _select_frequencies(params.freq_center, params.bandwidth, tx_n_wavelengths, db_thresh, df, xp)
    df = freq_plan.freq_step
    n_freq_full = round(2.0 * params.freq_center / df) + 1
    freq_idx_start = round(float(freq_plan.selected_freqs[0]) / df)

    correction_factor = 1.0 if tx_n_wavelengths == float("inf") else df
    correction_factor = correction_factor * params.element_width

    return PfieldPlan(
        selected_freqs=freq_plan.selected_freqs,
        pulse_spectrum=freq_plan.pulse_spectrum,
        probe_spectrum=freq_plan.probe_spectrum,
        n_sub=n_sub,
        seg_length=seg_length,
        correction_factor=correction_factor,
        freq_step=df,
        n_freq_full=n_freq_full,
        freq_idx_start=freq_idx_start,
    )


def pfield_compute(
    positions: Float[Array, "*grid_shape dim"],
    delays: Float[Array, " n_elements"],
    plan: PfieldPlan | FieldPlan,
    params: TransducerParams | Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    full_frequency_directivity: bool = False,
    strategy: PfieldStrategy | None = None,
    execution: ExecutionOptions | None = None,
) -> Float[Array, " *grid_shape"]:
    """Compute the RMS pressure field given a precomputed plan.

    Contains only static-shape operations and is suitable for JAX JIT
    compilation when ``plan`` and ``params`` are treated as static arguments.

    Args:
        execution: Optional bound on live numerical workspace; retained by new plans.
        positions: Grid positions in meters. Shape ``(*grid_shape, 2)``.
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        plan: Precomputed plan from ``pfield_precompute``.
        params: Transducer parameters.
        medium: Medium parameters.
        tx_apodization: Transmit apodization weights. Shape ``(n_elements,)``.
            Elements with NaN delays are automatically zeroed.
        full_frequency_directivity: If True, compute element directivity at
            every frequency. If False, use center-frequency-only directivity.
        strategy: Backend strategy for the frequency sweep. If None,
            auto-selects based on the detected array backend.

    Returns:
        RMS pressure field with shape ``(*grid_shape,)``.
    """
    if isinstance(params, Transducer):
        if type(plan) is not FieldPlan:
            raise ValueError("3D description requires a FieldPlan")
        if execution is not None and execution != plan.execution:
            raise ValueError("execution differs from plan")
        spectrum = field_spectrum(
            positions, delays, plan, params, medium, tx_apodization, full_frequency_directivity, strategy, rms=True
        )
        return spectrum
    if isinstance(plan, FieldPlan):
        raise ValueError("2D description requires a legacy plan")
    if execution is not None:
        if strategy == PfieldStrategy.METAL:
            raise NotImplementedError("Native strategy does not support an execution budget")
        return _legacy_field_blocks(
            positions, delays, plan, params, medium, tx_apodization, full_frequency_directivity, execution, rms=True
        )
    xp = array_namespace(positions, delays, tx_apodization)

    delays_clean, tx_apodization = _clean_transmit_inputs(delays, tx_apodization, params.n_elements, xp)

    grid_size = prod(positions.shape[:-1])
    selected = _select_strategy(xp, grid_size, params, full_frequency_directivity, strategy=strategy)

    if selected == PfieldStrategy.METAL:
        from fast_simus.kernels.metal_pfield import pfield_metal

        if TYPE_CHECKING:
            import mlx.core as mx

        pressure_accum = cast(
            Array,
            pfield_metal(
                positions=cast("mx.array", positions),
                params=params,
                plan=plan,
                medium=medium,
                delays_clean=cast("mx.array", delays_clean),
                tx_apodization=cast("mx.array", tx_apodization),
            ),
        )
    else:
        from fast_simus._pfield_strategies import _freq_outer_python, _freq_outer_scan

        sweep = _prepare_frequency_sweep(
            positions,
            delays_clean,
            tx_apodization,
            plan,
            params,
            medium,
            full_frequency_directivity=full_frequency_directivity,
            xp=xp,
        )
        driver = _freq_outer_scan if selected == PfieldStrategy.SCAN else _freq_outer_python
        pressure_accum = driver(**sweep._asdict(), xp=xp)

    return xp.sqrt(pressure_accum * plan.correction_factor)


@jaxtyped(typechecker=typechecker)
def pfield(
    positions: Float[Array, "*grid_shape dim"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams | Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    full_frequency_directivity: bool = False,
    element_splitting: int | tuple[int, int] | None = None,
    frequency_step: float | int = 1.0,
    strategy: PfieldStrategy | None = None,
    execution: ExecutionOptions | None = None,
) -> Float[Array, " *grid_shape"]:
    """Compute the RMS acoustic pressure field of a transducer array.

    Calculates the radiation pattern (root-mean-square of acoustic pressure)
    for a uniform linear or convex array whose elements are excited at
    different time delays. 2-D computation only (no elevation focusing).

    Algorithm
    ---------
    Implements Garcia 2022 Eq. 22, computing acoustic pressure by superposing
    contributions from all array elements:

        P(X,w,t) ~ P_TX(w) exp(-iwt) Sum_n W_n [exp(ikr_n)/r_n] D(theta_n,k) exp(iw*tau_n)

    Where:
      - P_TX(w): Transmit pulse spectrum (windowed sinusoid x transducer response)
      - r_n: Distance from sub-element n to field point
      - D(theta_n,k): Element directivity = sinc(kb*sin(theta)) x obliquity_factor
      - W_n: Transmit apodization weights
      - tau_n: Transmit time delays for focusing/steering

    Wide elements are split into nu sub-elements where nu = ceil(width/lambda_min)
    to satisfy far-field conditions. The RMS field is computed by integrating
    |P(X,w)|^2 over the frequency band (Garcia 2022 Eq. 41-42):

        P_RMS(X) = sqrt[Integral |P(X,w)|^2 dw] ~ sqrt[Delta_w Sum |P(X,w_j)|^2]

    Frequency sampling uses adaptive step Delta_w to avoid phase aliasing, ensuring
    (Delta_w/c)*r_max + Delta_w*tau_max < 2*pi everywhere in the region of interest.

    Implementation Notes
    --------------------
    - **2D mode**: Uses 1/sqrt(r) geometric spreading (no elevation focusing)
    - **Attenuation**: Frequency-linear absorption exp(-alpha*f*r) with alpha in dB/cm/MHz
    - **Baffle**: Obliquity factor depends on boundary condition (rigid/soft/custom)
    - **Directivity**: Can be frequency-dependent (slower) or center-frequency only

    Args:
        execution: Optional bound on live numerical workspace; retained by new plans.
        positions: Grid positions in meters. Shape ``(*grid_shape, 2)`` where
            ``positions[..., 0]`` is lateral (x) and ``positions[..., 1]`` is
            axial (z, into tissue).
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        params: Transducer parameters (geometry, frequency, bandwidth, baffle).
        medium: Medium parameters (speed of sound, attenuation).
        tx_apodization: Transmit apodization weights. Shape ``(n_elements,)``.
            Elements with NaN delays are automatically zeroed.
        tx_n_wavelengths: Number of wavelengths in the TX pulse.
        db_thresh: Threshold in dB for frequency component selection.
            Only components above this threshold (relative to peak) are used.
        full_frequency_directivity: If True, compute element directivity at
            every frequency. If False, use center-frequency-only directivity.
        element_splitting: Number of sub-elements per transducer element.
            If None, computed automatically as ceil(element_width / smallest_wavelength).
        frequency_step: Scaling factor for the frequency step.
            Values > 1 speed up computation; values < 1 give smoother results.
        strategy: Backend strategy for the frequency sweep. If None,
            auto-selects based on the detected array backend.

    Returns:
        RMS pressure field with shape ``(*grid_shape,)``.
    """
    if isinstance(params, Transducer):
        _validate_apodization(tx_apodization, delays)
    plan = pfield_precompute(
        positions,
        delays,
        params,
        medium,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
        execution=execution,
    )
    return pfield_compute(
        positions,
        delays,
        plan,
        params,
        medium,
        tx_apodization=tx_apodization,
        full_frequency_directivity=full_frequency_directivity,
        strategy=strategy,
        execution=execution,
    )


def pfield_spectrum(
    positions: Float[Array, "*grid_shape dim"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams | Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    full_frequency_directivity: bool = False,
    element_splitting: int | tuple[int, int] | None = None,
    frequency_step: float | int = 1.0,
    execution: ExecutionOptions | None = None,
) -> tuple[Complex[Array, "*grid_shape n_freq_selected"], PfieldSpectrumInfo | FieldSpectrumInfo]:
    """Compute the complex acoustic pressure spectrum of a transducer array.

    This uses the same frequency sweep as :func:`pfield`, but preserves the
    complex pressure at every selected temporal frequency. The spatial input
    shape is preserved and frequency is appended as the final axis::

        spectrum, info = pfield_spectrum(positions, delays, params, execution=execution)
        rms = rms_from_spectrum(spectrum, info)

    Unlike :func:`pfield`, this materializes ``O(grid * n_freq)`` values.

    Args:
        execution: Optional bound on live numerical workspace; retained by new plans.
        positions: Grid positions in meters. Shape ``(*grid_shape, 2)`` where
            ``positions[..., 0]`` is lateral (x) and ``positions[..., 1]`` is
            axial (z, into tissue).
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        params: Transducer parameters (geometry, frequency, bandwidth, baffle).
        medium: Medium parameters (speed of sound, attenuation).
        tx_apodization: Transmit apodization weights. Shape ``(n_elements,)``.
            Elements with NaN delays are automatically zeroed.
        tx_n_wavelengths: Number of wavelengths in the TX pulse.
        db_thresh: Threshold in dB for frequency component selection.
        full_frequency_directivity: If True, compute element directivity at
            every frequency. If False, use center-frequency-only directivity.
        element_splitting: Number of sub-elements per transducer element.
            If None, computed automatically.
        frequency_step: Scaling factor for the frequency step. Values below 1
            give a finer grid, and therefore a longer time record after an
            inverse FFT.

    Returns:
        Tuple of (spectrum, info) where spectrum has shape
        ``(*grid_shape, n_freq_selected)`` and is complex-valued.
    """
    if isinstance(params, Transducer):
        _validate_apodization(tx_apodization, delays)
    plan = pfield_precompute(
        positions,
        delays,
        params,
        medium,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
        execution=execution,
    )

    spectrum = pfield_spectrum_compute(
        positions,
        delays,
        plan,
        params,
        medium,
        tx_apodization=tx_apodization,
        full_frequency_directivity=full_frequency_directivity,
        execution=execution,
    )
    if isinstance(plan, FieldPlan):
        return spectrum, FieldSpectrumInfo(plan._grid)
    info = PfieldSpectrumInfo(
        selected_freqs=plan.selected_freqs,
        freq_idx_start=plan.freq_idx_start,
        n_freq_full=plan.n_freq_full,
        freq_step=plan.freq_step,
        correction_factor=plan.correction_factor,
    )
    return spectrum, info


def pfield_spectrum_compute(
    positions: Float[Array, "*grid_shape dim"],
    delays: Float[Array, " n_elements"],
    plan: PfieldPlan | FieldPlan,
    params: TransducerParams | Transducer,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    full_frequency_directivity: bool = False,
    execution: ExecutionOptions | None = None,
) -> Complex[Array, "*grid_shape n_freq_selected"]:
    """Compute a pressure spectrum from a precomputed static-shape plan.

    Bind ``plan``, ``params``, ``medium``, and keyword options in a closure to
    compile this function with :func:`fast_simus.jit`.
    """
    if isinstance(params, Transducer):
        if type(plan) is not FieldPlan:
            raise ValueError("3D description requires a FieldPlan")
        if execution is not None and execution != plan.execution:
            raise ValueError("execution differs from plan")
        spectrum = field_spectrum(positions, delays, plan, params, medium, tx_apodization, full_frequency_directivity)
        return spectrum
    if isinstance(plan, FieldPlan):
        raise ValueError("2D description requires a legacy plan")
    if execution is not None:
        return _legacy_field_blocks(
            positions, delays, plan, params, medium, tx_apodization, full_frequency_directivity, execution, rms=False
        )
    xp = array_namespace(positions, delays, tx_apodization)
    delays_clean, tx_apodization = _clean_transmit_inputs(delays, tx_apodization, params.n_elements, xp)

    from fast_simus._pfield_strategies import _freq_outer_python_complex

    sweep = _prepare_frequency_sweep(
        positions,
        delays_clean,
        tx_apodization,
        plan,
        params,
        medium,
        full_frequency_directivity=full_frequency_directivity,
        xp=xp,
    )
    return _freq_outer_python_complex(**sweep._asdict(), xp=xp)


def rms_from_spectrum(
    spectrum: Complex[Array, "*grid_shape n_freq_selected"],
    info: PfieldPlan | PfieldSpectrumInfo | FieldSpectrumInfo,
) -> Float[Array, " *grid_shape"]:
    """Rebuild the RMS pressure field from a ``pfield_spectrum`` result.

    This is the discrete form of Garcia 2022 Eq. 41-42: square, sum over the
    selected frequencies, scale by ``info.correction_factor``, then take the
    square root. It matches :func:`pfield` to floating-point tolerance because
    both use the same per-frequency pressure.

    Args:
        execution: Optional bound on live numerical workspace; retained by new plans.
        spectrum: Complex pressure from :func:`pfield_spectrum`.
        info: Metadata from the same call, or the ``PfieldPlan`` used by
            :func:`pfield_spectrum_compute`.

    Returns:
        RMS pressure with the spatial shape of ``spectrum`` (no frequency axis).
    """
    xp = array_namespace(spectrum)
    energy = xp.sum(xp.real(spectrum * xp.conj(spectrum)), axis=-1)
    return xp.sqrt(energy * info.correction_factor)


def _legacy_field_blocks(positions, delays, plan, params, medium, apodization, full_directivity, execution, *, rms):
    """Bound legacy point workspace while preserving source reduction order."""
    xp = array_namespace(positions, delays)
    size = legacy_point_count(execution, params.n_elements, plan.n_sub)
    flat = xp.reshape(positions, (-1, 2))
    parts = []
    for start in range(0, flat.shape[0], size):
        if rms:
            value = pfield_compute(
                flat[start : start + size, :],
                delays,
                plan,
                params,
                medium,
                tx_apodization=apodization,
                full_frequency_directivity=full_directivity,
                strategy=PfieldStrategy.VECTORIZED,
            )
        else:
            value = pfield_spectrum_compute(
                flat[start : start + size, :],
                delays,
                plan,
                params,
                medium,
                tx_apodization=apodization,
                full_frequency_directivity=full_directivity,
            )
        parts.append(value)
    tail = () if rms else (plan.selected_freqs.shape[0],)
    return xp.reshape(xp.concat(parts, axis=0), (*positions.shape[:-1], *tail))
