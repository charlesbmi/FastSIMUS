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
from math import ceil, inf, pi, prod
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple, cast

import array_api_extra as xpx
from array_api_compat import is_jax_namespace
from beartype import beartype as typechecker
from jaxtyping import Bool, Complex, Float, jaxtyped

from fast_simus._pfield_math import (
    _distances_and_angles,
    _init_exponentials,
    _obliquity_factor,
    _select_frequencies,
    _subelement_centroids,
)
from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import BaffleType, TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace, array_namespace, is_mlx_namespace
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

    Attributes:
        selected_freqs: Significant frequency samples in Hz.
        pulse_spectrum: Pulse spectrum at selected frequencies (complex).
        probe_spectrum: Probe response at selected frequencies (real).
        freq_start: First selected frequency in Hz (Python float for JIT compatibility).
        freq_step: Frequency step in Hz.
        n_sub: Number of sub-elements per transducer element.
        seg_length: Sub-element length in meters (element_width / n_sub).
        correction_factor: Scaling factor for the RMS integration
            (df * element_width, or element_width when tx_n_wavelengths=inf).
    """

    selected_freqs: Float[Array, " n_frequencies"]
    pulse_spectrum: Complex[Array, " n_frequencies"]
    probe_spectrum: Float[Array, " n_frequencies"]
    freq_start: float
    freq_step: float
    n_sub: int
    seg_length: float
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
    positions: Float[Array, "*grid_shape 2"],
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
    element_pos, theta_elements, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(params.n_elements)

    speed_of_sound = medium.speed_of_sound
    attenuation = medium.attenuation

    subelement_offsets = _subelement_centroids(params.element_width, plan.n_sub, theta_elements, xp)

    x = positions[..., 0]
    z = positions[..., 1]
    is_out = z < 0
    if params.radius != inf:
        is_out = is_out | ((x**2 + (z + apex_offset) ** 2) <= params.radius**2)

    distances, sin_theta, theta_arr = _distances_and_angles(
        positions, subelement_offsets, element_pos, theta_elements, speed_of_sound, params.freq_center, xp
    )

    obliquity_factor = _obliquity_factor(theta_arr, params.baffle, xp)
    phase_decay_init, phase_decay_step = _init_exponentials(
        plan.freq_start, speed_of_sound, attenuation, distances, obliquity_factor, plan.freq_step, xp
    )

    if not full_frequency_directivity:
        center_wavenumber = 2.0 * pi * params.freq_center / speed_of_sound
        sinc_arg = xp.asarray(center_wavenumber * plan.seg_length / 2.0) * sin_theta / pi
        phase_decay_init = phase_decay_init * xpx.sinc(sinc_arg, xp=xp)

    # Absorb delay+apodization into the geometric progression so loop
    # drivers don't need a per-frequency multiply for delays.
    delay_apod_init = xp.exp(xp.asarray(1j * 2.0 * pi * plan.freq_start) * delays_clean) * tx_apodization
    delay_apod_step = xp.exp(xp.asarray(1j * 2.0 * pi * plan.freq_step) * delays_clean)
    phase_decay_init = phase_decay_init * delay_apod_init[:, None]
    phase_decay_step = phase_decay_step * delay_apod_step[:, None]

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

    wavenumbers = xp.asarray(2.0 * pi) * plan.selected_freqs / speed_of_sound

    return _SweepInputs(
        phase_decay_init=phase_decay_init,
        phase_decay_step=phase_decay_step,
        is_out=is_out,
        wavenumbers=wavenumbers,
        pulse_spect=plan.pulse_spectrum,
        probe_spect=plan.probe_spectrum,
        seg_length=plan.seg_length,
        sin_theta=sin_theta,
        full_frequency_directivity=full_frequency_directivity,
    )


def _metal_supported(params: TransducerParams, full_frequency_directivity: bool) -> bool:
    """Check whether the Metal kernel supports the given configuration."""
    if full_frequency_directivity:
        return False
    if not isinstance(params.baffle, str | BaffleType):
        return False
    return params.baffle == BaffleType.SOFT


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
        if strategy == PfieldStrategy.METAL and not _metal_supported(params, full_frequency_directivity):
            unsupported = []
            if full_frequency_directivity:
                unsupported.append("full_frequency_directivity=True")
            if params.baffle != BaffleType.SOFT:
                unsupported.append(f"baffle={params.baffle!r} (only SOFT supported)")
            raise NotImplementedError(
                f"Metal kernel does not support: {', '.join(unsupported)}. Use strategy=None for auto-selection."
            )
        return strategy
    if is_jax_namespace(cast(ModuleType, xp)):
        return PfieldStrategy.SCAN
    if is_mlx_namespace(xp) and _metal_supported(params, full_frequency_directivity):
        return PfieldStrategy.METAL
    return PfieldStrategy.VECTORIZED


def pfield_precompute(
    positions: Float[Array, "*grid_shape 2"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    element_splitting: int | None = None,
    frequency_step: float | int = 1.0,
) -> PfieldPlan:
    """Precompute static quantities for pfield computation.

    Extracts all data-dependent scalars and
    dynamically-shaped arrays so that ``pfield_compute`` has static shapes
    suitable for JAX JIT compilation.

    Args:
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
    distances, _, _ = _distances_and_angles(
        positions, subelement_offsets, element_pos, theta_elements, speed_of_sound, params.freq_center, xp
    )

    # Frequency step: requires float() extraction from array
    df = 1.0 / (float(xp.max(distances)) / speed_of_sound + float(xp.max(delays_clean)))
    df = float(frequency_step) * df

    # Frequency selection: uses boolean masking -> dynamic n_frequencies
    freq_plan = _select_frequencies(params.freq_center, params.bandwidth, tx_n_wavelengths, db_thresh, df, xp)
    df = freq_plan.freq_step

    correction_factor = 1.0 if tx_n_wavelengths == float("inf") else df
    correction_factor = correction_factor * params.element_width

    return PfieldPlan(
        selected_freqs=freq_plan.selected_freqs,
        pulse_spectrum=freq_plan.pulse_spectrum,
        probe_spectrum=freq_plan.probe_spectrum,
        freq_start=float(freq_plan.selected_freqs[0]),
        freq_step=df,
        n_sub=n_sub,
        seg_length=seg_length,
        correction_factor=correction_factor,
    )


def pfield_compute(
    positions: Float[Array, "*grid_shape 2"],
    delays: Float[Array, " n_elements"],
    plan: PfieldPlan,
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    full_frequency_directivity: bool = False,
    strategy: PfieldStrategy | None = None,
) -> Float[Array, " *grid_shape"]:
    """Compute the RMS pressure field given a precomputed plan.

    Contains only static-shape operations and is suitable for JAX JIT
    compilation when ``plan`` and ``params`` are treated as static arguments.

    Args:
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
    xp = array_namespace(positions, delays, tx_apodization)

    if tx_apodization is None:
        tx_apodization = xp.ones(params.n_elements)

    nan_mask = xp.isnan(delays)
    tx_apodization = xp.where(nan_mask, xp.asarray(0.0), tx_apodization)
    delays_clean = xp.where(nan_mask, xp.asarray(0.0), delays)

    grid_size = prod(positions.shape[:-1])
    selected = _select_strategy(xp, grid_size, params, full_frequency_directivity, strategy=strategy)

    if selected == PfieldStrategy.METAL:
        from fast_simus.kernels.metal_pfield import pfield_metal  # noqa: PLC0415

        if TYPE_CHECKING:
            import mlx.core as mx  # noqa: PLC0415

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
        from fast_simus._pfield_strategies import _freq_outer_scan, _pfield_freq_vectorized  # noqa: PLC0415

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
        driver = _freq_outer_scan if selected == PfieldStrategy.SCAN else _pfield_freq_vectorized
        pressure_accum = driver(**sweep._asdict(), xp=xp)

    return xp.sqrt(pressure_accum * plan.correction_factor)


@jaxtyped(typechecker=typechecker)
def pfield(
    positions: Float[Array, "*grid_shape 2"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    full_frequency_directivity: bool = False,
    element_splitting: int | None = None,
    frequency_step: float | int = 1.0,
    strategy: PfieldStrategy | None = None,
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
    plan = pfield_precompute(
        positions,
        delays,
        params,
        medium,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
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
    )
