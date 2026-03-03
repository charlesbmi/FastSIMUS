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

from math import ceil, inf, log, pi
from typing import NamedTuple

import array_api_extra as xpx
from beartype import beartype as typechecker
from jaxtyping import Bool, Complex, Float, jaxtyped

from fast_simus.medium_params import MediumParams
from fast_simus.spectrum import probe_spectrum, pulse_spectrum
from fast_simus.transducer_params import BaffleType, TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace, array_namespace
from fast_simus.utils.geometry import element_positions

_DEFAULT_MEDIUM = MediumParams()

# Conversion factor: Nepers to dB
# 20/log(10) ≈ 8.6859
_NEPER_TO_DB = 20.0 / log(10.0)


class _FrequencyPlan(NamedTuple):
    """Frequency sampling plan for pfield computation.

    Attributes:
        selected_freqs: Selected frequencies, shape
        freq_mask: Boolean mask for selected frequencies, shape
        pulse_spectrum: Pulse spectrum at selected frequencies, shape
        probe_spectrum: Probe spectrum at selected frequencies, shape
        freq_step: Frequency step in Hz
    """

    selected_freqs: Float[Array, " n_frequencies"]
    freq_mask: Bool[Array, " n_freq"]
    pulse_spectrum: Complex[Array, " n_frequencies"]
    probe_spectrum: Float[Array, " n_frequencies"]
    freq_step: float


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


@jaxtyped(typechecker=typechecker)
def _subelement_centroids(
    element_width: float,
    n_sub: int,
    theta_e: Float[Array, " n_elements"],
    xp: _ArrayNamespace,
) -> Float[Array, "n_elements n_sub 2"]:
    """Compute sub-element centroid positions relative to element centers.

    Args:
        element_width: Element width in meters.
        n_sub: Number of sub-elements per element.
        theta_e: Element angular positions in radians.
        xp: Array namespace.

    Returns:
        Sub-element offsets with shape (n_elements, n_sub, 2) where [..., 0]
        is lateral (x) and [..., 1] is axial (z).
    """
    seg_length = element_width / n_sub
    seg_offsets = xp.asarray([-element_width / 2.0 + seg_length / 2.0 + i * seg_length for i in range(n_sub)])
    # Broadcasting: (n_sub,) -> (1, n_sub), (n_elements,) -> (n_elements, 1)
    seg_offsets_2d = xp.reshape(seg_offsets, (1, n_sub))
    cos_theta = xp.cos(theta_e)[:, None]
    sin_neg_theta = xp.sin(-theta_e)[:, None]
    subelement_dx = seg_offsets_2d * cos_theta
    subelement_dz = seg_offsets_2d * sin_neg_theta
    return xp.stack([subelement_dx, subelement_dz], axis=-1)


@jaxtyped(typechecker=typechecker)
def _distances_and_angles(
    points: Float[Array, "*batch 2"],
    subelement_offsets: Float[Array, "n_elements n_sub 2"],
    element_pos: Float[Array, "n_elements 2"],
    theta_e: Float[Array, " n_elements"],
    speed_of_sound: float,
    freq_center: float,
    xp: _ArrayNamespace,
) -> tuple[
    Float[Array, "*batch n_elements n_sub"],
    Float[Array, "*batch n_elements n_sub"],
    Float[Array, "*batch n_elements n_sub"],
]:
    """Compute distances and angles from grid points to sub-elements.

    Args:
        points: Grid point positions.
        subelement_offsets: Sub-element offsets.
        element_pos: Element positions.
        theta_e: Element angular positions.
        speed_of_sound: Speed of sound in m/s.
        freq_center: Center frequency in Hz.
        xp: Array namespace.

    Returns:
        Tuple of (distances, sin_theta, theta_arr):
        - distances: Distances from grid points to sub-elements.
        - sin_theta: Sine of angles relative to element normal.
        - theta_arr: Angles relative to element normal.
    """
    delta: Float[Array, "*batch n_elements n_sub xz=2"] = (
        points[..., None, None, :] - subelement_offsets - element_pos[:, None, :]
    )
    delta_x = delta[..., 0]
    delta_z = delta[..., 1]
    dist_squared = delta_x**2 + delta_z**2
    distances = xp.sqrt(dist_squared)

    # Distances with clipping (use unclipped sqrt for angle computation)
    min_distance = xp.asarray(speed_of_sound / freq_center / 2.0)
    distances_clipped = xp.where(distances < min_distance, min_distance, distances)

    # Angle relative to element normal
    _div_eps = xp.asarray(1e-16)  # Numerical stability for division
    theta_arr = xp.asin((delta_x + _div_eps) / (distances + _div_eps)) - theta_e[:, None]
    sin_theta = xp.sin(theta_arr)

    return distances_clipped, sin_theta, theta_arr


def _select_frequencies(
    fc: float,
    bandwidth: float,
    tx_n_wavelengths: float,
    db_thresh: float,
    max_freq_step: float,
    xp: _ArrayNamespace,
) -> _FrequencyPlan:
    """Select frequency samples for pfield computation.

    Args:
        fc: Center frequency in Hz.
        bandwidth: Fractional bandwidth.
        tx_n_wavelengths: Number of wavelengths in TX pulse.
        db_thresh: Threshold in dB for frequency component selection.
        max_freq_step: Upper bound for frequency step.
        xp: Array namespace.

    Returns:
        FrequencyPlan with selected frequencies and spectra.
    """
    # Frequency samples
    n_freq = int(2 * ceil(fc / max_freq_step) + 1)
    frequencies = xp.linspace(0, 2 * fc, n_freq)
    freq_step = float(frequencies[1])

    # Keep only significant components (dB threshold)
    angular_freqs_all = xp.asarray(2.0 * pi) * frequencies
    spectrum_magnitude = xp.abs(
        pulse_spectrum(angular_freqs_all, fc, tx_n_wavelengths) * probe_spectrum(angular_freqs_all, fc, bandwidth)
    )
    gain_db = 20.0 * xp.log10(xp.asarray(1e-200) + spectrum_magnitude / xp.max(spectrum_magnitude))
    above_threshold = gain_db > db_thresh
    idx_first, idx_last = _first_last_true(xp, above_threshold)
    all_indices = xp.arange(frequencies.shape[0])
    freq_mask = (all_indices >= idx_first) & (all_indices <= idx_last)

    selected_freqs = frequencies[freq_mask]

    angular_freqs_sel = xp.asarray(2.0 * pi) * selected_freqs
    pulse_spect = pulse_spectrum(angular_freqs_sel, fc, tx_n_wavelengths)
    probe_spect = probe_spectrum(angular_freqs_sel, fc, bandwidth)

    return _FrequencyPlan(selected_freqs, freq_mask, pulse_spect, probe_spect, freq_step)


@jaxtyped(typechecker=typechecker)
def _obliquity_factor(
    theta_arr: Float[Array, "*batch n_elements n_sub"],
    baffle: BaffleType | float,
    xp: _ArrayNamespace,
) -> Float[Array, "*batch n_elements n_sub"]:
    """Compute obliquity factor based on baffle type.

    Args:
        theta_arr: Angles relative to element normal.
        baffle: Baffle type or impedance ratio.
        xp: Array namespace.

    Returns:
        Obliquity factor.
    """
    non_rigid_baffle = baffle != BaffleType.RIGID
    _horizon_floor = xp.asarray(1e-16)  # Near-zero for beyond-hemisphere angles

    if non_rigid_baffle:
        if baffle == BaffleType.SOFT:
            obliquity_factor = xp.cos(theta_arr)
        else:
            cos_th = xp.cos(theta_arr)
            obliquity_factor = cos_th / (cos_th + float(baffle))
    else:
        obliquity_factor = xp.ones(theta_arr.shape)

    obliquity_factor = xp.where(
        xp.abs(theta_arr) >= xp.asarray(pi / 2),
        _horizon_floor,
        obliquity_factor,
    )

    return obliquity_factor


@jaxtyped(typechecker=typechecker)
def _init_exponentials(
    freq_start: float,
    speed_of_sound: float,
    attenuation: float,
    distances: Float[Array, "*batch n_elements n_sub"],
    obliquity_factor: Float[Array, "*batch n_elements n_sub"],
    freq_step: float,
    xp: _ArrayNamespace,
) -> tuple[
    Complex[Array, "*batch n_elements n_sub"],
    Complex[Array, "*batch n_elements n_sub"],
]:
    """Initialize exponential arrays for frequency loop.

    Args:
        freq_start: Initial frequency in Hz.
        speed_of_sound: Speed of sound in m/s.
        attenuation: Attenuation coefficient in dB/cm/MHz.
        distances: Distances.
        obliquity_factor: Obliquity factor.
        freq_step: Frequency step in Hz.
        xp: Array namespace.

    Returns:
        Tuple of (phase_decay, phase_decay_step):
        - phase_decay: Initial complex exponential array.
        - phase_decay_step: Complex exponential increment per frequency step.
    """
    wavenumber_init = 2.0 * pi * freq_start / speed_of_sound
    attenuation_wavenum = attenuation / _NEPER_TO_DB * freq_start / 1e6 * 1e2

    # exp(-kwa*distances + 1j*mod(kw*distances, 2pi))
    kw0_r = xp.asarray(wavenumber_init) * distances
    two_pi = xp.asarray(2.0 * pi)
    phase_mod = kw0_r - two_pi * xp.floor(kw0_r / two_pi)
    phase_decay = xp.exp(xp.asarray(-attenuation_wavenum) * distances + xp.asarray(1j) * phase_mod)

    wavenumber_step = 2.0 * pi * freq_step / speed_of_sound
    attenuation_step = attenuation / _NEPER_TO_DB * freq_step / 1e6 * 1e2
    phase_decay_step = xp.exp(xp.asarray(-attenuation_step + 1j * wavenumber_step) * distances)

    # Incorporate obliquity / sqrt(distances) (2D, no elevation)
    phase_decay = phase_decay * obliquity_factor / xp.sqrt(distances)

    return phase_decay, phase_decay_step


@jaxtyped(typechecker=typechecker)
def _pfield_freq_vectorized(
    phase_decay_init: Complex[Array, "*grid n_elements n_sub"],
    phase_decay_step: Complex[Array, "*grid n_elements n_sub"],
    delays_clean: Float[Array, " n_elements"],
    tx_apodization: Float[Array, " n_elements"],
    is_out: Bool[Array, " *grid"],
    wavenumbers: Float[Array, " n_freq"],
    speed_of_sound: float,
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    n_sub: int,
    seg_length: float,
    sin_theta: Float[Array, "*grid n_elements n_sub"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> Float[Array, " *grid"]:
    """Compute pressure field contribution for all frequencies at once.

    Replaces the sequential frequency loop with vectorized tensor operations.
    Uses the geometric progression: phase_decay[k] = init * step^k.

    The sub-element loop (n_sub iterations) avoids materializing the full
    (*grid, n_elements, n_sub, n_freq) tensor. Only (*grid, n_elements, n_freq)
    is materialized per sub-element, then immediately contracted over elements.

    Args:
        phase_decay_init: Complex propagation at first frequency.
        phase_decay_step: Complex multiplier per frequency step.
        delays_clean: Transmit time delays (NaN replaced with 0).
        tx_apodization: Transmit apodization weights.
        is_out: Out-of-field mask.
        wavenumbers: Angular wavenumbers for all frequencies (2*pi*f/c).
        speed_of_sound: Speed of sound in m/s.
        pulse_spect: Pulse spectrum at all selected frequencies.
        probe_spect: Probe response at all selected frequencies.
        n_sub: Number of sub-elements per element.
        seg_length: Sub-element length (element_width / n_sub).
        sin_theta: Sine of angles relative to element normal.
        full_frequency_directivity: If True, compute frequency-dependent directivity.
        xp: Array namespace.

    Returns:
        Sum of |P_k|^2 over all frequencies, shape (*grid,).
    """
    n_freq = wavenumbers.shape[0]
    exponents = xp.arange(n_freq, dtype=wavenumbers.dtype)  # match float dtype of inputs

    # Delay + apodization for all frequencies at once: (n_elements, n_freq)
    delay_apod = xp.exp(xp.asarray(1j) * wavenumbers * speed_of_sound * delays_clean[:, None]) * tx_apodization[:, None]

    # Accumulate over sub-elements to avoid materializing the 4D tensor
    pressure_all = xp.asarray(0.0 + 0j)  # broadcasts to (*grid, n_freq)

    for i in range(n_sub):
        # Phase at all frequencies for sub-element i: (*grid, n_elements, n_freq)
        phase_k = phase_decay_init[..., i, None] * phase_decay_step[..., i, None] ** exponents

        if full_frequency_directivity:
            sinc_arg = wavenumbers * seg_length / 2.0 * sin_theta[..., i, None] / pi
            phase_k = xpx.sinc(sinc_arg, xp=xp) * phase_k

        # Contract over elements: (*grid, n_elements, n_freq) -> (*grid, n_freq)
        pressure_all = pressure_all + xp.sum(phase_k * delay_apod, axis=-2)

    pressure_all = pressure_all / n_sub

    # Apply spectra: (n_freq,) broadcasts with (*grid, n_freq)
    pressure_all = pulse_spect * probe_spect * pressure_all

    # Zero out-of-field points
    pressure_all = xp.where(is_out[..., None], xp.asarray(0.0 + 0j), pressure_all)

    # Sum |P_k|^2 over frequencies -> (*grid,)
    return xp.sum(xp.abs(pressure_all) ** 2, axis=-1)


def _pfield_freq_element_accum(
    phase_decay_init: Complex[Array, "*grid n_elements n_sub"],
    phase_decay_step: Complex[Array, "*grid n_elements n_sub"],
    delays_clean: Float[Array, " n_elements"],
    tx_apodization: Float[Array, " n_elements"],
    is_out: Bool[Array, " *grid"],
    wavenumbers: Float[Array, " n_freq"],
    speed_of_sound: float,
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    n_sub: int,
    seg_length: float,
    sin_theta: Float[Array, "*grid n_elements n_sub"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
    *,
    chunk_e: int = 1,
) -> Float[Array, " *grid"]:
    """Element-chunked variant of ``_pfield_freq_vectorized``.

    Loops over element chunks of size ``chunk_e`` to avoid materializing the
    full ``(*grid, n_elements, n_freq)`` tensor. Produces bit-identical results
    to the baseline vectorized implementation.
    """
    n_freq = wavenumbers.shape[0]
    n_elements = phase_decay_init.shape[-2]
    exponents = xp.arange(n_freq, dtype=wavenumbers.dtype)

    delay_apod = xp.exp(xp.asarray(1j) * wavenumbers * speed_of_sound * delays_clean[:, None]) * tx_apodization[:, None]

    pressure_all = xp.asarray(0.0 + 0j)

    for i in range(n_sub):
        for e_start in range(0, n_elements, chunk_e):
            e_slice = slice(e_start, min(e_start + chunk_e, n_elements))
            phase_k = phase_decay_init[..., e_slice, i, None] * phase_decay_step[..., e_slice, i, None] ** exponents

            if full_frequency_directivity:
                sinc_arg = wavenumbers * seg_length / 2.0 * sin_theta[..., e_slice, i, None] / pi
                phase_k = xpx.sinc(sinc_arg, xp=xp) * phase_k

            pressure_all = pressure_all + xp.sum(phase_k * delay_apod[e_slice, ...], axis=-2)

    pressure_all = pressure_all / n_sub
    pressure_all = pulse_spect * probe_spect * pressure_all
    pressure_all = xp.where(is_out[..., None], xp.asarray(0.0 + 0j), pressure_all)
    return xp.sum(xp.abs(pressure_all) ** 2, axis=-1)


def _pfield_freq_chunk_freq(
    phase_decay_init: Complex[Array, "*grid n_elements n_sub"],
    phase_decay_step: Complex[Array, "*grid n_elements n_sub"],
    delays_clean: Float[Array, " n_elements"],
    tx_apodization: Float[Array, " n_elements"],
    is_out: Bool[Array, " *grid"],
    wavenumbers: Float[Array, " n_freq"],
    speed_of_sound: float,
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    n_sub: int,
    seg_length: float,
    sin_theta: Float[Array, "*grid n_elements n_sub"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
    *,
    chunk_freq: int = 1,
) -> Float[Array, " *grid"]:
    """Frequency-chunked variant of ``_pfield_freq_vectorized``.

    Loops over frequency chunks of size ``chunk_freq``, accumulating
    ``sum |P_f|^2`` incrementally. Peak memory is proportional to
    ``(*grid, n_elements, chunk_freq)`` instead of the full frequency axis.
    """
    n_freq = wavenumbers.shape[0]
    exponents = xp.arange(n_freq, dtype=wavenumbers.dtype)

    delay_apod = xp.exp(xp.asarray(1j) * wavenumbers * speed_of_sound * delays_clean[:, None]) * tx_apodization[:, None]

    pressure_sq = xp.zeros(is_out.shape, dtype=wavenumbers.dtype)

    for f_start in range(0, n_freq, chunk_freq):
        f_slice = slice(f_start, min(f_start + chunk_freq, n_freq))
        exps = exponents[f_slice]

        pressure_chunk = xp.asarray(0.0 + 0j)

        for i in range(n_sub):
            phase_k = phase_decay_init[..., i, None] * phase_decay_step[..., i, None] ** exps

            if full_frequency_directivity:
                sinc_arg = wavenumbers[f_slice] * seg_length / 2.0 * sin_theta[..., i, None] / pi
                phase_k = xpx.sinc(sinc_arg, xp=xp) * phase_k

            pressure_chunk = pressure_chunk + xp.sum(phase_k * delay_apod[:, f_slice], axis=-2)

        pressure_chunk = pressure_chunk / n_sub
        pressure_chunk = pulse_spect[f_slice] * probe_spect[f_slice] * pressure_chunk
        pressure_chunk = xp.where(is_out[..., None], xp.asarray(0.0 + 0j), pressure_chunk)
        pressure_sq = pressure_sq + xp.sum(xp.abs(pressure_chunk) ** 2, axis=-1)

    return pressure_sq


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

    Returns:
        RMS pressure field with shape ``(*grid_shape,)``.
    """
    xp = array_namespace(positions, delays, tx_apodization)

    element_pos, theta_elements, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(params.n_elements)

    speed_of_sound = medium.speed_of_sound
    attenuation = medium.attenuation

    if tx_apodization is None:
        tx_apodization = xp.ones(params.n_elements)

    nan_mask = xp.isnan(delays)
    tx_apodization = xp.where(nan_mask, xp.asarray(0.0), tx_apodization)
    delays_clean = xp.where(nan_mask, xp.asarray(0.0), delays)

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

    wavenumbers = xp.asarray(2.0 * pi) * plan.selected_freqs / speed_of_sound

    pressure_accum = _pfield_freq_vectorized(
        phase_decay_init=phase_decay_init,
        phase_decay_step=phase_decay_step,
        delays_clean=delays_clean,
        tx_apodization=tx_apodization,
        is_out=is_out,
        wavenumbers=wavenumbers,
        speed_of_sound=speed_of_sound,
        pulse_spect=plan.pulse_spectrum,
        probe_spect=plan.probe_spectrum,
        n_sub=plan.n_sub,
        seg_length=plan.seg_length,
        sin_theta=sin_theta,
        full_frequency_directivity=full_frequency_directivity,
        xp=xp,
    )

    return xp.sqrt(pressure_accum * plan.correction_factor)


def pfield_compute_chunked(
    positions: Float[Array, "*grid_shape 2"],
    delays: Float[Array, " n_elements"],
    plan: PfieldPlan,
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    grid_chunk_size: int | None = None,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    full_frequency_directivity: bool = False,
) -> Float[Array, " *grid_shape"]:
    """Compute the RMS pressure field by chunking the spatial grid.

    Wraps ``pfield_compute``, splitting the positions into flat batches of
    ``grid_chunk_size`` to limit peak memory. The ``plan`` must already be
    precomputed over the full positions array (for correct max-distance
    frequency sampling).

    Args:
        positions: Grid positions in meters. Shape ``(*grid_shape, 2)``.
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        plan: Precomputed plan from ``pfield_precompute`` (over full grid).
        params: Transducer parameters.
        medium: Medium parameters.
        grid_chunk_size: Maximum number of grid points per batch. If None,
            delegates directly to ``pfield_compute`` without chunking.
        tx_apodization: Transmit apodization weights. Shape ``(n_elements,)``.
        full_frequency_directivity: If True, compute frequency-dependent
            directivity at every frequency.

    Returns:
        RMS pressure field with shape ``(*grid_shape,)``.
    """
    if grid_chunk_size is None:
        return pfield_compute(
            positions,
            delays,
            plan,
            params,
            medium,
            tx_apodization=tx_apodization,
            full_frequency_directivity=full_frequency_directivity,
        )

    xp = array_namespace(positions, delays)
    grid_shape = positions.shape[:-1]
    n_total = 1
    for s in grid_shape:
        n_total *= s

    flat_pos = xp.reshape(positions, (n_total, 2))
    results = []

    for start in range(0, n_total, grid_chunk_size):
        end = min(start + grid_chunk_size, n_total)
        chunk_pos = xp.reshape(flat_pos[start:end, ...], (end - start, 1, 2))
        chunk_result = pfield_compute(
            chunk_pos,
            delays,
            plan,
            params,
            medium,
            tx_apodization=tx_apodization,
            full_frequency_directivity=full_frequency_directivity,
        )
        results.append(xp.reshape(chunk_result, (end - start,)))

    flat_result = xp.concat(results, axis=0)
    return xp.reshape(flat_result, grid_shape)


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
    )


def _first_last_true(xp: _ArrayNamespace, mask: Array) -> tuple[int, int]:
    """Find first and last True index in 1D boolean array. JAX-compatible (no nonzero)."""
    n = mask.shape[0]
    if n == 0:
        return 0, 0
    # Cast to int: argmax on bool not allowed by array_api_strict
    mask_int = xp.asarray(mask, dtype=xp.int32)
    first = int(xp.argmax(mask_int))
    if int(xp.max(mask_int)) == 0:
        return 0, 0
    last = n - 1 - int(xp.argmax(mask_int[::-1]))
    return first, last
