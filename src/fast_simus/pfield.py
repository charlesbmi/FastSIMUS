"""Pressure field computation for ultrasound transducer arrays.

Implements PFIELD algorithm for simulating ultrasound beam patterns from
phased/linear/convex arrays using Fraunhofer (far-field) approximation in
the azimuthal plane and Fresnel (paraxial) approximation in elevation.

All functions are Array API compliant and work with NumPy, JAX, CuPy backends.

References:
    Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
    Part I: theory & examples. CMPB, 2022;218:106726.

    Shahriari S, Garcia D. Meshfree simulations of ultrasound vector flow
    imaging using smoothed particle hydrodynamics. Phys Med Biol,
    2018;63:205011.
"""

from __future__ import annotations

from math import ceil, inf, log, pi, prod
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

    selected_freqs: Float[Array, " n_sampling"]
    freq_mask: Bool[Array, " n_freq"]
    pulse_spectrum: Float[Array, " n_sampling"]
    probe_spectrum: Float[Array, " n_sampling"]
    freq_step: float


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
    seg_offsets = xp.asarray(
        [-element_width / 2.0 + seg_length / 2.0 + i * seg_length for i in range(n_sub)],
        dtype=xp.float64,
    )
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
    # Broadcasting: points (*batch, 2) -> (*batch, 1, 1, 2)
    #               element_pos (n_elements, 2) -> (1, n_elements, 1, 2)
    #               subelement_offsets (n_elements, n_sub, 2) -> (1, n_elements, n_sub, 2)
    delta = points[..., None, None, :] - subelement_offsets[None, :, :, :] - element_pos[None, :, None, :]
    delta_x = delta[..., 0]
    delta_z = delta[..., 1]
    dist_squared = delta_x**2 + delta_z**2

    # distances with clipping
    distances = xp.sqrt(dist_squared)
    min_distance = xp.asarray(speed_of_sound / freq_center / 2.0)
    distances = xp.where(distances < min_distance, min_distance, distances)

    # Angle relative to element normal
    epsilon = xp.asarray(1e-16)  # Small epsilon to avoid division by zero
    sqrt_d2 = xp.sqrt(dist_squared)
    # Broadcasting: theta_e (n_elements,) -> (n_elements, 1)
    theta_arr = xp.asin((delta_x + epsilon) / (sqrt_d2 + epsilon)) - theta_e[:, None]
    sin_theta = xp.sin(theta_arr)

    return distances, sin_theta, theta_arr


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
    frequencies = xp.linspace(0, 2 * fc, n_freq, dtype=xp.float64)
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
    epsilon = xp.asarray(1e-16)

    if non_rigid_baffle:
        if baffle == BaffleType.SOFT:
            obliquity_factor = xp.cos(theta_arr)
        else:
            cos_th = xp.cos(theta_arr)
            obliquity_factor = cos_th / (cos_th + float(baffle))
    else:
        obliquity_factor = xp.ones(theta_arr.shape, dtype=xp.float64)

    obliquity_factor = xp.where(
        xp.abs(theta_arr) >= xp.asarray(pi / 2),
        epsilon,
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
    phase_decay = xp.exp(xp.asarray(-attenuation_wavenum) * distances) * (
        xp.cos(phase_mod) + xp.asarray(1j) * xp.sin(phase_mod)
    )

    wavenumber_step = 2.0 * pi * freq_step / speed_of_sound
    attenuation_step = attenuation / _NEPER_TO_DB * freq_step / 1e6 * 1e2
    phase_decay_step = xp.exp(xp.asarray(-attenuation_step + 1j * wavenumber_step) * distances)

    # Incorporate obliquity / sqrt(distances) (2D, no elevation)
    phase_decay = phase_decay * obliquity_factor / xp.sqrt(distances)

    return phase_decay, phase_decay_step


@jaxtyped(typechecker=typechecker)
def _pfield_freq_step(
    phase_decay: Complex[Array, "*batch n_elements n_sub"],
    delays_clean: Float[Array, " n_elements"],
    apodization: Float[Array, " n_elements"],
    is_out: Bool[Array, " *batch"],
    wavenumber: float,
    speed_of_sound: float,
    pulse_spect_k: complex,
    probe_spect_k: complex | float,
    n_sub: int,
    seg_length: float,
    sin_theta: Float[Array, "*batch n_elements n_sub"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> Float[Array, " *batch"]:
    """Compute pressure field contribution for a single frequency sample.

    Args:
        phase_decay: Complex exponential array with propagation and attenuation.
        delays_clean: Transmit time delays (NaN replaced with 0).
        apodization: Transmit apodization weights (zeroed where delays were NaN).
        is_out: Out-of-field mask.
        wavenumber: Angular wavenumber for this frequency (2*pi*f/c).
        speed_of_sound: Speed of sound in m/s.
        pulse_spect_k: Pulse spectrum value at this frequency.
        probe_spect_k: Probe spectrum value at this frequency.
        n_sub: Number of sub-elements per element.
        seg_length: Sub-element length (element_width / n_sub).
        sin_theta: Sine of angles relative to element normal.
        full_frequency_directivity: If True, compute frequency-dependent directivity.
        xp: Array namespace.

    Returns:
        Squared magnitude |P_k|^2 of the pressure field at this frequency.
    """
    # Directivity (frequency-dependent path)
    if full_frequency_directivity:
        # Use unnormalized sinc: sinc(x/pi) from array_api_extra
        sinc_arg_k = xp.asarray(wavenumber * seg_length / 2.0) * sin_theta / pi
        directivity_k = xpx.sinc(sinc_arg_k, xp=xp)

    # Single-element radiation patterns: average over sub-elements
    if full_frequency_directivity:
        element_pattern = xp.mean(directivity_k * phase_decay, axis=-1)
    elif n_sub > 1:
        element_pattern = xp.mean(phase_decay, axis=-1)
    else:
        # n_sub == 1, squeeze last dimension
        element_pattern = phase_decay[..., 0]

    # Transmit delays + apodization
    # delays_clean is 1-D (n_elements,), apply phase shift
    delay_exp = xp.exp(xp.asarray(1j * wavenumber * speed_of_sound) * delays_clean)
    delay_apodization = delay_exp * apodization

    # Sum across elements: element_pattern (*batch, n_elements), delay_apodization (n_elements,)
    pressure_k = xp.sum(element_pattern * delay_apodization, axis=-1)

    # Apply spectrum
    pressure_k = pulse_spect_k * pressure_k * probe_spect_k

    # Zero out-of-field (no mutation)
    pressure_k = xp.where(is_out, xp.asarray(0.0 + 0j), pressure_k)

    # Return squared magnitude
    return xp.abs(pressure_k) ** 2


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
    xp = array_namespace(positions, delays)

    # Compute element positions
    element_pos, theta_e, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, xp)

    return _pfield_core(
        point_positions=positions,
        delays=delays,
        element_pos=element_pos,
        theta_e=theta_e,
        apex_offset=apex_offset,
        fc=params.freq_center,
        element_width=params.element_width,
        bandwidth=params.bandwidth,
        baffle=params.baffle,
        n_elements=params.n_elements,
        radius_of_curvature=params.radius,
        medium=medium,
        tx_apodization=tx_apodization,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        full_frequency_directivity=full_frequency_directivity,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
    )


@jaxtyped(typechecker=typechecker)
def _pfield_core(
    point_positions: Float[Array, "*grid_shape 2"],
    delays: Float[Array, " n_elements"],
    element_pos: Float[Array, "n_elements 2"],
    theta_e: Float[Array, " n_elements"] | None,
    apex_offset: float,
    fc: float,
    element_width: float,
    bandwidth: float,
    baffle: BaffleType | float,
    n_elements: int,
    radius_of_curvature: float,
    medium: MediumParams,
    *,
    tx_apodization: Float[Array, " n_elements"] | None,
    tx_n_wavelengths: float,
    db_thresh: float,
    full_frequency_directivity: bool,
    element_splitting: int | None,
    frequency_step: float,
) -> Float[Array, "*grid_shape"]:
    """Core pfield implementation computing RMS pressure field.

    When SIMUS support is added, the per-frequency accumulation logic
    (currently inlined as |P_k|^2 summation) should be extracted into
    an accumulator protocol, so that SIMUS can plug in its own
    backscatter accumulation (Eq. 32) without branching inside the loop.
    See: Garcia 2022, Sec. 4 (Eqs. 29-34).
    """
    xp: _ArrayNamespace = array_namespace(point_positions, delays)

    # Extract medium parameters
    speed_of_sound = medium.speed_of_sound
    attenuation = medium.attenuation

    # Input validation
    n_points = prod(point_positions.shape[:-1])
    if n_points == 0:
        raise ValueError("Grid has no points")

    # TX apodization
    if tx_apodization is None:
        apodization = xp.ones(n_elements, dtype=xp.float64)
    else:
        apodization = xp.asarray(tx_apodization, dtype=xp.float64)

    # Zero apodization where delays are NaN; replace NaN delays with 0
    nan_mask = xp.isnan(delays)
    apodization = xp.where(nan_mask, xp.asarray(0.0), apodization)
    delays_clean = xp.where(nan_mask, xp.asarray(0.0), delays)

    # Element splitting
    if element_splitting is not None:
        n_sub = element_splitting
    else:
        lambda_min = speed_of_sound / (fc * (1.0 + bandwidth / 2.0))
        n_sub = ceil(element_width / lambda_min)

    seg_length = element_width / n_sub

    # Element positions (already computed, passed in)
    # element_pos: (n_elements, 2), theta_e: (n_elements,) or None
    if theta_e is None:
        theta_elements = xp.zeros(n_elements, dtype=xp.float64)
    else:
        theta_elements = theta_e

    # Sub-element centroids (shape: n_elements, n_sub, 2)
    subelement_offsets = _subelement_centroids(element_width, n_sub, theta_elements, xp)

    # Out-of-field mask
    x = point_positions[..., 0]
    z = point_positions[..., 1]
    is_out = z < 0
    if radius_of_curvature != inf:
        is_out = is_out | ((x**2 + (z + apex_offset) ** 2) <= radius_of_curvature**2)

    # Distances and angles (shape: *grid_shape, n_elements, n_sub)
    distances, sin_theta, theta_arr = _distances_and_angles(
        point_positions, subelement_offsets, element_pos, theta_elements, speed_of_sound, fc, xp
    )

    # Frequency selection

    # df chosen so phase increment 2*pi*(df*r/c + df*delay) < 2*pi
    # => df < 1/(r_max/c + delay_max)
    df = 1.0 / (float(xp.max(distances)) / speed_of_sound + float(xp.max(delays_clean)))
    df = frequency_step * df

    # Select frequencies
    freq_plan = _select_frequencies(fc, bandwidth, tx_n_wavelengths, db_thresh, df, xp)
    selected_freqs = freq_plan.selected_freqs
    n_sampling = selected_freqs.shape[0]
    pulse_spect = freq_plan.pulse_spectrum
    probe_spect = freq_plan.probe_spectrum
    df = freq_plan.freq_step

    # Initialization
    pressure_accum = xp.asarray(0.0, dtype=xp.float64)

    # Obliquity factor
    obliquity_factor = _obliquity_factor(theta_arr, baffle, xp)

    # Exponential arrays
    freq_start = float(selected_freqs[0])
    phase_decay, phase_decay_step = _init_exponentials(
        freq_start, speed_of_sound, attenuation, distances, obliquity_factor, df, xp
    )

    # Simplified directivity (center-frequency only)
    if not full_frequency_directivity:
        center_wavenumber = 2.0 * pi * fc / speed_of_sound
        # Use unnormalized sinc: sinc(x/pi) from array_api_extra
        sinc_arg = xp.asarray(center_wavenumber * seg_length / 2.0) * sin_theta / pi
        # array-api-extra does not have type interoperability
        directivity = xpx.sinc(sinc_arg, xp=xp)
        phase_decay = phase_decay * directivity

    # Frequency loop
    for freq_idx in range(n_sampling):
        current_freq = float(selected_freqs[freq_idx])
        wavenumber = 2.0 * pi * current_freq / speed_of_sound

        if freq_idx > 0:
            phase_decay = phase_decay * phase_decay_step

        # Compute |P_k|^2 for this frequency
        pressure_k_squared = _pfield_freq_step(
            phase_decay=phase_decay,
            delays_clean=delays_clean,
            apodization=apodization,
            is_out=is_out,
            wavenumber=wavenumber,
            speed_of_sound=speed_of_sound,
            pulse_spect_k=pulse_spect[freq_idx],
            probe_spect_k=probe_spect[freq_idx],
            n_sub=n_sub,
            seg_length=seg_length,
            sin_theta=sin_theta,
            full_frequency_directivity=full_frequency_directivity,
            xp=xp,
        )

        # Accumulate
        pressure_accum = pressure_accum + pressure_k_squared

    # Correcting factor
    correction_factor = 1.0 if tx_n_wavelengths == float("inf") else df
    correction_factor = correction_factor * element_width

    # RMS pressure
    pressure_rms = xp.sqrt(pressure_accum * correction_factor)
    return pressure_rms


def _first_last_true(xp: _ArrayNamespace, mask: Array) -> tuple[int, int]:
    """Find first and last True index in 1D boolean array."""
    indices = xp.nonzero(mask)[0]
    if indices.shape[0] == 0:
        return 0, 0
    return int(indices[0]), int(indices[-1])
