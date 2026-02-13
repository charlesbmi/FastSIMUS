"""Pressure field computation for ultrasound transducer arrays.

Computes the RMS acoustic pressure field radiated by a uniform linear or
convex array. The model uses Fraunhofer (far-field) equations in the x-z plane.

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
from typing import NamedTuple, cast

import array_api_extra as xpx
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

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
    """Frequency sampling plan for pfield computation."""

    selected_freqs: Array  # Selected frequencies, shape (n_sampling,)
    freq_mask: Array  # Boolean mask for selected frequencies, shape (n_freq,)
    pulse_spectrum: Array  # Pulse spectrum at selected frequencies, shape (n_sampling,)
    probe_spectrum: Array  # Probe spectrum at selected frequencies, shape (n_sampling,)
    freq_step: float  # Frequency step in Hz


class _SimusContext(NamedTuple):
    """Context for SIMUS-specific computation (internal use)."""

    reflectivity_coeffs: Array | None  # Scatterer reflectivity coefficients
    rx_delay: Array | None  # Receive delays
    freq_step: float | None  # Frequency step override


def _subelement_centroids(
    element_width: float,
    n_sub: int,
    theta_e: Array,
    xp: _ArrayNamespace,
) -> tuple[Array, Array]:
    """Compute sub-element centroid positions relative to element centers.

    Args:
        element_width: Element width in meters.
        n_sub: Number of sub-elements per element.
        theta_e: Element angular positions in radians. Shape (n_elements,).
        xp: Array namespace.

    Returns:
        Tuple of (subelement_dx, subelement_dz) each with shape (n_elements, n_sub):
        - subelement_dx: Lateral positions of sub-element centroids.
        - subelement_dz: Axial positions of sub-element centroids.
    """
    seg_length = element_width / n_sub
    seg_offsets = xp.asarray(
        [-element_width / 2.0 + seg_length / 2.0 + i * seg_length for i in range(n_sub)],
        dtype=xp.float64,
    )
    # Broadcasting: (n_sub,) -> (1, n_sub), (n_elements,) -> (n_elements, 1)
    seg_offsets_2d = xp.reshape(seg_offsets, (1, n_sub))
    cos_theta = xp.cos(theta_e)[:, None]  # (n_elements, 1)
    sin_neg_theta = xp.sin(-theta_e)[:, None]
    subelement_dx = seg_offsets_2d * cos_theta  # (n_elements, n_sub)
    subelement_dz = seg_offsets_2d * sin_neg_theta
    return subelement_dx, subelement_dz


def _distances_and_angles(
    x_flat: Array,
    z_flat: Array,
    subelement_dx: Array,
    subelement_dz: Array,
    element_x: Array,
    element_z: Array,
    theta_e: Array,
    speed_of_sound: float,
    freq_center: float,
    xp: _ArrayNamespace,
) -> tuple[Array, Array, Array]:
    """Compute distances and angles from grid points to sub-elements.

    Args:
        x_flat: Flattened x-coordinates of grid points. Shape (nx,).
        z_flat: Flattened z-coordinates of grid points. Shape (nx,).
        subelement_dx: Sub-element lateral positions. Shape (n_elements, n_sub).
        subelement_dz: Sub-element axial positions. Shape (n_elements, n_sub).
        element_x: Element lateral positions. Shape (n_elements,).
        element_z: Element axial positions. Shape (n_elements,).
        theta_e: Element angular positions. Shape (n_elements,).
        speed_of_sound: Speed of sound in m/s.
        freq_center: Center frequency in Hz.
        xp: Array namespace.

    Returns:
        Tuple of (distances, sin_theta, theta_arr):
        - distances: Distances with shape (nx, n_elements, n_sub).
        - sin_theta: Sine of angles with shape (nx, n_elements, n_sub).
        - theta_arr: Angles relative to element normal with shape (nx, n_elements, n_sub).
    """
    # Distances and angles (shape: nx, n_elements, n_sub)
    # Broadcasting: x_flat (nx,) -> (nx, 1, 1)
    #               element_x (n_elements,) -> (1, n_elements, 1)
    #               subelement_dx (n_elements, n_sub) -> (1, n_elements, n_sub)
    delta_x = x_flat[:, None, None] - subelement_dx[None, :, :] - element_x[None, :, None]
    delta_z = z_flat[:, None, None] - subelement_dz[None, :, :] - element_z[None, :, None]
    dist_squared = delta_x**2 + delta_z**2

    # distances with clipping
    distances = xp.sqrt(dist_squared)
    min_distance = xp.asarray(speed_of_sound / freq_center / 2.0)
    distances = xp.where(distances < min_distance, min_distance, distances)

    # Angle relative to element normal
    epsilon = xp.asarray(1e-16)  # Small epsilon to avoid division by zero
    sqrt_d2 = xp.sqrt(dist_squared)
    # Broadcasting: theta_e (n_elements,) -> (1, n_elements, 1)
    theta_arr = xp.asin((delta_x + epsilon) / (sqrt_d2 + epsilon)) - theta_e[None, :, None]
    sin_theta = xp.sin(theta_arr)

    return distances, sin_theta, theta_arr


def _select_frequencies(
    fc: float,
    bandwidth: float,
    tx_n_wavelengths: float,
    db_thresh: float,
    df_upper: float,
    xp: _ArrayNamespace,
) -> _FrequencyPlan:
    """Select frequency samples for pfield computation.

    Args:
        fc: Center frequency in Hz.
        bandwidth: Fractional bandwidth.
        tx_n_wavelengths: Number of wavelengths in TX pulse.
        db_thresh: Threshold in dB for frequency component selection.
        df_upper: Upper bound for frequency step.
        xp: Array namespace.

    Returns:
        FrequencyPlan with selected frequencies and spectra.
    """
    # Frequency samples
    n_freq = int(2 * ceil(fc / df_upper) + 1)
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


def _obliquity_factor(
    theta_arr: Array,
    baffle: BaffleType | float,
    xp: _ArrayNamespace,
) -> Array:
    """Compute obliquity factor based on baffle type.

    Args:
        theta_arr: Angles relative to element normal. Shape (nx, n_elements, n_sub).
        baffle: Baffle type or impedance ratio.
        xp: Array namespace.

    Returns:
        Obliquity factor with shape (nx, n_elements, n_sub).
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


def _init_exponentials(
    freq_start: float,
    speed_of_sound: float,
    alpha_db: float,
    distances: Array,
    obliquity_factor: Array,
    freq_step: float,
    xp: _ArrayNamespace,
) -> tuple[Array, Array]:
    """Initialize exponential arrays for frequency loop.

    Args:
        freq_start: Initial frequency in Hz.
        speed_of_sound: Speed of sound in m/s.
        alpha_db: Attenuation coefficient in dB/cm/MHz.
        distances: Distances. Shape (nx, n_elements, n_sub).
        obliquity_factor: Obliquity factor. Shape (nx, n_elements, n_sub).
        freq_step: Frequency step in Hz.
        xp: Array namespace.

    Returns:
        Tuple of (phase_decay, phase_decay_step):
        - phase_decay: Initial exponential array with shape (nx, n_elements, n_sub).
        - phase_decay_step: Exponential increment for frequency step, same shape.
    """
    wavenumber_init = 2.0 * pi * freq_start / speed_of_sound
    attenuation_wavenum = alpha_db / _NEPER_TO_DB * freq_start / 1e6 * 1e2

    # exp(-kwa*distances + 1j*mod(kw*distances, 2pi))
    kw0_r = xp.asarray(wavenumber_init) * distances
    two_pi = xp.asarray(2.0 * pi)
    phase_mod = kw0_r - two_pi * xp.floor(kw0_r / two_pi)
    phase_decay = xp.exp(xp.asarray(-attenuation_wavenum) * distances) * (
        xp.cos(phase_mod) + xp.asarray(1j) * xp.sin(phase_mod)
    )

    wavenumber_step = 2.0 * pi * freq_step / speed_of_sound
    attenuation_step = alpha_db / _NEPER_TO_DB * freq_step / 1e6 * 1e2
    phase_decay_step = xp.exp(xp.asarray(-attenuation_step + 1j * wavenumber_step) * distances)

    # Incorporate obliquity / sqrt(distances) (2D, no elevation)
    phase_decay = phase_decay * obliquity_factor / xp.sqrt(distances)

    return phase_decay, phase_decay_step


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

    # Extract x and z from positions
    x = positions[..., 0]
    z = positions[..., 1]

    # Compute element positions
    element_x, element_z, theta_e, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, xp)

    # _pfield_core returns Array when _is_simus=False
    result = _pfield_core(
        x=x,
        z=z,
        delays=delays,
        element_x=element_x,
        element_z=element_z,
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
        _is_simus=False,
        _simus_ctx=None,
    )
    return cast(Array, result)


def _pfield_core(
    x: Array,
    z: Array,
    delays: Array,
    element_x: Array,
    element_z: Array,
    theta_e: Array | None,
    apex_offset: float,
    fc: float,
    element_width: float,
    bandwidth: float,
    baffle: BaffleType | float,
    n_elements: int,
    radius_of_curvature: float,
    medium: MediumParams,
    *,
    tx_apodization: Array | None,
    tx_n_wavelengths: float,
    db_thresh: float,
    full_frequency_directivity: bool,
    element_splitting: int | None,
    frequency_step: float,
    _is_simus: bool,
    _simus_ctx: _SimusContext | None,
) -> Array | tuple[Array, Array, Array]:
    """Core pfield implementation shared by standalone pfield and simus."""
    xp: _ArrayNamespace = array_namespace(x, z, delays)

    # Extract medium parameters
    speed_of_sound = medium.speed_of_sound
    alpha_db = medium.attenuation

    # Validate inputs
    if x.shape != z.shape:
        msg = "x and z must have the same shape"
        raise ValueError(msg)

    if delays.ndim != 1:
        msg = f"delays must be 1-D, got shape {delays.shape}"
        raise ValueError(msg)

    if delays.shape[0] != n_elements:
        msg = f"delays has {delays.shape[0]} elements, expected {n_elements}"
        raise ValueError(msg)

    # Store original shape for output reshaping
    original_shape = x.shape
    n_points = prod(x.shape)

    # Early return for empty grid
    if n_points == 0:
        if _is_simus:
            empty = xp.zeros((0,), dtype=xp.float64)
            return empty, empty, empty
        return xp.zeros((0,), dtype=xp.float64)

    # Flatten to 1D
    x_flat = xp.reshape(x, (-1,))
    z_flat = xp.reshape(z, (-1,))

    # TX apodization
    if tx_apodization is None:
        apodization = xp.ones(n_elements, dtype=xp.float64)
    else:
        apodization = xp.asarray(tx_apodization, dtype=xp.float64)
        if apodization.ndim != 1 or apodization.shape[0] != n_elements:
            msg = f"tx_apodization must have shape ({n_elements},), got {apodization.shape}"
            raise ValueError(msg)

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

    # Element positions (already computed, passed in)
    # element_x, element_z, theta_e are 1-D arrays with shape (n_elements,)
    if theta_e is None:
        theta_elements = xp.zeros(n_elements, dtype=xp.float64)
    else:
        theta_elements = theta_e

    # Sub-element centroids (shape: n_elements, n_sub)
    subelement_dx, subelement_dz = _subelement_centroids(element_width, n_sub, theta_elements, xp)

    # Out-of-field mask
    is_out = z_flat < 0
    if radius_of_curvature != inf:
        is_out = is_out | ((x_flat**2 + (z_flat + apex_offset) ** 2) <= radius_of_curvature**2)

    # Distances and angles (shape: n_points, n_elements, n_sub)
    distances, sin_theta, theta_arr = _distances_and_angles(
        x_flat, z_flat, subelement_dx, subelement_dz, element_x, element_z, theta_elements, speed_of_sound, fc, xp
    )

    # Frequency step
    if _is_simus and _simus_ctx is not None and _simus_ctx.freq_step is not None:
        df = _simus_ctx.freq_step
    else:
        df = 1.0 / (float(xp.max(distances)) / speed_of_sound + float(xp.max(delays_clean)))
        df = frequency_step * df

    # Select frequencies
    freq_plan = _select_frequencies(fc, bandwidth, tx_n_wavelengths, db_thresh, df, xp)
    f_sel = freq_plan.selected_freqs
    n_sampling = f_sel.shape[0]
    pulse_spect = freq_plan.pulse_spectrum
    probe_spect = freq_plan.probe_spectrum
    freq_mask = freq_plan.freq_mask
    df = freq_plan.freq_step

    # Initialization
    pressure_accum = xp.asarray(0.0, dtype=xp.float64)
    if _is_simus:
        spectrum_rows = []  # Accumulate rows in a list, stack at end

    # Obliquity factor
    obliquity_factor = _obliquity_factor(theta_arr, baffle, xp)

    # Exponential arrays
    freq_start = float(f_sel[0])
    phase_decay, phase_decay_step = _init_exponentials(
        freq_start, speed_of_sound, alpha_db, distances, obliquity_factor, df, xp
    )

    # Simplified directivity (center-frequency only)
    if not full_frequency_directivity:
        center_wavenumber = 2.0 * pi * fc / speed_of_sound
        seg_length = element_width / n_sub
        # Use unnormalized sinc: sinc(x/pi) from array_api_extra
        sinc_arg = xp.asarray(center_wavenumber * seg_length / 2.0) * sin_theta / pi
        # array-api-extra does not have type interoperability
        directivity = xpx.sinc(sinc_arg, xp=xp)  # ty: ignore[invalid-argument-type]
        phase_decay = phase_decay * cast(Array, directivity)

    # Frequency loop
    for freq_idx in range(n_sampling):
        freq_k = float(f_sel[freq_idx])
        wavenumber = 2.0 * pi * freq_k / speed_of_sound

        if freq_idx > 0:
            phase_decay = phase_decay * phase_decay_step

        # Directivity (frequency-dependent path)
        if full_frequency_directivity:
            # Use unnormalized sinc: sinc(x/pi) from array_api_extra
            sinc_arg_k = xp.asarray(wavenumber * seg_length / 2.0) * sin_theta / pi
            directivity_k = xpx.sinc(sinc_arg_k, xp=xp)  # type: ignore[arg-type]

        # Single-element radiation patterns: average over sub-elements
        if full_frequency_directivity:
            element_pattern = xp.mean(directivity_k * phase_decay, axis=-1)  # ty: ignore[unsupported-operator]
        elif n_sub > 1:
            element_pattern = xp.mean(phase_decay, axis=-1)
        else:
            # n_sub == 1, squeeze last dimension
            element_pattern = xp.reshape(phase_decay, (phase_decay.shape[0], phase_decay.shape[1]))

        # Transmit delays + apodization
        # delays_clean is 1-D (n_elements,), apply phase shift
        delay_exp = xp.exp(xp.asarray(1j * wavenumber * speed_of_sound) * delays_clean)
        delay_apodization = delay_exp * apodization  # Element-wise, shape (n_elements,)

        # Sum across elements: element_pattern (n_points, n_elements), delay_apodization (n_elements,)
        pressure_k = element_pattern @ xp.reshape(delay_apodization, (-1, 1))

        # Apply spectrum
        pressure_k = pulse_spect[freq_idx] * pressure_k * probe_spect[freq_idx]

        # Zero out-of-field (no mutation)
        pressure_k = xp.where(is_out[:, None], xp.asarray(0.0 + 0j), pressure_k)

        # Accumulate
        if _is_simus:
            if _simus_ctx is None or _simus_ctx.reflectivity_coeffs is None:
                msg = "_simus_ctx.reflectivity_coeffs must be provided when _is_simus=True"
                raise ValueError(msg)
            spectrum_k = probe_spect[freq_idx]
            rp_flat = xp.reshape(pressure_k, (-1,))
            rc_flat = xp.reshape(_simus_ctx.reflectivity_coeffs, (-1,))
            weighted = xp.reshape(rp_flat * rc_flat, (1, -1))
            row = spectrum_k * xp.reshape(weighted @ element_pattern, (-1,))
            if _simus_ctx.rx_delay is not None:
                rx_exp = xp.exp(xp.asarray(1j * wavenumber * speed_of_sound) * _simus_ctx.rx_delay)
                row = row * rx_exp
            spectrum_rows.append(row)
        else:
            pressure_accum = pressure_accum + xp.abs(pressure_k) ** 2

    # Correcting factor
    correction_factor = 1.0 if tx_n_wavelengths == float("inf") else df
    correction_factor = correction_factor * element_width

    if _is_simus:
        # Stack accumulated rows into spectrum array
        spectrum = xp.stack(spectrum_rows, axis=0) * correction_factor
        pressure_accum = pressure_accum * correction_factor
        return pressure_accum, spectrum, freq_mask

    # RMS pressure, reshape to original grid shape
    pressure_rms = xp.sqrt(pressure_accum * correction_factor)
    return xp.reshape(pressure_rms, original_shape)


def _first_last_true(xp: _ArrayNamespace, mask: Array) -> tuple[int, int]:
    """Find first and last True index in 1D boolean array."""
    indices = xp.nonzero(mask)[0]
    if indices.shape[0] == 0:
        return 0, 0
    return int(indices[0]), int(indices[-1])
