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

    f_sel: Array  # Selected frequencies, shape (n_sampling,)
    freq_mask: Array  # Boolean mask for selected frequencies, shape (n_freq,)
    pulse_spect: Array  # Pulse spectrum at selected frequencies, shape (n_sampling,)
    probe_spect: Array  # Probe spectrum at selected frequencies, shape (n_sampling,)
    df: float  # Frequency step in Hz


class _SimusContext(NamedTuple):
    """Context for SIMUS-specific computation (internal use)."""

    rc: Array | None  # Scatterer reflectivity coefficients
    rx_delay: Array | None  # Receive delays
    df: float | None  # Frequency step override


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
        Tuple of (xi, zi) each with shape (n_elements, n_sub):
        - xi: Lateral positions of sub-element centroids.
        - zi: Axial positions of sub-element centroids.
    """
    seg_length = element_width / n_sub
    seg_offsets = xp.asarray(
        [-element_width / 2.0 + seg_length / 2.0 + i * seg_length for i in range(n_sub)],
        dtype=xp.float64,
    )
    # Broadcasting: (n_sub,) -> (1, n_sub), (n_elements,) -> (n_elements, 1)
    seg_2d = xp.reshape(seg_offsets, (1, n_sub))
    cos_the = xp.cos(theta_e)[:, None]  # (n_elements, 1)
    sin_neg_the = xp.sin(-theta_e)[:, None]
    xi = seg_2d * cos_the  # (n_elements, n_sub)
    zi = seg_2d * sin_neg_the
    return xi, zi


def _distances_and_angles(
    x_flat: Array,
    z_flat: Array,
    xi: Array,
    zi: Array,
    xe: Array,
    ze: Array,
    theta_e: Array,
    c: float,
    fc: float,
    xp: _ArrayNamespace,
) -> tuple[Array, Array, Array]:
    """Compute distances and angles from grid points to sub-elements.

    Args:
        x_flat: Flattened x-coordinates of grid points. Shape (nx,).
        z_flat: Flattened z-coordinates of grid points. Shape (nx,).
        xi: Sub-element lateral positions. Shape (n_elements, n_sub).
        zi: Sub-element axial positions. Shape (n_elements, n_sub).
        xe: Element lateral positions. Shape (n_elements,).
        ze: Element axial positions. Shape (n_elements,).
        theta_e: Element angular positions. Shape (n_elements,).
        c: Speed of sound in m/s.
        fc: Center frequency in Hz.
        xp: Array namespace.

    Returns:
        Tuple of (r, sin_theta, theta_arr):
        - r: Distances with shape (nx, n_elements, n_sub).
        - sin_theta: Sine of angles with shape (nx, n_elements, n_sub).
        - theta_arr: Angles relative to element normal with shape (nx, n_elements, n_sub).
    """
    # Distances and angles (shape: nx, n_elements, n_sub)
    # Broadcasting: x_flat (nx,) -> (nx, 1, 1)
    #               xe (n_elements,) -> (1, n_elements, 1)
    #               xi (n_elements, n_sub) -> (1, n_elements, n_sub)
    dxi = x_flat[:, None, None] - xi[None, :, :] - xe[None, :, None]
    dz_arr = z_flat[:, None, None] - zi[None, :, :] - ze[None, :, None]
    d2 = dxi**2 + dz_arr**2

    # r with clipping
    r = xp.sqrt(d2)
    small_d = xp.asarray(c / fc / 2.0)
    r = xp.where(r < small_d, small_d, r)

    # Angle relative to element normal
    eps_sp = xp.asarray(1e-16)  # Small epsilon to avoid division by zero
    sqrt_d2 = xp.sqrt(d2)
    # Broadcasting: theta_e (n_elements,) -> (1, n_elements, 1)
    theta_arr = xp.asin((dxi + eps_sp) / (sqrt_d2 + eps_sp)) - theta_e[None, :, None]
    sin_theta = xp.sin(theta_arr)

    return r, sin_theta, theta_arr


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
    f = xp.linspace(0, 2 * fc, n_freq, dtype=xp.float64)
    df = float(f[1])

    # Keep only significant components (dB threshold)
    w_all = xp.asarray(2.0 * pi) * f
    s_mag = xp.abs(pulse_spectrum(w_all, fc, tx_n_wavelengths) * probe_spectrum(w_all, fc, bandwidth))
    g_db = 20.0 * xp.log10(xp.asarray(1e-200) + s_mag / xp.max(s_mag))
    above = g_db > db_thresh
    idx_first, idx_last = _first_last_true(xp, above)
    all_indices = xp.arange(f.shape[0])
    freq_mask = (all_indices >= idx_first) & (all_indices <= idx_last)

    f_sel = f[freq_mask]

    w_f = xp.asarray(2.0 * pi) * f_sel
    pulse_spect = pulse_spectrum(w_f, fc, tx_n_wavelengths)
    probe_spect = probe_spectrum(w_f, fc, bandwidth)

    return _FrequencyPlan(f_sel, freq_mask, pulse_spect, probe_spect, df)


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
    eps_sp = xp.asarray(1e-16)

    if non_rigid_baffle:
        if baffle == BaffleType.SOFT:
            obli_fac = xp.cos(theta_arr)
        else:
            cos_th = xp.cos(theta_arr)
            obli_fac = cos_th / (cos_th + float(baffle))
    else:
        obli_fac = xp.ones(theta_arr.shape, dtype=xp.float64)

    obli_fac = xp.where(
        xp.abs(theta_arr) >= xp.asarray(pi / 2),
        eps_sp,
        obli_fac,
    )

    return obli_fac


def _init_exponentials(
    f0: float,
    c: float,
    alpha_db: float,
    r: Array,
    obli_fac: Array,
    df: float,
    xp: _ArrayNamespace,
) -> tuple[Array, Array]:
    """Initialize exponential arrays for frequency loop.

    Args:
        f0: Initial frequency in Hz.
        c: Speed of sound in m/s.
        alpha_db: Attenuation coefficient in dB/cm/MHz.
        r: Distances. Shape (nx, n_elements, n_sub).
        obli_fac: Obliquity factor. Shape (nx, n_elements, n_sub).
        df: Frequency step in Hz.
        xp: Array namespace.

    Returns:
        Tuple of (exp_arr, exp_df):
        - exp_arr: Initial exponential array with shape (nx, n_elements, n_sub).
        - exp_df: Exponential increment for frequency step, same shape.
    """
    kw0 = 2.0 * pi * f0 / c
    kwa0 = alpha_db / _NEPER_TO_DB * f0 / 1e6 * 1e2

    # exp(-kwa*r + 1j*mod(kw*r, 2pi))
    kw0_r = xp.asarray(kw0) * r
    two_pi = xp.asarray(2.0 * pi)
    phase_mod = kw0_r - two_pi * xp.floor(kw0_r / two_pi)
    exp_arr = xp.exp(xp.asarray(-kwa0) * r) * (xp.cos(phase_mod) + xp.asarray(1j) * xp.sin(phase_mod))

    dkw = 2.0 * pi * df / c
    dkwa = alpha_db / _NEPER_TO_DB * df / 1e6 * 1e2
    exp_df = xp.exp(xp.asarray(-dkwa + 1j * dkw) * r)

    # Incorporate obliquity / sqrt(r) (2D, no elevation)
    exp_arr = exp_arr * obli_fac / xp.sqrt(r)

    return exp_arr, exp_df


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
    xe, ze, theta_e, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, xp)

    # _pfield_core returns Array when _is_simus=False
    result = _pfield_core(
        x=x,
        z=z,
        delays=delays,
        xe=xe,
        ze=ze,
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
    xe: Array,
    ze: Array,
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
    c = medium.speed_of_sound
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
    nx = prod(x.shape)

    # Early return for empty grid
    if nx == 0:
        if _is_simus:
            empty = xp.zeros((0,), dtype=xp.float64)
            return empty, empty, empty
        return xp.zeros((0,), dtype=xp.float64)

    # Flatten to 1D
    x_flat = xp.reshape(x, (-1,))
    z_flat = xp.reshape(z, (-1,))

    # TX apodization
    if tx_apodization is None:
        apod = xp.ones(n_elements, dtype=xp.float64)
    else:
        apod = xp.asarray(tx_apodization, dtype=xp.float64)
        if apod.ndim != 1 or apod.shape[0] != n_elements:
            msg = f"tx_apodization must have shape ({n_elements},), got {apod.shape}"
            raise ValueError(msg)

    # Zero apodization where delays are NaN; replace NaN delays with 0
    nan_mask = xp.isnan(delays)
    apod = xp.where(nan_mask, xp.asarray(0.0), apod)
    delays_clean = xp.where(nan_mask, xp.asarray(0.0), delays)

    # Element splitting
    if element_splitting is not None:
        n_sub = element_splitting
    else:
        lambda_min = c / (fc * (1.0 + bandwidth / 2.0))
        n_sub = ceil(element_width / lambda_min)

    # Element positions (already computed, passed in)
    # xe, ze, theta_e are 1-D arrays with shape (n_elements,)
    if theta_e is None:
        the_1d = xp.zeros(n_elements, dtype=xp.float64)
    else:
        the_1d = theta_e

    h = float(apex_offset)

    # Sub-element centroids (shape: n_elements, n_sub)
    xi, zi = _subelement_centroids(element_width, n_sub, the_1d, xp)

    # Out-of-field mask
    is_out = z_flat < 0
    if radius_of_curvature != inf:
        is_out = is_out | ((x_flat**2 + (z_flat + h) ** 2) <= radius_of_curvature**2)

    # Distances and angles (shape: nx, n_elements, n_sub)
    r, sin_theta, theta_arr = _distances_and_angles(x_flat, z_flat, xi, zi, xe, ze, the_1d, c, fc, xp)

    # Frequency step
    if _is_simus and _simus_ctx is not None and _simus_ctx.df is not None:
        df = _simus_ctx.df
    else:
        df = 1.0 / (float(xp.max(r)) / c + float(xp.max(delays_clean)))
        df = frequency_step * df

    # Select frequencies
    freq_plan = _select_frequencies(fc, bandwidth, tx_n_wavelengths, db_thresh, df, xp)
    f_sel = freq_plan.f_sel
    n_sampling = f_sel.shape[0]
    pulse_spect = freq_plan.pulse_spect
    probe_spect = freq_plan.probe_spect
    idx_out = freq_plan.freq_mask
    df = freq_plan.df

    # Initialization
    rp_accum = xp.asarray(0.0, dtype=xp.float64)
    if _is_simus:
        spect_rows = []  # Accumulate rows in a list, stack at end

    # Obliquity factor
    obli_fac = _obliquity_factor(theta_arr, baffle, xp)

    # Exponential arrays
    f0 = float(f_sel[0])
    exp_arr, exp_df = _init_exponentials(f0, c, alpha_db, r, obli_fac, df, xp)

    # Simplified directivity (center-frequency only)
    if not full_frequency_directivity:
        kc = 2.0 * pi * fc / c
        seg_length = element_width / n_sub
        # Use unnormalized sinc: sinc(x/pi) from array_api_extra
        sinc_arg = xp.asarray(kc * seg_length / 2.0) * sin_theta / pi
        # array-api-extra does not have type interoperability
        dir_arr = xpx.sinc(sinc_arg, xp=xp)  # ty: ignore[invalid-argument-type]
        exp_arr = exp_arr * cast(Array, dir_arr)

    # Frequency loop
    for k in range(n_sampling):
        fk = float(f_sel[k])
        kw = 2.0 * pi * fk / c

        if k > 0:
            exp_arr = exp_arr * exp_df

        # Directivity (frequency-dependent path)
        if full_frequency_directivity:
            # Use unnormalized sinc: sinc(x/pi) from array_api_extra
            sinc_arg_k = xp.asarray(kw * seg_length / 2.0) * sin_theta / pi
            dir_k = xpx.sinc(sinc_arg_k, xp=xp)  # type: ignore[arg-type]

        # Single-element radiation patterns: average over sub-elements
        if full_frequency_directivity:
            rp_mono = xp.mean(dir_k * exp_arr, axis=-1)  # ty: ignore[unsupported-operator]
        elif n_sub > 1:
            rp_mono = xp.mean(exp_arr, axis=-1)
        else:
            # n_sub == 1, squeeze last dimension
            rp_mono = xp.reshape(exp_arr, (exp_arr.shape[0], exp_arr.shape[1]))

        # Transmit delays + apodization
        # delays_clean is 1-D (n_elements,), apply phase shift
        delay_exp = xp.exp(xp.asarray(1j * kw * c) * delays_clean)
        delapod = delay_exp * apod  # Element-wise, shape (n_elements,)

        # Sum across elements: rp_mono (nx, n_elements), delapod (n_elements,)
        rp_k = rp_mono @ xp.reshape(delapod, (-1, 1))

        # Apply spectrum
        rp_k = pulse_spect[k] * rp_k * probe_spect[k]

        # Zero out-of-field (no mutation)
        rp_k = xp.where(is_out[:, None], xp.asarray(0.0 + 0j), rp_k)

        # Accumulate
        if _is_simus:
            if _simus_ctx is None or _simus_ctx.rc is None:
                msg = "_simus_ctx.rc must be provided when _is_simus=True"
                raise ValueError(msg)
            spect_k = probe_spect[k]
            rp_flat = xp.reshape(rp_k, (-1,))
            rc_flat = xp.reshape(_simus_ctx.rc, (-1,))
            weighted = xp.reshape(rp_flat * rc_flat, (1, -1))
            row = spect_k * xp.reshape(weighted @ rp_mono, (-1,))
            if _simus_ctx.rx_delay is not None:
                rx_exp = xp.exp(xp.asarray(1j * kw * c) * _simus_ctx.rx_delay)
                row = row * rx_exp
            spect_rows.append(row)
        else:
            rp_accum = rp_accum + xp.abs(rp_k) ** 2

    # Correcting factor
    cor_fac = 1.0 if tx_n_wavelengths == float("inf") else df
    cor_fac = cor_fac * element_width

    if _is_simus:
        # Stack accumulated rows into spect array
        spect = xp.stack(spect_rows, axis=0) * cor_fac
        rp_accum = rp_accum * cor_fac
        return rp_accum, spect, idx_out

    # RMS pressure, reshape to original grid shape
    rp = xp.sqrt(rp_accum * cor_fac)
    return xp.reshape(rp, original_shape)


def _first_last_true(xp: _ArrayNamespace, mask: Array) -> tuple[int, int]:
    """Find first and last True index in 1D boolean array."""
    indices = xp.nonzero(mask)[0]
    if indices.shape[0] == 0:
        return 0, 0
    return int(indices[0]), int(indices[-1])
