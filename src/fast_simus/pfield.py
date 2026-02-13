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

from math import ceil, inf, pi
from typing import cast

import array_api_extra as xpx

from fast_simus.medium_params import MediumParams
from fast_simus.spectrum import probe_spectrum, pulse_spectrum
from fast_simus.transducer_params import BaffleType, TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace, array_namespace
from fast_simus.utils.geometry import element_positions

_DEFAULT_MEDIUM = MediumParams()


def pfield(
    positions: Array,
    delays: Array,
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Array | None = None,
    tx_n_wavelengths: float = 1.0,
    db_thresh: float = -60.0,
    full_frequency_directivity: bool = False,
    element_splitting: int | None = None,
    frequency_step: float = 1.0,
) -> Array:
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
        _rc=None,
        _rx_delay=None,
        _simus_df=None,
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
    _rc: Array | None,
    _rx_delay: Array | None,
    _simus_df: float | None,
) -> Array | tuple[Array, Array, Array]:
    """Core pfield implementation shared by standalone pfield and simus."""
    xp: _ArrayNamespace = array_namespace(x, z, delays)  # type: ignore[assignment]

    # --- Extract medium parameters ---
    c = medium.speed_of_sound
    alpha_db = medium.attenuation

    # --- Validate inputs ---
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
    nx = 1
    for s in x.shape:
        nx *= s

    # Early return for empty grid
    if nx == 0:
        if _is_simus:
            empty = xp.zeros((0,), dtype=xp.float64)
            return empty, empty, empty
        return xp.zeros((0,), dtype=xp.float64)

    # Flatten to 1D
    x_flat = xp.reshape(x, (-1,)) if x.ndim > 1 else x
    z_flat = xp.reshape(z, (-1,)) if z.ndim > 1 else z

    # --- TX apodization ---
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

    # --- Baffle type ---
    non_rigid_baffle = baffle != BaffleType.RIGID

    # --- Element splitting ---
    if element_splitting is not None:
        n_sub = element_splitting
    else:
        lambda_min = c / (fc * (1.0 + bandwidth / 2.0))
        n_sub = ceil(element_width / lambda_min)

    # --- Element positions (already computed, passed in) ---
    # xe, ze, theta_e are 1-D arrays with shape (n_elements,)
    if theta_e is None:
        the_1d = xp.zeros(n_elements, dtype=xp.float64)
    else:
        the_1d = theta_e

    h = float(apex_offset)

    # --- Sub-element centroids (shape: n_elements, n_sub) ---
    seg_length = element_width / n_sub
    seg_offsets = xp.asarray(
        [-element_width / 2.0 + seg_length / 2.0 + i * seg_length for i in range(n_sub)],
        dtype=xp.float64,
    )
    # Broadcasting: (n_sub,) -> (1, n_sub), (n_elements,) -> (n_elements, 1)
    seg_2d = xp.reshape(seg_offsets, (1, n_sub))
    cos_the = xp.cos(the_1d)[:, None]  # (n_elements, 1)
    sin_neg_the = xp.sin(-the_1d)[:, None]
    xi = seg_2d * cos_the  # (n_elements, n_sub)
    zi = seg_2d * sin_neg_the

    # --- Out-of-field mask ---
    is_out = z_flat < 0
    if radius_of_curvature != inf:
        is_out = is_out | ((x_flat**2 + (z_flat + h) ** 2) <= radius_of_curvature**2)

    # --- Distances and angles (shape: nx, n_elements, n_sub) ---
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
    # Broadcasting: the_1d (n_elements,) -> (1, n_elements, 1)
    theta_arr = xp.asin((dxi + eps_sp) / (sqrt_d2 + eps_sp)) - the_1d[None, :, None]
    sin_theta = xp.sin(theta_arr)

    # --- Frequency step ---
    if _is_simus and _simus_df is not None:
        df = _simus_df
    else:
        df = 1.0 / (float(xp.max(r)) / c + float(xp.max(delays_clean)))
        df = frequency_step * df

    # --- Frequency samples ---
    n_freq = int(2 * ceil(fc / df) + 1)
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
    idx_out = freq_mask

    f_sel = f[freq_mask]
    n_sampling = f_sel.shape[0]

    w_f = xp.asarray(2.0 * pi) * f_sel
    pulse_spect = pulse_spectrum(w_f, fc, tx_n_wavelengths)
    probe_spect = probe_spectrum(w_f, fc, bandwidth)

    # --- Initialization ---
    rp_accum = xp.asarray(0.0, dtype=xp.float64)
    if _is_simus:
        spect_rows = []  # Accumulate rows in a list, stack at end

    # --- Obliquity factor ---
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

    # --- Exponential arrays ---
    f0 = float(f_sel[0])
    kw0 = 2.0 * pi * f0 / c
    kwa0 = alpha_db / 8.69 * f0 / 1e6 * 1e2

    # exp(-kwa*r + 1j*mod(kw*r, 2pi))
    kw0_r = xp.asarray(kw0) * r
    two_pi = xp.asarray(2.0 * pi)
    phase_mod = kw0_r - two_pi * xp.floor(kw0_r / two_pi)
    exp_arr = xp.exp(xp.asarray(-kwa0) * r) * (xp.cos(phase_mod) + xp.asarray(1j) * xp.sin(phase_mod))

    dkw = 2.0 * pi * df / c
    dkwa = alpha_db / 8.69 * df / 1e6 * 1e2
    exp_df = xp.exp(xp.asarray(-dkwa + 1j * dkw) * r)

    # Incorporate obliquity / sqrt(r) (2D, no elevation)
    exp_arr = exp_arr * obli_fac / xp.sqrt(r)

    # --- Simplified directivity (center-frequency only) ---
    if not full_frequency_directivity:
        kc = 2.0 * pi * fc / c
        # Use unnormalized sinc: sinc(x/pi) from array_api_extra
        sinc_arg = xp.asarray(kc * seg_length / 2.0) * sin_theta / pi
        # array-api-extra does not have type interoperability
        dir_arr = xpx.sinc(sinc_arg, xp=xp)  # ty: ignore[invalid-argument-type]
        exp_arr = exp_arr * cast(Array, dir_arr)

    # --- Frequency loop ---

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

        # --- Accumulate ---
        if _is_simus:
            if _rc is None:
                msg = "_rc must be provided when _is_simus=True"
                raise ValueError(msg)
            spect_k = probe_spect[k]
            rp_flat = xp.reshape(rp_k, (-1,))
            rc_flat = xp.reshape(_rc, (-1,))
            weighted = xp.reshape(rp_flat * rc_flat, (1, -1))
            row = spect_k * xp.reshape(weighted @ rp_mono, (-1,))
            if _rx_delay is not None:
                rx_exp = xp.exp(xp.asarray(1j * kw * c) * _rx_delay)
                row = row * rx_exp
            spect_rows.append(row)
        else:
            rp_accum = rp_accum + xp.abs(rp_k) ** 2

    # --- Correcting factor ---
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
