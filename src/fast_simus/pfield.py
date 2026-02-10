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

import math
from math import ceil, inf, pi
from typing import TYPE_CHECKING

import numpy as np
from array_api_compat import array_namespace

from fast_simus.medium_params import MediumParams
from fast_simus.spectrum import mysinc, probe_spectrum_fn, pulse_spectrum_fn
from fast_simus.transducer_params import BaffleType, TransducerParams
from fast_simus.utils._array_api import Array
from fast_simus.utils.geometry import element_positions

if TYPE_CHECKING:
    from fast_simus.utils._array_api import _ArrayNamespace

_DEFAULT_MEDIUM = MediumParams()

# Single-precision machine epsilon
_EPS_SINGLE: float = 1.1920928955078125e-07


def pfield(
    x: Array,
    z: Array,
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
        x: Lateral positions in meters. Shape ``(*grid_shape,)``.
        z: Axial positions in meters. Shape ``(*grid_shape,)``.
            Must have the same shape as ``x``. Positive z = into tissue.
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
            Must be non-negative.
        params: Transducer parameters (geometry, frequency, bandwidth, baffle).
        medium: Medium parameters (speed of sound, attenuation).
            Defaults to soft tissue (1540 m/s, 0 attenuation).
        tx_apodization: Transmit apodization weights. Shape ``(n_elements,)``.
            Defaults to uniform (all ones). Elements with NaN delays are
            automatically zeroed.
        tx_n_wavelengths: Number of wavelengths in the TX pulse.
            Defaults to 1.0.
        db_thresh: Threshold in dB for frequency component selection.
            Only components above this threshold (relative to peak) are used.
            Lower values are more accurate but slower. Defaults to -60.
        full_frequency_directivity: If True, compute element directivity at
            every frequency. If False (default), use center-frequency-only
            directivity (faster, accurate except very near the array).
        element_splitting: Number of sub-elements per transducer element.
            If None (default), computed automatically as
            ceil(element_width / smallest_wavelength).
        frequency_step: Scaling factor for the frequency step.
            Values > 1 speed up computation; values < 1 give smoother results.
            Defaults to 1.0.

    Returns:
        RMS pressure field with same shape as input ``x`` and ``z``.
    """
    return _pfield_core(
        x=x,
        z=z,
        delays=delays,
        params=params,
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


def _pfield_core(
    x: Array,
    z: Array,
    delays: Array,
    params: TransducerParams,
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
    xp = array_namespace(x, z, delays)

    # --- Extract parameters ---
    fc = params.freq_center
    n_elements = params.n_elements
    element_width = params.element_width
    radius_of_curvature = params.radius
    c = medium.speed_of_sound
    alpha_db = medium.attenuation

    # --- Validate inputs ---
    if x.shape != z.shape:
        msg = "x and z must have the same shape"
        raise ValueError(msg)

    # Store original shape for output reshaping
    siz0 = x.shape if x.ndim > 1 else (x.shape[0], 1)
    nx = 1
    for s in x.shape:
        nx *= s

    # Flatten to 1D, cast to float32 (matching PyMUST precision)
    x_flat = _flatten_f32(xp, x)
    z_flat = _flatten_f32(xp, z)

    # --- Delays: ensure 2D (1, n_elements), handle NaN ---
    delays_f32 = xp.asarray(delays, dtype=xp.float32)
    if delays_f32.ndim == 1:
        delays_2d = xp.reshape(delays_f32, (1, -1))
    else:
        delays_2d = delays_f32

    if delays_2d.shape[-1] != n_elements:
        msg = f"delays has {delays_2d.shape[-1]} elements, expected {n_elements}"
        raise ValueError(msg)

    # --- TX apodization ---
    if tx_apodization is None:
        apod = xp.ones(n_elements, dtype=xp.float32)
    else:
        apod = xp.reshape(xp.asarray(tx_apodization, dtype=xp.float32), (-1,))
        if apod.shape[0] != n_elements:
            msg = f"tx_apodization has {apod.shape[0]} elements, expected {n_elements}"
            raise ValueError(msg)

    # Zero apodization where delays are NaN; replace NaN delays with 0
    nan_mask = _isnan(xp, delays_2d)
    nan_any = _any_axis0(xp, nan_mask)
    apod = xp.where(nan_any, xp.asarray(0.0, dtype=xp.float32), apod)
    delays_2d = xp.where(nan_mask, xp.asarray(0.0, dtype=xp.float32), delays_2d)

    # --- Baffle type ---
    baffle = params.baffle
    if baffle == BaffleType.RIGID:
        non_rigid_baffle = False
    elif baffle == BaffleType.SOFT:
        non_rigid_baffle = True
    else:
        non_rigid_baffle = True

    # --- Element splitting ---
    if element_splitting is not None:
        n_sub = element_splitting
    else:
        bandwidth_pct = params.bandwidth * 100.0
        lambda_min = c / (fc * (1.0 + bandwidth_pct / 200.0))
        n_sub = ceil(element_width / lambda_min)

    # --- simus early return ---
    if _is_simus and nx == 0:
        empty = xp.zeros((0,), dtype=xp.float32)
        return empty, empty, empty

    # --- Element positions ---
    xe, ze, theta_e, apex_offset = element_positions(
        n_elements, params.pitch, radius_of_curvature, xp
    )
    xe = xp.reshape(xp.asarray(xe, dtype=xp.float32), (1, -1))
    ze = xp.reshape(xp.asarray(ze, dtype=xp.float32), (1, -1))

    if theta_e is None:
        the_2d = xp.zeros((1, n_elements), dtype=xp.float32)
    else:
        the_2d = xp.reshape(xp.asarray(theta_e, dtype=xp.float32), (1, -1))

    h = float(apex_offset)

    # --- Sub-element centroids (shape: 1, n_elements, n_sub) ---
    seg_length = element_width / n_sub
    seg_offsets = xp.asarray(
        [-element_width / 2.0 + seg_length / 2.0 + i * seg_length for i in range(n_sub)],
        dtype=xp.float32,
    )
    seg_3d = xp.reshape(seg_offsets, (1, 1, n_sub))
    cos_the = xp.cos(the_2d)[:, :, None]  # (1, N, 1)
    sin_neg_the = xp.sin(-the_2d)[:, :, None]
    xi = seg_3d * cos_the
    zi = seg_3d * sin_neg_the

    # --- Out-of-field mask ---
    is_out = z_flat < 0
    if radius_of_curvature != inf:
        is_out = is_out | (
            (x_flat**2 + (z_flat + xp.asarray(h, dtype=xp.float32)) ** 2)
            <= xp.asarray(radius_of_curvature**2, dtype=xp.float32)
        )

    # --- Distances and angles (shape: nx, n_elements, n_sub) ---
    dxi = x_flat[:, None, None] - xi - xe[:, :, None]
    dz_arr = z_flat[:, None, None] - zi - ze[:, :, None]
    d2 = dxi**2 + dz_arr**2

    # r in float32 (matching PyMUST)
    r = xp.sqrt(xp.asarray(d2, dtype=xp.float32))
    small_d = xp.asarray(c / fc / 2.0, dtype=xp.float32)
    r = xp.where(r < small_d, small_d, r)

    # Angle relative to element normal
    eps_sp = xp.asarray(_EPS_SINGLE, dtype=xp.float32)
    sqrt_d2 = xp.sqrt(xp.asarray(d2, dtype=xp.float32))
    theta_arr = xp.asin((dxi + eps_sp) / (sqrt_d2 + eps_sp)) - the_2d[:, :, None]
    sin_theta = xp.sin(theta_arr)

    # --- Spectrum functions ---
    pulse_fn = pulse_spectrum_fn(fc, tx_n_wavelengths)
    probe_fn = probe_spectrum_fn(fc, params.bandwidth)

    # --- Frequency step ---
    if _is_simus and _simus_df is not None:
        df = _simus_df
    else:
        df = 1.0 / (float(xp.max(r)) / c + float(xp.max(delays_2d)))
        df = frequency_step * df

    # --- Frequency samples ---
    n_freq = int(2 * ceil(fc / df) + 1)
    f = xp.linspace(0, 2 * fc, n_freq, dtype=xp.float64)
    df = float(f[1])

    # Keep only significant components (dB threshold)
    w_all = xp.asarray(2.0 * pi) * f
    s_mag = xp.abs(pulse_fn(w_all) * probe_fn(w_all))
    g_db = 20.0 * _log10(xp, xp.asarray(1e-200) + s_mag / xp.max(s_mag))
    above = g_db > db_thresh
    idx_first, idx_last = _first_last_true(xp, above)
    all_indices = xp.arange(f.shape[0])
    freq_mask = (all_indices >= idx_first) & (all_indices <= idx_last)
    idx_out = freq_mask

    f_sel = _masked_select(xp, f, freq_mask)
    n_sampling = f_sel.shape[0]

    w_f = xp.asarray(2.0 * pi) * f_sel
    pulse_spect = pulse_fn(w_f)
    probe_spect = probe_fn(w_f)

    # --- Initialization ---
    rp_accum = xp.asarray(0.0, dtype=xp.float64)
    if _is_simus:
        spect = xp.zeros((n_sampling, n_elements), dtype=xp.complex64)
    else:
        spect = xp.zeros((n_sampling, nx), dtype=xp.complex64)

    # --- Obliquity factor ---
    if non_rigid_baffle:
        if baffle == BaffleType.SOFT:
            obli_fac = xp.cos(theta_arr)
        else:
            cos_th = xp.cos(theta_arr)
            obli_fac = cos_th / (cos_th + float(baffle))
    else:
        obli_fac = xp.ones(theta_arr.shape, dtype=xp.float32)

    obli_fac = xp.where(
        xp.abs(theta_arr) >= xp.asarray(pi / 2, dtype=xp.float32),
        eps_sp,
        obli_fac,
    )

    # --- Exponential arrays (complex64 matching PyMUST) ---
    f0 = float(f_sel[0])
    kw0 = 2.0 * pi * f0 / c
    kwa0 = alpha_db / 8.69 * f0 / 1e6 * 1e2

    # exp(-kwa*r + 1j*mod(kw*r, 2pi)) then cast to complex64
    r_f64 = xp.asarray(r, dtype=xp.float64)
    kw0_r = xp.asarray(kw0) * r_f64
    two_pi = xp.asarray(2.0 * pi)
    phase_mod = kw0_r - two_pi * xp.floor(kw0_r / two_pi)
    exp_arr = _to_complex64(
        xp,
        xp.exp(xp.asarray(-kwa0) * r_f64) * (xp.cos(phase_mod) + xp.asarray(1j) * xp.sin(phase_mod)),
    )

    dkw = 2.0 * pi * df / c
    dkwa = alpha_db / 8.69 * df / 1e6 * 1e2
    exp_df = _to_complex64(
        xp,
        xp.exp(xp.asarray(-dkwa + 1j * dkw) * r_f64),
    )

    # Incorporate obliquity / sqrt(r) (2D, no elevation)
    exp_arr = exp_arr * xp.asarray(obli_fac, dtype=xp.complex64) / xp.asarray(xp.sqrt(r), dtype=xp.complex64)

    # --- Simplified directivity (center-frequency only) ---
    if not full_frequency_directivity:
        kc = 2.0 * pi * fc / c
        dir_arr = mysinc(xp.asarray(kc * seg_length / 2.0) * xp.asarray(sin_theta, dtype=xp.float64))
        exp_arr = exp_arr * xp.asarray(dir_arr, dtype=xp.complex64)

    # --- Frequency loop ---
    exp_arr = _to_complex64(xp, exp_arr)

    for k in range(n_sampling):
        fk = float(f_sel[k])
        kw = 2.0 * pi * fk / c

        if k > 0:
            exp_arr = exp_arr * exp_df

        # Directivity (frequency-dependent path)
        if full_frequency_directivity:
            dir_k = mysinc(xp.asarray(kw * seg_length / 2.0) * xp.asarray(sin_theta, dtype=xp.float64))

        # Single-element radiation patterns: average over sub-elements
        if full_frequency_directivity:
            rp_mono = _mean_last(xp, xp.asarray(dir_k, dtype=xp.complex64) * exp_arr)
        elif n_sub > 1:
            rp_mono = _mean_last(xp, exp_arr)
        else:
            rp_mono = exp_arr

        # Remove trailing size-1 dim
        if rp_mono.ndim == 3 and rp_mono.shape[2] == 1:
            rp_mono = xp.reshape(rp_mono, (rp_mono.shape[0], rp_mono.shape[1]))

        # Transmit delays + apodization
        delay_exp = xp.exp(
            xp.asarray(1j * kw * c) * xp.asarray(delays_2d, dtype=xp.complex64)
        )
        delapod = xp.sum(delay_exp, axis=0) * xp.asarray(apod, dtype=xp.complex64)

        # Sum across elements
        rp_k = rp_mono @ xp.reshape(delapod, (-1, 1))

        # Apply spectrum
        rp_k = xp.asarray(pulse_spect[k], dtype=xp.complex64) * rp_k * xp.asarray(probe_spect[k], dtype=xp.complex64)

        # Zero out-of-field (no mutation)
        rp_k = xp.where(is_out[:, None], xp.asarray(0.0 + 0j, dtype=xp.complex64), rp_k)

        # --- Accumulate ---
        if _is_simus:
            if _rc is None:
                msg = "_rc must be provided when _is_simus=True"
                raise ValueError(msg)
            spect_k = xp.asarray(probe_spect[k], dtype=xp.complex64)
            rp_flat = xp.reshape(rp_k, (-1,))
            rc_flat = xp.reshape(xp.asarray(_rc, dtype=xp.complex64), (-1,))
            weighted = xp.reshape(rp_flat * rc_flat, (1, -1))
            row = spect_k * xp.reshape(weighted @ rp_mono, (-1,))
            if _rx_delay is not None:
                rx_exp = xp.exp(xp.asarray(1j * kw * c) * xp.asarray(_rx_delay, dtype=xp.complex64))
                row = row * rx_exp
            spect = _set_row(xp, spect, k, n_sampling, row)
        else:
            rp_accum = rp_accum + xp.abs(rp_k) ** 2
            spect = _set_row(xp, spect, k, n_sampling, xp.reshape(rp_k, (-1,)))

    # --- Correcting factor ---
    cor_fac = 1.0 if tx_n_wavelengths == float("inf") else df
    cor_fac = cor_fac * element_width

    spect = spect * cor_fac
    rp_accum = rp_accum * cor_fac

    if _is_simus:
        return rp_accum, spect, idx_out

    # RMS pressure, reshape to original grid
    rp = xp.sqrt(rp_accum)
    return xp.reshape(rp, siz0)


# ---------------------------------------------------------------------------
# Array API helper functions
# ---------------------------------------------------------------------------


def _flatten_f32(xp: _ArrayNamespace, arr: Array) -> Array:
    """Flatten array to 1D float32."""
    flat = xp.reshape(arr, (-1,)) if arr.ndim > 1 else arr
    return xp.asarray(flat, dtype=xp.float32)


def _to_complex64(xp: _ArrayNamespace, arr: Array) -> Array:
    """Cast to complex64."""
    return xp.asarray(arr, dtype=xp.complex64)


def _isnan(xp: _ArrayNamespace, arr: Array) -> Array:
    """Element-wise NaN check."""
    if hasattr(xp, "isnan"):
        return xp.isnan(arr)
    return arr != arr  # NaN != NaN


def _any_axis0(xp: _ArrayNamespace, arr: Array) -> Array:
    """Reduce boolean along axis 0 with 'any'."""
    if hasattr(xp, "any"):
        return xp.any(arr, axis=0)
    return xp.sum(xp.asarray(arr, dtype=xp.int32), axis=0) > 0


def _log10(xp: _ArrayNamespace, arr: Array) -> Array:
    """Log base 10."""
    if hasattr(xp, "log10"):
        return xp.log10(arr)
    return xp.log(arr) / math.log(10)


def _first_last_true(xp: _ArrayNamespace, mask: Array) -> tuple[int, int]:
    """Find first and last True index in 1D boolean array."""
    m = np.asarray(mask) if hasattr(mask, "__array__") else np.array(mask)
    indices = np.where(m)[0]
    if len(indices) == 0:
        return 0, 0
    return int(indices[0]), int(indices[-1])


def _masked_select(xp: _ArrayNamespace, arr: Array, mask: Array) -> Array:
    """Select elements where mask is True."""
    a = np.asarray(arr) if hasattr(arr, "__array__") else np.array(arr)
    m = np.asarray(mask) if hasattr(mask, "__array__") else np.array(mask)
    return xp.asarray(a[m], dtype=arr.dtype)


def _mean_last(xp: _ArrayNamespace, arr: Array) -> Array:
    """Mean along last axis."""
    if hasattr(xp, "mean"):
        return xp.mean(arr, axis=-1)
    return xp.sum(arr, axis=-1) / arr.shape[-1]


def _set_row(xp: _ArrayNamespace, arr: Array, idx: int, n_rows: int, row: Array) -> Array:
    """Set row idx in 2D array without mutation."""
    indices = xp.arange(n_rows)
    mask = (indices == idx)[:, None]
    row_2d = xp.reshape(xp.asarray(row, dtype=arr.dtype), (1, -1))
    return xp.where(mask, xp.broadcast_to(row_2d, arr.shape), arr)
