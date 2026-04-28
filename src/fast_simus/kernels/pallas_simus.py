"""Pallas GPU kernel for simus RF spectrum on NVIDIA GPUs (SM 70+, Triton backend).

Architecture (v2 -- fused TX beamforming):
  TX kernel:  grid (n_scat_tiles,).
    Carries phase[tile_s, n_elem] as fori_loop state over n_freq frequencies.
    At each step: beamform (sum over elements), apply pulse*probe + rc + mask,
    write p_weighted[tile_s] to output column.
    Single compilation, single launch.

  RX kernel:  grid (n_scat_tiles, n_elem).
    For each (tile, receiving element), reads p_weighted from TX output,
    combines with RX geometric progression, reduces over scatterers.
    Single compilation, single launch.

  Host accumulation:  sum RX partial spectra over scatterer tiles.

Phase computation uses geometric progression (ALU-only inner loop, O(1) SFU
per element per scatterer per kernel).

Supports n_sub=1 (single sub-element per element, most common case).
"""

from __future__ import annotations

import os
from math import inf, pi
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils.geometry import element_positions

if TYPE_CHECKING:
    from fast_simus.simus import SimusPlan

os.environ.setdefault("JAX_PALLAS_USE_MOSAIC_GPU", "0")

_NEPER_TO_DB = 8.685889638065036


def _pad_to(n: int, tile: int) -> int:
    return ((n + tile - 1) // tile) * tile


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_geometry(
    params: TransducerParams,
    plan: "SimusPlan",
    medium: MediumParams,
    delays_clean: jax.Array,
    tx_apodization: jax.Array,
) -> dict:
    n_elem = params.n_elements
    n_sub = plan.n_sub
    n_es = n_elem * n_sub
    n_freq = int(plan.selected_freqs.shape[0])

    element_pos, theta_raw, apex_offset = element_positions(
        n_elem, params.pitch, params.radius, np,
    )
    if theta_raw is None:
        theta_raw = np.zeros(n_elem, dtype=np.float32)

    elem_x = np.ascontiguousarray(element_pos[:, 0], dtype=np.float32)
    elem_z = np.ascontiguousarray(element_pos[:, 1], dtype=np.float32)

    seg_length = params.element_width / n_sub
    seg_offsets = np.array(
        [-params.element_width / 2.0 + seg_length / 2.0 + i * seg_length
         for i in range(n_sub)], dtype=np.float32,
    )
    cos_th = np.cos(theta_raw).astype(np.float32)
    sin_neg_th = np.sin(-theta_raw).astype(np.float32)
    sub_dx = np.zeros(n_es, dtype=np.float32)
    sub_dz = np.zeros(n_es, dtype=np.float32)
    for e in range(n_elem):
        for s in range(n_sub):
            sub_dx[e * n_sub + s] = seg_offsets[s] * cos_th[e]
            sub_dz[e * n_sub + s] = seg_offsets[s] * sin_neg_th[e]

    freq_start = float(plan.selected_freqs[0])
    freq_step = (
        float(plan.selected_freqs[1] - plan.selected_freqs[0])
        if n_freq > 1 else 0.0
    )
    c = medium.speed_of_sound
    attenuation = medium.attenuation
    fc = params.freq_center

    delays_np = np.asarray(delays_clean, dtype=np.float32)
    tx_apod_np = np.asarray(tx_apodization, dtype=np.float32)

    spectra = np.asarray(plan.pulse_spectrum * plan.probe_spectrum)
    pp_re = np.real(spectra).astype(np.float32)
    pp_im = np.imag(spectra).astype(np.float32)
    probe_raw = np.asarray(plan.probe_spectrum)
    probe = (
        np.abs(probe_raw).astype(np.float32)
        if np.iscomplexobj(probe_raw)
        else probe_raw.astype(np.float32)
    )

    radius_val = params.radius if params.radius != inf else 1e31

    return {
        "elem_x": elem_x, "elem_z": elem_z,
        "cos_te": cos_th, "sin_neg_te": sin_neg_th,
        "sub_dx": sub_dx, "sub_dz": sub_dz,
        "delays": delays_np, "tx_apod": tx_apod_np,
        "pp_re": jnp.asarray(pp_re), "pp_im": jnp.asarray(pp_im),
        "probe": jnp.asarray(probe),
        "n_elem": n_elem, "n_sub": n_sub, "n_es": n_es, "n_freq": n_freq,
        "freq_start": freq_start, "freq_step": freq_step,
        "kw_init": float(2.0 * pi * freq_start / c),
        "alpha_init": float(attenuation / _NEPER_TO_DB * freq_start / 1e6 * 1e2),
        "kw_step": float(2.0 * pi * freq_step / c),
        "alpha_step": float(attenuation / _NEPER_TO_DB * freq_step / 1e6 * 1e2),
        "min_dist": float(c / fc / 2.0),
        "seg_length": float(seg_length),
        "center_kw": float(2.0 * pi * fc / c),
        "inv_nsub": float(1.0 / n_sub),
        "radius": float(radius_val),
        "apex_offset": float(apex_offset),
    }


# ---------------------------------------------------------------------------
# TX beamform kernel: 2D carry [tile_s, n_elem] for all-element phase state
# ---------------------------------------------------------------------------

def _make_tx_beamform_kernel(
    *, n_freq_padded: int,
    kw_init: float, alpha_init: float,
    kw_step: float, alpha_step: float,
    min_dist: float, seg_length: float,
    center_kw: float, inv_nsub: float,
    freq_start: float, freq_step: float,
):
    """TX beamform kernel processing ALL elements simultaneously.

    Carry: phase[tile_s, n_elem] -- geometric progression for all elements.
    Output: p_weighted[tile_s, n_freq_padded] -- beamformed, weighted TX pressure.
    """

    def kernel(
        sx_ref, sz_ref, rc_ref, is_out_ref,
        all_ex_ref, all_ez_ref, all_ct_ref, all_snt_ref,
        da_re0_ref, da_im0_ref, das_re_ref, das_im_ref,
        pp_re_ref, pp_im_ref,
        out_re_ref, out_im_ref,
    ):
        sx = sx_ref[:]          # (tile_s,)
        sz = sz_ref[:]
        rc = rc_ref[:]
        is_out = is_out_ref[:]
        all_ex = all_ex_ref[:]  # (n_elem,)
        all_ez = all_ez_ref[:]
        all_ct = all_ct_ref[:]
        all_snt = all_snt_ref[:]

        # Distance from each scatterer to each element: (tile_s, n_elem)
        dx = sx[:, None] - all_ex[None, :]
        dz = sz[:, None] - all_ez[None, :]
        r2 = dx * dx + dz * dz
        inv_r = jax.lax.rsqrt(r2 + 1e-30)
        r_val = r2 * inv_r
        rc_clamp = jnp.maximum(r_val, min_dist)

        # Obliquity and sinc directivity
        sin_th = (dx * all_ct[None, :] + dz * all_snt[None, :]) * inv_r
        cos_th_val = (dz * all_ct[None, :] - dx * all_snt[None, :]) * inv_r
        obliq = jnp.where(cos_th_val <= 0.0, 1e-16, cos_th_val)
        sa = center_kw * seg_length * 0.5 * sin_th
        sv_sinc = jnp.where(jnp.abs(sa) < 1e-8, 1.0, jnp.sin(sa) / sa)
        amp = obliq * sv_sinc * jax.lax.rsqrt(rc_clamp) * inv_nsub

        # Initial TX phase
        kr = kw_init * rc_clamp
        ar = alpha_init * rc_clamp
        a0 = amp * jnp.exp(-ar)
        ph_re_init = a0 * jnp.cos(kr)    # (tile_s, n_elem)
        ph_im_init = a0 * jnp.sin(kr)

        # Phase step (constant across freq loop)
        kr_s = kw_step * rc_clamp
        ar_s = alpha_step * rc_clamp
        stp_mag = jnp.exp(-ar_s)
        step_re = stp_mag * jnp.cos(kr_s)
        step_im = stp_mag * jnp.sin(kr_s)

        # Delay+apodization initial and step
        da_re0 = da_re0_ref[:]   # (n_elem,)
        da_im0 = da_im0_ref[:]
        das_re = das_re_ref[:]
        das_im = das_im_ref[:]

        mask = 1.0 - is_out      # (tile_s,) -- 0 for out-of-field

        def body(f, carry):
            ph_re, ph_im, d_re, d_im = carry

            # TX beamform: p[s] = sum_e(phase[s,e] * delay_apod[e])
            p_elem_re = ph_re * d_re[None, :] - ph_im * d_im[None, :]
            p_elem_im = ph_re * d_im[None, :] + ph_im * d_re[None, :]
            p_re = jnp.sum(p_elem_re, axis=1)   # (tile_s,)
            p_im = jnp.sum(p_elem_im, axis=1)

            # Apply pulse*probe spectrum weight
            pp_r = pp_re_ref[f]
            pp_i = pp_im_ref[f]
            pw_re = (pp_r * p_re - pp_i * p_im) * rc * mask
            pw_im = (pp_r * p_im + pp_i * p_re) * rc * mask

            out_re_ref[:, f] = pw_re
            out_im_ref[:, f] = pw_im

            # Geometric progression
            new_ph_re = ph_re * step_re - ph_im * step_im
            new_ph_im = ph_re * step_im + ph_im * step_re
            new_d_re = d_re * das_re - d_im * das_im
            new_d_im = d_re * das_im + d_im * das_re

            return new_ph_re, new_ph_im, new_d_re, new_d_im

        jax.lax.fori_loop(
            0, n_freq_padded, body,
            (ph_re_init, ph_im_init, da_re0, da_im0),
        )

    return kernel


# ---------------------------------------------------------------------------
# TX fallback kernel: nested fori_loop (outer=elements, inner=frequencies)
# Uses only 1D carry. Falls back here if 2D carry fails.
# ---------------------------------------------------------------------------

def _make_tx_fallback_kernel(
    *, n_freq_padded: int, n_elem: int,
    kw_init: float, alpha_init: float,
    kw_step: float, alpha_step: float,
    min_dist: float, seg_length: float,
    center_kw: float, inv_nsub: float,
    freq_start: float, freq_step: float,
):
    """TX kernel with nested fori_loop: outer=elements, inner=frequencies.

    Accumulates TX contributions via read-modify-write on output ref.
    Slower (L2-bound) but avoids 2D carry requirement.
    """

    def kernel(
        sx_ref, sz_ref, rc_ref, is_out_ref,
        all_ex_ref, all_ez_ref, all_ct_ref, all_snt_ref,
        da_re0_ref, da_im0_ref, das_re_ref, das_im_ref,
        pp_re_ref, pp_im_ref,
        out_re_ref, out_im_ref,
    ):
        sx = sx_ref[:]
        sz = sz_ref[:]
        rc = rc_ref[:]
        is_out = is_out_ref[:]
        mask = 1.0 - is_out

        # Initialize output to zero
        def zero_body(f, _):
            out_re_ref[:, f] = jnp.zeros_like(sx)
            out_im_ref[:, f] = jnp.zeros_like(sx)
            return None
        jax.lax.fori_loop(0, n_freq_padded, zero_body, None)

        def elem_body(e, _):
            ex = all_ex_ref[e]
            ez = all_ez_ref[e]
            ct = all_ct_ref[e]
            snt = all_snt_ref[e]

            dx = sx - ex
            dz = sz - ez
            r2 = dx * dx + dz * dz
            inv_r = jax.lax.rsqrt(r2 + 1e-30)
            r_val = r2 * inv_r
            rc_clamp = jnp.maximum(r_val, min_dist)

            sin_th = (dx * ct + dz * snt) * inv_r
            cos_th_val = (dz * ct - dx * snt) * inv_r
            obliq = jnp.where(cos_th_val <= 0.0, 1e-16, cos_th_val)
            sa = center_kw * seg_length * 0.5 * sin_th
            sv_sinc = jnp.where(jnp.abs(sa) < 1e-8, 1.0, jnp.sin(sa) / sa)
            amp = obliq * sv_sinc * jax.lax.rsqrt(rc_clamp) * inv_nsub

            kr = kw_init * rc_clamp
            ar = alpha_init * rc_clamp
            a0 = amp * jnp.exp(-ar)
            cv_re = a0 * jnp.cos(kr)
            cv_im = a0 * jnp.sin(kr)

            kr_s = kw_step * rc_clamp
            ar_s = alpha_step * rc_clamp
            stp_mag = jnp.exp(-ar_s)
            sv_re = stp_mag * jnp.cos(kr_s)
            sv_im = stp_mag * jnp.sin(kr_s)

            da_re = da_re0_ref[e]
            da_im = da_im0_ref[e]
            da_sr = das_re_ref[e]
            da_si = das_im_ref[e]

            def freq_body(f, carry):
                cr, ci, dr, di = carry
                # TX contribution for this element at this freq
                c_re = cr * dr - ci * di
                c_im = cr * di + ci * dr
                # Apply pulse*probe, rc, mask and accumulate
                pp_r = pp_re_ref[f]
                pp_i = pp_im_ref[f]
                wr = (pp_r * c_re - pp_i * c_im) * rc * mask
                wi = (pp_r * c_im + pp_i * c_re) * rc * mask
                out_re_ref[:, f] = out_re_ref[:, f] + wr
                out_im_ref[:, f] = out_im_ref[:, f] + wi
                # Geo progression
                new_cr = cr * sv_re - ci * sv_im
                new_ci = cr * sv_im + ci * sv_re
                new_dr = dr * da_sr - di * da_si
                new_di = dr * da_si + di * da_sr
                return new_cr, new_ci, new_dr, new_di

            da_re_bc = jnp.broadcast_to(da_re, cv_re.shape)
            da_im_bc = jnp.broadcast_to(da_im, cv_re.shape)
            jax.lax.fori_loop(
                0, n_freq_padded, freq_body,
                (cv_re, cv_im, da_re_bc, da_im_bc),
            )
            return None

        jax.lax.fori_loop(0, n_elem, elem_body, None)

    return kernel


# ---------------------------------------------------------------------------
# RX kernel: grid (n_scat_tiles, n_elem)
# ---------------------------------------------------------------------------

def _make_rx_kernel(
    *, n_freq_padded: int,
    kw_init: float, alpha_init: float,
    kw_step: float, alpha_step: float,
    min_dist: float, seg_length: float,
    center_kw: float, inv_nsub: float,
):
    """RX kernel for one receiving element per block.

    Grid: (n_scat_tiles, n_elem).
    Reads p_weighted from TX output, combines with RX phase, reduces over
    scatterers to produce partial spectrum per (tile, element).
    """

    def kernel(
        sx_ref, sz_ref,
        p_re_ref, p_im_ref,
        probe_ref,
        ex_ref, ez_ref, ct_ref, snt_ref,
        out_re_ref, out_im_ref,
    ):
        sx = sx_ref[:]     # (tile_s,)
        sz = sz_ref[:]
        ex = ex_ref[0]     # scalar (block selected by grid dim)
        ez = ez_ref[0]
        ct = ct_ref[0]
        snt = snt_ref[0]

        dx = sx - ex
        dz = sz - ez
        r2 = dx * dx + dz * dz
        inv_r = jax.lax.rsqrt(r2 + 1e-30)
        r_val = r2 * inv_r
        rc_clamp = jnp.maximum(r_val, min_dist)

        sin_th = (dx * ct + dz * snt) * inv_r
        cos_th_val = (dz * ct - dx * snt) * inv_r
        obliq = jnp.where(cos_th_val <= 0.0, 1e-16, cos_th_val)
        sa = center_kw * seg_length * 0.5 * sin_th
        sv_sinc = jnp.where(jnp.abs(sa) < 1e-8, 1.0, jnp.sin(sa) / sa)
        amp = obliq * sv_sinc * jax.lax.rsqrt(rc_clamp) * inv_nsub

        kr = kw_init * rc_clamp
        ar = alpha_init * rc_clamp
        a0 = amp * jnp.exp(-ar)
        cv_re = a0 * jnp.cos(kr)
        cv_im = a0 * jnp.sin(kr)

        kr_s = kw_step * rc_clamp
        ar_s = alpha_step * rc_clamp
        stp_mag = jnp.exp(-ar_s)
        sv_re = stp_mag * jnp.cos(kr_s)
        sv_im = stp_mag * jnp.sin(kr_s)

        def body(f, carry):
            cr, ci = carry
            pr = p_re_ref[:, f]
            pi_ = p_im_ref[:, f]

            comb_re = pr * cr - pi_ * ci
            comb_im = pr * ci + pi_ * cr

            out_re_ref[0, 0, f] = jnp.sum(comb_re) * probe_ref[f]
            out_im_ref[0, 0, f] = jnp.sum(comb_im) * probe_ref[f]

            new_cr = cr * sv_re - ci * sv_im
            new_ci = cr * sv_im + ci * sv_re
            return new_cr, new_ci

        jax.lax.fori_loop(0, n_freq_padded, body, (cv_re, cv_im))

    return kernel


# ---------------------------------------------------------------------------
# JIT-compiled compute pipeline builder
# ---------------------------------------------------------------------------

_compute_cache: dict[tuple, callable] = {}


def _build_compute_fn(
    *, tile_s: int, n_tiles: int, scat_padded: int,
    n_elem: int, n_freq_padded: int, n_freq: int,
    use_2d_carry: bool,
    kw_init: float, alpha_init: float,
    kw_step: float, alpha_step: float,
    min_dist: float, seg_length: float,
    center_kw: float, inv_nsub: float,
    freq_start: float, freq_step: float,
) -> callable:
    """Build and JIT-compile the TX->RX->accumulate pipeline.

    Returns a JIT-compiled function that takes pure JAX arrays and returns
    (spect_re, spect_im) of shape (n_elem, n_freq).
    """
    common = dict(
        kw_init=kw_init, alpha_init=alpha_init,
        kw_step=kw_step, alpha_step=alpha_step,
        min_dist=min_dist, seg_length=seg_length,
        center_kw=center_kw, inv_nsub=inv_nsub,
    )

    if use_2d_carry:
        tx_kern = _make_tx_beamform_kernel(
            n_freq_padded=n_freq_padded,
            freq_start=freq_start, freq_step=freq_step,
            **common,
        )
    else:
        tx_kern = _make_tx_fallback_kernel(
            n_freq_padded=n_freq_padded, n_elem=n_elem,
            freq_start=freq_start, freq_step=freq_step,
            **common,
        )

    rx_kern = _make_rx_kernel(n_freq_padded=n_freq_padded, **common)

    tx_call = pl.pallas_call(
        tx_kern,
        out_shape=[
            jax.ShapeDtypeStruct((scat_padded, n_freq_padded), jnp.float32),
            jax.ShapeDtypeStruct((scat_padded, n_freq_padded), jnp.float32),
        ],
        grid=(n_tiles,),
        in_specs=[
            pl.BlockSpec((tile_s,), lambda i: (i,)),
            pl.BlockSpec((tile_s,), lambda i: (i,)),
            pl.BlockSpec((tile_s,), lambda i: (i,)),
            pl.BlockSpec((tile_s,), lambda i: (i,)),
            pl.BlockSpec((n_elem,), lambda i: (0,)),
            pl.BlockSpec((n_elem,), lambda i: (0,)),
            pl.BlockSpec((n_elem,), lambda i: (0,)),
            pl.BlockSpec((n_elem,), lambda i: (0,)),
            pl.BlockSpec((n_elem,), lambda i: (0,)),
            pl.BlockSpec((n_elem,), lambda i: (0,)),
            pl.BlockSpec((n_elem,), lambda i: (0,)),
            pl.BlockSpec((n_elem,), lambda i: (0,)),
            pl.BlockSpec((n_freq_padded,), lambda i: (0,)),
            pl.BlockSpec((n_freq_padded,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((tile_s, n_freq_padded), lambda i: (i, 0)),
            pl.BlockSpec((tile_s, n_freq_padded), lambda i: (i, 0)),
        ],
    )

    rx_call = pl.pallas_call(
        rx_kern,
        out_shape=[
            jax.ShapeDtypeStruct(
                (n_tiles, n_elem, n_freq_padded), jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (n_tiles, n_elem, n_freq_padded), jnp.float32,
            ),
        ],
        grid=(n_tiles, n_elem),
        in_specs=[
            pl.BlockSpec((tile_s,), lambda i, e: (i,)),
            pl.BlockSpec((tile_s,), lambda i, e: (i,)),
            pl.BlockSpec((tile_s, n_freq_padded), lambda i, e: (i, 0)),
            pl.BlockSpec((tile_s, n_freq_padded), lambda i, e: (i, 0)),
            pl.BlockSpec((n_freq_padded,), lambda i, e: (0,)),
            pl.BlockSpec((1,), lambda i, e: (e,)),
            pl.BlockSpec((1,), lambda i, e: (e,)),
            pl.BlockSpec((1,), lambda i, e: (e,)),
            pl.BlockSpec((1,), lambda i, e: (e,)),
        ],
        out_specs=[
            pl.BlockSpec((1, 1, n_freq_padded), lambda i, e: (i, e, 0)),
            pl.BlockSpec((1, 1, n_freq_padded), lambda i, e: (i, e, 0)),
        ],
    )

    @jax.jit
    def compute(
        scat_x, scat_z, rc_f32, is_out_f,
        all_ex, all_ez, all_ct, all_snt,
        da_re0, da_im0, das_re, das_im,
        pp_re, pp_im, probe,
    ):
        tx_re, tx_im = tx_call(
            scat_x, scat_z, rc_f32, is_out_f,
            all_ex, all_ez, all_ct, all_snt,
            da_re0, da_im0, das_re, das_im,
            pp_re, pp_im,
        )
        rx_part_re, rx_part_im = rx_call(
            scat_x, scat_z, tx_re, tx_im, probe,
            all_ex, all_ez, all_ct, all_snt,
        )
        spect_re = jnp.sum(rx_part_re, axis=0)
        spect_im = jnp.sum(rx_part_im, axis=0)
        return spect_re[:, :n_freq], spect_im[:, :n_freq]

    return compute


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simus_pallas(
    scatterers: jax.Array,
    rc: jax.Array,
    params: TransducerParams,
    plan: "SimusPlan",
    medium: MediumParams,
    delays_clean: jax.Array,
    tx_apodization: jax.Array,
    *,
    tile_s: int = 64,
    use_2d_carry: bool = True,
) -> jax.Array:
    """Compute simus RF spectrum via Pallas kernels (Triton backend).

    Two-pass architecture: TX beamform -> RX combine+reduce.
    JIT-compiled: first call compiles, subsequent calls reuse compiled code.

    Returns:
        Complex RF spectrum, shape (n_freq, n_elements).
    """
    geo = _build_geometry(params, plan, medium, delays_clean, tx_apodization)
    n_elem = geo["n_elem"]
    n_freq = geo["n_freq"]
    n_scat = int(scatterers.shape[0])
    n_freq_padded = _next_pow2(n_freq)

    scat_x = scatterers[:, 0].astype(jnp.float32)
    scat_z = scatterers[:, 1].astype(jnp.float32)
    rc_f32 = rc.astype(jnp.float32)

    scat_padded = _pad_to(n_scat, tile_s)
    n_tiles = scat_padded // tile_s

    if scat_padded > n_scat:
        pad = scat_padded - n_scat
        scat_x = jnp.pad(scat_x, (0, pad))
        scat_z = jnp.pad(scat_z, (0, pad), constant_values=-1.0)
        rc_f32 = jnp.pad(rc_f32, (0, pad))

    radius_val = geo["radius"]
    apex_offset = geo["apex_offset"]
    is_out_f = jnp.where(scat_z < 0.0, 1.0, 0.0)
    if radius_val < 1e30:
        is_out_f = jnp.where(
            (scat_x**2 + (scat_z + apex_offset)**2) <= radius_val**2,
            1.0, is_out_f,
        )

    all_ex = jnp.asarray(geo["elem_x"])
    all_ez = jnp.asarray(geo["elem_z"])
    all_ct = jnp.asarray(geo["cos_te"])
    all_snt = jnp.asarray(geo["sin_neg_te"])

    delays_np = geo["delays"]
    tx_apod_np = geo["tx_apod"]
    freq_start = geo["freq_start"]
    freq_step = geo["freq_step"]

    da_re0 = jnp.asarray(
        [a * np.cos(2 * pi * freq_start * d)
         for d, a in zip(delays_np, tx_apod_np)],
        dtype=jnp.float32,
    )
    da_im0 = jnp.asarray(
        [a * np.sin(2 * pi * freq_start * d)
         for d, a in zip(delays_np, tx_apod_np)],
        dtype=jnp.float32,
    )
    das_re = jnp.asarray(
        [np.cos(2 * pi * freq_step * d) for d in delays_np],
        dtype=jnp.float32,
    )
    das_im = jnp.asarray(
        [np.sin(2 * pi * freq_step * d) for d in delays_np],
        dtype=jnp.float32,
    )

    pp_re = jnp.pad(geo["pp_re"], (0, n_freq_padded - n_freq))
    pp_im = jnp.pad(geo["pp_im"], (0, n_freq_padded - n_freq))
    probe = jnp.pad(geo["probe"], (0, n_freq_padded - n_freq))

    # Get or build JIT-compiled compute function (cached by shape+physics)
    cache_key = (
        tile_s, n_tiles, scat_padded, n_elem, n_freq_padded, n_freq,
        use_2d_carry,
        geo["kw_init"], geo["alpha_init"], geo["kw_step"], geo["alpha_step"],
        geo["min_dist"], geo["seg_length"], geo["center_kw"], geo["inv_nsub"],
        freq_start, freq_step,
    )
    if cache_key not in _compute_cache:
        _compute_cache[cache_key] = _build_compute_fn(
            tile_s=tile_s, n_tiles=n_tiles, scat_padded=scat_padded,
            n_elem=n_elem, n_freq_padded=n_freq_padded, n_freq=n_freq,
            use_2d_carry=use_2d_carry,
            kw_init=geo["kw_init"], alpha_init=geo["alpha_init"],
            kw_step=geo["kw_step"], alpha_step=geo["alpha_step"],
            min_dist=geo["min_dist"], seg_length=geo["seg_length"],
            center_kw=geo["center_kw"], inv_nsub=geo["inv_nsub"],
            freq_start=freq_start, freq_step=freq_step,
        )
    compute_fn = _compute_cache[cache_key]

    spect_re, spect_im = compute_fn(
        scat_x, scat_z, rc_f32, is_out_f,
        all_ex, all_ez, all_ct, all_snt,
        da_re0, da_im0, das_re, das_im,
        pp_re, pp_im, probe,
    )

    spect_complex = (spect_re + 1j * spect_im).T
    return spect_complex.astype(jnp.complex64)
