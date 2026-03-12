"""Custom Metal kernel for pfield computation on Apple Silicon.

Fuses geometry, phase initialization, and frequency sweep into a single
GPU kernel. One thread per grid point computes the full pressure contribution
on-the-fly, avoiding large intermediate arrays.

Requires: MLX (mlx package) on Apple Silicon.

Limitations:
    - Soft baffle only (BaffleType.SOFT assumed)
    - Center-frequency directivity only (full_frequency_directivity=False)
    - Linear arrays only (convex array support needs testing)
"""

from __future__ import annotations

from math import inf, log, pi
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx

from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import _ArrayNamespace
from fast_simus.utils.geometry import element_positions

if TYPE_CHECKING:
    from fast_simus.pfield import PfieldPlan

_NEPER_TO_DB = 20.0 / log(10.0)

# Metal Shading Language source for the pfield kernel.
# Uses metal::precise:: for geometry transcendentals to stay within rtol=1e-4.
# Template constants N_ELEM, N_SUB, N_FREQ, N_ES are injected at compile time.
_METAL_PFIELD_SOURCE = """
    uint g = thread_position_in_grid.x;

    float gx = grid_x[g];
    float gz = grid_z[g];

    float kw_init    = scalars[0];
    float alpha_init = scalars[1];
    float kw_step    = scalars[2];
    float alpha_step = scalars[3];
    float min_dist   = scalars[4];
    float seg_len    = scalars[5];
    float center_kw  = scalars[6];
    float eff_corr   = scalars[7];

    float2 cur[N_ES];
    float2 stp[N_ES];

    for (int e = 0; e < N_ELEM; e++) {
        float ex = elem_x[e];
        float ez = elem_z[e];
        float te = theta_e[e];
        float di_re = da_init_re[e], di_im = da_init_im[e];
        float ds_re = da_step_re[e], ds_im = da_step_im[e];

        for (int s = 0; s < N_SUB; s++) {
            int idx = e * N_SUB + s;

            float dx = gx - ex - sub_dx[idx];
            float dz = gz - ez - sub_dz[idx];
            float r = metal::precise::sqrt(dx * dx + dz * dz);
            float rc = max(r, min_dist);

            // Angle relative to element normal (unclipped distance for angle)
            float th = metal::precise::asin((dx + 1e-16f) / (r + 1e-16f)) - te;

            // Soft baffle obliquity
            float obliq = (fabs(th) >= M_PI_2_F) ? 1e-16f : metal::precise::cos(th);

            // Phase init: obliq/sqrt(r) * exp(-alpha*r + j*wrap(k*r, 2pi))
            float kwr = kw_init * rc;
            float TWO_PI = 2.0f * M_PI_F;
            float ph_wrap = kwr - TWO_PI * metal::precise::floor(kwr / TWO_PI);
            float ai = obliq / metal::precise::sqrt(rc) * metal::precise::exp(-alpha_init * rc);
            float2 pi_ = float2(ai * metal::precise::cos(ph_wrap),
                                ai * metal::precise::sin(ph_wrap));

            // Phase step: exp((-alpha_step + j*k_step) * r)
            float as_ = metal::precise::exp(-alpha_step * rc);
            float phs = kw_step * rc;
            float2 ps_ = float2(as_ * metal::precise::cos(phs),
                                as_ * metal::precise::sin(phs));

            // Center-frequency sinc directivity
            float sa = center_kw * seg_len * 0.5f * metal::precise::sin(th);
            float sv = (fabs(sa) < 1e-8f) ? 1.0f : metal::precise::sin(sa) / sa;
            pi_ *= sv;

            // Absorb delay+apodization (complex multiply)
            cur[idx] = float2(
                pi_.x * di_re - pi_.y * di_im,
                pi_.x * di_im + pi_.y * di_re
            );
            stp[idx] = float2(
                ps_.x * ds_re - ps_.y * ds_im,
                ps_.x * ds_im + ps_.y * ds_re
            );
        }
    }

    // Frequency sweep: accumulate sum_f |pulse_probe_f|^2 * |sum_es phase_es_f|^2
    float acc = 0.0f;
    for (int f = 0; f < N_FREQ; f++) {
        float sr = 0.0f, si = 0.0f;
        for (int j = 0; j < N_ES; j++) {
            sr += cur[j].x;
            si += cur[j].y;
            float cr = cur[j].x, ci = cur[j].y;
            float tr = stp[j].x, ti = stp[j].y;
            cur[j] = float2(cr * tr - ci * ti, cr * ti + ci * tr);
        }
        acc += pp_mag_sq[f] * (sr * sr + si * si);
    }

    pressure[g] = (is_out[g] > 0.5f) ? 0.0f : acc * eff_corr;
"""

_kernel_cache: dict[tuple[int, int, int], Any] = {}


def build_pfield_kernel(n_elem: int, n_sub: int, n_freq: int) -> Any:
    """Build (or retrieve cached) Metal kernel for given dimensions.

    Args:
        n_elem: Number of transducer elements.
        n_sub: Number of sub-elements per element.
        n_freq: Number of frequency samples.

    Returns:
        Compiled Metal kernel callable.
    """
    key = (n_elem, n_sub, n_freq)
    if key in _kernel_cache:
        return _kernel_cache[key]

    n_es = n_elem * n_sub
    header = f"#define N_ELEM {n_elem}\n#define N_SUB {n_sub}\n#define N_FREQ {n_freq}\n#define N_ES {n_es}\n"
    kernel = mx.fast.metal_kernel(
        name=f"pfield_{n_elem}_{n_sub}_{n_freq}",
        input_names=[
            "grid_x",
            "grid_z",
            "elem_x",
            "elem_z",
            "theta_e",
            "sub_dx",
            "sub_dz",
            "da_init_re",
            "da_init_im",
            "da_step_re",
            "da_step_im",
            "pp_mag_sq",
            "is_out",
            "scalars",
        ],
        output_names=["pressure"],
        header=header,
        source=_METAL_PFIELD_SOURCE,
    )
    _kernel_cache[key] = kernel
    return kernel


def _subelement_offsets_flat(
    element_width: float,
    n_sub: int,
    theta_e: mx.array,
) -> tuple[mx.array, mx.array]:
    """Compute flattened subelement offsets (n_elem*n_sub,) for x and z."""
    seg_length = element_width / n_sub
    seg_offsets = mx.array(
        [-element_width / 2.0 + seg_length / 2.0 + i * seg_length for i in range(n_sub)],
        dtype=mx.float32,
    )
    seg_2d = mx.reshape(seg_offsets, (1, n_sub))
    cos_th = mx.cos(theta_e)[:, None]
    sin_neg_th = mx.sin(-theta_e)[:, None]
    sub_dx = (seg_2d * cos_th).reshape(-1)
    sub_dz = (seg_2d * sin_neg_th).reshape(-1)
    return sub_dx, sub_dz


def pfield_metal(
    positions: mx.array,
    params: TransducerParams,
    plan: PfieldPlan,
    medium: MediumParams,
    delays_clean: mx.array,
    tx_apodization: mx.array,
) -> mx.array:
    """Compute pressure field using a custom Metal kernel.

    Computes geometry on-the-fly per grid point, avoiding large intermediate
    arrays (*grid, n_elements, n_sub). Returns raw pressure accumulation
    (sum of |P_k|^2 * correction), NOT the final sqrt -- the caller applies
    sqrt after the dispatch block.

    Args:
        positions: Grid positions (x, z) in meters. Shape ``(*grid_shape, 2)``.
        params: Transducer parameters.
        plan: Precomputed frequency plan from ``pfield_precompute``.
        medium: Medium parameters.
        delays_clean: NaN-cleaned delays. Shape ``(n_elements,)``.
        tx_apodization: Per-element apodization (NaN-zeroed). Shape ``(n_elements,)``.

    Returns:
        Raw pressure accumulation, shape ``(*grid_shape,)``.
        Caller must apply ``xp.sqrt(result)`` to get RMS pressure.
    """
    c = medium.speed_of_sound
    alpha = medium.attenuation
    n_elem = params.n_elements
    n_sub = plan.n_sub
    n_freq = int(plan.selected_freqs.shape[0])
    grid_shape = positions.shape[:-1]

    # Element geometry
    elem_pos, theta_e, apex_offset = element_positions(
        n_elem,
        params.pitch,
        params.radius,
        cast(_ArrayNamespace, mx),
    )
    if theta_e is None:
        theta_e = mx.zeros(n_elem, dtype=mx.float32)

    # Subelement offsets
    sub_dx, sub_dz = _subelement_offsets_flat(params.element_width, n_sub, theta_e)

    # is_out mask (float32: 1.0=out, 0.0=in)
    x_flat = positions[..., 0].reshape(-1)
    z_flat = positions[..., 1].reshape(-1)
    is_out = (z_flat < 0).astype(mx.float32)
    if params.radius != inf:
        in_arc = (x_flat**2 + (z_flat + apex_offset) ** 2) <= params.radius**2
        is_out = mx.maximum(is_out, in_arc.astype(mx.float32))

    # Delay+apodization split into real/imag
    ph_init = mx.array(2.0 * pi * plan.freq_start, dtype=mx.float32) * delays_clean
    da_init_re = (mx.cos(ph_init) * tx_apodization).astype(mx.float32)
    da_init_im = (mx.sin(ph_init) * tx_apodization).astype(mx.float32)

    ph_step = mx.array(2.0 * pi * plan.freq_step, dtype=mx.float32) * delays_clean
    da_step_re = mx.cos(ph_step).astype(mx.float32)
    da_step_im = mx.sin(ph_step).astype(mx.float32)

    # |pulse_spectrum * probe_spectrum|^2
    _pulse = cast(mx.array, plan.pulse_spectrum)
    _probe = cast(mx.array, plan.probe_spectrum)
    pp_mag_sq = mx.abs(_pulse).astype(mx.float32) ** 2 * _probe.astype(mx.float32) ** 2

    # Scalar physics parameters
    wavenumber_init = 2.0 * pi * plan.freq_start / c
    attenuation_init = alpha / _NEPER_TO_DB * plan.freq_start / 1e6 * 1e2
    wavenumber_step = 2.0 * pi * plan.freq_step / c
    attenuation_step = alpha / _NEPER_TO_DB * plan.freq_step / 1e6 * 1e2
    min_distance = c / params.freq_center / 2.0
    center_wavenumber = 2.0 * pi * params.freq_center / c
    # 1/n_sub^2 because kernel sums (not means) over sub-elements.
    # correction_factor is applied by the caller uniformly across all strategies.
    effective_correction = 1.0 / (n_sub**2)

    scalars = mx.array(
        [
            wavenumber_init,
            attenuation_init,
            wavenumber_step,
            attenuation_step,
            min_distance,
            plan.seg_length,
            center_wavenumber,
            effective_correction,
        ],
        dtype=mx.float32,
    )

    # Build kernel and dispatch
    n_grid = int(x_flat.shape[0])
    kernel = build_pfield_kernel(n_elem, n_sub, n_freq)

    outputs = kernel(
        inputs=[
            x_flat.astype(mx.float32),
            z_flat.astype(mx.float32),
            elem_pos[:, 0].astype(mx.float32),
            elem_pos[:, 1].astype(mx.float32),
            theta_e.astype(mx.float32),
            sub_dx.astype(mx.float32),
            sub_dz.astype(mx.float32),
            da_init_re,
            da_init_im,
            da_step_re,
            da_step_im,
            pp_mag_sq,
            is_out.astype(mx.float32),
            scalars,
        ],
        output_shapes=[(n_grid,)],
        output_dtypes=[mx.float32],
        grid=(n_grid, 1, 1),
        threadgroup=(256, 1, 1),
    )

    # Return raw accumulation (acc / n_sub^2). The caller applies
    # sqrt(pressure_accum * correction_factor) uniformly for all strategies.
    return outputs[0].reshape(grid_shape)
