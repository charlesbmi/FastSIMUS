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

from math import inf, pi
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx

from fast_simus._pfield_math import NEPER_TO_DB, _subelement_centroids
from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace
from fast_simus.utils.geometry import element_positions

if TYPE_CHECKING:
    from fast_simus.pfield import PfieldPlan

_metal_source_cache: str | None = None


def _load_kernel_source() -> str:
    """Read the Metal kernel body from ``pfield.metal`` (cached)."""
    global _metal_source_cache
    if _metal_source_cache is None:
        _metal_source_cache = (Path(__file__).parent / "pfield.metal").read_text()
    return _metal_source_cache


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
        source=_load_kernel_source(),
    )
    _kernel_cache[key] = kernel
    return kernel


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

    # Subelement offsets -- reuse shared geometry, reshape to flat (n_elem*n_sub,)
    xp_mx = cast(_ArrayNamespace, mx)
    offsets = _subelement_centroids(params.element_width, n_sub, cast("Array", theta_e), xp_mx)
    sub_dx = cast(mx.array, offsets[..., 0]).reshape(-1)
    sub_dz = cast(mx.array, offsets[..., 1]).reshape(-1)

    # is_out mask (float32: 1.0=out, 0.0=in)
    x_flat = positions[..., 0].reshape(-1)
    z_flat = positions[..., 1].reshape(-1)
    is_out = (z_flat < 0).astype(mx.float32)
    if params.radius != inf:
        in_arc = (x_flat**2 + (z_flat + apex_offset) ** 2) <= params.radius**2
        is_out = mx.maximum(is_out, in_arc.astype(mx.float32))

    # Derive freq_start / freq_step from the canonical selected_freqs array.
    freq_start = float(plan.selected_freqs[0])
    freq_step = float(plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freq > 1 else 0.0

    # Delay+apodization split into real/imag
    ph_init = mx.array(2.0 * pi * freq_start, dtype=mx.float32) * delays_clean
    da_init_re = (mx.cos(ph_init) * tx_apodization).astype(mx.float32)
    da_init_im = (mx.sin(ph_init) * tx_apodization).astype(mx.float32)

    ph_step = mx.array(2.0 * pi * freq_step, dtype=mx.float32) * delays_clean
    da_step_re = mx.cos(ph_step).astype(mx.float32)
    da_step_im = mx.sin(ph_step).astype(mx.float32)

    # |pulse_spectrum * probe_spectrum|^2
    _pulse = cast(mx.array, plan.pulse_spectrum)
    _probe = cast(mx.array, plan.probe_spectrum)
    pp_mag_sq = mx.abs(_pulse).astype(mx.float32) ** 2 * _probe.astype(mx.float32) ** 2

    # Scalar physics parameters
    wavenumber_init = 2.0 * pi * freq_start / c
    attenuation_init = alpha / NEPER_TO_DB * freq_start / 1e6 * 1e2
    wavenumber_step = 2.0 * pi * freq_step / c
    attenuation_step = alpha / NEPER_TO_DB * freq_step / 1e6 * 1e2
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
