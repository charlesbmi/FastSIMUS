"""Custom Metal kernel for simus RF spectrum on Apple Silicon.

Fuses geometry, phase initialization, and frequency sweep into a single GPU
kernel. One thread per scatterer computes the full TX->scatter->RX chain,
accumulating the complex RF spectrum via atomic adds to the output.

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
    from fast_simus.simus import SimusPlan

_metal_source_cache: str | None = None


def _load_kernel_source() -> str:
    """Read the Metal kernel body from ``simus.metal`` (cached)."""
    global _metal_source_cache
    if _metal_source_cache is None:
        _metal_source_cache = (Path(__file__).parent / "simus.metal").read_text()
    return _metal_source_cache


_kernel_cache: dict[tuple[int, int, int, int], Any] = {}


def build_simus_kernel(n_elem: int, n_sub: int, n_freq: int, n_scat: int) -> Any:
    """Build (or retrieve cached) Metal kernel for given dimensions.

    Args:
        n_elem: Number of transducer elements.
        n_sub: Number of sub-elements per element.
        n_freq: Number of selected frequency samples.
        n_scat: Number of scatterers.

    Returns:
        Compiled Metal kernel callable.
    """
    key = (n_elem, n_sub, n_freq, n_scat)
    if key in _kernel_cache:
        return _kernel_cache[key]

    n_es = n_elem * n_sub
    header = (
        f"#define N_ELEM {n_elem}\n"
        f"#define N_SUB {n_sub}\n"
        f"#define N_FREQ {n_freq}\n"
        f"#define N_ES {n_es}\n"
        f"#define N_SCAT {n_scat}\n"
    )
    kernel = mx.fast.metal_kernel(
        name=f"simus_{n_elem}_{n_sub}_{n_freq}_{n_scat}",
        input_names=[
            "scat_x",
            "scat_z",
            "elem_x",
            "elem_z",
            "theta_e",
            "sub_dx",
            "sub_dz",
            "da_init_re",
            "da_init_im",
            "da_step_re",
            "da_step_im",
            "pp_re",
            "pp_im",
            "probe",
            "rc",
            "is_out",
            "scalars",
        ],
        output_names=["spect_re", "spect_im"],
        header=header,
        source=_load_kernel_source(),
        atomic_outputs=True,
    )
    _kernel_cache[key] = kernel
    return kernel


def simus_metal(
    scatterers: mx.array,
    rc: mx.array,
    params: TransducerParams,
    plan: SimusPlan,
    medium: MediumParams,
    delays_clean: mx.array,
    tx_apodization: mx.array,
) -> mx.array:
    """Compute simus RF spectrum using a custom Metal kernel.

    Computes geometry on-the-fly per scatterer, avoiding large intermediate
    arrays. Returns the complex RF spectrum (n_freq, n_elements) corresponding
    to the selected frequency band.

    Args:
        scatterers: Scatterer positions (x, z) in meters. Shape ``(n_scat, 2)``.
        rc: Reflection coefficients. Shape ``(n_scat,)``.
        params: Transducer parameters.
        plan: Precomputed frequency plan from ``simus_precompute``.
        medium: Medium parameters.
        delays_clean: NaN-cleaned delays. Shape ``(n_elements,)``.
        tx_apodization: Per-element apodization (NaN-zeroed). Shape ``(n_elements,)``.

    Returns:
        Complex RF spectrum, shape ``(n_freq, n_elements)``.
    """
    c = medium.speed_of_sound
    alpha = medium.attenuation
    n_elem = params.n_elements
    n_sub = plan.n_sub
    n_freq = int(plan.selected_freqs.shape[0])
    n_scat = int(scatterers.shape[0])

    elem_pos, theta_e, apex_offset = element_positions(
        n_elem,
        params.pitch,
        params.radius,
        cast(_ArrayNamespace, mx),
    )
    if theta_e is None:
        theta_e = mx.zeros(n_elem, dtype=mx.float32)

    xp_mx = cast(_ArrayNamespace, mx)
    offsets = _subelement_centroids(params.element_width, n_sub, cast("Array", theta_e), xp_mx)
    sub_dx = cast(mx.array, offsets[..., 0]).reshape(-1)
    sub_dz = cast(mx.array, offsets[..., 1]).reshape(-1)

    # is_out mask (float32: 1.0=out, 0.0=in)
    x_flat = scatterers[:, 0]
    z_flat = scatterers[:, 1]
    is_out = (z_flat < 0).astype(mx.float32)
    if params.radius != inf:
        in_arc = (x_flat**2 + (z_flat + apex_offset) ** 2) <= params.radius**2
        is_out = mx.maximum(is_out, in_arc.astype(mx.float32))

    freq_start = float(plan.selected_freqs[0])
    freq_step = float(plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freq > 1 else 0.0

    # Delay+apodization geometric progression (complex)
    ph_init = mx.array(2.0 * pi * freq_start, dtype=mx.float32) * delays_clean
    da_init_re = (mx.cos(ph_init) * tx_apodization).astype(mx.float32)
    da_init_im = (mx.sin(ph_init) * tx_apodization).astype(mx.float32)

    ph_step = mx.array(2.0 * pi * freq_step, dtype=mx.float32) * delays_clean
    da_step_re = mx.cos(ph_step).astype(mx.float32)
    da_step_im = mx.sin(ph_step).astype(mx.float32)

    # Pulse * probe spectrum (complex) and probe-only (real)
    _pulse = cast(mx.array, plan.pulse_spectrum)
    _probe = cast(mx.array, plan.probe_spectrum)
    pp_complex = _pulse * _probe
    pp_re = mx.real(pp_complex).astype(mx.float32)
    pp_im = mx.imag(pp_complex).astype(mx.float32)
    probe_real = _probe.astype(mx.float32)

    # Scalar physics parameters
    wavenumber_init = 2.0 * pi * freq_start / c
    attenuation_init = alpha / NEPER_TO_DB * freq_start / 1e6 * 1e2
    wavenumber_step = 2.0 * pi * freq_step / c
    attenuation_step = alpha / NEPER_TO_DB * freq_step / 1e6 * 1e2
    min_distance = c / params.freq_center / 2.0
    center_wavenumber = 2.0 * pi * params.freq_center / c
    inv_n_sub = 1.0 / n_sub

    scalars = mx.array(
        [
            wavenumber_init,
            attenuation_init,
            wavenumber_step,
            attenuation_step,
            min_distance,
            plan.seg_length,
            center_wavenumber,
            inv_n_sub,
        ],
        dtype=mx.float32,
    )

    kernel = build_simus_kernel(n_elem, n_sub, n_freq, n_scat)

    output_size = n_freq * n_elem
    outputs = kernel(
        inputs=[
            scatterers[:, 0].astype(mx.float32),
            scatterers[:, 1].astype(mx.float32),
            elem_pos[:, 0].astype(mx.float32),
            elem_pos[:, 1].astype(mx.float32),
            theta_e.astype(mx.float32),
            sub_dx.astype(mx.float32),
            sub_dz.astype(mx.float32),
            da_init_re,
            da_init_im,
            da_step_re,
            da_step_im,
            pp_re,
            pp_im,
            probe_real,
            rc.astype(mx.float32),
            is_out.astype(mx.float32),
            scalars,
        ],
        output_shapes=[(output_size,), (output_size,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_scat, 1, 1),
        threadgroup=(min(256, n_scat), 1, 1),
        init_value=0.0,
    )

    spect_re = outputs[0].reshape(n_freq, n_elem)
    spect_im = outputs[1].reshape(n_freq, n_elem)

    return (spect_re + 1j * spect_im).astype(mx.complex64)
