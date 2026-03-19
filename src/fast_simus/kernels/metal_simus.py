"""Custom Metal kernel for simus RF spectrum on Apple Silicon.

Two-kernel architecture for optimal GPU occupancy:
  - Kernel A (TX): Element-tiled progression with shared-memory geometry.
    One threadgroup per scatterer; threads cooperatively compute geometry,
    then each thread processes sub-element tiles with ALU-only geometric
    progression. TILE_SE=16, threadgroup=64.
  - Kernel B (RX): SIMD-reduce RX with SCAT_REDUCE scatterers per
    threadgroup.  Adjacent SIMD threads handle the same element from
    different scatterers and use simd_shuffle_xor to sum contributions
    before a single atomic write.  Cuts atomic ops by SCAT_REDUCE (4x)
    while preserving coalesced output access.
    Threadgroup size = N_ELEM * SCAT_REDUCE (128 for P4-2v with SR=2).

For large scatterer counts, scatterers are processed in chunks that fit
within ``MAX_TX_INTERMEDIATE_BYTES``, with the split-path spectrum
accumulated across chunks via simple addition.

Falls back to a single fused kernel only when explicitly requested.

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

_KERNELS_DIR = Path(__file__).parent

MAX_TX_INTERMEDIATE_BYTES = 256 * 1024 * 1024  # 256 MB

_FUSED_THREADGROUP = 64
_TX_TILE_SE = 16
_TX_TILE_TG = 64
_RX_SCAT_REDUCE = 2

# TX tiled kernel: register pressure is only TILE_SE * 2 * 8 bytes per thread
# (256 bytes for TILE_SE=16). No register spills, so chunks can be larger
# than the old progression kernel which spilled at ~5000 scatterers.
_TX_OPTIMAL_CHUNK: dict[int, int] = {
    64: 10_000,  # P4-2v class (64 elem, 256B registers/thread)
    128: 5_000,  # L11-5v class (128 elem, 256B registers/thread)
}
_TX_DEFAULT_CHUNK = 10_000

# ---------------------------------------------------------------------------
# Source caching
# ---------------------------------------------------------------------------
_source_cache: dict[str, str] = {}


def _load_source(filename: str) -> str:
    if filename not in _source_cache:
        _source_cache[filename] = (_KERNELS_DIR / filename).read_text()
    return _source_cache[filename]


# ---------------------------------------------------------------------------
# Kernel builders (cached by dimension tuple)
# ---------------------------------------------------------------------------
_kernel_cache: dict[tuple, Any] = {}


def _make_header(n_elem: int, n_sub: int, n_freq: int, n_scat: int) -> str:
    return (
        f"#define N_ELEM {n_elem}\n"
        f"#define N_SUB {n_sub}\n"
        f"#define N_FREQ {n_freq}\n"
        f"#define N_ES {n_elem * n_sub}\n"
        f"#define N_SCAT {n_scat}\n"
    )


def _build_fused(n_elem: int, n_sub: int, n_freq: int, n_scat: int) -> Any:
    key = ("fused", n_elem, n_sub, n_freq, n_scat)
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
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
            header=_make_header(n_elem, n_sub, n_freq, n_scat),
            source=_load_source("simus.metal"),
            atomic_outputs=True,
        )
    return _kernel_cache[key]


def _build_tx(n_elem: int, n_sub: int, n_freq: int, n_scat: int) -> Any:
    """Build the tiled TX kernel (element-tiled progression with shared geometry)."""
    key = ("tx_tiled", n_elem, n_sub, n_freq, n_scat)
    if key not in _kernel_cache:
        tg = _TX_TILE_TG
        header = (
            _make_header(n_elem, n_sub, n_freq, n_scat)
            + f"#define TILE_SE {_TX_TILE_SE}\n"
            + f"#define TG_SIZE {tg}\n"
            + f"#define MAX_FPT (({n_freq} + {tg} - 1) / {tg})\n"
        )
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"simus_tx_tiled_{n_elem}_{n_sub}_{n_freq}_{n_scat}",
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
                "delay_phase_step",
                "pp_re",
                "pp_im",
                "is_out",
                "scalars",
            ],
            output_names=["tx_re", "tx_im"],
            header=header,
            source=_load_source("simus_tx_tiled.metal"),
        )
    return _kernel_cache[key]


def _build_rx(n_elem: int, n_sub: int, n_freq: int, n_scat: int) -> Any:
    """Build the SIMD-reduce RX kernel.

    Groups SCAT_REDUCE scatterers per threadgroup.  Adjacent threads handle
    the same element from different scatterers and use simd_shuffle_xor to
    sum contributions before writing a single atomic.  Cuts atomic writes by
    SCAT_REDUCE while preserving coalesced output access.
    """
    sr = _RX_SCAT_REDUCE
    key = ("rx_simd", n_elem, n_sub, n_freq, n_scat, sr)
    if key not in _kernel_cache:
        header = _make_header(n_elem, n_sub, n_freq, n_scat) + f"#define SCAT_REDUCE {sr}\n"
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"simus_rx_simd_{n_elem}_{n_sub}_{n_freq}_{n_scat}_{sr}",
            input_names=[
                "scat_x",
                "scat_z",
                "elem_x",
                "elem_z",
                "theta_e",
                "sub_dx",
                "sub_dz",
                "tx_re",
                "tx_im",
                "probe",
                "rc",
                "scalars",
            ],
            output_names=["spect_re", "spect_im"],
            header=header,
            source=_load_source("simus_rx_simd.metal"),
            atomic_outputs=True,
        )
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Input preparation (shared by both paths)
# ---------------------------------------------------------------------------
def _prepare_common(
    scatterers: mx.array,
    rc: mx.array,
    params: TransducerParams,
    plan: SimusPlan,
    medium: MediumParams,
    delays_clean: mx.array,
    tx_apodization: mx.array,
) -> dict[str, Any]:
    """Prepare all GPU-side inputs from plan and params."""
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

    x_flat = scatterers[:, 0]
    z_flat = scatterers[:, 1]
    is_out = (z_flat < 0).astype(mx.float32)
    if params.radius != inf:
        in_arc = (x_flat**2 + (z_flat + apex_offset) ** 2) <= params.radius**2
        is_out = mx.maximum(is_out, in_arc.astype(mx.float32))

    freq_start = float(plan.selected_freqs[0])
    freq_step = float(plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freq > 1 else 0.0

    ph_init = mx.array(2.0 * pi * freq_start, dtype=mx.float32) * delays_clean
    da_init_re = (mx.cos(ph_init) * tx_apodization).astype(mx.float32)
    da_init_im = (mx.sin(ph_init) * tx_apodization).astype(mx.float32)

    ph_step = mx.array(2.0 * pi * freq_step, dtype=mx.float32) * delays_clean
    delay_phase_step = ph_step.astype(mx.float32)
    da_step_re = mx.cos(ph_step).astype(mx.float32)
    da_step_im = mx.sin(ph_step).astype(mx.float32)

    _pulse = cast(mx.array, plan.pulse_spectrum)
    _probe = cast(mx.array, plan.probe_spectrum)
    pp_complex = _pulse * _probe
    pp_re = mx.real(pp_complex).astype(mx.float32)
    pp_im = mx.imag(pp_complex).astype(mx.float32)
    probe_real = _probe.astype(mx.float32)

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

    return {
        "x_flat": x_flat.astype(mx.float32),
        "z_flat": z_flat.astype(mx.float32),
        "elem_x": elem_pos[:, 0].astype(mx.float32),
        "elem_z": elem_pos[:, 1].astype(mx.float32),
        "theta_e": theta_e.astype(mx.float32),
        "sub_dx": sub_dx.astype(mx.float32),
        "sub_dz": sub_dz.astype(mx.float32),
        "da_init_re": da_init_re,
        "da_init_im": da_init_im,
        "delay_phase_step": delay_phase_step,
        "da_step_re": da_step_re,
        "da_step_im": da_step_im,
        "pp_re": pp_re,
        "pp_im": pp_im,
        "probe_real": probe_real,
        "rc": rc.astype(mx.float32),
        "is_out": is_out,
        "scalars": scalars,
        "n_elem": n_elem,
        "n_sub": n_sub,
        "n_freq": n_freq,
        "n_scat": n_scat,
    }


# ---------------------------------------------------------------------------
# Dispatch paths
# ---------------------------------------------------------------------------
def _dispatch_fused(d: dict[str, Any]) -> mx.array:
    """Single-kernel path (fused TX+RX). Lower memory, higher register pressure."""
    n_elem, n_sub, n_freq, n_scat = d["n_elem"], d["n_sub"], d["n_freq"], d["n_scat"]
    kernel = _build_fused(n_elem, n_sub, n_freq, n_scat)
    output_size = n_freq * n_elem
    outputs = kernel(
        inputs=[
            d["x_flat"],
            d["z_flat"],
            d["elem_x"],
            d["elem_z"],
            d["theta_e"],
            d["sub_dx"],
            d["sub_dz"],
            d["da_init_re"],
            d["da_init_im"],
            d["da_step_re"],
            d["da_step_im"],
            d["pp_re"],
            d["pp_im"],
            d["probe_real"],
            d["rc"],
            d["is_out"],
            d["scalars"],
        ],
        output_shapes=[(output_size,), (output_size,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_scat, 1, 1),
        threadgroup=(min(_FUSED_THREADGROUP, n_scat), 1, 1),
        init_value=0.0,
    )
    spect_re = outputs[0].reshape(n_freq, n_elem)
    spect_im = outputs[1].reshape(n_freq, n_elem)
    return (spect_re + 1j * spect_im).astype(mx.complex64)


def _dispatch_split(d: dict[str, Any]) -> mx.array:
    """Two-kernel path with automatic chunking for large scatterer counts.

    Scatterers are processed in chunks that fit the TX intermediate buffer
    within ``MAX_TX_INTERMEDIATE_BYTES``. Each chunk runs TX then RX, and
    the per-chunk spectra are summed on the host.
    """
    n_elem, n_sub, n_freq, n_scat = d["n_elem"], d["n_sub"], d["n_freq"], d["n_scat"]
    spect_size = n_freq * n_elem

    # Use TX-throughput-optimal chunk sizes, capped by memory budget
    bytes_per_scat = n_freq * 4 * 2  # float32 re + im
    mem_chunk = max(1, MAX_TX_INTERMEDIATE_BYTES // bytes_per_scat)
    perf_chunk = _TX_OPTIMAL_CHUNK.get(n_elem, _TX_DEFAULT_CHUNK)
    chunk_size = min(mem_chunk, perf_chunk)

    # Geometry arrays shared across all chunks
    geom_tx = [
        d["elem_x"],
        d["elem_z"],
        d["theta_e"],
        d["sub_dx"],
        d["sub_dz"],
        d["da_init_re"],
        d["da_init_im"],
        d["delay_phase_step"],
        d["pp_re"],
        d["pp_im"],
    ]
    geom_rx = [d["elem_x"], d["elem_z"], d["theta_e"], d["sub_dx"], d["sub_dz"]]
    probe = d["probe_real"]
    scalars = d["scalars"]

    # Build kernels for the standard chunk size (cached, compiled once per probe)
    k_tx = _build_tx(n_elem, n_sub, n_freq, chunk_size)
    k_rx = _build_rx(n_elem, n_sub, n_freq, chunk_size)

    total_re = mx.zeros(spect_size, dtype=mx.float32)
    total_im = mx.zeros(spect_size, dtype=mx.float32)

    for start in range(0, n_scat, chunk_size):
        end = min(start + chunk_size, n_scat)
        cn = end - start

        cx = d["x_flat"][start:end]
        cz = d["z_flat"][start:end]
        crc = d["rc"][start:end]
        c_out = d["is_out"][start:end]

        # TX kernel: one threadgroup per scatterer (tiled progression)
        tg = _TX_TILE_TG
        tx_out = k_tx(
            inputs=[cx, cz, *geom_tx, c_out, scalars],
            output_shapes=[(cn * n_freq,), (cn * n_freq,)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(cn * tg, 1, 1),
            threadgroup=(tg, 1, 1),
        )

        # RX kernel: SCAT_REDUCE scatterers per threadgroup, SIMD reduction
        sr = _RX_SCAT_REDUCE
        rx_tg = n_elem * sr
        n_tgs = (cn + sr - 1) // sr
        rx_out = k_rx(
            inputs=[cx, cz, *geom_rx, tx_out[0], tx_out[1], probe, crc, scalars],
            output_shapes=[(spect_size,), (spect_size,)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(n_tgs * rx_tg, 1, 1),
            threadgroup=(rx_tg, 1, 1),
            init_value=0.0,
        )

        total_re = total_re + rx_out[0]
        total_im = total_im + rx_out[1]

    spect_re = total_re.reshape(n_freq, n_elem)
    spect_im = total_im.reshape(n_freq, n_elem)
    return (spect_re + 1j * spect_im).astype(mx.complex64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def simus_metal(
    scatterers: mx.array,
    rc: mx.array,
    params: TransducerParams,
    plan: SimusPlan,
    medium: MediumParams,
    delays_clean: mx.array,
    tx_apodization: mx.array,
) -> mx.array:
    """Compute simus RF spectrum using custom Metal kernels.

    Uses a two-kernel TX/RX split with automatic chunking for large
    scatterer counts. Each chunk fits within the TX intermediate memory
    budget, and chunk spectra are accumulated via simple addition.

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
    d = _prepare_common(scatterers, rc, params, plan, medium, delays_clean, tx_apodization)
    return _dispatch_split(d)
