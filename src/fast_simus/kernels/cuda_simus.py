"""CUDA GPU dispatch for simus RF spectrum on NVIDIA GPUs.

Uses a custom fused TX+RX kernel compiled via NVRTC at runtime.
The kernel mirrors the Metal tiled-progression approach:
  - One block per scatterer
  - Shared-memory geometry cache
  - Element-tiled geometric progression (ALU-only inner loop)
  - Fused TX+RX to eliminate intermediate buffer

Data stays on GPU: uses JAX device pointers and CUDA driver API
to avoid host round-trips for large arrays.
"""

from __future__ import annotations

import ctypes
from math import inf, pi
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils.geometry import element_positions

if TYPE_CHECKING:
    from fast_simus.simus import SimusPlan

_NEPER_TO_DB = 8.685889638065036  # 20/log(10)
_KERNEL_SOURCE: str | None = None
_TG_SIZE = 128
_TILE_SE = 4

_device_cache: dict[str, ctypes.c_void_p] = {}


def _load_kernel_source() -> str:
    global _KERNEL_SOURCE
    if _KERNEL_SOURCE is None:
        kernel_path = Path(__file__).parent / "simus_fused.cu"
        _KERNEL_SOURCE = kernel_path.read_text()
    return _KERNEL_SOURCE


def _get_compute_cap() -> str:
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuInit(0)
        device = ctypes.c_int()
        cuda.cuDeviceGet(ctypes.byref(device), 0)
        major, minor = ctypes.c_int(), ctypes.c_int()
        cuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device)
        return f"sm_{major.value}{minor.value}"
    except Exception:
        return "sm_86"


def _upload_const(key: str, data: np.ndarray) -> ctypes.c_void_p:
    """Upload constant array to device, caching by key."""
    from fast_simus.kernels.cuda_runtime import device_alloc, memcpy_htod

    if key in _device_cache:
        return _device_cache[key]
    ptr = device_alloc(data.nbytes)
    memcpy_htod(ptr, data)
    _device_cache[key] = ptr
    return ptr


def _jax_ptr(arr: jax.Array) -> ctypes.c_void_p:
    """Get raw CUDA device pointer from a JAX array."""
    arr.block_until_ready()
    buf = arr.addressable_data(0)
    return ctypes.c_void_p(buf.unsafe_buffer_pointer())


def simus_cuda(
    scatterers: jax.Array,
    rc: jax.Array,
    params: TransducerParams,
    plan: "SimusPlan",
    medium: MediumParams,
    delays_clean: jax.Array,
    tx_apodization: jax.Array,
) -> jax.Array:
    """Compute simus RF spectrum on NVIDIA GPU via custom fused CUDA kernel.

    One block per scatterer, tiled progression through elements,
    geometric progression through frequencies. ALU-only inner loop.
    Data stays on GPU -- no host round-trip for scatterer arrays.

    Returns:
        Complex RF spectrum, shape (n_freq, n_elements).
    """
    from fast_simus.kernels.cuda_runtime import (
        compile_kernel,
        launch_kernel,
        synchronize,
        _get_cuda,
        _get_context,
    )

    n_elem = params.n_elements
    n_scat = int(scatterers.shape[0])
    n_freq = int(plan.selected_freqs.shape[0])
    n_sub = plan.n_sub
    n_es = n_elem * n_sub

    # --- Prepare scatterer data (stays on GPU via JAX) ---
    scat_f32 = jnp.asarray(scatterers, dtype=jnp.float32)
    scat_x = scat_f32[:, 0].copy()
    scat_z = scat_f32[:, 1].copy()
    rc_f32 = jnp.asarray(rc, dtype=jnp.float32)

    d_scat_x = _jax_ptr(scat_x)
    d_scat_z = _jax_ptr(scat_z)
    d_rc = _jax_ptr(rc_f32)

    # --- Prepare constant data (cached on device) ---
    cache_key = f"{n_elem}_{n_sub}_{n_freq}_{params.pitch}_{params.radius}"

    element_pos, theta_elements_raw, apex_offset = element_positions(
        n_elem, params.pitch, params.radius, np,
    )
    if theta_elements_raw is None:
        theta_elements_raw = np.zeros(n_elem, dtype=np.float32)

    elem_x_np = np.ascontiguousarray(element_pos[:, 0], dtype=np.float32)
    elem_z_np = np.ascontiguousarray(element_pos[:, 1], dtype=np.float32)
    theta_e_np = np.ascontiguousarray(theta_elements_raw, dtype=np.float32)

    seg_length = params.element_width / n_sub
    seg_offsets = np.array(
        [-params.element_width / 2.0 + seg_length / 2.0 + i * seg_length for i in range(n_sub)],
        dtype=np.float32,
    )

    cos_th = np.cos(theta_e_np)
    sin_neg_th = np.sin(-theta_e_np)
    sub_dx_np = np.zeros(n_es, dtype=np.float32)
    sub_dz_np = np.zeros(n_es, dtype=np.float32)
    for e in range(n_elem):
        for s in range(n_sub):
            idx = e * n_sub + s
            sub_dx_np[idx] = seg_offsets[s] * cos_th[e]
            sub_dz_np[idx] = seg_offsets[s] * sin_neg_th[e]

    d_elem_x = _upload_const(f"{cache_key}_elem_x", elem_x_np)
    d_elem_z = _upload_const(f"{cache_key}_elem_z", elem_z_np)
    d_theta_e = _upload_const(f"{cache_key}_theta_e", theta_e_np)
    d_sub_dx = _upload_const(f"{cache_key}_sub_dx", sub_dx_np)
    d_sub_dz = _upload_const(f"{cache_key}_sub_dz", sub_dz_np)

    freq_start = float(plan.selected_freqs[0])
    freq_step_val = (
        float(plan.selected_freqs[1] - plan.selected_freqs[0])
        if n_freq > 1
        else 0.0
    )

    speed_of_sound = medium.speed_of_sound
    attenuation = medium.attenuation
    freq_center = params.freq_center

    kw_init = 2.0 * pi * freq_start / speed_of_sound
    alpha_init = attenuation / _NEPER_TO_DB * freq_start / 1e6 * 1e2
    kw_step = 2.0 * pi * freq_step_val / speed_of_sound
    alpha_step = attenuation / _NEPER_TO_DB * freq_step_val / 1e6 * 1e2
    min_dist_val = speed_of_sound / freq_center / 2.0
    center_kw = 2.0 * pi * freq_center / speed_of_sound
    inv_nsub = 1.0 / n_sub

    delays_np = np.asarray(delays_clean, dtype=np.float32)
    tx_apod_np = np.asarray(tx_apodization, dtype=np.float32)
    da_phase_init = 2.0 * pi * freq_start * delays_np
    da_init_re_np = (np.cos(da_phase_init) * tx_apod_np).astype(np.float32)
    da_init_im_np = (np.sin(da_phase_init) * tx_apod_np).astype(np.float32)
    dps_np = (2.0 * pi * freq_step_val * delays_np).astype(np.float32)

    d_da_init_re = _upload_const(f"{cache_key}_da_re_{hash(delays_np.tobytes())}", da_init_re_np)
    d_da_init_im = _upload_const(f"{cache_key}_da_im_{hash(delays_np.tobytes())}", da_init_im_np)
    d_dps = _upload_const(f"{cache_key}_dps_{hash(delays_np.tobytes())}", dps_np)

    spectra = np.asarray(plan.pulse_spectrum * plan.probe_spectrum)
    pp_re_np = np.real(spectra).astype(np.float32)
    pp_im_np = np.imag(spectra).astype(np.float32)
    probe_raw = np.asarray(plan.probe_spectrum)
    probe_np = (
        np.abs(probe_raw).astype(np.float32)
        if np.iscomplexobj(probe_raw)
        else probe_raw.astype(np.float32)
    )

    d_pp_re = _upload_const(f"{cache_key}_pp_re", pp_re_np)
    d_pp_im = _upload_const(f"{cache_key}_pp_im", pp_im_np)
    d_probe = _upload_const(f"{cache_key}_probe", probe_np)

    # --- Compile kernel ---
    max_fpt = (n_freq + _TG_SIZE - 1) // _TG_SIZE
    defines = {
        "N_ELEM": n_elem,
        "N_SUB": n_sub,
        "N_FREQ": n_freq,
        "N_ES": n_es,
        "TILE_SE": _TILE_SE,
        "TG_SIZE": _TG_SIZE,
        "MAX_FPT": max_fpt,
    }
    source = _load_kernel_source()
    arch = _get_compute_cap()

    _, func = compile_kernel(
        source, "simus_fused_kernel", defines=tuple(sorted(defines.items())), arch=arch,
    )

    # --- Allocate output as JAX arrays (stays on GPU, no host roundtrip) ---
    out_size = n_freq * n_elem
    spect_re_jax = jnp.zeros(out_size, dtype=jnp.float32)
    spect_im_jax = jnp.zeros(out_size, dtype=jnp.float32)
    d_spect_re = _jax_ptr(spect_re_jax)
    d_spect_im = _jax_ptr(spect_im_jax)

    # --- Shared memory size ---
    shmem_bytes = (9 * n_es + 3 * n_elem + 2 * n_freq) * 4

    radius_val = params.radius if params.radius != inf else 1e31

    # --- Launch kernel ---
    args = [
        d_scat_x, d_scat_z, d_rc,
        d_elem_x, d_elem_z, d_theta_e,
        d_sub_dx, d_sub_dz,
        d_da_init_re, d_da_init_im, d_dps,
        d_pp_re, d_pp_im, d_probe,
        d_spect_re, d_spect_im,
        ctypes.c_int(n_scat),
        ctypes.c_float(kw_init),
        ctypes.c_float(alpha_init),
        ctypes.c_float(kw_step),
        ctypes.c_float(alpha_step),
        ctypes.c_float(min_dist_val),
        ctypes.c_float(seg_length),
        ctypes.c_float(center_kw),
        ctypes.c_float(inv_nsub),
        ctypes.c_float(radius_val),
        ctypes.c_float(apex_offset),
    ]

    launch_kernel(
        func,
        grid=(n_scat, 1, 1),
        block=(_TG_SIZE, 1, 1),
        args=args,
        shared_mem=shmem_bytes,
    )
    synchronize()

    # --- Build result on GPU (no host roundtrip) ---
    # Kernel writes element-major: (n_elem, n_freq). Transpose to (n_freq, n_elem).
    spect_re_2d = spect_re_jax.reshape(n_elem, n_freq)
    spect_im_2d = spect_im_jax.reshape(n_elem, n_freq)
    spect_complex = (spect_re_2d + 1j * spect_im_2d).T
    return spect_complex.astype(jnp.complex64)
