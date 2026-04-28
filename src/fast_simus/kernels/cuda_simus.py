"""CuPy CUDA backend for simus.

Compiles the v25c register-resident TX kernel via NVRTC at runtime
(``cupy.RawModule``) -- no nanobind, no setuptools build step. Pinned to
``(B_SCAT=9, ELEM_TILE=2)`` for RTX 4090 / sm_89 / P4-2v; performance may
regress on other probes / GPUs (see exp22 + the FastSIMUS-cuda-tune
follow-up).

Output layout matches ``metal_simus.simus_metal``: complex64
``(n_freq, n_elements)``. The shipped kernel does its own per-scatterer
Phase-1 geometry from a flat input set, so the
``(n_scat, n_elem, n_sub)`` phase tensors that ``_simus_freq_outer_python``
consumes are *not* fed in here.

Requires: cupy (``cupy-cuda12x`` or ``cupy-cuda11x``) on a CUDA host.
"""

from __future__ import annotations

from math import inf, pi
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cupy as cp

from fast_simus._pfield_math import NEPER_TO_DB, _subelement_centroids
from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import _ArrayNamespace
from fast_simus.utils.geometry import element_positions

if TYPE_CHECKING:
    from fast_simus.simus import SimusPlan

_KERNELS_DIR = Path(__file__).parent
_SOURCE_NAME = "simus_fused.cu"

# Pinned tuning -- see docs/progress/experiments/exp22-svshmem-et2.md.
# These constants are RTX 4090 / sm_89 / P4-2v optimal; not autotuned.
_B_SCAT = 9
_ELEM_TILE = 2
_TG_SIZE = 128
_TILE_SE = 16
_GRID_BLOCKS = 256  # 2 * 128 SMs on RTX 4090

# CuPy / NVRTC auto-derives ``--gpu-architecture`` from the current device,
# so we don't pin it here. Tuning constants (B_SCAT, ELEM_TILE, TG_SIZE)
# are still hardwired for sm_89 and may need adjustment for sm_80 / sm_90.

# Default 48 KB CUDA dynamic-shmem cap. We assert against this so a future
# probe that pushes shmem over the cap fails fast instead of silently
# truncating the kernel launch.
_DEFAULT_SHMEM_CAP_BYTES = 48 * 1024

_source_cache: dict[str, str] = {}
_kernel_cache: dict[tuple[int, int, int], Any] = {}


def _load_source(filename: str) -> str:
    if filename not in _source_cache:
        _source_cache[filename] = (_KERNELS_DIR / filename).read_text()
    return _source_cache[filename]


def _shmem_bytes(n_elem: int, n_sub: int) -> int:
    """Bytes of dynamic shared memory required by the v25c kernel.

    Layout (see ``simus_fused.cu``):
        7 * B_SCAT * N_ES floats of TX/RX geometry + 3 * N_ELEM floats of
        per-element broadcast (da_init_re, da_init_im, dps).
    """
    n_es = n_elem * n_sub
    return (7 * _B_SCAT * n_es + 3 * n_elem) * 4


def _get_kernel(n_elem: int, n_sub: int, n_freq: int) -> Any:
    """Compile + cache simus_fused_kernel for the given problem shape.

    The cache key is ``(n_elem, n_sub, n_freq)`` -- ``n_scat`` is not in
    the key because the kernel grid-strides over scatterers (one fused
    launch covers the whole sweep, unlike the Metal split-kernel path).
    """
    key = (n_elem, n_sub, n_freq)
    if key in _kernel_cache:
        return _kernel_cache[key]

    n_es = n_elem * n_sub
    max_fpt = (n_freq + _TG_SIZE - 1) // _TG_SIZE

    options = (
        "--std=c++17",
        "--use_fast_math",
        "--extra-device-vectorization",
        f"-DN_ELEM={n_elem}",
        f"-DN_SUB={n_sub}",
        f"-DN_FREQ={n_freq}",
        f"-DN_ES={n_es}",
        f"-DTILE_SE={_TILE_SE}",
        f"-DTG_SIZE={_TG_SIZE}",
        f"-DMAX_FPT={max_fpt}",
        f"-DB_SCAT={_B_SCAT}",
        f"-DELEM_TILE={_ELEM_TILE}",
    )

    module = cp.RawModule(
        code=_load_source(_SOURCE_NAME),
        backend="nvrtc",
        options=options,
        name_expressions=("simus_fused_kernel",),
    )
    kernel = module.get_function("simus_fused_kernel")
    _kernel_cache[key] = kernel
    return kernel


def _prepare_inputs(
    scatterers: Any,
    rc: Any,
    delays_clean: Any,
    tx_apodization: Any,
    plan: SimusPlan,
    params: TransducerParams,
    medium: MediumParams,
) -> dict[str, Any]:
    """Pack the 15 input arrays + 12 scalars the v25c kernel expects.

    Mirrors ``metal_simus._prepare_common`` but without the
    ``(n_scat, n_elem, n_sub)`` expansion: v25c does its own Phase-1
    geometry from the flat per-element / per-sub-element inputs.
    """
    c = medium.speed_of_sound
    alpha = medium.attenuation
    n_elem = params.n_elements
    n_sub = plan.n_sub
    n_freq = int(plan.selected_freqs.shape[0])

    # element_positions with `xp=cp` returns CuPy arrays directly.
    xp_cp: _ArrayNamespace = cp  # type: ignore[assignment]
    elem_pos, theta_e, apex_offset = element_positions(n_elem, params.pitch, params.radius, xp_cp)
    if theta_e is None:
        theta_e = cp.zeros(n_elem, dtype=cp.float32)

    # Sub-element offsets per (elem, sub) flattened to N_ES with se = elem*n_sub + sub
    # (see kernel line `int elem = se / N_SUB;`).
    offsets = cast(cp.ndarray, _subelement_centroids(params.element_width, n_sub, theta_e, xp_cp))
    sub_dx = cp.ascontiguousarray(offsets[..., 0].reshape(-1).astype(cp.float32))
    sub_dz = cp.ascontiguousarray(offsets[..., 1].reshape(-1).astype(cp.float32))

    cos_te = cp.ascontiguousarray(cp.cos(theta_e).astype(cp.float32))
    sin_neg_te = cp.ascontiguousarray(cp.sin(-theta_e).astype(cp.float32))

    # Frequency-grid scalars
    freq_start = float(plan.selected_freqs[0])
    freq_step = float(plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freq > 1 else 0.0

    # Delay+apodization as separate per-element arrays. The kernel folds
    # tx_apodization into the initial value and steps phase by 2*pi*freq_step
    # per outer-frequency iteration.
    da_init_re = cp.ascontiguousarray((cp.cos(2 * pi * freq_start * delays_clean) * tx_apodization).astype(cp.float32))
    da_init_im = cp.ascontiguousarray((cp.sin(2 * pi * freq_start * delays_clean) * tx_apodization).astype(cp.float32))
    dps = cp.ascontiguousarray((2 * pi * freq_step * delays_clean).astype(cp.float32))

    # Pulse * probe (complex), and probe magnitude separately for the RX leg.
    pulse_probe = cast(cp.ndarray, plan.pulse_spectrum * plan.probe_spectrum).astype(cp.complex64)
    pp_re = cp.ascontiguousarray(cp.real(pulse_probe).astype(cp.float32))
    pp_im = cp.ascontiguousarray(cp.imag(pulse_probe).astype(cp.float32))
    probe_real = cp.ascontiguousarray(cp.asarray(plan.probe_spectrum).astype(cp.float32))

    # Convex array radius is float('inf') for linear arrays. Replace with
    # 1e31 so the kernel's `radius * radius` stays finite in fp32.
    radius_v = params.radius if params.radius != inf else 1e31

    return {
        "scat_x": cp.ascontiguousarray(scatterers[:, 0].astype(cp.float32)),
        "scat_z": cp.ascontiguousarray(scatterers[:, 1].astype(cp.float32)),
        "rc": cp.ascontiguousarray(rc.astype(cp.float32)),
        "elem_x": cp.ascontiguousarray(elem_pos[:, 0].astype(cp.float32)),
        "elem_z": cp.ascontiguousarray(elem_pos[:, 1].astype(cp.float32)),
        "cos_te": cos_te,
        "sin_neg_te": sin_neg_te,
        "sub_dx": sub_dx,
        "sub_dz": sub_dz,
        "da_init_re": da_init_re,
        "da_init_im": da_init_im,
        "dps": dps,
        "pp_re": pp_re,
        "pp_im": pp_im,
        "probe_real": probe_real,
        "n_scat": int(scatterers.shape[0]),
        "kw_init": 2 * pi * freq_start / c,
        "alpha_init": alpha / NEPER_TO_DB * freq_start / 1e6 * 1e2,
        "kw_step": 2 * pi * freq_step / c,
        "alpha_step": alpha / NEPER_TO_DB * freq_step / 1e6 * 1e2,
        "min_dist": c / params.freq_center / 2.0,
        "seg_len": plan.seg_length,
        "center_kw": 2 * pi * params.freq_center / c,
        "inv_nsub": 1.0 / n_sub,
        "radius_v": radius_v,
        "apex_offset": apex_offset,
        "n_elem": n_elem,
        "n_sub": n_sub,
        "n_freq": n_freq,
    }


def simus_cuda(
    scatterers: Any,
    rc: Any,
    params: TransducerParams,
    plan: SimusPlan,
    medium: MediumParams,
    delays_clean: Any,
    tx_apodization: Any,
) -> Any:
    """Compute simus RF spectrum using the v25c CUDA kernel via CuPy/NVRTC.

    Single fused TX+RX kernel that grid-strides over scatterers; no
    chunking is needed since per-thread state is in registers.

    Args:
        scatterers: Scatterer positions (x, z) in meters. Shape ``(n_scat, 2)``.
        rc: Reflection coefficients. Shape ``(n_scat,)``.
        params: Transducer parameters.
        plan: Precomputed frequency plan from ``simus_precompute``.
        medium: Medium parameters.
        delays_clean: NaN-cleaned delays. Shape ``(n_elements,)``.
        tx_apodization: Per-element apodization (NaN-zeroed). Shape ``(n_elements,)``.

    Returns:
        Complex RF spectrum, shape ``(n_freq, n_elements)``, dtype ``complex64``.
    """
    d = _prepare_inputs(scatterers, rc, delays_clean, tx_apodization, plan, params, medium)
    n_elem, n_sub, n_freq = d["n_elem"], d["n_sub"], d["n_freq"]

    shmem = _shmem_bytes(n_elem, n_sub)
    if shmem > _DEFAULT_SHMEM_CAP_BYTES:
        # Cross this bridge when an autotune probe pushes us over -- would
        # need cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) here.
        msg = (
            f"v25c shmem {shmem} B exceeds default {_DEFAULT_SHMEM_CAP_BYTES} B cap "
            f"for (n_elem={n_elem}, n_sub={n_sub}); raise the cap explicitly."
        )
        raise RuntimeError(msg)

    kernel = _get_kernel(n_elem, n_sub, n_freq)

    # Output buffers; kernel uses atomicAdd into spect_re[elem*N_FREQ + f].
    spect_re = cp.zeros(n_elem * n_freq, dtype=cp.float32)
    spect_im = cp.zeros(n_elem * n_freq, dtype=cp.float32)

    args = (
        d["scat_x"],
        d["scat_z"],
        d["rc"],
        d["elem_x"],
        d["elem_z"],
        d["cos_te"],
        d["sin_neg_te"],
        d["sub_dx"],
        d["sub_dz"],
        d["da_init_re"],
        d["da_init_im"],
        d["dps"],
        d["pp_re"],
        d["pp_im"],
        d["probe_real"],
        spect_re,
        spect_im,
        cp.int32(d["n_scat"]),
        cp.float32(d["kw_init"]),
        cp.float32(d["alpha_init"]),
        cp.float32(d["kw_step"]),
        cp.float32(d["alpha_step"]),
        cp.float32(d["min_dist"]),
        cp.float32(d["seg_len"]),
        cp.float32(d["center_kw"]),
        cp.float32(d["inv_nsub"]),
        cp.float32(d["radius_v"]),
        cp.float32(d["apex_offset"]),
    )

    kernel(
        grid=(_GRID_BLOCKS, 1, 1),
        block=(_TG_SIZE, 1, 1),
        args=args,
        shared_mem=shmem,
    )

    # Row-major (n_elem, n_freq) -> column-major (n_freq, n_elem) complex64
    # to match metal_simus / _simus_freq_outer_python output convention.
    spect = (spect_re + 1j * spect_im).reshape(n_elem, n_freq).T
    return spect.astype(cp.complex64)
