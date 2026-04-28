"""Profile a v21 idealized-ceiling variant via NVRTC compile + single launch.

Designed to be wrapped by ncu:
    sudo /usr/local/cuda-12.2/bin/ncu \
        --target-processes all \
        --launch-skip 1 --launch-count 1 \
        --set full \
        -f -o 4090_v21_idealized.ncu-rep \
        $(which uv) run python tools/ncu_profile_v21.py \
            --variant idealized --b-scat 5 --elem-tile 8 --blocks 256

Runs a warmup + one timed launch so ncu can replay the single profiled launch.
"""
import argparse
import ctypes
import os
import sys
import time
from math import inf, pi
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from fast_simus.kernels.cuda_runtime import (
    _get_context,
    _get_cuda,
    compile_module,
    device_alloc,
    get_function,
    launch_kernel,
    memcpy_htod,
    set_max_dynamic_shared_mem,
    synchronize,
)
from fast_simus.medium_params import MediumParams
from fast_simus.simus import simus_precompute
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils.geometry import element_positions

_NEPER_TO_DB = 8.685889638065036
_TG_SIZE = 128
_TILE_SE = 16

VARIANT_PATHS = {
    "noatom":     "src/fast_simus/kernels/simus_fused_v21_noatom.cu",
    "constfreq":  "src/fast_simus/kernels/simus_fused_v21_constfreq.cu",
    "singleelem": "src/fast_simus/kernels/simus_fused_v21_singleelem.cu",
    "idealized":  "src/fast_simus/kernels/simus_fused_v21_idealized.cu",
}


def compute_shmem_v21(variant, b_scat, n_es, n_freq, n_elem):
    tx_half_bytes = 2 * b_scat * n_freq * 2
    tx_half_floats = (tx_half_bytes + 3) // 4
    if variant == "noatom":
        geo_floats = 7 * b_scat * n_es
        delay_floats = 3 * n_elem
    elif variant == "constfreq":
        geo_floats = 3 * b_scat * n_es
        delay_floats = 3 * n_elem
    elif variant == "singleelem":
        geo_floats = 7 * b_scat
        delay_floats = 3
    elif variant == "idealized":
        geo_floats = 3 * b_scat
        delay_floats = 3
    else:
        raise ValueError(f"unknown variant: {variant}")
    return (geo_floats + tx_half_floats + delay_floats) * 4


def prepare(n_scat):
    params = P4_2v()
    medium = MediumParams()
    n_elem = params.n_elements
    rng = np.random.default_rng(42)
    scat_np = np.column_stack([
        rng.uniform(-0.02, 0.02, n_scat),
        rng.uniform(0.01, 0.08, n_scat),
    ]).astype(np.float32)
    rc_np = rng.standard_normal(n_scat).astype(np.float32)
    delays_np = np.zeros(n_elem, dtype=np.float32)
    plan = simus_precompute(scat_np, rc_np, delays_np, params, medium)
    nf = int(plan.selected_freqs.shape[0])
    ns = plan.n_sub
    nes = n_elem * ns
    element_pos, theta_raw, apex_offset = element_positions(
        n_elem, params.pitch, params.radius, np)
    if theta_raw is None:
        theta_raw = np.zeros(n_elem, dtype=np.float32)
    elem_x = np.ascontiguousarray(element_pos[:, 0], dtype=np.float32)
    elem_z = np.ascontiguousarray(element_pos[:, 1], dtype=np.float32)
    seg_length = params.element_width / ns
    seg_offsets = np.array(
        [-params.element_width / 2.0 + seg_length / 2.0 + i * seg_length
         for i in range(ns)], dtype=np.float32)
    cos_th = np.cos(theta_raw).astype(np.float32)
    sin_neg_th = np.sin(-theta_raw).astype(np.float32)
    sub_dx = np.zeros(nes, dtype=np.float32)
    sub_dz = np.zeros(nes, dtype=np.float32)
    for e in range(n_elem):
        for s in range(ns):
            sub_dx[e * ns + s] = seg_offsets[s] * cos_th[e]
            sub_dz[e * ns + s] = seg_offsets[s] * sin_neg_th[e]
    freq_start = float(plan.selected_freqs[0])
    freq_step = float(plan.selected_freqs[1] - plan.selected_freqs[0]) if nf > 1 else 0.0
    c = medium.speed_of_sound
    spectra = np.asarray(plan.pulse_spectrum * plan.probe_spectrum)
    probe_raw = np.asarray(plan.probe_spectrum)
    return dict(
        scat_np=scat_np, rc_np=rc_np, elem_x=elem_x, elem_z=elem_z,
        cos_te=cos_th, sin_neg_te=sin_neg_th,
        sub_dx=sub_dx, sub_dz=sub_dz,
        da_init_re=np.cos(2*pi*freq_start*delays_np).astype(np.float32),
        da_init_im=np.sin(2*pi*freq_start*delays_np).astype(np.float32),
        dps_np=(2*pi*freq_step*delays_np).astype(np.float32),
        pp_re=np.real(spectra).astype(np.float32),
        pp_im=np.imag(spectra).astype(np.float32),
        probe_np=(np.abs(probe_raw) if np.iscomplexobj(probe_raw) else probe_raw).astype(np.float32),
        n_scat=n_scat, n_freq=nf, n_elem=n_elem, n_sub=ns, n_es=nes,
        kw_init=2*pi*freq_start/c,
        alpha_init=medium.attenuation/_NEPER_TO_DB*freq_start/1e6*1e2,
        kw_step=2*pi*freq_step/c,
        alpha_step=medium.attenuation/_NEPER_TO_DB*freq_step/1e6*1e2,
        min_dist=c/params.freq_center/2.0,
        seg_length=seg_length,
        center_kw=2*pi*params.freq_center/c,
        inv_nsub=1.0/ns,
        radius_v=params.radius if params.radius != inf else 1e31,
        apex_offset=apex_offset,
    )


def upload(data):
    p = device_alloc(data.nbytes)
    memcpy_htod(p, data)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANT_PATHS), default="idealized")
    parser.add_argument("--b-scat", type=int, default=5)
    parser.add_argument("--elem-tile", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=256)
    parser.add_argument("--n-scat", type=int, default=100_000)
    args = parser.parse_args()

    kernel_path = VARIANT_PATHS[args.variant]
    source = Path(kernel_path).read_text()
    d = prepare(args.n_scat)
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]

    cuda = _get_cuda()
    _get_context()

    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE
    defs = {
        "N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
        "TILE_SE": _TILE_SE, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt,
        "B_SCAT": args.b_scat, "ELEM_TILE": args.elem_tile,
    }
    shmem = compute_shmem_v21(args.variant, args.b_scat, nes, nf, ne)

    mod = compile_module(source, defines=tuple(sorted(defs.items())))
    func = get_function(mod, "simus_fused_kernel")
    if shmem > 49152:
        set_max_dynamic_shared_mem(func, shmem)

    CU_FUNC_ATTR_NUM_REGS = 4
    CU_FUNC_ATTR_LOCAL_SIZE_BYTES = 3
    regs = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(regs), CU_FUNC_ATTR_NUM_REGS, func)
    local_mem = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(local_mem), CU_FUNC_ATTR_LOCAL_SIZE_BYTES, func)
    occ = ctypes.c_int()
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(occ), func, _TG_SIZE, shmem)

    out_size = nf * ne
    d_ore = device_alloc(out_size * 4)
    d_oim = device_alloc(out_size * 4)
    d_sx = upload(np.ascontiguousarray(d["scat_np"][:, 0]))
    d_sz = upload(np.ascontiguousarray(d["scat_np"][:, 1]))
    d_rc = upload(d["rc_np"])
    d_ex = upload(d["elem_x"]); d_ez = upload(d["elem_z"])
    d_ct = upload(d["cos_te"]); d_snt = upload(d["sin_neg_te"])
    d_sdx = upload(d["sub_dx"]); d_sdz = upload(d["sub_dz"])
    d_dare = upload(d["da_init_re"]); d_daim = upload(d["da_init_im"])
    d_dps = upload(d["dps_np"])
    d_ppre = upload(d["pp_re"]); d_ppim = upload(d["pp_im"])
    d_probe = upload(d["probe_np"])

    kernel_args = [
        d_sx, d_sz, d_rc, d_ex, d_ez, d_ct, d_snt, d_sdx, d_sdz,
        d_dare, d_daim, d_dps, d_ppre, d_ppim, d_probe,
        d_ore, d_oim,
        ctypes.c_int(d["n_scat"]),
        ctypes.c_float(d["kw_init"]), ctypes.c_float(d["alpha_init"]),
        ctypes.c_float(d["kw_step"]), ctypes.c_float(d["alpha_step"]),
        ctypes.c_float(d["min_dist"]), ctypes.c_float(d["seg_length"]),
        ctypes.c_float(d["center_kw"]), ctypes.c_float(d["inv_nsub"]),
        ctypes.c_float(d["radius_v"]), ctypes.c_float(d["apex_offset"]),
    ]

    grid = (args.blocks, 1, 1)
    block = (_TG_SIZE, 1, 1)

    print(f"v21 {args.variant} | B={args.b_scat} ET={args.elem_tile} "
          f"| N_SCAT={args.n_scat:,}, N_FREQ={nf}, N_ELEM={ne}, N_ES={nes}")
    print(f"shmem={shmem/1024:.1f} KB | regs={regs.value} "
          f"local={local_mem.value}B | blk/SM={occ.value}")

    cuda.cuMemsetD32_v2(d_ore, 0, out_size)
    cuda.cuMemsetD32_v2(d_oim, 0, out_size)
    synchronize()
    launch_kernel(func, grid, block, kernel_args, shmem)
    synchronize()

    cuda.cuMemsetD32_v2(d_ore, 0, out_size)
    cuda.cuMemsetD32_v2(d_oim, 0, out_size)
    synchronize()
    t0 = time.perf_counter()
    launch_kernel(func, grid, block, kernel_args, shmem)
    synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    sps = args.n_scat / (t1 - t0)
    print(f"wall: {ms:.2f} ms = {sps/1e6:.2f} M scat/s")


if __name__ == "__main__":
    main()
