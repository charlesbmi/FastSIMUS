"""Reusable ncu profiling harness for kernel experiments.

Usage:
    uv run python tools/ncu_profile.py <kernel_path> [--b-scat N] [--elem-tile N] [--blocks N] [--n-scat N]

The script compiles the kernel, runs it, and prints timing. When wrapped with ncu,
the second kernel launch (after warmup) gets profiled.

After ncu finishes, run:
    sudo /usr/local/cuda-12.2/bin/ncu --import <report>.ncu-rep --page raw 2>&1 | python tools/ncu_parse.py
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
    compile_module, device_alloc, get_function,
    launch_kernel, memcpy_dtoh, memcpy_htod, synchronize,
    set_max_dynamic_shared_mem, _get_cuda, _get_context,
)
from fast_simus.medium_params import MediumParams
from fast_simus.simus import simus_precompute
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils.geometry import element_positions

_NEPER_TO_DB = 8.685889638065036
_TG_SIZE = 128


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


def compute_shmem(kernel_name, b_scat, n_es, n_freq, n_elem, extra_defs=None):
    if "v6" in kernel_name or "v5" in kernel_name:
        return (9 * n_es + 3 * n_elem) * 4
    n_es_pad = extra_defs.get("N_ES_PAD", n_es) if extra_defs else n_es
    base = 7 * b_scat * n_es_pad + 2 * b_scat * n_freq + 3 * n_elem
    if "shmem" in kernel_name:
        base += 4 * n_es
    return base * 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_path", help="Path to .cu kernel file")
    parser.add_argument("--b-scat", type=int, default=4)
    parser.add_argument("--elem-tile", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=192)
    parser.add_argument("--n-scat", type=int, default=100_000)
    parser.add_argument("--extra-defines", type=str, default="",
                        help="Extra defines as KEY=VAL,KEY=VAL")
    args = parser.parse_args()

    d = prepare(args.n_scat)
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE

    defs = {"N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
            "TILE_SE": 16, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt}

    kernel_name = Path(args.kernel_path).stem
    if "v6" not in kernel_name:
        defs["B_SCAT"] = args.b_scat
        defs["ELEM_TILE"] = args.elem_tile

    if args.extra_defines:
        for pair in args.extra_defines.split(","):
            k, v = pair.split("=")
            defs[k.strip()] = int(v.strip())

    shmem = compute_shmem(kernel_name, args.b_scat, nes, nf, ne, defs)

    print(f"Kernel: {args.kernel_path}")
    print(f"N_SCAT={args.n_scat:,}, N_FREQ={nf}, N_ELEM={ne}")
    print(f"B_SCAT={defs.get('B_SCAT','N/A')}, ELEM_TILE={defs.get('ELEM_TILE','N/A')}, blocks={args.blocks}")
    print(f"Shared memory: {shmem} bytes ({shmem/1024:.1f} KB)")
    print(f"Defines: {defs}")

    print("Compiling...", flush=True)
    source = Path(args.kernel_path).read_text()
    mod = compile_module(source, defines=tuple(sorted(defs.items())))
    func = get_function(mod, "simus_fused_kernel")
    cuda = _get_cuda(); _get_context()
    if shmem > 49152:
        set_max_dynamic_shared_mem(func, shmem)

    CU_FUNC_ATTR_NUM_REGS = 4
    CU_FUNC_ATTR_LOCAL_SIZE_BYTES = 3
    regs = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(regs), CU_FUNC_ATTR_NUM_REGS, func)
    local_mem = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(local_mem), CU_FUNC_ATTR_LOCAL_SIZE_BYTES, func)
    occ = ctypes.c_int()
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(occ), func, _TG_SIZE, shmem)
    print(f"Registers/thread: {regs.value}, Local mem: {local_mem.value}B, Max blocks/SM: {occ.value}")

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

    print("Warmup...", flush=True)
    cuda.cuMemsetD32_v2(d_ore, 0, out_size)
    cuda.cuMemsetD32_v2(d_oim, 0, out_size)
    synchronize()
    launch_kernel(func, grid, block, kernel_args, shmem)
    synchronize()

    print("Profiled launch...", flush=True)
    cuda.cuMemsetD32_v2(d_ore, 0, out_size)
    cuda.cuMemsetD32_v2(d_oim, 0, out_size)
    synchronize()
    t0 = time.perf_counter()
    launch_kernel(func, grid, block, kernel_args, shmem)
    synchronize()
    t1 = time.perf_counter()
    print(f"Kernel: {(t1-t0)*1000:.1f}ms ({args.n_scat/(t1-t0):,.0f} scat/s)")
    print("Done.")


if __name__ == "__main__":
    main()
