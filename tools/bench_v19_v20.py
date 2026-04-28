"""Benchmark v15 (champion), v19 (fp16 half2 cmul), v20 (2-deep pipeline).

Usage:
    uv run python tools/bench_v19_v20.py [--skip-accuracy]
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
N_SCAT = 100_000
BLOCKS = 256


def prepare():
    params = P4_2v()
    medium = MediumParams()
    n_elem = params.n_elements
    rng = np.random.default_rng(42)
    scat_np = np.column_stack([
        rng.uniform(-0.02, 0.02, N_SCAT),
        rng.uniform(0.01, 0.08, N_SCAT),
    ]).astype(np.float32)
    rc_np = rng.standard_normal(N_SCAT).astype(np.float32)
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
        n_scat=N_SCAT, n_freq=nf, n_elem=n_elem, n_sub=ns, n_es=nes,
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


def compute_shmem_v15(b_scat, n_es, n_freq, n_elem):
    geo_floats = 7 * b_scat * n_es
    tx_half_bytes = 2 * b_scat * n_freq * 2
    tx_half_floats = (tx_half_bytes + 3) // 4
    delay_floats = 3 * n_elem
    return (geo_floats + tx_half_floats + delay_floats) * 4


def bench_kernel(label, kernel_path, b_scat, elem_tile, d, kernel_args,
                 d_ore, d_oim, out_size, nf, ne, nes, ns, reps=5):
    source = Path(kernel_path).read_text()
    cuda = _get_cuda()
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE
    defs = {
        "N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
        "TILE_SE": 16, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt,
        "B_SCAT": b_scat, "ELEM_TILE": elem_tile,
    }

    shmem = compute_shmem_v15(b_scat, nes, nf, ne)

    try:
        mod = compile_module(source, defines=tuple(sorted(defs.items())))
        func = get_function(mod, "simus_fused_kernel")
    except Exception as e:
        print(f"{label:35s} | COMPILE ERROR: {e}")
        return None, None

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

    grid = (BLOCKS, 1, 1)
    block = (_TG_SIZE, 1, 1)

    cuda.cuMemsetD32_v2(d_ore, 0, out_size)
    cuda.cuMemsetD32_v2(d_oim, 0, out_size)
    synchronize()
    launch_kernel(func, grid, block, kernel_args, shmem)
    synchronize()

    out_re = np.empty(out_size, dtype=np.float32)
    out_im = np.empty(out_size, dtype=np.float32)
    memcpy_dtoh(out_re, d_ore)
    memcpy_dtoh(out_im, d_oim)

    cuda.cuMemsetD32_v2(d_ore, 0, out_size)
    cuda.cuMemsetD32_v2(d_oim, 0, out_size)
    synchronize()

    times = []
    for _ in range(reps):
        cuda.cuMemsetD32_v2(d_ore, 0, out_size)
        cuda.cuMemsetD32_v2(d_oim, 0, out_size)
        synchronize()
        t0 = time.perf_counter()
        launch_kernel(func, grid, block, kernel_args, shmem)
        synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    best = min(times)
    sps = N_SCAT / (best / 1000)
    print(f"{label:35s} | R={regs.value:3d} L={local_mem.value:4d}B occ={occ.value} "
          f"| {best:.2f}ms {sps/1e6:.2f}M scat/s")
    return best, np.stack([out_re, out_im])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args()

    d = prepare()
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    cuda = _get_cuda()
    _get_context()

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

    print(f"v19/v20 Benchmark | N_SCAT={N_SCAT:,}, blocks={BLOCKS}")
    print(f"N_FREQ={nf}, N_ELEM={ne}, N_ES={nes}")
    print("=" * 90)

    configs = [
        ("v15 champion (B=9 ET=4)", "src/fast_simus/kernels/simus_fused_v15.cu", 9, 4),
        ("v19 fp16 half2 (B=9 ET=4)", "src/fast_simus/kernels/simus_fused_v19.cu", 9, 4),
        ("v20 pipeline (B=9 ET=4)", "src/fast_simus/kernels/simus_fused_v20.cu", 9, 4),
        ("v19 fp16 half2 (B=12 ET=4)", "src/fast_simus/kernels/simus_fused_v19.cu", 12, 4),
        ("v19 fp16 half2 (B=9 ET=8)", "src/fast_simus/kernels/simus_fused_v19.cu", 9, 8),
        ("v20 pipeline (B=5 ET=4)", "src/fast_simus/kernels/simus_fused_v20.cu", 5, 4),
    ]

    results = {}
    for label, path, b, et in configs:
        t, out = bench_kernel(label, path, b, et, d, kernel_args,
                              d_ore, d_oim, out_size, nf, ne, nes, ns)
        results[label] = (t, out)

    if not args.skip_accuracy:
        ref_label = "v15 champion (B=9 ET=4)"
        _, ref_out = results[ref_label]
        if ref_out is not None:
            print("\n" + "=" * 90)
            print("Accuracy (relative error vs v15 champion):")
            for label, (t, out) in results.items():
                if label == ref_label or out is None:
                    continue
                mask = np.abs(ref_out) > 1e-10
                if mask.sum() == 0:
                    print(f"  {label}: NO VALID OUTPUTS")
                    continue
                rel_err = np.abs(out[mask] - ref_out[mask]) / np.abs(ref_out[mask])
                print(f"  {label}: max={rel_err.max():.2e} mean={rel_err.mean():.2e} "
                      f"p99={np.percentile(rel_err, 99):.2e}")

    print("\n" + "=" * 90)
    print("Target: 3.33ms = 30.0M scat/s")


if __name__ == "__main__":
    main()
