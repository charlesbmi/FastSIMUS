"""Compare output accuracy between v11 (fp32 TX) and v15 (fp16 TX).

Runs both kernels on identical input and reports relative error statistics.
"""
import ctypes
import os
import sys
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

sys.path.insert(0, "tools")
from ncu_profile import prepare, upload

_TG_SIZE = 128
N_SCAT = 10_000
BLOCKS = 96


def run_kernel(source_path, shmem_func, b_scat, elem_tile, d, extra_defs=None):
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE
    defs = {
        "N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
        "TILE_SE": 16, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt,
        "B_SCAT": b_scat, "ELEM_TILE": elem_tile,
    }
    if extra_defs:
        defs.update(extra_defs)

    shmem = shmem_func(b_scat, nes, nf, ne)
    source = Path(source_path).read_text()
    mod = compile_module(source, defines=tuple(sorted(defs.items())))
    func = get_function(mod, "simus_fused_kernel")
    cuda = _get_cuda()
    if shmem > 49152:
        set_max_dynamic_shared_mem(func, shmem)

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

    cuda.cuMemsetD32_v2(d_ore, 0, out_size)
    cuda.cuMemsetD32_v2(d_oim, 0, out_size)
    synchronize()
    launch_kernel(func, (BLOCKS, 1, 1), (_TG_SIZE, 1, 1), kernel_args, shmem)
    synchronize()

    out_re = np.empty(out_size, dtype=np.float32)
    out_im = np.empty(out_size, dtype=np.float32)
    memcpy_dtoh(out_re, d_ore)
    memcpy_dtoh(out_im, d_oim)
    return out_re, out_im


def shmem_v11(b, nes, nf, ne):
    return (7 * b * nes + 2 * b * nf + 3 * ne) * 4


def shmem_v15(b, nes, nf, ne):
    geo = 7 * b * nes
    tx_half_bytes = 2 * b * nf * 2
    tx_half_floats = (tx_half_bytes + 3) // 4
    delay = 3 * ne
    return (geo + tx_half_floats + delay) * 4


def main():
    d = prepare(N_SCAT)
    print(f"N_SCAT={N_SCAT}, N_FREQ={d['n_freq']}, N_ELEM={d['n_elem']}")

    print("\nRunning v11 (fp32 TX, B=5 ET=8)...")
    v11_re, v11_im = run_kernel(
        "src/fast_simus/kernels/simus_fused_v11.cu", shmem_v11, 5, 8, d)

    print("Running v15 (fp16 TX, B=5 ET=8)...")
    v15_re, v15_im = run_kernel(
        "src/fast_simus/kernels/simus_fused_v15.cu", shmem_v15, 5, 8, d)

    mag_v11 = np.sqrt(v11_re**2 + v11_im**2)
    mag_v15 = np.sqrt(v15_re**2 + v15_im**2)

    mask = mag_v11 > 1e-10
    rel_err_re = np.abs(v15_re[mask] - v11_re[mask]) / (np.abs(v11_re[mask]) + 1e-30)
    rel_err_im = np.abs(v15_im[mask] - v11_im[mask]) / (np.abs(v11_im[mask]) + 1e-30)
    rel_err_mag = np.abs(mag_v15[mask] - mag_v11[mask]) / (mag_v11[mask] + 1e-30)

    print(f"\n{'Metric':<25} {'re':>12} {'im':>12} {'mag':>12}")
    print("-" * 63)
    print(f"{'Max relative error':<25} {rel_err_re.max():.6e} {rel_err_im.max():.6e} {rel_err_mag.max():.6e}")
    print(f"{'Mean relative error':<25} {rel_err_re.mean():.6e} {rel_err_im.mean():.6e} {rel_err_mag.mean():.6e}")
    print(f"{'Median relative error':<25} {np.median(rel_err_re):.6e} {np.median(rel_err_im):.6e} {np.median(rel_err_mag):.6e}")
    print(f"{'P99 relative error':<25} {np.percentile(rel_err_re, 99):.6e} {np.percentile(rel_err_im, 99):.6e} {np.percentile(rel_err_mag, 99):.6e}")
    print(f"\nNon-zero output elements: {mask.sum()} / {len(mask)}")
    print(f"v11 output range: [{v11_re.min():.4f}, {v11_re.max():.4f}]")
    print(f"v15 output range: [{v15_re.min():.4f}, {v15_re.max():.4f}]")

    tol = 1e-2
    ok = rel_err_mag.max() < tol
    print(f"\nAccuracy check (rtol={tol}): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
