"""Validate v23 / v23b / v25 numerical equivalence vs v11 (fp32 baseline).

All three are built directly on v11 (no precision change). Phase 3 stage
ordering (v23/v23b) and TX storage location (v25 reg vs v11 shmem) are
the only differences; arithmetic is otherwise identical. Expected: max
mag rel err vs v11 below ~1e-3 (fp32 FMA reorder), mean / p99 at the
level of bit-noise.
"""
import ctypes
import os
import sys
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from fast_simus.kernels.cuda_runtime import (
    compile_module, device_alloc, get_function,
    launch_kernel, memcpy_dtoh, synchronize,
    set_max_dynamic_shared_mem, _get_cuda,
)

sys.path.insert(0, "tools")
from ncu_profile import prepare, upload

_TG_SIZE = 128
N_SCAT = 10_000
BLOCKS = 96


def run_kernel(source_path, shmem_bytes, b_scat, elem_tile, d):
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE
    defs = {
        "N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
        "TILE_SE": 16, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt,
        "B_SCAT": b_scat, "ELEM_TILE": elem_tile,
    }
    source = Path(source_path).read_text()
    mod = compile_module(source, defines=tuple(sorted(defs.items())))
    func = get_function(mod, "simus_fused_kernel")
    cuda = _get_cuda()
    if shmem_bytes > 49152:
        set_max_dynamic_shared_mem(func, shmem_bytes)

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

    args = [
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
    launch_kernel(func, (BLOCKS, 1, 1), (_TG_SIZE, 1, 1), args, shmem_bytes)
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


def report(name, ref_name, ref_re, ref_im, test_re, test_im):
    mag_ref = np.sqrt(ref_re**2 + ref_im**2)
    mag_test = np.sqrt(test_re**2 + test_im**2)
    mask = mag_ref > 1e-10
    rer = np.abs(test_re[mask] - ref_re[mask]) / (np.abs(ref_re[mask]) + 1e-30)
    rei = np.abs(test_im[mask] - ref_im[mask]) / (np.abs(ref_im[mask]) + 1e-30)
    rem = np.abs(mag_test[mask] - mag_ref[mask]) / (mag_ref[mask] + 1e-30)
    print(f"\n=== {name} vs {ref_name} (n={mask.sum()}) ===")
    print(f"  max rel err  re={rer.max():.3e}  im={rei.max():.3e}  mag={rem.max():.3e}")
    print(f"  p99 rel err  re={np.percentile(rer,99):.3e}  im={np.percentile(rei,99):.3e}  mag={np.percentile(rem,99):.3e}")
    print(f"  mean rel err re={rer.mean():.3e}  im={rei.mean():.3e}  mag={rem.mean():.3e}")
    return rem.max()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-scat", type=int, default=9)
    ap.add_argument("--elem-tile", type=int, default=4)
    args = ap.parse_args()
    b, et = args.b_scat, args.elem_tile
    print(f"Validation: B={b} ET={et}, N_SCAT={N_SCAT}, BLOCKS={BLOCKS}")
    d = prepare(N_SCAT)
    print(f"N_FREQ={d['n_freq']}, N_ELEM={d['n_elem']}, N_ES={d['n_es']}")

    print("\nRunning v11 (fp32 baseline)...")
    v11_re, v11_im = run_kernel(
        "src/fast_simus/kernels/simus_fused_v11.cu",
        shmem_v11(b, d["n_es"], d["n_freq"], d["n_elem"]), b, et, d)
    print("Running v23 (chain-split, fp32)...")
    v23_re, v23_im = run_kernel(
        "src/fast_simus/kernels/simus_fused_v23_chainsplit.cu",
        shmem_v11(b, d["n_es"], d["n_freq"], d["n_elem"]), b, et, d)
    print("Running v23b (advance-split, fp32)...")
    v23b_re, v23b_im = run_kernel(
        "src/fast_simus/kernels/simus_fused_v23b_advsplit.cu",
        shmem_v11(b, d["n_es"], d["n_freq"], d["n_elem"]), b, et, d)
    print("Running v25 (register TX, fp32)...")
    shmem_v25 = (7 * b * d["n_es"] + 3 * d["n_elem"]) * 4
    v25_re, v25_im = run_kernel(
        "src/fast_simus/kernels/simus_fused_v25_regtx.cu",
        shmem_v25, b, et, d)
    print("Running v25b (register TX, unrolled fi, fp32)...")
    v25b_re, v25b_im = run_kernel(
        "src/fast_simus/kernels/simus_fused_v25b_regtx_unroll.cu",
        shmem_v25, b, et, d)

    def gate(name, ref_re, ref_im, x_re, x_im):
        e_max = report(name, "v11", ref_re, ref_im, x_re, x_im)
        mag_r = np.sqrt(ref_re**2 + ref_im**2)
        mag_x = np.sqrt(x_re**2 + x_im**2)
        mask = mag_r > 1e-10
        rem = np.abs(mag_x[mask] - mag_r[mask]) / (mag_r[mask] + 1e-30)
        p99 = float(np.percentile(rem, 99))
        mean = float(rem.mean())
        ok = e_max < 1e-3 and p99 < 1e-5 and mean < 1e-6
        print(f"  {name} vs v11:  max={e_max:.2e}  p99={p99:.2e}  mean={mean:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
        return ok

    print("\n=== Pass/Fail (vs v11, fp32 reorder noise only) ===")
    gate("v23",  v11_re, v11_im, v23_re,  v23_im)
    gate("v23b", v11_re, v11_im, v23b_re, v23b_im)
    gate("v25",  v11_re, v11_im, v25_re,  v25_im)
    gate("v25b", v11_re, v11_im, v25b_re, v25b_im)


if __name__ == "__main__":
    main()
