"""Bench v11 vs v23_chainsplit on the correct kernel sweep configs.

v11 is the fp32-precision correct kernel (no fp16 TX). v23 applies the
exp20 ILP restructure (explicit prep/FMA/advance stages) to v11 without
breaking the cmul rotation chain or changing precision.

For comparison the v15 (fp16 TX) row is also included as a reference for
where a concurrent precision-drop sits.

Usage:
    uv run python tools/bench_v23.py
    uv run python tools/bench_v23.py --b-scat 5 9 --elem-tile 4 8
"""
import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from fast_simus.kernels.cuda_runtime import (
    _get_context, _get_cuda, compile_module, device_alloc,
    get_function, launch_kernel, set_max_dynamic_shared_mem, synchronize,
)

sys.path.insert(0, "tools")
from bench_idealized_ceiling import compute_shmem_v15, prepare, upload


def compute_shmem_v11(b, nes, nf, ne):
    """v11 / v23 fp32 TX shmem (no fp16 packing)."""
    return (7 * b * nes + 2 * b * nf + 3 * ne) * 4

_TG_SIZE = 128
_TILE_SE = 16
_BLOCKS = 256
_N_SCAT = 100_000

VARIANTS = {
    "v11":         ("src/fast_simus/kernels/simus_fused_v11.cu",            "fp32"),
    "v23_split":   ("src/fast_simus/kernels/simus_fused_v23_chainsplit.cu", "fp32"),
    "v23b_adv":    ("src/fast_simus/kernels/simus_fused_v23b_advsplit.cu",  "fp32"),
    "v25_regtx":   ("src/fast_simus/kernels/simus_fused_v25_regtx.cu",      "fp32_regtx"),
    "v25b_unroll": ("src/fast_simus/kernels/simus_fused_v25b_regtx_unroll.cu", "fp32_regtx"),
    "v15":         ("src/fast_simus/kernels/simus_fused_v15.cu",            "fp16"),
}


def bench_one(cuda, path, precision, b_scat, elem_tile, blocks, d, kernel_args,
              d_ore, d_oim, out_size, reps=5):
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    if precision == "fp16":
        shmem = compute_shmem_v15(b_scat, nes, nf, ne)
    elif precision == "fp32_regtx":
        shmem = (7 * b_scat * nes + 3 * ne) * 4
    else:
        shmem = compute_shmem_v11(b_scat, nes, nf, ne)
    source = Path(path).read_text()
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE
    defs = {
        "N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
        "TILE_SE": _TILE_SE, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt,
        "B_SCAT": b_scat, "ELEM_TILE": elem_tile,
    }
    try:
        mod = compile_module(source, defines=tuple(sorted(defs.items())))
        func = get_function(mod, "simus_fused_kernel")
    except Exception as e:
        return {"error": f"COMPILE: {str(e)[:80]}"}
    if shmem > 49152:
        try:
            set_max_dynamic_shared_mem(func, shmem)
        except Exception as e:
            return {"error": f"SHMEM_SET: {str(e)[:80]}"}
    regs = ctypes.c_int(); local_mem = ctypes.c_int(); occ = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(regs), 4, func)
    cuda.cuFuncGetAttribute(ctypes.byref(local_mem), 3, func)
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(occ), func, _TG_SIZE, shmem)
    grid = (blocks, 1, 1); block = (_TG_SIZE, 1, 1)
    cuda.cuMemsetD32_v2(d_ore, 0, out_size); cuda.cuMemsetD32_v2(d_oim, 0, out_size)
    synchronize()
    try:
        launch_kernel(func, grid, block, kernel_args, shmem); synchronize()
    except Exception as e:
        return {"error": f"LAUNCH: {str(e)[:80]}", "regs": regs.value,
                "local": local_mem.value, "occ": occ.value, "shmem": shmem}
    times = []
    for _ in range(reps):
        cuda.cuMemsetD32_v2(d_ore, 0, out_size); cuda.cuMemsetD32_v2(d_oim, 0, out_size)
        synchronize()
        t0 = time.perf_counter()
        launch_kernel(func, grid, block, kernel_args, shmem); synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "shmem": shmem, "regs": regs.value, "local": local_mem.value,
        "occ": occ.value, "best_ms": min(times),
        "med_ms": sorted(times)[len(times) // 2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-scat", type=int, nargs="+", default=[5, 9])
    parser.add_argument("--elem-tile", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--blocks", type=int, default=_BLOCKS)
    parser.add_argument("--n-scat", type=int, default=_N_SCAT)
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()

    d = prepare(args.n_scat)
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    cuda = _get_cuda(); _get_context()

    out_size = nf * ne
    d_ore = device_alloc(out_size * 4); d_oim = device_alloc(out_size * 4)
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

    print(f"v11 vs v23_chainsplit (fp32) | N_SCAT={args.n_scat:,}, N_FREQ={nf}, "
          f"N_ELEM={ne}, N_ES={nes}, blocks={args.blocks}")
    print("(v15 fp16 included as reference -- not the precision target.)")
    print("=" * 105)
    header = (f"{'variant':>11} {'prec':>5} {'B':>3} {'ET':>3} {'shmem':>7} "
              f"{'regs':>5} {'spill':>7} {'blk/SM':>7} "
              f"{'best_ms':>9} {'med_ms':>8} {'M scat/s':>10}")
    print(header); print("-" * len(header))
    rows = {}
    for variant_name, (path, precision) in VARIANTS.items():
        for b in args.b_scat:
            for et in args.elem_tile:
                res = bench_one(cuda, path, precision, b, et, args.blocks, d,
                                kernel_args, d_ore, d_oim, out_size,
                                reps=args.reps)
                if "error" in res:
                    print(f"{variant_name:>11} {precision:>5} {b:3d} {et:3d} ERR ... {res['error']}")
                    continue
                sps = args.n_scat / (res["best_ms"] / 1000.0)
                print(f"{variant_name:>11} {precision:>5} {b:3d} {et:3d} "
                      f"{res['shmem']/1024:6.1f}K {res['regs']:5d} "
                      f"{res['local']:6d}B {res['occ']:7d} "
                      f"{res['best_ms']:9.2f} {res['med_ms']:8.2f} "
                      f"{sps/1e6:10.2f}")
                rows[(variant_name, b, et)] = res

    print("=" * 105)
    print("\nfp32 delta vs v11 baseline:")
    for cand in ["v23_split", "v23b_adv", "v25_regtx", "v25b_unroll"]:
        print(f"\n  -- {cand}")
        for b in args.b_scat:
            for et in args.elem_tile:
                v11 = rows.get(("v11", b, et))
                vx = rows.get((cand, b, et))
                if not v11 or not vx or "error" in v11 or "error" in vx:
                    continue
                speedup = v11["best_ms"] / vx["best_ms"]
                m11 = args.n_scat / (v11["best_ms"] / 1000.0) / 1e6
                mx = args.n_scat / (vx["best_ms"] / 1000.0) / 1e6
                print(f"     B={b} ET={et}: v11 {v11['best_ms']:6.2f} ms ({m11:5.2f} M) "
                      f"-> {cand} {vx['best_ms']:6.2f} ms ({mx:5.2f} M) "
                      f"= {speedup:.3f}x  (regs {v11['regs']}->{vx['regs']}, "
                      f"spill {v11['local']}->{vx['local']}B, "
                      f"blk/SM {v11['occ']}->{vx['occ']})")


if __name__ == "__main__":
    main()
