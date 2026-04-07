"""Profiling harness for v16 Grid-Y element group partitioning kernel.

Grid is (blocks_x, N_ELEM_GROUPS, 1) where each Y-dimension selects one element group.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ncu_profile import prepare, upload, compute_shmem

from fast_simus.kernels.cuda_runtime import (
    compile_module, device_alloc, get_function,
    launch_kernel, synchronize, set_max_dynamic_shared_mem,
    _get_cuda, _get_context,
)

_TG_SIZE = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_path", help="Path to .cu kernel file")
    parser.add_argument("--b-scat", type=int, default=5)
    parser.add_argument("--elem-tile", type=int, default=8)
    parser.add_argument("--blocks-x", type=int, default=12)
    parser.add_argument("--n-scat", type=int, default=100_000)
    args = parser.parse_args()

    d = prepare(args.n_scat)
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE
    n_elem_groups = nes // args.elem_tile

    defs = {"N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
            "TILE_SE": 16, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt,
            "B_SCAT": args.b_scat, "ELEM_TILE": args.elem_tile}

    shmem = compute_shmem("v16", args.b_scat, nes, nf, ne)
    total_blocks = args.blocks_x * n_elem_groups

    print(f"Kernel: {args.kernel_path}")
    print(f"N_SCAT={args.n_scat:,}, N_FREQ={nf}, N_ELEM={ne}")
    print(f"B_SCAT={args.b_scat}, ELEM_TILE={args.elem_tile}")
    print(f"Grid: ({args.blocks_x}, {n_elem_groups}, 1) = {total_blocks} total blocks")
    print(f"Shared memory: {shmem} bytes ({shmem/1024:.1f} KB)")

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
    d_sx = upload(d["scat_np"][:, 0].copy())
    d_sz = upload(d["scat_np"][:, 1].copy())
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

    grid = (args.blocks_x, n_elem_groups, 1)
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
