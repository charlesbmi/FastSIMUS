"""Profile v25 / v25b regtx variants via NVRTC + single launch.

Designed to be wrapped by ncu:
    sudo /usr/local/cuda/bin/ncu \\
        --target-processes all \\
        --launch-skip 1 --launch-count 1 \\
        --set full \\
        -f -o 4090_v25b_b7et4.ncu-rep \\
        $(which uv) run python tools/ncu_profile_v25.py \\
            --variant v25b --b-scat 7 --elem-tile 4 --blocks 256
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

sys.path.insert(0, "tools")
from ncu_profile_v22 import prepare, upload  # reuse harness

VARIANT_PATHS = {
    "v25":  "src/fast_simus/kernels/simus_fused_v25_regtx.cu",
    "v25b": "src/fast_simus/kernels/simus_fused_v25b_regtx_unroll.cu",
    "v25c": "src/fast_simus/kernels/simus_fused_v25c_svshmem.cu",
    "v26":  "src/fast_simus/kernels/simus_fused_v26_freqchunk.cu",
    "v11":  "src/fast_simus/kernels/simus_fused_v11.cu",
    "v15":  "src/fast_simus/kernels/simus_fused_v15.cu",
}
DEFAULT_TG = 128
DEFAULT_TILE_SE = 16


def compute_shmem(b_scat, nes, nf, ne, regtx):
    """Regtx variants drop the 2*B*N_FREQ TX block."""
    if regtx:
        return (7 * b_scat * nes + 3 * ne) * 4
    return (7 * b_scat * nes + 2 * b_scat * nf + 3 * ne) * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_PATHS), default="v25b")
    ap.add_argument("--b-scat", type=int, default=7)
    ap.add_argument("--elem-tile", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--n-scat", type=int, default=100_000)
    ap.add_argument("--tg-size", type=int, default=DEFAULT_TG)
    ap.add_argument("--tile-se", type=int, default=DEFAULT_TILE_SE)
    args = ap.parse_args()

    kernel_path = VARIANT_PATHS[args.variant]
    source = Path(kernel_path).read_text()
    d = prepare(args.n_scat)
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]

    cuda = _get_cuda()
    _get_context()

    max_fpt = (nf + args.tg_size - 1) // args.tg_size
    defs = {
        "N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
        "TILE_SE": args.tile_se, "TG_SIZE": args.tg_size, "MAX_FPT": max_fpt,
        "B_SCAT": args.b_scat, "ELEM_TILE": args.elem_tile,
    }
    regtx = args.variant.startswith("v25")
    shmem = compute_shmem(args.b_scat, nes, nf, ne, regtx=regtx)

    mod = compile_module(source, defines=tuple(sorted(defs.items())))
    func = get_function(mod, "simus_fused_kernel")
    if shmem > 49152:
        set_max_dynamic_shared_mem(func, shmem)

    CU_FUNC_ATTR_NUM_REGS = 4
    CU_FUNC_ATTR_LOCAL_SIZE_BYTES = 3
    regs = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(regs), CU_FUNC_ATTR_NUM_REGS, func)
    local_mem = ctypes.c_int()
    cuda.cuFuncGetAttribute(
        ctypes.byref(local_mem), CU_FUNC_ATTR_LOCAL_SIZE_BYTES, func)
    occ = ctypes.c_int()
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(occ), func, args.tg_size, shmem)

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
    block = (args.tg_size, 1, 1)

    print(f"{args.variant} | B={args.b_scat} ET={args.elem_tile} "
          f"TG={args.tg_size} TILE_SE={args.tile_se} "
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
