"""Profiling harness for Exp 13: L2 persistence hints.

Uses cuCtxSetLimit + CU_MEM_RANGE_ATTRIBUTE to pin the output array in L2 cache.
Falls back to cudaAccessPropertyPersisting via CUDA runtime if driver API unavailable.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ncu_profile import prepare, upload, compute_shmem

from fast_simus.kernels.cuda_runtime import (
    compile_module, device_alloc, get_function,
    launch_kernel, synchronize, set_max_dynamic_shared_mem,
    _get_cuda, _get_context,
)

_TG_SIZE = 128

CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 5
CU_MEM_ADVISE_SET_ACCESSED_BY = 5


def set_l2_persistence(cuda, d_ptr, nbytes, l2_persist_bytes):
    """Try to pin a device memory range in L2 cache via stream attribute."""
    try:
        status = cuda.cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, l2_persist_bytes)
        if status != 0:
            print(f"  cuCtxSetLimit(PERSISTING_L2): status={status} (may not be supported)")
            return False
        print(f"  L2 persistence limit set to {l2_persist_bytes/1024:.0f} KB")

        CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
        stream = ctypes.c_void_p(0)

        class CUaccessPolicyWindow(ctypes.Structure):
            _fields_ = [
                ("base_ptr", ctypes.c_void_p),
                ("num_bytes", ctypes.c_size_t),
                ("hitRatio", ctypes.c_float),
                ("hitProp", ctypes.c_int),
                ("missProp", ctypes.c_int),
            ]

        CU_ACCESS_PROPERTY_PERSISTING = 2
        CU_ACCESS_PROPERTY_STREAMING = 1

        window = CUaccessPolicyWindow()
        window.base_ptr = d_ptr.value if hasattr(d_ptr, 'value') else d_ptr
        window.num_bytes = nbytes
        window.hitRatio = 1.0
        window.hitProp = CU_ACCESS_PROPERTY_PERSISTING
        window.missProp = CU_ACCESS_PROPERTY_STREAMING

        class CUstreamAttrValue(ctypes.Union):
            _fields_ = [
                ("accessPolicyWindow", CUaccessPolicyWindow),
            ]

        attr_val = CUstreamAttrValue()
        attr_val.accessPolicyWindow = window

        status = cuda.cuStreamSetAttribute(
            stream,
            CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            ctypes.byref(attr_val),
            ctypes.sizeof(CUstreamAttrValue),
        )
        if status != 0:
            print(f"  cuStreamSetAttribute: status={status} (stream attribute not supported)")
            return False

        print(f"  L2 persistence set for {nbytes/1024:.0f} KB output array")
        return True
    except Exception as e:
        print(f"  L2 persistence failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_path", help="Path to .cu kernel file")
    parser.add_argument("--b-scat", type=int, default=5)
    parser.add_argument("--elem-tile", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=192)
    parser.add_argument("--n-scat", type=int, default=100_000)
    parser.add_argument("--no-persist", action="store_true", help="Disable L2 persistence (control)")
    args = parser.parse_args()

    d = prepare(args.n_scat)
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE

    defs = {"N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
            "TILE_SE": 16, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt,
            "B_SCAT": args.b_scat, "ELEM_TILE": args.elem_tile}

    kernel_name = Path(args.kernel_path).stem
    shmem = compute_shmem(kernel_name, args.b_scat, nes, nf, ne)
    out_size = nf * ne
    out_bytes = out_size * 4 * 2

    print(f"Kernel: {args.kernel_path}")
    print(f"N_SCAT={args.n_scat:,}, N_FREQ={nf}, N_ELEM={ne}")
    print(f"B_SCAT={args.b_scat}, ELEM_TILE={args.elem_tile}, blocks={args.blocks}")
    print(f"Output array: {out_bytes/1024:.0f} KB ({out_size*2} floats)")
    print(f"L2 persistence: {'DISABLED' if args.no_persist else 'ENABLED'}")

    print("Compiling...", flush=True)
    source = Path(args.kernel_path).read_text()
    mod = compile_module(source, defines=tuple(sorted(defs.items())))
    func = get_function(mod, "simus_fused_kernel")
    cuda = _get_cuda(); _get_context()
    if shmem > 49152:
        set_max_dynamic_shared_mem(func, shmem)

    CU_FUNC_ATTR_NUM_REGS = 4
    regs = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(regs), CU_FUNC_ATTR_NUM_REGS, func)
    print(f"Registers/thread: {regs.value}")

    d_ore = device_alloc(out_size * 4)
    d_oim = device_alloc(out_size * 4)

    if not args.no_persist:
        print("Setting L2 persistence hints...")
        ok_re = set_l2_persistence(cuda, d_ore, out_size * 4, out_bytes)
        if not ok_re:
            print("WARNING: L2 persistence not available, running without it")

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

    results = []
    for run in range(3):
        cuda.cuMemsetD32_v2(d_ore, 0, out_size)
        cuda.cuMemsetD32_v2(d_oim, 0, out_size)
        synchronize()
        t0 = time.perf_counter()
        launch_kernel(func, grid, block, kernel_args, shmem)
        synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        results.append(ms)
        print(f"  Run {run+1}: {ms:.1f}ms ({args.n_scat/(t1-t0):,.0f} scat/s)")

    median = sorted(results)[1]
    print(f"Median: {median:.1f}ms ({args.n_scat/(median/1000):,.0f} scat/s)")
    print("Done.")


if __name__ == "__main__":
    main()
