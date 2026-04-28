"""Sweep v22 6-blocks/SM variants across B_SCAT and ELEM_TILE.

For each (variant, B, ET) config, compiles, launches, measures best/median
time over 5 repeats, and records occupancy + register + shmem metrics.

Baselines are hardcoded (from exp19):
  v15 champion   B=9 ET=4: 6.44 ms = 15.53 M scat/s
  v21 idealized  B=5 ET=8: 4.72 ms = 21.17 M scat/s

Usage:
    uv run python tools/bench_6blk_ceiling.py
    uv run python tools/bench_6blk_ceiling.py --variants lb6 nochain
    uv run python tools/bench_6blk_ceiling.py --b-scat 5 --elem-tile 8
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
_BLOCKS = 256
_N_SCAT = 100_000

# Hardcoded baselines from exp19.
_CHAMP_MS = 6.44   # v15 B=9 ET=4
_IDEAL_MS = 4.72   # v21 idealized B=5 ET=8

VARIANTS = ["lb6", "nochain", "nochain_ilp", "tg64", "floor"]
VARIANT_PATHS = {
    "lb6":         "src/fast_simus/kernels/simus_fused_v22_lb6.cu",
    "nochain":     "src/fast_simus/kernels/simus_fused_v22_nochain.cu",
    "nochain_ilp": "src/fast_simus/kernels/simus_fused_v22_nochain_ilp.cu",
    "tg64":        "src/fast_simus/kernels/simus_fused_v22_tg64.cu",
    "floor":       "src/fast_simus/kernels/simus_fused_v22_floor.cu",
}
TG_SIZE_BY_VARIANT = {
    "lb6":         128,
    "nochain":     128,
    "nochain_ilp": 128,
    "tg64":        64,
    "floor":       128,
}
TILE_SE_BY_VARIANT = {
    "lb6":         16,
    "nochain":     16,
    "nochain_ilp": 16,
    "tg64":        16,
    "floor":       1,
}


def compute_shmem_v22(b_scat, n_es, n_freq, n_elem):
    """All v22 variants share the idealized shmem layout.

    geo:   3 * B_SCAT              floats  (AMP + STP_RX_RE + STP_RX_IM)
    tx:    2 * B_SCAT * N_FREQ     ushort  (fp16 bits)
    delay: 3                       floats
    """
    tx_half_bytes = 2 * b_scat * n_freq * 2
    tx_half_floats = (tx_half_bytes + 3) // 4
    geo_floats = 3 * b_scat
    delay_floats = 3
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


def bench_one(cuda, variant, d, kernel_args,
              d_ore, d_oim, out_size, b_scat, elem_tile, blocks, reps=5):
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    tg_size = TG_SIZE_BY_VARIANT[variant]
    tile_se = TILE_SE_BY_VARIANT[variant]
    shmem = compute_shmem_v22(b_scat, nes, nf, ne)
    source = Path(VARIANT_PATHS[variant]).read_text()
    max_fpt = (nf + tg_size - 1) // tg_size
    defs = {
        "N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
        "TILE_SE": tile_se, "TG_SIZE": tg_size, "MAX_FPT": max_fpt,
        "B_SCAT": b_scat, "ELEM_TILE": elem_tile,
    }
    try:
        mod = compile_module(source, defines=tuple(sorted(defs.items())))
        func = get_function(mod, "simus_fused_kernel")
    except Exception as e:
        return {"error": f"COMPILE: {str(e)[:80]}", "shmem": shmem}

    if shmem > 49152:
        try:
            set_max_dynamic_shared_mem(func, shmem)
        except Exception as e:
            return {"error": f"SHMEM_SET: {str(e)[:80]}", "shmem": shmem}

    CU_FUNC_ATTR_NUM_REGS = 4
    CU_FUNC_ATTR_LOCAL_SIZE_BYTES = 3
    regs = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(regs), CU_FUNC_ATTR_NUM_REGS, func)
    local_mem = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(local_mem), CU_FUNC_ATTR_LOCAL_SIZE_BYTES, func)
    occ = ctypes.c_int()
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(occ), func, tg_size, shmem)

    grid = (blocks, 1, 1)
    block = (tg_size, 1, 1)

    cuda.cuMemsetD32_v2(d_ore, 0, out_size)
    cuda.cuMemsetD32_v2(d_oim, 0, out_size)
    synchronize()
    try:
        launch_kernel(func, grid, block, kernel_args, shmem)
        synchronize()
    except Exception as e:
        return {"error": f"LAUNCH: {str(e)[:80]}", "shmem": shmem,
                "regs": regs.value, "local": local_mem.value, "occ": occ.value}

    times = []
    for _ in range(reps):
        cuda.cuMemsetD32_v2(d_ore, 0, out_size)
        cuda.cuMemsetD32_v2(d_oim, 0, out_size)
        synchronize()
        t0 = time.perf_counter()
        launch_kernel(func, grid, block, kernel_args, shmem)
        synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    best = min(times)
    median = sorted(times)[len(times) // 2]
    return {
        "shmem": shmem,
        "regs": regs.value,
        "local": local_mem.value,
        "occ": occ.value,
        "best_ms": best,
        "med_ms": median,
        "tg_size": tg_size,
    }


def iter_configs(variant, b_scats, elem_tiles):
    """For v22_floor: only B=1 ET=1 (per plan). Others: cartesian sweep."""
    if variant == "floor":
        yield (1, 1)
        return
    for b in b_scats:
        for et in elem_tiles:
            yield (b, et)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=VARIANTS,
                        choices=VARIANTS)
    parser.add_argument("--b-scat", type=int, nargs="+",
                        default=[1, 3, 5, 9])
    parser.add_argument("--elem-tile", type=int, nargs="+",
                        default=[1, 2, 4, 8])
    parser.add_argument("--blocks", type=int, default=_BLOCKS)
    parser.add_argument("--n-scat", type=int, default=_N_SCAT)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    d = prepare(args.n_scat)
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

    print(f"6-blocks/SM Ceiling | N_SCAT={args.n_scat:,}, N_FREQ={nf}, "
          f"N_ELEM={ne}, N_ES={nes}, N_SUB={ns}, blocks={args.blocks}")
    print(f"Baselines: v15 champion={_CHAMP_MS} ms, "
          f"v21 idealized={_IDEAL_MS} ms")
    print("=" * 108)
    header = (f"{'variant':>11} {'TG':>3} {'B':>3} {'ET':>3} {'shmem':>7} "
              f"{'regs':>5} {'spill':>6} {'blk/SM':>7} "
              f"{'best_ms':>9} {'med_ms':>8} {'M scat/s':>10} "
              f"{'vs champ':>9} {'vs ideal':>9}")
    print(header)
    print("-" * len(header))

    per_variant_best = {}     # variant -> (b, et, ms, res)
    best_at_6blk = None       # (variant, b, et, ms, res)
    fastest_overall = None    # (variant, b, et, ms, res)

    for variant in args.variants:
        for b_scat, elem_tile in iter_configs(variant, args.b_scat, args.elem_tile):
            res = bench_one(
                cuda, variant, d, kernel_args,
                d_ore, d_oim, out_size, b_scat, elem_tile,
                args.blocks, reps=args.reps)
            tg = TG_SIZE_BY_VARIANT[variant]
            if "error" in res:
                print(f"{variant:>11} {tg:3d} {b_scat:3d} {elem_tile:3d} "
                      f"{res.get('shmem', 0)/1024:6.1f}K "
                      f"{'-':>5} {'-':>6} {'-':>7} "
                      f"{'-':>9} {'-':>8} {'-':>10} "
                      f"{'-':>9} {'-':>9}  {res['error']}")
                continue
            sps = args.n_scat / (res["best_ms"] / 1000.0)
            r_champ = _CHAMP_MS / res["best_ms"]
            r_ideal = _IDEAL_MS / res["best_ms"]
            print(f"{variant:>11} {tg:3d} {b_scat:3d} {elem_tile:3d} "
                  f"{res['shmem']/1024:6.1f}K {res['regs']:5d} "
                  f"{res['local']:5d}B {res['occ']:7d} "
                  f"{res['best_ms']:9.2f} {res['med_ms']:8.2f} "
                  f"{sps/1e6:10.2f} "
                  f"{r_champ:8.3f}x {r_ideal:8.3f}x")
            if (variant not in per_variant_best or
                    res["best_ms"] < per_variant_best[variant][2]):
                per_variant_best[variant] = (b_scat, elem_tile, res["best_ms"], res)
            if res["occ"] >= 6:
                if best_at_6blk is None or res["best_ms"] < best_at_6blk[3]:
                    best_at_6blk = (variant, b_scat, elem_tile, res["best_ms"], res)
            if fastest_overall is None or res["best_ms"] < fastest_overall[3]:
                fastest_overall = (variant, b_scat, elem_tile, res["best_ms"], res)

    print("=" * 108)
    for v, (b, et, ms, r) in per_variant_best.items():
        sps = args.n_scat / (ms / 1000.0)
        print(f"Best {v:>11}: B={b} ET={et} | {ms:.2f} ms = {sps/1e6:.2f} M scat/s "
              f"(regs={r['regs']}, spill={r['local']}B, blk/SM={r['occ']})")

    print()
    if best_at_6blk is not None:
        v, b, et, ms, r = best_at_6blk
        sps = args.n_scat / (ms / 1000.0)
        print(f"Best at >=6 blk/SM: {v} B={b} ET={et} "
              f"| {ms:.2f} ms = {sps/1e6:.2f} M scat/s "
              f"(regs={r['regs']}, spill={r['local']}B, blk/SM={r['occ']})")
        print(f"  vs v21 idealized ({_IDEAL_MS} ms): {_IDEAL_MS/ms:.3f}x")
    else:
        print("Best at >=6 blk/SM: NONE (no config reached 6 blocks/SM)")

    if fastest_overall is not None:
        v, b, et, ms, r = fastest_overall
        sps = args.n_scat / (ms / 1000.0)
        print(f"Fastest overall:    {v} B={b} ET={et} "
              f"| {ms:.2f} ms = {sps/1e6:.2f} M scat/s "
              f"(regs={r['regs']}, spill={r['local']}B, blk/SM={r['occ']})")
        print(f"  vs v21 idealized ({_IDEAL_MS} ms): {_IDEAL_MS/ms:.3f}x")


if __name__ == "__main__":
    main()
