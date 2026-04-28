"""Two-pass TX/RX benchmark on RTX 4090 with L2-chunked execution.

Tests the hypothesis from HANDOFF-RTX4090.md: with 72MB L2, two-pass
architecture enables Grid-Y element partitioning (8x atomic reduction)
without redundant Phase 1+2 compute.

Chunking: processes scatterers in chunks of ~10K to keep TX buffer in L2.
TX buffer per chunk: 10K * 854 * 2 * 4 = 68MB (fits in 72MB L2).

Compares against v11 B=5 ET=8 baseline for correctness and speed.
"""
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
    compile_module, device_alloc, device_free, get_function,
    launch_kernel, memcpy_dtoh, memcpy_htod, synchronize,
    set_max_dynamic_shared_mem, _get_cuda, _get_context,
)
from fast_simus.medium_params import MediumParams
from fast_simus.simus import simus_precompute
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils.geometry import element_positions

_NEPER_TO_DB = 8.685889638065036
_TG_SIZE = 128
NUM_SMS = 128


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
    n_freq = int(plan.selected_freqs.shape[0])
    n_sub = plan.n_sub
    n_es = n_elem * n_sub
    element_pos, theta_raw, apex_offset = element_positions(
        n_elem, params.pitch, params.radius, np)
    if theta_raw is None:
        theta_raw = np.zeros(n_elem, dtype=np.float32)
    elem_x = np.ascontiguousarray(element_pos[:, 0], dtype=np.float32)
    elem_z = np.ascontiguousarray(element_pos[:, 1], dtype=np.float32)
    seg_length = params.element_width / n_sub
    seg_offsets = np.array(
        [-params.element_width / 2.0 + seg_length / 2.0 + i * seg_length
         for i in range(n_sub)], dtype=np.float32)
    cos_th = np.cos(theta_raw).astype(np.float32)
    sin_neg_th = np.sin(-theta_raw).astype(np.float32)
    sub_dx = np.zeros(n_es, dtype=np.float32)
    sub_dz = np.zeros(n_es, dtype=np.float32)
    for e in range(n_elem):
        for s in range(n_sub):
            sub_dx[e * n_sub + s] = seg_offsets[s] * cos_th[e]
            sub_dz[e * n_sub + s] = seg_offsets[s] * sin_neg_th[e]
    freq_start = float(plan.selected_freqs[0])
    freq_step = float(plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freq > 1 else 0.0
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
        n_scat=n_scat, n_freq=n_freq, n_elem=n_elem, n_sub=n_sub, n_es=n_es,
        kw_init=2*pi*freq_start/c,
        alpha_init=medium.attenuation/_NEPER_TO_DB*freq_start/1e6*1e2,
        kw_step=2*pi*freq_step/c,
        alpha_step=medium.attenuation/_NEPER_TO_DB*freq_step/1e6*1e2,
        min_dist=c/params.freq_center/2.0,
        seg_length=seg_length,
        center_kw=2*pi*params.freq_center/c,
        inv_nsub=1.0/n_sub,
        radius_v=params.radius if params.radius != inf else 1e31,
        apex_offset=apex_offset,
    )


def upload(data):
    p = device_alloc(data.nbytes)
    memcpy_htod(p, data)
    return p


def run_v11_reference(d, n_scat):
    """Run v11 B=5 ET=8 as reference."""
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE
    blocks = 2 * NUM_SMS

    defs = {"N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
            "TILE_SE": 16, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt,
            "B_SCAT": 5, "ELEM_TILE": 8}
    shmem = (7 * 5 * nes + 2 * 5 * nf + 3 * ne) * 4

    source = Path("src/fast_simus/kernels/simus_fused_v11.cu").read_text()
    mod = compile_module(source, defines=tuple(sorted(defs.items())))
    func = get_function(mod, "simus_fused_kernel")
    cuda = _get_cuda(); _get_context()
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

    args = [
        d_sx, d_sz, d_rc, d_ex, d_ez, d_ct, d_snt, d_sdx, d_sdz,
        d_dare, d_daim, d_dps, d_ppre, d_ppim, d_probe,
        d_ore, d_oim,
        ctypes.c_int(n_scat),
        ctypes.c_float(d["kw_init"]), ctypes.c_float(d["alpha_init"]),
        ctypes.c_float(d["kw_step"]), ctypes.c_float(d["alpha_step"]),
        ctypes.c_float(d["min_dist"]), ctypes.c_float(d["seg_length"]),
        ctypes.c_float(d["center_kw"]), ctypes.c_float(d["inv_nsub"]),
        ctypes.c_float(d["radius_v"]), ctypes.c_float(d["apex_offset"]),
    ]

    grid = (blocks, 1, 1)
    block = (_TG_SIZE, 1, 1)

    for _ in range(5):
        cuda.cuMemsetD32_v2(d_ore, 0, out_size)
        cuda.cuMemsetD32_v2(d_oim, 0, out_size)
        synchronize()
        launch_kernel(func, grid, block, args, shmem)
        synchronize()

    times = []
    for _ in range(10):
        cuda.cuMemsetD32_v2(d_ore, 0, out_size)
        cuda.cuMemsetD32_v2(d_oim, 0, out_size)
        synchronize()
        t0 = time.perf_counter()
        launch_kernel(func, grid, block, args, shmem)
        synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    re = np.empty(out_size, dtype=np.float32)
    im = np.empty(out_size, dtype=np.float32)
    memcpy_dtoh(re, d_ore); memcpy_dtoh(im, d_oim)
    med = np.median(times)
    print(f"  v11 B=5 ET=8: best5=[{', '.join(f'{t*1000:.1f}' for t in sorted(times)[:5])}] "
          f"median={med*1000:.1f}ms  {n_scat/med:,.0f} scat/s")
    return re + 1j * im, med


def run_two_pass(d, n_scat, chunk_size, elem_tile, rx_blocks_x):
    """Run two-pass TX/RX with chunked execution."""
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE
    n_elem_groups = (nes + elem_tile - 1) // elem_tile

    tx_defs = {"N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
               "TILE_SE": 16, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt}
    tx_shmem = (7 * nes + 3 * ne) * 4

    rx_defs = {"N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf, "N_ES": nes,
               "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt, "ELEM_TILE": elem_tile}

    print(f"  Compiling TX kernel...")
    tx_source = Path("src/fast_simus/kernels/simus_tx_v14.cu").read_text()
    tx_mod = compile_module(tx_source, defines=tuple(sorted(tx_defs.items())))
    tx_func = get_function(tx_mod, "simus_tx_kernel")

    print(f"  Compiling RX Grid-Y kernel (ELEM_TILE={elem_tile}, groups={n_elem_groups})...")
    rx_source = Path("src/fast_simus/kernels/simus_rx_gridy_v18.cu").read_text()
    rx_mod = compile_module(rx_source, defines=tuple(sorted(rx_defs.items())))
    rx_func = get_function(rx_mod, "simus_rx_gridy_kernel")

    cuda = _get_cuda(); _get_context()
    if tx_shmem > 49152:
        set_max_dynamic_shared_mem(tx_func, tx_shmem)

    tx_regs = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(tx_regs), 4, tx_func)
    rx_regs = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(rx_regs), 4, rx_func)
    rx_local = ctypes.c_int()
    cuda.cuFuncGetAttribute(ctypes.byref(rx_local), 3, rx_func)
    rx_occ = ctypes.c_int()
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(rx_occ), rx_func, _TG_SIZE, 0)
    print(f"  TX regs: {tx_regs.value}, RX regs: {rx_regs.value}, RX local: {rx_local.value}B, RX blocks/SM: {rx_occ.value}")

    out_size = nf * ne
    d_ore = device_alloc(out_size * 4)
    d_oim = device_alloc(out_size * 4)

    tx_buf_size = n_scat * nf
    d_tx_re = device_alloc(tx_buf_size * 4)
    d_tx_im = device_alloc(tx_buf_size * 4)

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

    scalar_args = [
        ctypes.c_float(d["kw_init"]), ctypes.c_float(d["alpha_init"]),
        ctypes.c_float(d["kw_step"]), ctypes.c_float(d["alpha_step"]),
        ctypes.c_float(d["min_dist"]), ctypes.c_float(d["seg_length"]),
        ctypes.c_float(d["center_kw"]), ctypes.c_float(d["inv_nsub"]),
        ctypes.c_float(d["radius_v"]), ctypes.c_float(d["apex_offset"]),
    ]

    def run_once():
        cuda.cuMemsetD32_v2(d_ore, 0, out_size)
        cuda.cuMemsetD32_v2(d_oim, 0, out_size)
        synchronize()

        for chunk_start in range(0, n_scat, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_scat)
            chunk_n = chunk_end - chunk_start

            sx_off = ctypes.c_void_p(d_sx.value + chunk_start * 4)
            sz_off = ctypes.c_void_p(d_sz.value + chunk_start * 4)
            rc_off = ctypes.c_void_p(d_rc.value + chunk_start * 4)
            txre_off = ctypes.c_void_p(d_tx_re.value + chunk_start * nf * 4)
            txim_off = ctypes.c_void_p(d_tx_im.value + chunk_start * nf * 4)

            tx_args = [
                sx_off, sz_off, rc_off, d_ex, d_ez, d_ct, d_snt, d_sdx, d_sdz,
                d_dare, d_daim, d_dps, d_ppre, d_ppim,
                txre_off, txim_off,
                ctypes.c_int(chunk_n),
                *scalar_args,
            ]
            launch_kernel(tx_func, (chunk_n, 1, 1), (_TG_SIZE, 1, 1), tx_args, tx_shmem)

            rx_args = [
                d_sx, d_sz, d_ex, d_ez, d_ct, d_snt, d_sdx, d_sdz,
                d_tx_re, d_tx_im, d_probe,
                d_ore, d_oim,
                ctypes.c_int(chunk_n),
                *scalar_args,
                ctypes.c_int(chunk_start),
            ]
            launch_kernel(rx_func,
                          (rx_blocks_x, n_elem_groups, 1),
                          (_TG_SIZE, 1, 1),
                          rx_args, 0)

        synchronize()

    print(f"  Warmup (chunk_size={chunk_size}, rx_blocks_x={rx_blocks_x})...")
    for _ in range(3):
        run_once()

    print(f"  Benchmarking...")
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    re = np.empty(out_size, dtype=np.float32)
    im = np.empty(out_size, dtype=np.float32)
    memcpy_dtoh(re, d_ore); memcpy_dtoh(im, d_oim)
    med = np.median(times)
    label = f"two-pass ET={elem_tile} chunk={chunk_size} rxblk={rx_blocks_x}"
    print(f"  {label}: best5=[{', '.join(f'{t*1000:.1f}' for t in sorted(times)[:5])}] "
          f"median={med*1000:.1f}ms  {n_scat/med:,.0f} scat/s")

    device_free(d_tx_re)
    device_free(d_tx_im)
    return re + 1j * im, med


if __name__ == "__main__":
    n_scat = 100_000
    d = prepare(n_scat)
    nf, ne, nes = d["n_freq"], d["n_elem"], d["n_es"]
    print(f"N_SCAT={n_scat:,}, N_FREQ={nf}, N_ELEM={ne}, N_ES={nes}")

    print(f"\n--- v11 B=5 ET=8 reference ---")
    out_v11, t_v11 = run_v11_reference(d, n_scat)

    configs = [
        (10_000, 8, 32),
        (10_000, 8, 64),
        (10_000, 8, 128),
        (10_000, 8, 256),
        (5_000, 8, 64),
        (20_000, 8, 64),
        (100_000, 8, 64),
        (10_000, 16, 64),
        (10_000, 4, 64),
    ]

    print(f"\n--- Two-pass Grid-Y configs ---")
    results = {}
    for chunk_size, elem_tile, rx_blocks_x in configs:
        label = f"chunk={chunk_size} ET={elem_tile} rxblk={rx_blocks_x}"
        print(f"\n--- {label} ---")
        try:
            out, t = run_two_pass(d, n_scat, chunk_size, elem_tile, rx_blocks_x)
            norm = np.abs(out_v11).max()
            rd = np.abs(out_v11 - out).max() / norm if norm > 0 else 0
            sp = t_v11 / t
            ok = "PASS" if rd < 1e-3 else "FAIL"
            print(f"  Correctness: {rd:.2e} {ok}  Speedup vs v11: {sp:.2f}x")
            results[(chunk_size, elem_tile, rx_blocks_x)] = (t, sp, rd, ok)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== Summary (vs v11={t_v11*1000:.1f}ms) ===")
    for (cs, et, rb), (t, sp, rd, ok) in sorted(
            results.items(), key=lambda x: x[1][0] if x[1][0] else 1e9):
        print(f"  chunk={cs:>6} ET={et:>2} rxblk={rb:>3}: {t*1000:.1f}ms  "
              f"{sp:.2f}x  {n_scat/t:,.0f} scat/s  {ok}")
