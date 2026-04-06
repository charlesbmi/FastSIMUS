"""Benchmark v11 (element-tiled Phase 3) vs v6/v10 baselines."""

import os
import time

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import ctypes
from math import inf, pi
from pathlib import Path

import numpy as np

from fast_simus.kernels.cuda_runtime import (
    compile_module, device_alloc, device_free, get_function,
    launch_kernel, memcpy_dtoh, memcpy_htod, synchronize,
    set_max_dynamic_shared_mem,
    _get_cuda, _get_context,
)
from fast_simus.medium_params import MediumParams
from fast_simus.simus import simus_precompute
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils.geometry import element_positions

_NEPER_TO_DB = 8.685889638065036
_TG_SIZE = 128
NUM_SMS = 48


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


def bench(name, source_path, d, grid_blocks, defines, shmem, nw=10, nr=10):
    source = Path(source_path).read_text()
    mod = compile_module(source, defines=tuple(sorted(defines.items())))
    func = get_function(mod, "simus_fused_kernel")
    cuda = _get_cuda(); _get_context()

    if shmem > 49152:
        set_max_dynamic_shared_mem(func, shmem)

    nf, ne = d["n_freq"], d["n_elem"]
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

    grid = (grid_blocks, 1, 1)
    block = (_TG_SIZE, 1, 1)

    for _ in range(nw):
        cuda.cuMemsetD32_v2(d_ore, 0, out_size)
        cuda.cuMemsetD32_v2(d_oim, 0, out_size)
        synchronize()
        launch_kernel(func, grid, block, args, shmem)
        synchronize()

    times = []
    for _ in range(nr):
        cuda.cuMemsetD32_v2(d_ore, 0, out_size)
        cuda.cuMemsetD32_v2(d_oim, 0, out_size)
        synchronize()
        t0 = time.perf_counter()
        launch_kernel(func, grid, block, args, shmem)
        synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    med = np.median(times)
    re = np.empty(out_size, dtype=np.float32)
    im = np.empty(out_size, dtype=np.float32)
    memcpy_dtoh(re, d_ore); memcpy_dtoh(im, d_oim)

    print(f"  {name}: best5=[{', '.join(f'{t*1000:.1f}' for t in times[:5])}] "
          f"median={med*1000:.1f}ms  {d['n_scat']/med:,.0f} scat/s")
    return re + 1j * im, med


if __name__ == "__main__":
    n_scat = 100_000
    d = prepare(n_scat)
    nf, ne, nes, ns = d["n_freq"], d["n_elem"], d["n_es"], d["n_sub"]
    max_fpt = (nf + _TG_SIZE - 1) // _TG_SIZE
    print(f"N_SCAT={n_scat:,}, N_FREQ={nf}, N_ELEM={ne}, N_SUB={ns}, N_ES={nes}")

    v6_defs = {"N_ELEM": ne, "N_SUB": ns, "N_FREQ": nf,
               "N_ES": nes, "TILE_SE": 16, "TG_SIZE": _TG_SIZE, "MAX_FPT": max_fpt}
    v6_shmem = (9 * nes + 3 * ne) * 4

    print(f"\n--- v6 baseline ---")
    out_v6, t_v6 = bench("v6", "src/fast_simus/kernels/simus_fused_v6.cu",
                          d, 192, v6_defs, v6_shmem)

    results = {}
    configs = [
        (2, 4, 192),  (2, 8, 192),  (2, 16, 192),
        (4, 4, 192),  (4, 8, 192),  (4, 16, 192),
        (2, 4, 96),   (4, 4, 96),
    ]

    for b_scat, etile, npb in configs:
        v11_defs = dict(v6_defs)
        v11_defs["B_SCAT"] = b_scat
        v11_defs["ELEM_TILE"] = etile
        v11_shmem = (7 * b_scat * nes + 2 * b_scat * nf + 3 * ne) * 4

        name = f"v11 B={b_scat} ET={etile} blk={npb}"
        print(f"\n--- {name} (shmem={v11_shmem:,}) ---")
        try:
            out, t = bench(name,
                           "src/fast_simus/kernels/simus_fused_v11.cu",
                           d, npb, v11_defs, v11_shmem)
            norm = np.abs(out_v6).max()
            rd = np.abs(out_v6 - out).max() / norm if norm > 0 else 0
            sp = t_v6 / t
            ok = "PASS" if rd < 1e-3 else "FAIL"
            print(f"  Correctness: {rd:.2e} {ok}  Speedup: {sp:.2f}x")
            results[(b_scat, etile, npb)] = (t, sp, rd, ok)
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n=== Summary (vs v6={t_v6*1000:.1f}ms) ===")
    for (b, et, npb), (t, sp, rd, ok) in sorted(
            results.items(), key=lambda x: x[1][0] if x[1][0] else 1e9):
        if t is not None:
            print(f"  B={b} ET={et:>2} blk={npb:>3}: {t*1000:.1f}ms  {sp:.2f}x  "
                  f"{n_scat/t:,.0f} scat/s  {ok}")
