"""Benchmark: progression vs shared vs tiled TX kernel approaches.

Compares four TX kernel architectures:
  1. Geometric progression (current production kernel)
  2. Shared-memory direct (one threadgroup per scatterer, direct cos/sin)
  3. Tiled progression (shared geometry + element-tiled geometric progression)

Usage:
    uv run python bench_direct_vs_progression.py [--n-scat N] [--warmup W] [--trials T]
"""

from __future__ import annotations

import argparse
import time
from math import pi
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from fast_simus.backends.mlx import ensure_compat

ensure_compat(mx)

from fast_simus.kernels.metal_simus import (
    _TX_THREADGROUP,
    _build_tx,
    _load_source,
    _make_header,
    _prepare_common,
)
from fast_simus.medium_params import MediumParams
from fast_simus.simus import simus_precompute
from fast_simus.transducer_presets import P4_2v

_KERNELS_DIR = Path(__file__).parent / "src" / "fast_simus" / "kernels"

_kernel_cache: dict[tuple, Any] = {}


def _build_tx_direct(n_elem: int, n_sub: int, n_freq: int, n_scat: int) -> Any:
    key = ("tx_direct", n_elem, n_sub, n_freq, n_scat)
    if key not in _kernel_cache:
        source = (_KERNELS_DIR / "simus_tx_direct.metal").read_text()
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"simus_tx_direct_{n_elem}_{n_sub}_{n_freq}_{n_scat}",
            input_names=[
                "scat_x", "scat_z", "elem_x", "elem_z", "theta_e",
                "sub_dx", "sub_dz", "da_init_re", "da_init_im",
                "delay_phase_step", "pp_re", "pp_im", "is_out", "scalars",
            ],
            output_names=["tx_re", "tx_im"],
            header=_make_header(n_elem, n_sub, n_freq, n_scat),
            source=source,
        )
    return _kernel_cache[key]


def _build_tx_shared(n_elem: int, n_sub: int, n_freq: int, n_scat: int) -> Any:
    key = ("tx_shared", n_elem, n_sub, n_freq, n_scat)
    if key not in _kernel_cache:
        source = (_KERNELS_DIR / "simus_tx_shared.metal").read_text()
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"simus_tx_shared_{n_elem}_{n_sub}_{n_freq}_{n_scat}",
            input_names=[
                "scat_x", "scat_z", "elem_x", "elem_z", "theta_e",
                "sub_dx", "sub_dz", "da_init_re", "da_init_im",
                "delay_phase_step", "pp_re", "pp_im", "is_out", "scalars",
            ],
            output_names=["tx_re", "tx_im"],
            header=_make_header(n_elem, n_sub, n_freq, n_scat),
            source=source,
        )
    return _kernel_cache[key]


def _make_tiled_header(
    n_elem: int, n_sub: int, n_freq: int, n_scat: int,
    tile_se: int, tg_size: int,
) -> str:
    return (
        _make_header(n_elem, n_sub, n_freq, n_scat)
        + f"#define TILE_SE {tile_se}\n"
        + f"#define TG_SIZE {tg_size}\n"
        + f"#define MAX_FPT (({n_freq} + {tg_size} - 1) / {tg_size})\n"
    )


def _build_tx_tiled(
    n_elem: int, n_sub: int, n_freq: int, n_scat: int,
    tile_se: int = 16, tg_size: int = 256,
) -> Any:
    key = ("tx_tiled", n_elem, n_sub, n_freq, n_scat, tile_se, tg_size)
    if key not in _kernel_cache:
        source = (_KERNELS_DIR / "simus_tx_tiled.metal").read_text()
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"simus_tx_tiled_{n_elem}_{n_sub}_{n_freq}_{n_scat}_{tile_se}",
            input_names=[
                "scat_x", "scat_z", "elem_x", "elem_z", "theta_e",
                "sub_dx", "sub_dz", "da_init_re", "da_init_im",
                "delay_phase_step", "pp_re", "pp_im", "is_out", "scalars",
            ],
            output_names=["tx_re", "tx_im"],
            header=_make_tiled_header(n_elem, n_sub, n_freq, n_scat, tile_se, tg_size),
            source=source,
        )
    return _kernel_cache[key]


def run_progression(d: dict[str, Any]) -> tuple[mx.array, mx.array]:
    n_elem, n_sub, n_freq, n_scat = d["n_elem"], d["n_sub"], d["n_freq"], d["n_scat"]
    k = _build_tx(n_elem, n_sub, n_freq, n_scat)
    outputs = k(
        inputs=[
            d["x_flat"], d["z_flat"], d["elem_x"], d["elem_z"], d["theta_e"],
            d["sub_dx"], d["sub_dz"], d["da_init_re"], d["da_init_im"],
            d["da_step_re"], d["da_step_im"], d["pp_re"], d["pp_im"],
            d["is_out"], d["scalars"],
        ],
        output_shapes=[(n_scat * n_freq,), (n_scat * n_freq,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_scat, 1, 1),
        threadgroup=(min(_TX_THREADGROUP, n_scat), 1, 1),
    )
    return outputs[0], outputs[1]


def run_direct(d: dict[str, Any]) -> tuple[mx.array, mx.array]:
    n_elem, n_sub, n_freq, n_scat = d["n_elem"], d["n_sub"], d["n_freq"], d["n_scat"]
    k = _build_tx_direct(n_elem, n_sub, n_freq, n_scat)
    total_threads = n_scat * n_freq
    outputs = k(
        inputs=[
            d["x_flat"], d["z_flat"], d["elem_x"], d["elem_z"], d["theta_e"],
            d["sub_dx"], d["sub_dz"], d["da_init_re"], d["da_init_im"],
            d["delay_phase_step"], d["pp_re"], d["pp_im"],
            d["is_out"], d["scalars"],
        ],
        output_shapes=[(n_scat * n_freq,), (n_scat * n_freq,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(total_threads, 1, 1),
        threadgroup=(min(256, total_threads), 1, 1),
    )
    return outputs[0], outputs[1]


_SHARED_TG = 256


def run_shared(d: dict[str, Any], tg_size: int = _SHARED_TG) -> tuple[mx.array, mx.array]:
    n_elem, n_sub, n_freq, n_scat = d["n_elem"], d["n_sub"], d["n_freq"], d["n_scat"]
    k = _build_tx_shared(n_elem, n_sub, n_freq, n_scat)
    tg = min(tg_size, n_freq)
    outputs = k(
        inputs=[
            d["x_flat"], d["z_flat"], d["elem_x"], d["elem_z"], d["theta_e"],
            d["sub_dx"], d["sub_dz"], d["da_init_re"], d["da_init_im"],
            d["delay_phase_step"], d["pp_re"], d["pp_im"],
            d["is_out"], d["scalars"],
        ],
        output_shapes=[(n_scat * n_freq,), (n_scat * n_freq,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_scat * tg, 1, 1),
        threadgroup=(tg, 1, 1),
    )
    return outputs[0], outputs[1]


def run_tiled(
    d: dict[str, Any], tile_se: int = 16, tg_size: int = 64,
) -> tuple[mx.array, mx.array]:
    n_elem, n_sub, n_freq, n_scat = d["n_elem"], d["n_sub"], d["n_freq"], d["n_scat"]
    k = _build_tx_tiled(n_elem, n_sub, n_freq, n_scat, tile_se, tg_size)
    outputs = k(
        inputs=[
            d["x_flat"], d["z_flat"], d["elem_x"], d["elem_z"], d["theta_e"],
            d["sub_dx"], d["sub_dz"], d["da_init_re"], d["da_init_im"],
            d["delay_phase_step"], d["pp_re"], d["pp_im"],
            d["is_out"], d["scalars"],
        ],
        output_shapes=[(n_scat * n_freq,), (n_scat * n_freq,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_scat * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
    )
    return outputs[0], outputs[1]


def time_kernel(fn, d: dict[str, Any], warmup: int, trials: int) -> float:
    for _ in range(warmup):
        out = fn(d)
        mx.eval(out[0], out[1])

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn(d)
        mx.eval(out[0], out[1])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times))


def rel_error(ref_re: mx.array, ref_im: mx.array, test_re: mx.array, test_im: mx.array) -> float:
    ref_mag = mx.sqrt(ref_re * ref_re + ref_im * ref_im)
    test_mag = mx.sqrt(test_re * test_re + test_im * test_im)
    denom = mx.maximum(ref_mag, mx.array(1e-30))
    return float(mx.max(mx.abs(ref_mag - test_mag) / denom).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="TX kernel architecture benchmark")
    parser.add_argument("--n-scat", type=int, nargs="+",
                        default=[100, 500, 1000, 2000, 5000, 10000])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--tile-se", type=int, nargs="+", default=[16])
    parser.add_argument("--skip-direct", action="store_true",
                        help="Skip the direct kernel (slow at large N)")
    args = parser.parse_args()

    params = P4_2v()
    medium = MediumParams()
    depth_m = 0.08

    n_elem = params.n_elements
    scat_np = np.column_stack([np.zeros(10), np.linspace(0.005, depth_m, 10)])
    rc_np = np.ones(10)
    delays_np = np.zeros(n_elem)
    plan = simus_precompute(scat_np, rc_np, delays_np, params, medium)

    from fast_simus.simus import SimusPlan
    plan = SimusPlan(
        selected_freqs=mx.array(np.asarray(plan.selected_freqs), dtype=mx.float32),
        pulse_spectrum=mx.array(np.asarray(plan.pulse_spectrum)),
        probe_spectrum=mx.array(np.asarray(plan.probe_spectrum), dtype=mx.float32),
        n_sub=plan.n_sub,
        seg_length=plan.seg_length,
        correction_factor=plan.correction_factor,
        n_freq_full=plan.n_freq_full,
        freq_idx_start=plan.freq_idx_start,
        n_fft=plan.n_fft,
    )

    n_freq = int(plan.selected_freqs.shape[0])
    freq_start = float(plan.selected_freqs[0])
    freq_step = float(plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freq > 1 else 0.0

    print(f"Config: {n_elem} elements, {plan.n_sub} sub, {n_freq} frequencies")
    print(f"Tile sizes: {args.tile_se}")
    print()

    # Build header
    cols = ["n_scat", "prog", "shared"]
    if not args.skip_direct:
        cols.append("direct")
    for ts in args.tile_se:
        cols.append(f"tiled_{ts}")
    cols.append("best")
    hdr = " | ".join(f"{c:>12}" for c in cols)
    print(hdr)
    print("-" * len(hdr))

    for n_scat in args.n_scat:
        scatterers = mx.array(
            np.column_stack([
                np.random.default_rng(42).uniform(-0.02, 0.02, n_scat),
                np.random.default_rng(43).uniform(0.005, 0.065, n_scat),
            ]),
            dtype=mx.float32,
        )
        rc_mx = mx.ones(n_scat, dtype=mx.float32)
        delays_mx = mx.zeros(n_elem, dtype=mx.float32)
        tx_apod = mx.ones(n_elem, dtype=mx.float32)

        d = _prepare_common(scatterers, rc_mx, params, plan, medium, delays_mx, tx_apod)

        ph_step = mx.array(2.0 * pi * freq_step, dtype=mx.float32) * delays_mx
        d["delay_phase_step"] = ph_step.astype(mx.float32)

        # Reference: progression
        prog_re, prog_im = run_progression(d)
        mx.eval(prog_re, prog_im)

        results: dict[str, tuple[float, float]] = {}

        # Progression timing
        t = time_kernel(run_progression, d, args.warmup, args.trials)
        results["prog"] = (t, 0.0)

        # Shared
        shared_re, shared_im = run_shared(d)
        mx.eval(shared_re, shared_im)
        err = rel_error(prog_re, prog_im, shared_re, shared_im)
        t = time_kernel(run_shared, d, args.warmup, args.trials)
        results["shared"] = (t, err)

        # Direct (optional)
        if not args.skip_direct:
            direct_re, direct_im = run_direct(d)
            mx.eval(direct_re, direct_im)
            err = rel_error(prog_re, prog_im, direct_re, direct_im)
            t = time_kernel(run_direct, d, args.warmup, args.trials)
            results["direct"] = (t, err)

        # Tiled variants
        for ts in args.tile_se:
            tiled_re, tiled_im = run_tiled(d, tile_se=ts)
            mx.eval(tiled_re, tiled_im)
            err = rel_error(prog_re, prog_im, tiled_re, tiled_im)
            t = time_kernel(lambda d_: run_tiled(d_, tile_se=ts), d, args.warmup, args.trials)
            results[f"tiled_{ts}"] = (t, err)

        # Find best
        best_name = min(results, key=lambda k: results[k][0])
        best_t = results[best_name][0]

        # Print times
        vals = [f"{n_scat:>12}"]
        for col in cols[1:-1]:
            if col in results:
                ms = results[col][0] * 1000
                vals.append(f"{ms:>9.2f} ms")
            else:
                vals.append(f"{'--':>12}")
        vals.append(f"{best_name:>12}")
        print(" | ".join(vals))

        # Print throughput and error
        vals2 = [f"{'':>12}"]
        for col in cols[1:-1]:
            if col in results:
                tp = n_scat / results[col][0] / 1e3
                vals2.append(f"{tp:>8.0f}K/s")
            else:
                vals2.append(f"{'':>12}")
        errs = [f"e={results[k][1]:.0e}" for k in results if results[k][1] > 0]
        vals2.append(" ".join(errs) if errs else "")
        print(" | ".join(vals2))
        print()


if __name__ == "__main__":
    main()
