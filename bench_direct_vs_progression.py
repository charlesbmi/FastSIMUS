"""Benchmark: geometric progression vs direct per-frequency TX computation.

Compares the current simus_tx kernel (serial geometric progression across
frequencies) against a new direct computation kernel (one thread per
scatterer*frequency pair, no register arrays).

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


def run_progression(d: dict[str, Any]) -> tuple[mx.array, mx.array]:
    """Run the current geometric-progression TX kernel."""
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
    """Run the direct per-frequency TX kernel."""
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
    """Run the shared-memory geometry + direct frequency TX kernel."""
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


def time_kernel(fn, d: dict[str, Any], warmup: int, trials: int) -> float:
    """Time a kernel, returning median time in seconds."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Progression vs Direct TX kernel benchmark")
    parser.add_argument("--n-scat", type=int, nargs="+", default=[100, 500, 1000, 5000])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    params = P4_2v()
    medium = MediumParams()
    depth_m = 0.08

    n_elem = params.n_elements
    scat_np = np.column_stack([np.zeros(10), np.linspace(0.005, depth_m, 10)])
    rc_np = np.ones(10)
    delays_np = np.zeros(n_elem)
    plan = simus_precompute(scat_np, rc_np, delays_np, params, medium)

    # Convert plan arrays to MLX so _prepare_common works
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
    print(f"Configuration: {n_elem} elements, {plan.n_sub} sub, {n_freq} frequencies")
    print(f"{'n_scat':>8} | {'Progression':>14} | {'Direct':>14} | {'Shared':>14} | {'Best':>6}")
    print("-" * 80)

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

        freq_start = float(plan.selected_freqs[0])
        freq_step = float(plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freq > 1 else 0.0
        ph_step = mx.array(2.0 * pi * freq_step, dtype=mx.float32) * delays_mx
        d["delay_phase_step"] = ph_step.astype(mx.float32)

        # Accuracy: compare all three against progression
        prog_re, prog_im = run_progression(d)
        direct_re, direct_im = run_direct(d)
        shared_re, shared_im = run_shared(d)
        mx.eval(prog_re, prog_im, direct_re, direct_im, shared_re, shared_im)

        prog_mag = mx.sqrt(prog_re * prog_re + prog_im * prog_im)
        max_mag = mx.maximum(prog_mag, mx.array(1e-30))

        direct_mag = mx.sqrt(direct_re * direct_re + direct_im * direct_im)
        direct_err = float(mx.max(mx.abs(prog_mag - direct_mag) / max_mag).item())

        shared_mag = mx.sqrt(shared_re * shared_re + shared_im * shared_im)
        shared_err = float(mx.max(mx.abs(prog_mag - shared_mag) / max_mag).item())

        # Timing
        t_prog = time_kernel(run_progression, d, args.warmup, args.trials)
        t_direct = time_kernel(run_direct, d, args.warmup, args.trials)
        t_shared = time_kernel(run_shared, d, args.warmup, args.trials)

        best = "prog"
        best_t = t_prog
        if t_direct < best_t:
            best, best_t = "dir", t_direct
        if t_shared < best_t:
            best, best_t = "shr", t_shared

        print(
            f"{n_scat:>8} | {t_prog*1000:>11.2f} ms | {t_direct*1000:>11.2f} ms | "
            f"{t_shared*1000:>11.2f} ms | {best:>6}"
        )
        print(
            f"{'':>8} | {n_scat/t_prog/1e3:>10.1f}K/s | {n_scat/t_direct/1e3:>10.1f}K/s | "
            f"{n_scat/t_shared/1e3:>10.1f}K/s | err: d={direct_err:.1e} s={shared_err:.1e}"
        )


if __name__ == "__main__":
    main()
