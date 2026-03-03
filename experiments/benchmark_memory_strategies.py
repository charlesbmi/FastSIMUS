#!/usr/bin/env python
"""Benchmark memory-reduction strategies for pfield computation.

Compares baseline (fully vectorized), element accumulation, frequency chunking,
and grid chunking strategies across grid sizes and backends. Measures wall-clock
time, peak memory, and verifies correctness.

Usage:
    uv run --group test python experiments/benchmark_memory_strategies.py
    uv run --group test python experiments/benchmark_memory_strategies.py --backend jax
    uv run --group test python experiments/benchmark_memory_strategies.py --grids 50 200 --repeats 5
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from math import inf, pi
from pathlib import Path

import array_api_extra as xpx
import numpy as np

from fast_simus.medium_params import MediumParams
from fast_simus.pfield import (
    PfieldPlan,
    _distances_and_angles,
    _init_exponentials,
    _obliquity_factor,
    _pfield_freq_chunk_freq,
    _pfield_freq_element_accum,
    _pfield_freq_vectorized,
    _subelement_centroids,
    pfield,
    pfield_compute_chunked,
    pfield_precompute,
)
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils._array_api import _ArrayNamespace
from fast_simus.utils.geometry import element_positions

MEDIUM = MediumParams()
PARAMS = P4_2v()


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    strategy: str
    grid_size: int
    backend: str
    chunk_param: int | None
    time_s: float
    peak_mem_mb: float
    theoretical_peak_mb: float
    correct: bool
    n_elements: int
    n_freq: int
    n_sub: int


def _make_positions(n: int) -> np.ndarray:
    x = np.linspace(-4e-2, 4e-2, n)
    z = np.linspace(PARAMS.pitch, 10e-2, n)
    xg, zg = np.meshgrid(x, z)
    return np.stack([xg, zg], axis=-1)


def _theoretical_peak_mb(n_grid: int, n_elements: int, n_freq: int, strategy: str, chunk: int | None) -> float:
    """Estimate peak intermediate memory in MB (complex64 = 8 bytes)."""
    cx = 8
    if strategy == "baseline":
        return n_grid * n_elements * n_freq * cx / 1e6
    if strategy == "element_accum":
        c = chunk or 1
        return (n_grid * c * n_freq * cx + n_grid * n_freq * cx) / 1e6
    if strategy == "freq_chunk":
        c = chunk or 1
        return n_grid * n_elements * c * cx / 1e6
    if strategy == "grid_chunk":
        c = chunk or n_grid
        return c * n_elements * n_freq * cx / 1e6
    return 0.0


def _build_intermediates(positions, delays, plan: PfieldPlan, xp: _ArrayNamespace):
    """Build the intermediate arrays that strategy functions consume."""
    element_pos, theta_elements, apex_offset = element_positions(PARAMS.n_elements, PARAMS.pitch, PARAMS.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(PARAMS.n_elements)

    speed_of_sound = MEDIUM.speed_of_sound
    attenuation = MEDIUM.attenuation
    tx_apodization = xp.ones(PARAMS.n_elements)
    delays_clean = xp.where(xp.isnan(delays), xp.asarray(0.0), delays)
    tx_apodization = xp.where(xp.isnan(delays), xp.asarray(0.0), tx_apodization)

    subelement_offsets = _subelement_centroids(PARAMS.element_width, plan.n_sub, theta_elements, xp)

    x = positions[..., 0]
    z = positions[..., 1]
    is_out = z < 0
    if PARAMS.radius != inf:
        is_out = is_out | ((x**2 + (z + apex_offset) ** 2) <= PARAMS.radius**2)

    distances, sin_theta, theta_arr = _distances_and_angles(
        positions,
        subelement_offsets,
        element_pos,
        theta_elements,
        speed_of_sound,
        PARAMS.freq_center,
        xp,
    )
    obliquity_factor = _obliquity_factor(theta_arr, PARAMS.baffle, xp)
    phase_decay_init, phase_decay_step = _init_exponentials(
        plan.freq_start,
        speed_of_sound,
        attenuation,
        distances,
        obliquity_factor,
        plan.freq_step,
        xp,
    )

    center_wavenumber = 2.0 * pi * PARAMS.freq_center / speed_of_sound
    sinc_arg = xp.asarray(center_wavenumber * plan.seg_length / 2.0) * sin_theta / pi
    phase_decay_init = phase_decay_init * xpx.sinc(sinc_arg, xp=xp)

    wavenumbers = xp.asarray(2.0 * pi) * plan.selected_freqs / speed_of_sound

    return dict(
        phase_decay_init=phase_decay_init,
        phase_decay_step=phase_decay_step,
        delays_clean=delays_clean,
        tx_apodization=tx_apodization,
        is_out=is_out,
        wavenumbers=wavenumbers,
        speed_of_sound=speed_of_sound,
        pulse_spect=plan.pulse_spectrum,
        probe_spect=plan.probe_spectrum,
        n_sub=plan.n_sub,
        seg_length=plan.seg_length,
        sin_theta=sin_theta,
        full_frequency_directivity=False,
        xp=xp,
    )


def _run_strategy(
    strategy: str,
    positions,
    delays,
    plan: PfieldPlan,
    intermediates: dict,
    xp: _ArrayNamespace,
    chunk: int | None,
) -> np.ndarray:
    """Execute a strategy and return the result as numpy array."""
    if strategy == "baseline":
        accum = _pfield_freq_vectorized(**intermediates)
        return np.asarray(xp.sqrt(accum * plan.correction_factor))
    if strategy == "element_accum":
        accum = _pfield_freq_element_accum(**intermediates, chunk_e=chunk or 1)
        return np.asarray(xp.sqrt(accum * plan.correction_factor))
    if strategy == "freq_chunk":
        accum = _pfield_freq_chunk_freq(**intermediates, chunk_freq=chunk or 1)
        return np.asarray(xp.sqrt(accum * plan.correction_factor))
    if strategy == "grid_chunk":
        result = pfield_compute_chunked(
            positions,
            delays,
            plan,
            PARAMS,
            MEDIUM,
            grid_chunk_size=chunk,
        )
        return np.asarray(result)
    msg = f"Unknown strategy: {strategy}"
    raise ValueError(msg)


def _timed_run(fn, repeats: int) -> float:
    """Return median wall-clock time over repeats."""
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def _measure_peak_memory(fn) -> tuple[float, object]:
    """Run fn under tracemalloc and return (peak_mb, result)."""
    gc.collect()
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6, result


def _skip_if_too_large(n_grid: int, n_elements: int, n_freq: int, strategy: str, chunk: int | None) -> bool:
    """Return True if estimated peak exceeds 100 GB (skip to avoid swapping)."""
    peak_mb = _theoretical_peak_mb(n_grid, n_elements, n_freq, strategy, chunk)
    return peak_mb > 100_000


STRATEGY_CONFIGS = [
    ("baseline", None),
    ("element_accum", 1),
    ("element_accum", 8),
    ("element_accum", 32),
    ("freq_chunk", 1),
    ("freq_chunk", 4),
    ("freq_chunk", 16),
    ("grid_chunk", 10000),
    ("grid_chunk", 50000),
]


def run_benchmarks(
    grid_sizes: list[int],
    backend_name: str,
    repeats: int,
    reference_grid: int = 50,
) -> list[BenchmarkResult]:
    """Run all strategy benchmarks for given grid sizes."""
    xp: _ArrayNamespace
    if backend_name == "numpy":
        xp = np
    elif backend_name == "jax":
        import jax.numpy as jnp  # noqa: PLC0415

        xp = jnp
    else:
        msg = f"Unknown backend: {backend_name}"
        raise ValueError(msg)

    results: list[BenchmarkResult] = []

    # Compute reference at small grid for correctness checking
    ref_positions_np = _make_positions(reference_grid)
    ref_positions = xp.asarray(ref_positions_np)
    ref_delays = xp.zeros(PARAMS.n_elements)
    ref_result = np.asarray(pfield(ref_positions, ref_delays, PARAMS))

    for n in grid_sizes:
        positions_np = _make_positions(n)
        positions = xp.asarray(positions_np)
        delays = xp.zeros(PARAMS.n_elements)

        plan = pfield_precompute(positions, delays, PARAMS)
        n_freq = plan.selected_freqs.shape[0]

        # Build intermediates once per grid size (reused by A, B, baseline)
        intermediates = _build_intermediates(positions, delays, plan, xp)

        # Reference for this grid (only at small sizes)
        use_correctness_check = n == reference_grid

        print(f"\n{'=' * 70}")
        print(f"Grid: {n}x{n} ({n * n:,} points)  |  E={PARAMS.n_elements}  F={n_freq}  n_sub={plan.n_sub}")
        print(f"{'=' * 70}")

        for strategy, chunk in STRATEGY_CONFIGS:
            n_grid = n * n
            if _skip_if_too_large(n_grid, PARAMS.n_elements, n_freq, strategy, chunk):
                print(f"  {strategy:16s} chunk={chunk!s:>6s}  SKIPPED (>{100:.0f} GB estimated)")
                continue

            theoretical = _theoretical_peak_mb(n_grid, PARAMS.n_elements, n_freq, strategy, chunk)

            # Capture loop variables for closure
            _s, _p, _d, _pl, _im, _xp, _c = strategy, positions, delays, plan, intermediates, xp, chunk

            def _run(_s=_s, _p=_p, _d=_d, _pl=_pl, _im=_im, _xp=_xp, _c=_c):
                return _run_strategy(_s, _p, _d, _pl, _im, _xp, _c)

            try:
                peak_mb, first_result = _measure_peak_memory(_run)
                median_time = _timed_run(_run, repeats)
            except Exception as e:
                print(f"  {strategy:16s} chunk={chunk!s:>6s}  ERROR: {e}")
                continue

            # Correctness check
            correct = True
            if use_correctness_check:
                max_err = np.max(np.abs(first_result - ref_result))
                peak = np.max(np.abs(ref_result))
                correct = (max_err / peak) < 1e-6 if peak > 0 else True

            fits_16gb = theoretical < 16_000

            result = BenchmarkResult(
                strategy=strategy,
                grid_size=n,
                backend=backend_name,
                chunk_param=chunk,
                time_s=median_time,
                peak_mem_mb=peak_mb,
                theoretical_peak_mb=theoretical,
                correct=correct,
                n_elements=PARAMS.n_elements,
                n_freq=n_freq,
                n_sub=plan.n_sub,
            )
            results.append(result)

            status = "OK" if correct else "MISMATCH"
            gpu_fit = "16GB-OK" if fits_16gb else "16GB-NO"
            print(
                f"  {strategy:16s} chunk={chunk!s:>6s}  "
                f"time={median_time:8.3f}s  "
                f"mem={peak_mb:8.1f}MB  "
                f"theory={theoretical:8.1f}MB  "
                f"{gpu_fit:7s}  {status}"
            )

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"{'Strategy':16s} {'Chunk':>6s} {'Grid':>6s} {'Time(s)':>8s} "
        f"{'Mem(MB)':>9s} {'Theory(MB)':>10s} {'16GB?':>6s} {'OK?':>4s}"
    )
    print("-" * 90)
    for r in results:
        fits = "YES" if r.theoretical_peak_mb < 16_000 else "NO"
        ok = "OK" if r.correct else "FAIL"
        print(
            f"{r.strategy:16s} {r.chunk_param!s:>6s} {r.grid_size:>6d} "
            f"{r.time_s:>8.3f} {r.peak_mem_mb:>9.1f} {r.theoretical_peak_mb:>10.1f} "
            f"{fits:>6s} {ok:>4s}"
        )


def main():
    """Run benchmark suite and print results."""
    parser = argparse.ArgumentParser(description="Benchmark pfield memory strategies")
    parser.add_argument("--grids", type=int, nargs="+", default=[50, 200, 512], help="Grid sizes to test")
    parser.add_argument("--backend", choices=["numpy", "jax"], default="numpy", help="Array backend")
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats (median)")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    print(f"Platform: {platform.platform()}")
    print(f"Python:   {sys.version}")
    print(f"Backend:  {args.backend}")
    print(f"Grids:    {args.grids}")
    print(f"Repeats:  {args.repeats}")
    print(f"Probe:    P4-2v (n_elements={PARAMS.n_elements})")

    results = run_benchmarks(args.grids, args.backend, args.repeats)
    print_summary(results)

    if args.output:
        out_path = Path(args.output)
        data = [asdict(r) for r in results]
        for d in data:
            d["correct"] = bool(d["correct"])
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
