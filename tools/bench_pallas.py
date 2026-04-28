"""Benchmark JAX baseline (lax.scan) and Pallas kernel for simus.

Usage:
    uv run python tools/bench_pallas.py
"""
import os
import sys
import time

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PALLAS_USE_MOSAIC_GPU"] = "0"

nvidia_base = os.path.join(
    os.path.dirname(__file__), "..", ".venv", "lib", "python3.12",
    "site-packages", "nvidia",
)
nvidia_libs = ":".join(
    os.path.join(nvidia_base, pkg, "lib")
    for pkg in ["cusparse", "cublas", "cuda_runtime", "cufft", "cusolver", "cudnn", "nvjitlink"]
    if os.path.isdir(os.path.join(nvidia_base, pkg, "lib"))
)
if nvidia_libs:
    os.environ["LD_LIBRARY_PATH"] = nvidia_libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")
sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np

from fast_simus.medium_params import MediumParams
from fast_simus.simus import SimusStrategy, simus_compute, simus_precompute
from fast_simus.transducer_presets import P4_2v

N_SCAT = 100_000
WARMUP = 2
REPS = 5


def make_data():
    params = P4_2v()
    medium = MediumParams()
    rng = np.random.default_rng(42)
    scat_np = np.column_stack([
        rng.uniform(-0.02, 0.02, N_SCAT),
        rng.uniform(0.01, 0.08, N_SCAT),
    ]).astype(np.float32)
    rc_np = rng.standard_normal(N_SCAT).astype(np.float32)
    delays_np = np.zeros(params.n_elements, dtype=np.float32)

    scat_jax = jnp.asarray(scat_np)
    rc_jax = jnp.asarray(rc_np)
    delays_jax = jnp.asarray(delays_np)

    plan = simus_precompute(scat_jax, rc_jax, delays_jax, params, medium)
    return scat_jax, rc_jax, delays_jax, plan, params, medium


def bench_scan():
    """Benchmark the lax.scan (SCAN) strategy."""
    scat, rc, delays, plan, params, medium = make_data()
    n_freq = int(plan.selected_freqs.shape[0])
    n_elem = params.n_elements

    print(f"JAX lax.scan baseline | N_SCAT={N_SCAT:,} N_FREQ={n_freq} N_ELEM={n_elem}")
    print(f"Backend: {jax.default_backend()}")
    print("=" * 70)

    for i in range(WARMUP):
        result = simus_compute(
            scat, rc, delays, plan, params, medium,
            strategy=SimusStrategy.SCAN,
        )
        result.rf.block_until_ready()

    times = []
    for i in range(REPS):
        t0 = time.perf_counter()
        result = simus_compute(
            scat, rc, delays, plan, params, medium,
            strategy=SimusStrategy.SCAN,
        )
        result.rf.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    best = min(times)
    median = sorted(times)[len(times) // 2]
    sps = N_SCAT / (best / 1000)
    print(f"lax.scan: best={best:.1f}ms median={median:.1f}ms -> {sps/1e6:.2f}M scat/s")

    ref_spect = result.spectrum
    return best, ref_spect


def bench_cuda():
    """Benchmark the CUDA strategy for comparison."""
    scat, rc, delays, plan, params, medium = make_data()

    for _ in range(WARMUP):
        result = simus_compute(
            scat, rc, delays, plan, params, medium,
            strategy=SimusStrategy.CUDA,
        )
        result.rf.block_until_ready()

    times = []
    for i in range(REPS):
        t0 = time.perf_counter()
        result = simus_compute(
            scat, rc, delays, plan, params, medium,
            strategy=SimusStrategy.CUDA,
        )
        result.rf.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    best = min(times)
    sps = N_SCAT / (best / 1000)
    print(f"CUDA:     best={best:.1f}ms -> {sps/1e6:.2f}M scat/s")
    return best, result.spectrum


def bench_pallas():
    """Benchmark the Pallas kernel."""
    from fast_simus.kernels.pallas_simus import simus_pallas

    scat, rc, delays, plan, params, medium = make_data()

    print("Pallas fused kernel:")
    for tile_s in [256]:
        for tile_f in [32]:
            for i in range(WARMUP):
                result = simus_pallas(
                    scat, rc, params, plan, medium,
                    delays, jnp.ones(params.n_elements),
                    tile_s=tile_s, tile_f=tile_f,
                )
                result.block_until_ready()

            times = []
            for i in range(REPS):
                t0 = time.perf_counter()
                result = simus_pallas(
                    scat, rc, params, plan, medium,
                    delays, jnp.ones(params.n_elements),
                    tile_s=tile_s, tile_f=tile_f,
                )
                result.block_until_ready()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            best = min(times)
            sps = N_SCAT / (best / 1000)
            print(f"  TS={tile_s} TF={tile_f}: best={best:.1f}ms -> {sps/1e6:.2f}M scat/s")

    return best, result


def main():
    scan_time, scan_spect = bench_scan()
    print()

    try:
        pallas_time, pallas_spect = bench_pallas()
        print()
        pallas_s = np.asarray(pallas_spect)
        scan_s = np.asarray(scan_spect)
        mask = np.abs(scan_s) > 1e-10
        if mask.sum() > 0:
            rel = np.abs(scan_s[mask] - pallas_s[mask]) / np.abs(scan_s[mask])
            print(f"Pallas vs SCAN: max_rel={rel.max():.2e} mean_rel={rel.mean():.2e}")
    except Exception as e:
        print(f"Pallas failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTarget: 3.33ms = 30.0M scat/s")
    print(f"Speedup needed over SCAN: {scan_time/3.33:.1f}x")


if __name__ == "__main__":
    main()
