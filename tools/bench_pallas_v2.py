"""Benchmark Pallas v2 kernel: compilation + execution timing, throughput.

Usage:
    uv run python tools/bench_pallas_v2.py
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
    for pkg in ["cusparse", "cublas", "cuda_runtime", "cufft", "cusolver",
                "cudnn", "nvjitlink"]
    if os.path.isdir(os.path.join(nvidia_base, pkg, "lib"))
)
if nvidia_libs:
    os.environ["LD_LIBRARY_PATH"] = (
        nvidia_libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np

from fast_simus.medium_params import MediumParams
from fast_simus.simus import SimusStrategy, simus_compute, simus_precompute
from fast_simus.transducer_presets import P4_2v
from fast_simus.kernels.pallas_simus import simus_pallas

print(f"Backend: {jax.default_backend()}")

N_SCAT = 50000
N_RUNS = 5

params = P4_2v()
medium = MediumParams()
rng = np.random.default_rng(42)
scat_np = np.column_stack([
    rng.uniform(-0.02, 0.02, N_SCAT),
    rng.uniform(0.01, 0.08, N_SCAT),
]).astype(np.float32)
rc_np = rng.standard_normal(N_SCAT).astype(np.float32)
delays_np = np.zeros(params.n_elements, dtype=np.float32)

scat = jnp.asarray(scat_np)
rc = jnp.asarray(rc_np)
delays = jnp.asarray(delays_np)
plan = simus_precompute(scat, rc, delays, params, medium)
n_freq = int(plan.selected_freqs.shape[0])
n_elem = params.n_elements
print(f"N_SCAT={N_SCAT} N_FREQ={n_freq} N_ELEM={n_elem} N_SUB={plan.n_sub}")


def bench(label, tile_s, use_2d):
    print(f"\n--- {label} (tile_s={tile_s}, 2d={use_2d}) ---")
    # Warmup
    t0 = time.perf_counter()
    out = simus_pallas(
        scat, rc, params, plan, medium, delays, jnp.ones(n_elem),
        tile_s=tile_s, use_2d_carry=use_2d,
    )
    out.block_until_ready()
    t_warmup = time.perf_counter() - t0
    print(f"Warmup: {t_warmup*1000:.0f}ms")

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        out = simus_pallas(
            scat, rc, params, plan, medium, delays, jnp.ones(n_elem),
            tile_s=tile_s, use_2d_carry=use_2d,
        )
        out.block_until_ready()
        times.append(time.perf_counter() - t0)

    t_med = sorted(times)[len(times) // 2]
    t_min = min(times)
    print(f"Median: {t_med*1000:.1f}ms  ({N_SCAT/t_med/1e6:.3f}M scat/s)")
    print(f"Min:    {t_min*1000:.1f}ms  ({N_SCAT/t_min/1e6:.3f}M scat/s)")
    return out


# Reference
print("\n--- lax.scan reference ---")
t0 = time.perf_counter()
ref = simus_compute(scat, rc, delays, plan, params, medium,
                    strategy=SimusStrategy.SCAN)
ref.spectrum.block_until_ready()
t_scan = time.perf_counter() - t0
print(f"lax.scan: {t_scan*1000:.0f}ms  ({N_SCAT/t_scan/1e6:.3f}M scat/s)")

# Test different configs
configs = [
    ("2D tile=8",   8,  True),
    ("2D tile=16",  16, True),
    ("2D tile=32",  32, True),
    ("2D tile=64",  64, True),
    ("Fallback t=64", 64, False),
]

for label, tile_s, use_2d in configs:
    try:
        out = bench(label, tile_s, use_2d)
        # Correctness check
        ref_sel = ref.spectrum[plan.freq_idx_start:plan.freq_idx_start + n_freq, :]
        corrected = out * plan.correction_factor
        ref_np_arr = np.asarray(ref_sel)
        pal_np = np.asarray(corrected)
        mask = np.abs(ref_np_arr) > 1e-10
        if mask.sum() > 0:
            rel = np.abs(ref_np_arr[mask] - pal_np[mask]) / np.abs(ref_np_arr[mask])
            print(f"Accuracy: max_rel={rel.max():.4e} mean_rel={rel.mean():.4e}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

print("\n=DONE=")
