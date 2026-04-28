"""Small-scale Pallas kernel test for correctness.

Usage:
    uv run python tools/bench_pallas_small.py
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

N_SCAT = 256

print(f"Backend: {jax.default_backend()}")
print(f"Small-scale test with N_SCAT={N_SCAT}")

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
print(f"N_FREQ={n_freq} N_ELEM={n_elem} N_SUB={plan.n_sub}")

print("\n--- Reference (lax.scan) ---")
ref = simus_compute(scat, rc, delays, plan, params, medium, strategy=SimusStrategy.SCAN)
ref_spect = ref.spectrum
print(f"Ref spectrum shape: {ref_spect.shape}")

print("\n--- Pallas kernel ---")
from fast_simus.kernels.pallas_simus import simus_pallas

try:
    t0 = time.perf_counter()
    pallas_spect = simus_pallas(
        scat, rc, params, plan, medium, delays, jnp.ones(n_elem),
        tile_s=64,
    )
    pallas_spect.block_until_ready()
    t1 = time.perf_counter()
    print(f"Pallas spectrum shape: {pallas_spect.shape}")
    print(f"Time: {(t1-t0)*1000:.0f}ms")

    ref_sel = ref_spect[plan.freq_idx_start : plan.freq_idx_start + n_freq, :]
    correction = plan.correction_factor
    pallas_corrected = pallas_spect * correction

    ref_np = np.asarray(ref_sel)
    pal_np = np.asarray(pallas_corrected)
    mask = np.abs(ref_np) > 1e-10
    if mask.sum() > 0:
        rel = np.abs(ref_np[mask] - pal_np[mask]) / np.abs(ref_np[mask])
        print(f"Pallas vs Ref: max_rel={rel.max():.4e} mean_rel={rel.mean():.4e} n_compared={mask.sum()}")
    else:
        print("No significant values to compare")

    print(f"Ref max: {np.abs(ref_np).max():.4e}")
    print(f"Pallas max: {np.abs(pal_np).max():.4e}")
    print(f"Diff max: {np.abs(ref_np - pal_np).max():.4e}")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=DONE=")
