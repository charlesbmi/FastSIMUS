"""Benchmark tiled vmap+scan approach: pure JAX, no Pallas.

Tiles scatterers into small groups (tile_s), vmaps over tiles (GPU-parallel),
lax.scan over frequencies within each tile (phase carry in L1 cache).

Usage:
    uv run python tools/bench_tiled_scan.py
"""
import os
import sys
import time
from math import pi

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.stdout.reconfigure(line_buffering=True)

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

import jax
import jax.numpy as jnp
import numpy as np

from fast_simus.medium_params import MediumParams
from fast_simus.simus import SimusStrategy, simus_compute, simus_precompute
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils.geometry import element_positions

print(f"Backend: {jax.default_backend()}")

_NEPER_TO_DB = 8.685889638065036


def build_tiled_fn(params, plan, medium, delays_clean, tx_apodization, tile_s=64):
    """Build JIT-compiled tiled vmap+scan compute function."""
    n_elem = params.n_elements
    n_freq = int(plan.selected_freqs.shape[0])

    element_pos, theta_raw, apex_offset = element_positions(
        n_elem, params.pitch, params.radius, np,
    )
    if theta_raw is None:
        theta_raw = np.zeros(n_elem, dtype=np.float32)

    elem_x = jnp.asarray(element_pos[:, 0], dtype=jnp.float32)
    elem_z = jnp.asarray(element_pos[:, 1], dtype=jnp.float32)
    cos_te = jnp.asarray(np.cos(theta_raw), dtype=jnp.float32)
    sin_neg_te = jnp.asarray(np.sin(-theta_raw), dtype=jnp.float32)

    c = medium.speed_of_sound
    att = medium.attenuation
    fc = params.freq_center
    seg_length = params.element_width / plan.n_sub
    inv_nsub = 1.0 / plan.n_sub
    center_kw = 2.0 * pi * fc / c
    min_dist = c / fc / 2.0

    freq_start = float(plan.selected_freqs[0])
    freq_step = float(plan.selected_freqs[1] - plan.selected_freqs[0]) if n_freq > 1 else 0.0
    kw_init = 2.0 * pi * freq_start / c
    alpha_init = att / _NEPER_TO_DB * freq_start / 1e6 * 1e2
    kw_step = 2.0 * pi * freq_step / c
    alpha_step = att / _NEPER_TO_DB * freq_step / 1e6 * 1e2

    delays_np = np.asarray(delays_clean, dtype=np.float32)
    tx_apod_np = np.asarray(tx_apodization, dtype=np.float32)

    spectra = np.asarray(plan.pulse_spectrum * plan.probe_spectrum)
    pp_re = jnp.asarray(np.real(spectra), dtype=jnp.float32)
    pp_im = jnp.asarray(np.imag(spectra), dtype=jnp.float32)
    probe_raw = np.asarray(plan.probe_spectrum)
    probe = jnp.asarray(
        np.abs(probe_raw) if np.iscomplexobj(probe_raw) else probe_raw,
        dtype=jnp.float32,
    )

    da_re0 = jnp.asarray(
        [a * np.cos(2*pi*freq_start*d) for d, a in zip(delays_np, tx_apod_np)],
        dtype=jnp.float32,
    )
    da_im0 = jnp.asarray(
        [a * np.sin(2*pi*freq_start*d) for d, a in zip(delays_np, tx_apod_np)],
        dtype=jnp.float32,
    )
    das_re = jnp.asarray([np.cos(2*pi*freq_step*d) for d in delays_np], dtype=jnp.float32)
    das_im = jnp.asarray([np.sin(2*pi*freq_step*d) for d in delays_np], dtype=jnp.float32)

    radius_val = params.radius if params.radius != float('inf') else 1e31

    def process_one_tile(tile_data):
        """Process one tile: vmap-friendly, returns (n_freq, n_elem) spectrum."""
        sx, sz, rc_t, is_out_t = tile_data

        # Distances: (tile_s, n_elem)
        dx = sx[:, None] - elem_x[None, :]
        dz = sz[:, None] - elem_z[None, :]
        r2 = dx * dx + dz * dz
        inv_r = jax.lax.rsqrt(r2 + 1e-30)
        r_val = r2 * inv_r
        rc_clamp = jnp.maximum(r_val, min_dist)

        sin_th = (dx * cos_te[None, :] + dz * sin_neg_te[None, :]) * inv_r
        cos_th_val = (dz * cos_te[None, :] - dx * sin_neg_te[None, :]) * inv_r
        obliq = jnp.where(cos_th_val <= 0.0, 1e-16, cos_th_val)
        sa = center_kw * seg_length * 0.5 * sin_th
        sv_sinc = jnp.where(jnp.abs(sa) < 1e-8, 1.0, jnp.sin(sa) / sa)
        amp = obliq * sv_sinc * jax.lax.rsqrt(rc_clamp) * inv_nsub

        kr = kw_init * rc_clamp
        ar = alpha_init * rc_clamp
        a0 = amp * jnp.exp(-ar)
        ph_re0 = a0 * jnp.cos(kr)
        ph_im0 = a0 * jnp.sin(kr)

        kr_s = kw_step * rc_clamp
        ar_s = alpha_step * rc_clamp
        stp_mag = jnp.exp(-ar_s)
        step_re = stp_mag * jnp.cos(kr_s)
        step_im = stp_mag * jnp.sin(kr_s)

        mask = (1.0 - is_out_t)[:, None]  # (tile_s, 1)
        rc_m = rc_t[:, None] * mask         # (tile_s, 1)

        def freq_body(carry, k):
            ph_re, ph_im, d_re, d_im = carry

            # TX beamform
            p_e_re = ph_re * d_re[None, :] - ph_im * d_im[None, :]
            p_e_im = ph_re * d_im[None, :] + ph_im * d_re[None, :]
            p_re = jnp.sum(p_e_re, axis=1)  # (tile_s,)
            p_im = jnp.sum(p_e_im, axis=1)

            pw_re = (pp_re[k] * p_re - pp_im[k] * p_im)[:, None] * rc_m
            pw_im = (pp_re[k] * p_im + pp_im[k] * p_re)[:, None] * rc_m

            # RX
            spect_re_k = jnp.sum(pw_re * ph_re - pw_im * ph_im, axis=0) * probe[k]
            spect_im_k = jnp.sum(pw_re * ph_im + pw_im * ph_re, axis=0) * probe[k]

            new_ph_re = ph_re * step_re - ph_im * step_im
            new_ph_im = ph_re * step_im + ph_im * step_re
            new_d_re = d_re * das_re - d_im * das_im
            new_d_im = d_re * das_im + d_im * das_re

            return (new_ph_re, new_ph_im, new_d_re, new_d_im), (spect_re_k, spect_im_k)

        init_carry = (ph_re0, ph_im0, da_re0, da_im0)
        _, (spect_re_all, spect_im_all) = jax.lax.scan(
            freq_body, init_carry, jnp.arange(n_freq),
        )
        return spect_re_all, spect_im_all  # (n_freq, n_elem) each

    vmapped_process = jax.vmap(process_one_tile)

    @jax.jit
    def compute(scat_x, scat_z, rc_arr, is_out_arr):
        n_tiles = scat_x.shape[0] // tile_s
        sx_tiles = scat_x.reshape(n_tiles, tile_s)
        sz_tiles = scat_z.reshape(n_tiles, tile_s)
        rc_tiles = rc_arr.reshape(n_tiles, tile_s)
        io_tiles = is_out_arr.reshape(n_tiles, tile_s)

        tile_spect_re, tile_spect_im = vmapped_process(
            (sx_tiles, sz_tiles, rc_tiles, io_tiles),
        )  # (n_tiles, n_freq, n_elem)

        spect_re = jnp.sum(tile_spect_re, axis=0)
        spect_im = jnp.sum(tile_spect_im, axis=0)
        return spect_re + 1j * spect_im

    return compute


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

# Prepare inputs
scat_x = scat[:, 0].astype(jnp.float32)
scat_z = scat[:, 1].astype(jnp.float32)
rc_f32 = rc.astype(jnp.float32)

radius_val = params.radius if params.radius != float('inf') else 1e31
_, _, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, np)
is_out_f = jnp.where(scat_z < 0.0, 1.0, 0.0)
if radius_val < 1e30:
    is_out_f = jnp.where(
        (scat_x**2 + (scat_z + apex_offset)**2) <= radius_val**2,
        1.0, is_out_f,
    )

# Reference (with warmup)
print("\n--- lax.scan reference ---")
ref_fn = jax.jit(lambda s, r, d: simus_compute(
    s, r, d, plan, params, medium, strategy=SimusStrategy.SCAN,
))
t0 = time.perf_counter()
ref = ref_fn(scat, rc, delays)
ref.spectrum.block_until_ready()
t_warmup = time.perf_counter() - t0
print(f"Warmup (compile): {t_warmup*1000:.0f}ms")

times_scan = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    ref = ref_fn(scat, rc, delays)
    ref.spectrum.block_until_ready()
    times_scan.append(time.perf_counter() - t0)
t_med = sorted(times_scan)[len(times_scan)//2]
t_min = min(times_scan)
print(f"Execution med: {t_med*1000:.1f}ms  ({N_SCAT/t_med/1e6:.3f}M scat/s)")
print(f"Execution min: {t_min*1000:.1f}ms  ({N_SCAT/t_min/1e6:.3f}M scat/s)")

# Tiled vmap+scan
for tile_s in [16, 32, 64, 128, 256]:
    pad = (tile_s - N_SCAT % tile_s) % tile_s
    sx_p = jnp.pad(scat_x, (0, pad))
    sz_p = jnp.pad(scat_z, (0, pad), constant_values=-1.0)
    rc_p = jnp.pad(rc_f32, (0, pad))
    io_p = jnp.pad(is_out_f, (0, pad), constant_values=1.0)

    n_tiles = (N_SCAT + pad) // tile_s
    print(f"\n--- tiled vmap+scan tile_s={tile_s} ({n_tiles} tiles) ---")
    compute = build_tiled_fn(params, plan, medium, delays, jnp.ones(n_elem), tile_s=tile_s)

    t0 = time.perf_counter()
    out = compute(sx_p, sz_p, rc_p, io_p)
    out.block_until_ready()
    t_warmup = time.perf_counter() - t0
    print(f"Warmup (compile): {t_warmup*1000:.0f}ms")

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        out = compute(sx_p, sz_p, rc_p, io_p)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)

    t_med = sorted(times)[len(times)//2]
    t_min = min(times)
    print(f"Execution med: {t_med*1000:.1f}ms  ({N_SCAT/t_med/1e6:.3f}M scat/s)")
    print(f"Execution min: {t_min*1000:.1f}ms  ({N_SCAT/t_min/1e6:.3f}M scat/s)")

    # Correctness
    ref_sel = ref.spectrum[plan.freq_idx_start:plan.freq_idx_start + n_freq, :]
    correction = plan.correction_factor
    corrected = out * correction
    ref_np_arr = np.asarray(ref_sel)
    pal_np = np.asarray(corrected)
    mask = np.abs(ref_np_arr) > 1e-10
    if mask.sum() > 0:
        rel_err = np.abs(ref_np_arr[mask] - pal_np[mask]) / np.abs(ref_np_arr[mask])
        print(f"Accuracy: max_rel={rel_err.max():.4e} mean_rel={rel_err.mean():.4e}")

print("\n=DONE=")
