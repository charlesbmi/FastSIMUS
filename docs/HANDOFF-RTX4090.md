# FastSIMUS CUDA Optimization: RTX 4090 Hand-Off

## Goal

Reach **10M scatterers/second** (100K scatterers in 10ms) for the SIMUS
ultrasound simulation algorithm on an RTX 4090.

## Current State (RTX A4000)

**Champion kernel**: `src/fast_simus/kernels/simus_fused_v11.cu` with
`B_SCAT=5, ELEM_TILE=8, TG_SIZE=128, 192 blocks`.

| Metric | Value |
|--------|-------|
| Best time (100K scat) | 32.5ms |
| Throughput | 3.08M scat/s |
| Registers/thread | 228 |
| Shmem/block | 42.9 KB |
| Blocks/SM | 2 |
| Achieved occupancy | 16.7% |
| SM throughput | 57.1% |
| FMA pipe utilization | 42.5% |
| L2 atomic pressure | 21.7% |
| Atomic requests | 69.1M |
| IPC | 2.28 |

**Cross-platform validation**: Apple M4 Max (17 TFLOPS) achieves ~3.0M scat/s
with Metal kernels. The A4000 (19.2 TFLOPS) achieves 3.08M scat/s. Throughput
tracks FP32 TFLOPS linearly, confirming the algorithm hits a fundamental
compute+atomic efficiency ceiling at ~9x theoretical minimum.

## RTX 4090 Projection: 7-9ms (11-14M scat/s)

| Spec | A4000 | RTX 4090 | Ratio |
|------|-------|----------|-------|
| SMs | 48 | 128 | 2.67x |
| Clock | 1,560 MHz | ~2,520 MHz | 1.62x |
| FP32 TFLOPS | 19.2 | 82.6 | 4.30x |
| **L2 cache** | **4 MB** | **72 MB** | **18x** |
| Shmem/SM | 128 KB | 128 KB | 1.0x |
| Memory BW | 448 GB/s | 1,008 GB/s | 2.25x |
| Compute capability | sm_86 | sm_89 | - |

The output array is 64 x 854 x 2 x 4 = 438 KB. On the A4000 (4 MB L2) this
competes for cache residency. On the 4090 (72 MB L2) the entire output sits
comfortably in L2; `atomicAdd` never needs DRAM.

Projected breakdown:

| Component | A4000 time | Scaling | 4090 projected |
|-----------|-----------|---------|----------------|
| Compute (57%) | 18.5ms | 4.3x (SM x clock) | 4.3ms |
| L2 atomic (22%) | 7.2ms | ~6x (L2 size + slices) | 1.2ms |
| Memory/other (21%) | 6.8ms | ~3x (SM + BW) | 2.3ms |
| **Total** | **32.5ms** | | **~7.8ms** |

## Recommended Experiment Plan

### Phase 1: Baseline validation (run first)

1. **Lock GPU clocks** for reproducible benchmarks:
   ```bash
   sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 2520
   ```

2. **Run v11 B=5 ET=8 as-is** (change `sm_86` -> `sm_89` in cuda_runtime.py):
   ```bash
   uv run python test_fused_v11.py
   ```
   Expected: 7-9ms. If significantly off, re-profile with ncu.

3. **Profile with ncu**:
   ```bash
   sudo ncu --target-processes all --launch-skip 1 --launch-count 1 \
     --set full -f -o 4090_v11_baseline \
     $(which uv) run python tools/ncu_profile.py \
       src/fast_simus/kernels/simus_fused_v11.cu --b-scat 5 --elem-tile 8
   ```
   Then extract metrics:
   ```bash
   sudo ncu --import 4090_v11_baseline.ncu-rep --page raw 2>&1 | \
     uv run python tools/ncu_parse.py
   ```

### Phase 2: B_SCAT sweep (highest priority)

On the A4000, `B_SCAT > 5` hits the register wall (255 regs/thread) and the
resulting spills negate the atomic reduction. On the 4090, L2 atomic pressure
should drop dramatically (72 MB L2), so spill costs may be acceptable.

```bash
# Sweep B=5,6,7,8,10,12 with ET=8
for B in 5 6 7 8 10 12; do
  uv run python tools/ncu_profile.py \
    src/fast_simus/kernels/simus_fused_v11.cu --b-scat $B --elem-tile 8
done
```

**What to look for**: Does higher B reduce total time despite register spills?
Check `l1tex__t_output_wavefronts_pipe_lsu_mem_local_op_ld.sum` (spill loads)
and `lts__d_atomic_input_cycles_active` (L2 atomic pressure).

### Phase 3: Two-pass TX/RX re-test

This failed on the A4000 (Exp 8b was 43% slower) because the TX buffer in DRAM
added latency. On the 4090, the output (438 KB) is fully L2-resident and the
TX buffer may be partially cached.

Relevant kernels from the A4000 experiments:
- `src/fast_simus/kernels/simus_tx_v14.cu` (TX pass)
- `src/fast_simus/kernels/simus_rx_tiled_v14.cu` (RX pass, tiled shmem output)
- `src/fast_simus/kernels/simus_rx_v14.cu` (RX pass, Metal-style shuffle)

The tiled variant (8b) is the better starting point. The shuffle variant (8a)
produced 79x more atomics on CUDA due to thread mapping mismatch.

### Phase 4: fp16 TX with high B_SCAT

`src/fast_simus/kernels/simus_fused_v15.cu` uses fp16 for the TX buffer,
halving shmem. On the A4000 this gave only 2.5% improvement (31.6ms vs 32.5ms)
because the register wall prevented higher B_SCAT. On the 4090, if L2 atomics
are no longer the bottleneck, the register spill cost from higher B_SCAT is
offset by fewer total iterations.

```bash
# Sweep with fp16 TX
uv run python tools/ncu_profile_v15.py --sweep
```

### Phase 5: Architecture experiments (if needed)

If v11 already hits <10ms on the 4090, these are lower priority:

- **Warp shuffle + SCAT_REDUCE hybrid**: Remap threads so adjacent threads
  handle the same element from different scatterers (like Metal's
  `simus_rx_simd.metal`). On the 4090 with reduced L2 pressure, this could
  further cut atomics.

- **Cooperative groups for grid-wide reduction**: Instead of persistent blocks
  with atomics, use cooperative launch with grid-wide barriers.

## File Inventory

### Essential files (needed to run experiments)

| File | Purpose |
|------|---------|
| `src/fast_simus/kernels/simus_fused_v11.cu` | Champion kernel (fused TX+RX, batched+tiled) |
| `src/fast_simus/kernels/simus_fused_v15.cu` | fp16 TX variant of v11 |
| `src/fast_simus/kernels/cuda_runtime.py` | NVRTC compile + CUDA driver API launch |
| `src/fast_simus/kernels/cuda_simus.py` | High-level dispatch (integrates with simus.py) |
| `tools/ncu_profile.py` | Parameterized profiling harness |
| `tools/ncu_profile_v15.py` | Profiling harness for v15 (fp16 TX) |
| `tools/ncu_parse.py` | Extract key metrics from ncu raw output |
| `tools/accuracy_compare.py` | Compare v15 vs v11 accuracy |
| `test_fused_v11.py` | Benchmark v11 configs + correctness vs v6 |

### Reference kernels (for architecture experiments)

| File | Purpose |
|------|---------|
| `src/fast_simus/kernels/simus_fused_v6.cu` | Pre-batching baseline (68.9ms) |
| `src/fast_simus/kernels/simus_fused_v12.cu` | Per-block output (failed: 3-5x slower) |
| `src/fast_simus/kernels/simus_fused_v13_freqchunk.cu` | Freq-chunked (failed: 49% slower) |
| `src/fast_simus/kernels/simus_tx_v14.cu` | Two-pass TX (for re-test on 4090) |
| `src/fast_simus/kernels/simus_rx_tiled_v14.cu` | Two-pass RX tiled shmem (for re-test) |
| `src/fast_simus/kernels/simus_rx_v14.cu` | Two-pass RX shuffle (failed on A4000) |
| `src/fast_simus/kernels/simus_fused_v11_stagger.cu` | Staggered element groups (no effect) |

### Documentation

| File | Purpose |
|------|---------|
| `docs/progress/cuda-kernel-optimization.md` | Main progress tracker |
| `docs/progress/architecture-exploration.md` | Root cause analysis + hypothesis docs |
| `docs/progress/experiments/exp*.md` | Individual experiment write-ups |

### Metal reference (read-only, for architectural inspiration)

| File | Purpose |
|------|---------|
| `src/fast_simus/kernels/simus_tx_tiled.metal` | Metal TX kernel (element-tiled progression) |
| `src/fast_simus/kernels/simus_rx_simd.metal` | Metal RX kernel (SIMD shuffle reduction) |

## Key Learnings (Do NOT Repeat These Failures)

1. **Per-block output buffers** (Exp 6): Output (438 KB) >> L1 cache (48 KB).
   L2 read-modify-write is SLOWER than hardware atomicAdd on Ampere.
   *May be worth re-testing on 4090 ONLY if output fits in L2.*

2. **Frequency chunking** (Exp 7): Geometry recomputation per chunk costs ~40%
   of kernel time. Only viable if geometry is cached in global memory between
   launches.

3. **Metal-style SIMD shuffle on CUDA** (Exp 8a): The thread mapping
   (`tid = freq * N_ELEM + elem`) means each thread writes to a unique
   (elem, freq) pair. Warp shuffle has nothing to reduce. To make shuffle work,
   threads must map as `tid = elem * SCAT_REDUCE + scat_batch`.

4. **`__launch_bounds__` register reduction** (Exp 3): Shmem, not registers,
   limits occupancy. Forcing fewer registers causes 6.7x spills with zero
   occupancy improvement.

5. **Staggering element group order** (Exp 9b): L2 atomic contention comes
   from aggregate volume, not cache line collisions between blocks.

6. **The 2-block/SM cliff**: Going from 2 to 1 block/SM drops performance 45%.
   All experiments must maintain >= 2 blocks/SM.

## Algorithm Summary

The SIMUS algorithm computes:
```
output[elem][freq] = sum_over_scatterers(
    TX[scat][freq] * RX_phasor[scat][elem][freq]
) * probe[freq]
```

- TX[scat][freq] = sum_over_sub_elements(phasor(scat, sub_elem, freq))
- RX_phasor = geometric_progression(distance, kw_step) per frequency
- Output: 64 elements x 854 frequencies x 2 (re/im) = 438 KB
- Input: 100K scatterers (x, z, reflection_coefficient)

The v11 kernel fuses TX+RX in a single persistent kernel:
1. **Phase 1**: Compute geometry (distances, angles) for B_SCAT scatterers
2. **Phase 2**: TX sweep -- geometric progression over frequencies per sub-element
3. **Phase 3**: RX sweep -- element-tiled geometric progression, accumulate
   B_SCAT scatterers in registers, atomicAdd to output

Total FLOPs: ~65.5 GFLOPS (100K x 64 x 854 x ~12 FLOPs per complex multiply).

## Setup on RTX 4090

```bash
git clone <repo> && cd FastSIMUS
uv sync

# Change compute capability
# In src/fast_simus/kernels/cuda_runtime.py, change sm_86 -> sm_89
# In tools/ncu_profile.py, if arch is hardcoded, change similarly

# Lock clocks
sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 2520

# Verify
nvidia-smi -q -d CLOCK | grep -A3 "Max Clocks"

# Run baseline
uv run python test_fused_v11.py
```

## Profiling Quick Reference

```bash
# Full ncu profile (requires sudo for GPU perf counters)
sudo ncu --target-processes all --launch-skip 1 --launch-count 1 \
  --set full -f -o report_name \
  $(which uv) run python tools/ncu_profile.py <kernel.cu> [--b-scat N] [--elem-tile N]

# Parse results
sudo ncu --import report_name.ncu-rep --page raw 2>&1 | uv run python tools/ncu_parse.py

# Key metrics to watch:
#   gpu__time_duration.sum                           -- kernel time
#   lts__d_atomic_input_cycles_active.avg.pct...     -- L2 atomic pressure
#   sm__pipe_fma_cycles_active.avg.pct...            -- FMA utilization
#   l1tex__t_output_wavefronts_pipe_lsu_mem_local... -- register spills
#   sm__inst_executed.avg.per_cycle_elapsed           -- IPC
```
