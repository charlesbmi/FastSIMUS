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

## Complete A4000 Experiment History (Exp 1-14)

| # | Experiment | Result | Key Finding |
|---|-----------|--------|------------|
| 1 | v6 baseline ncu profile | Reference | L2 atomic bound (51%), 42% occupancy wasted |
| 2 | v11 B=2 (higher occupancy) | -10% | 46% L2 atomic pressure kills the gain |
| 3 | v11 + __launch_bounds__(128,4) | **-200%** | Shmem is occupancy limiter, not regs; 6.7x spills |
| 4 | v11 + shmem padding + SFU precompute | +3% | SFU 21%->13%, but LSU 37%->50%; bank pad N/A |
| 5 | v11 B_SCAT sweep (B=2,4,5,6,8) | **+8%** | B=5 optimal; 2->1 block/SM cliff at B=6 |
| 6 | v12 per-block output buffers | **-200%** | Output >> L1; R-M-W slower than atomicAdd |
| 7 | v13 freq-chunked multi-launch | -49% | Geometry recomputed per chunk dominates |
| 8a | v14 two-pass Metal-style SIMD RX | **-392%** | 79x more atomics than v11 (wrong thread map) |
| 8b | v14 two-pass tiled shmem output | -43% | Geometry per tile + shmem overhead > savings |
| 9a | v15 fp16 TX buffer | **+2.5%** | 31.6ms; register wall blocks higher B_SCAT |
| 9b | v11-stagger element group order | 0% | Atomic contention from volume, not collisions |
| 10 | Combined analysis + GPU roadmap | Analysis | A4000 near-optimal; 4090 projected 7-9ms |
| 11 | Block count sweep (48-384) | 0% | Wave quantization matters; L2 contention disproved |
| 12 | v16 Grid-Y element group partitioning | **-261%** | 8x redundant Phase 1+2 dominates atomic savings |
| 13 | L2 persistence hints | 0% | Output already L2-resident; no effect |
| 14 | v17 SCAT_REDUCE warp shuffle | **DNF** | Register spills from dynamic loop; catastrophic |

### New findings from Exp 11-14

**Exp 11: Block count sweep** -- Tested 48, 96, 128, 192, 256, 384 blocks.
Performance is identical for multiples of 96 (= 2 blocks/SM x 48 SMs). Non-
multiples suffer wave quantization tail effects up to 48%. **L2 contention is
NOT a factor at B_SCAT=5** (halving blocks from 192 to 96 has zero effect).
On 4090: set blocks to `2 * 128 = 256`.

**Exp 12: Grid-Y element group partitioning** -- Extended grid to `(blocks_x,
N_ELEM_GROUPS=8, 1)` so each block handles one element group. Reduces atomics
per output entry by 8x, but requires 8x redundant Phase 1+2 (geometry + TX).
Result: 117ms vs 32.5ms (-261%). Phase 1+2 accounts for ~60% of kernel time.
**On 4090: this approach only works with a two-pass architecture** where TX is
precomputed once and shared via L2.

**Exp 13: L2 persistence hints** -- Set `CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW`
with `hitProp=PERSISTING` for the output array. No measurable effect (32.7ms vs
32.6ms). The 438 KB output is already L2-resident during kernel execution.

**Exp 14: SCAT_REDUCE warp-cooperative reduction** -- Created v17 with
`__shfl_xor_sync` reduction across SCAT_REDUCE adjacent threads. Catastrophic
failure: the dynamic scatterer loop prevents compiler unrolling, causing all
geometric progression states to spill to local memory (336 bytes of spills at
B=4 ET=8 SR=2). Kernel cannot complete even 1K scatterers.
**On 4090: only viable with a two-pass RX-only kernel** (no fused Phase 1+2).

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

3. **Important**: Set block count to 256 (= 2 x 128 SMs) for optimal wave
   quantization. The `--blocks` parameter defaults to 192 which is NOT a
   multiple of 256 on the 4090.

4. **Profile with ncu**:
   ```bash
   sudo ncu --target-processes all --launch-skip 1 --launch-count 1 \
     --set full -f -o 4090_v11_baseline \
     $(which uv) run python tools/ncu_profile.py \
       src/fast_simus/kernels/simus_fused_v11.cu --b-scat 5 --elem-tile 8 --blocks 256
   ```

### Phase 2: B_SCAT sweep (highest priority)

On the A4000, `B_SCAT > 5` hits the register wall (255 regs/thread) and the
resulting spills negate the atomic reduction. On the 4090, L2 atomic pressure
should drop dramatically (72 MB L2), so spill costs may be acceptable.

```bash
for B in 5 6 7 8 10 12; do
  uv run python tools/ncu_profile.py \
    src/fast_simus/kernels/simus_fused_v11.cu --b-scat $B --elem-tile 8 --blocks 256
done
```

**Critical check from Exp 11**: On A4000, B=6 drops to 1 block/SM (45% penalty).
On the 4090, verify if B=6 still causes this cliff. If shmem at B=6 (~51 KB)
stays under 64 KB (128/2 per SM), you may get 2 blocks/SM.

### Phase 3: Two-pass TX/RX re-test (HIGHEST 4090-SPECIFIC PRIORITY)

This failed on A4000 (Exp 8b: -43%) due to TX buffer DRAM latency. On the
4090, the situation changes fundamentally:
- Output (438 KB) is fully L2-resident
- TX buffer: 100K x 854 x 2 x 4 = 684 MB does NOT fit L2, but is streamed
- **Grid-Y partitioning becomes viable** (Exp 12 showed -261% on fused kernel,
  but with two-pass TX/RX, Phase 1+2 is NOT duplicated)
- **SCAT_REDUCE becomes viable** (Exp 14 failed because fused kernel's register
  pressure prevents unrolling; a separate RX kernel has no Phase 1+2 overhead)

**New architecture for 4090**:
1. TX kernel: one block per scatterer batch, writes TX[scat][freq] to global memory
2. RX kernel: Grid-Y `(blocks_x, N_ELEM_GROUPS, 1)`, each block reads TX from L2
   and writes to 1/8th of output with SCAT_REDUCE=2

Relevant kernels from A4000 experiments:
- `src/fast_simus/kernels/simus_tx_v14.cu` (TX pass)
- `src/fast_simus/kernels/simus_rx_tiled_v14.cu` (RX pass, tiled shmem output)
- `src/fast_simus/kernels/simus_fused_v16.cu` (Grid-Y element partitioning, for reference)
- `src/fast_simus/kernels/simus_fused_v17.cu` (SCAT_REDUCE, for reference -- only useful as RX-only)

### Phase 4: fp16 TX with high B_SCAT

`src/fast_simus/kernels/simus_fused_v15.cu` uses fp16 for the TX buffer,
halving shmem. On A4000: only +2.5% because register wall prevented higher
B_SCAT. On 4090 with reduced L2 atomic pressure, higher B_SCAT may be viable.

```bash
uv run python tools/ncu_profile_v15.py --sweep
```

### Phase 5: Architecture experiments (if needed)

If the two-pass approach (Phase 3) works, combine with:
- Grid-Y partitioning (Exp 12 approach, viable without Phase 1+2 duplication)
- SCAT_REDUCE (Exp 14 approach, viable without fused kernel register pressure)
- Cooperative groups for grid-wide reduction

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
| `tools/ncu_profile_v16.py` | Profiling harness for v16 (Grid-Y partitioning) |
| `tools/ncu_profile_l2persist.py` | L2 persistence hint test harness |
| `tools/ncu_parse.py` | Extract key metrics from ncu raw output |
| `tools/accuracy_compare.py` | Compare v15 vs v11 accuracy |
| `test_fused_v11.py` | Benchmark v11 configs + correctness vs v6 |

### Reference kernels (for architecture experiments)

| File | Purpose |
|------|---------|
| `src/fast_simus/kernels/simus_fused_v6.cu` | Pre-batching baseline (68.9ms) |
| `src/fast_simus/kernels/simus_fused_v12.cu` | Per-block output (failed: 3-5x slower) |
| `src/fast_simus/kernels/simus_fused_v13_freqchunk.cu` | Freq-chunked (failed: 49% slower) |
| `src/fast_simus/kernels/simus_fused_v16.cu` | Grid-Y element partitioning (failed fused: -261%) |
| `src/fast_simus/kernels/simus_fused_v17.cu` | SCAT_REDUCE (failed fused: DNF, spills) |
| `src/fast_simus/kernels/simus_tx_v14.cu` | Two-pass TX (for re-test on 4090) |
| `src/fast_simus/kernels/simus_rx_tiled_v14.cu` | Two-pass RX tiled shmem (for re-test) |
| `src/fast_simus/kernels/simus_rx_v14.cu` | Two-pass RX shuffle (failed on A4000) |
| `src/fast_simus/kernels/simus_fused_v11_stagger.cu` | Staggered element groups (no effect) |

### Documentation

| File | Purpose |
|------|---------|
| `docs/progress/cuda-kernel-optimization.md` | Main progress tracker |
| `docs/progress/architecture-exploration.md` | Root cause analysis + hypothesis docs |
| `docs/progress/experiments/exp1-*.md` through `exp14-*.md` | Individual experiment write-ups |

### Metal reference (read-only, for architectural inspiration)

| File | Purpose |
|------|---------|
| `src/fast_simus/kernels/simus_tx_tiled.metal` | Metal TX kernel (element-tiled progression) |
| `src/fast_simus/kernels/simus_rx_simd.metal` | Metal RX kernel (SIMD shuffle reduction) |

## Key Learnings (Do NOT Repeat These Failures)

### Failures confirmed on A4000 (do NOT retry on A4000)

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

7. **Block count reduction** (Exp 11): Reducing blocks from 192 to 96 has zero
   effect on L2 contention at B_SCAT=5. Block count MUST be a multiple of
   `2 * n_SMs` for wave quantization.

8. **Grid-Y element partitioning in fused kernel** (Exp 12): The 8x redundant
   Phase 1+2 compute vastly outweighs the 8x atomic reduction. Only viable
   with a two-pass architecture.

9. **L2 persistence hints** (Exp 13): The output array is already L2-resident
   during kernel execution. Hints have zero effect.

10. **SCAT_REDUCE in fused kernel** (Exp 14): Dynamic scatterer iteration
    prevents compiler unrolling, causing catastrophic register spills (336 bytes).
    Only viable in a separate RX kernel without Phase 1+2 register pressure.

### Approaches that MIGHT work on 4090 (need re-testing)

1. **Two-pass TX/RX with Grid-Y RX** (combining Exp 8, 12):
   - TX pass writes to global memory (no atomics)
   - RX pass uses Grid-Y to handle 1/8th of output per block
   - 72 MB L2 keeps output resident, TX is streamed
   - No redundant Phase 1+2 because TX is separate

2. **SCAT_REDUCE in RX-only kernel** (combining Exp 8, 14):
   - Without fused Phase 1+2, the RX kernel has fewer registers
   - Geometric progression states can stay in registers
   - Combined with Grid-Y: could reduce atomics by 8x * 2x = 16x

3. **Higher B_SCAT with register spills** (Exp 5 extrapolation):
   - On A4000, B=6 causes 1 block/SM cliff. On 4090, check if shmem allows 2 blocks/SM at B=6
   - If L2 atomics are near-free on 4090, spills from B=8+ might be acceptable

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

# Run baseline (note: use --blocks 256 for 4090's 128 SMs)
uv run python tools/ncu_profile.py \
  src/fast_simus/kernels/simus_fused_v11.cu --b-scat 5 --elem-tile 8 --blocks 256
```

## Profiling Quick Reference

```bash
# Full ncu profile (requires sudo for GPU perf counters)
sudo ncu --target-processes all --launch-skip 1 --launch-count 1 \
  --set full -f -o report_name \
  $(which uv) run python tools/ncu_profile.py <kernel.cu> [--b-scat N] [--elem-tile N] [--blocks 256]

# Parse results
sudo ncu --import report_name.ncu-rep --page raw 2>&1 | uv run python tools/ncu_parse.py

# Key metrics to watch:
#   gpu__time_duration.sum                           -- kernel time
#   lts__d_atomic_input_cycles_active.avg.pct...     -- L2 atomic pressure
#   sm__pipe_fma_cycles_active.avg.pct...            -- FMA utilization
#   l1tex__t_output_wavefronts_pipe_lsu_mem_local... -- register spills
#   sm__inst_executed.avg.per_cycle_elapsed           -- IPC
```
