# Eliminate Geometric Progression via Direct Frequency Parallelism

**Date**: 2026-03-13
**Status**: Prototype complete, analysis done
**Bead**: FastSIMUS-3y8

## Problem

The current `simus_tx.metal` kernel uses a **geometric progression** to sweep
through frequencies: each thread maintains `float2 cur[N_ES]` and `float2
stp[N_ES]` arrays (1024 bytes for 64-element probes) and advances them via
complex multiplication at each frequency step. This creates:

1. **Serial dependency** across frequencies (frequency f+1 depends on cur after f)
2. **Severe register pressure**: 512 16-bit registers needed vs Apple Silicon
   limit of 256 -- guaranteed 50% register spill to device memory
3. **Low GPU utilization**: ~3% of peak FP32 throughput at small batch sizes

## Background: Apple Silicon Register Architecture

From Alyssa Rosenzweig's reverse-engineering (Asahi Linux project):
- Apple M-series GPUs support up to **256 16-bit registers per thread** (128 float32)
- Beyond 256 registers: compiler spills to device memory (DRAM)
- Our `float2[64] + float2[64]` = 512 16-bit registers = **2x the hardware limit**
- Register spills cause 3-10x slowdown (L1 is only 8KB, spills hit L2/DRAM)

MLX's FFT kernels use a similar strategy -- `metal::fast::cos/sin` for twiddle
factors and threadgroup shared memory for intermediate results, avoiding
per-thread register arrays.

## Approaches Tested

### 1. Geometric Progression (baseline)

Thread per scatterer. Each thread maintains `cur[64]` + `stp[64]` arrays,
advancing via complex multiply (6 flops/elem/freq). ~300K flops per scatterer.

### 2. Direct Per-Frequency (simus_tx_direct.metal)

Thread per (scatterer, frequency) pair. Each thread computes phase directly at
one specific frequency using `metal::fast::` trig. No register arrays at all.
~4-5K flops per thread, ~3.2M flops per scatterer (10x more than progression).

### 3. Shared-Memory Geometry Cache (simus_tx_shared.metal)

One threadgroup per scatterer. Geometry (distance, amplitude, phase slopes) is
computed cooperatively into `threadgroup` shared memory (~1.5KB). Each thread then
handles a stripe of frequencies using only scalar accumulators. ~1.6M flops per
scatterer (5x more than progression, 2x less than direct).

## Benchmark Results

P4-2v probe (64 elements, 1 sub-element), 812 frequencies, Apple Silicon.
Throughput = scatterers per second through TX kernel only.

| n_scat | Progression | Direct | Shared | Best |
|--------|-------------|--------|--------|------|
| 100 | 42K/s | 340K/s | **428K/s** | Shared (10x vs prog) |
| 500 | 207K/s | 532K/s | **1084K/s** | Shared (5.2x vs prog) |
| 1000 | 420K/s | 617K/s | **1362K/s** | Shared (3.2x vs prog) |
| 2000 | 827K/s | 637K/s | **1494K/s** | Shared (1.8x vs prog) |
| 3000 | 1122K/s | 652K/s | **1592K/s** | Shared (1.4x vs prog) |
| 5000 | **1841K/s** | 665K/s | 1673K/s | Progression (1.1x vs shared) |
| 10000 | **1959K/s** | 675K/s | 1739K/s | Progression (1.1x vs shared) |

### Key Observations

1. **Shared-memory kernel dominates for n_scat < ~4500**, up to 10x faster
   than progression at small batch sizes.

2. **Even at the optimal chunk size (5000), shared is only 9% slower** than
   progression -- remarkable given it does ~5x more FLOPs.

3. **Progression only wins at large N** because enough threads (5000+) hide the
   register spill latency. At small N, the spill penalty is catastrophic.

4. **Direct (no shared memory)** is a useful but dominated approach -- shared
   memory eliminates redundant geometry recomputation, giving 2x improvement.

5. **Accuracy**: All approaches match within 0.1-0.2% relative error.
   `metal::fast::cos/sin` is safe since there is no accumulated error.
   Shared kernel is slightly more accurate (0.17% vs 0.19% at 10K).

## FLOP Analysis

Per scatterer (N_ES=64, N_FREQ=812):

| Approach | Init flops | Per-freq flops | Total flops | Register bytes |
|----------|-----------|---------------|-------------|---------------|
| Progression | 2,560 | 384 | 314K | 1,024 (spills) |
| Direct | 0 | 4,992 | 4,054K | 0 |
| Shared | 2,560 (shared) | 1,920 | 1,561K | 0 |

The 5x flop increase (progression -> shared) is compensated by:
- Zero register pressure -> full occupancy (1024 vs 384 threads/threadgroup)
- No register spill traffic (saves ~50% of L2 bandwidth)
- Better latency hiding from higher occupancy

## Production Recommendations

### Option A: Shared kernel as default (simpler)

Replace the progression TX kernel with the shared-memory kernel. Pros:
- Consistent throughput (~1.4-1.7M/s) regardless of chunk size
- No register pressure -> smaller chunks viable (less intermediate memory)
- Better accuracy (no accumulated phase error)
- Only 9% slower at 5000-scatterer chunks

### Option B: Hybrid dispatch (optimal)

Select kernel based on chunk size:
- chunk < 4500: shared-memory kernel
- chunk >= 4500: geometric progression kernel

This gives best-of-both-worlds but adds code complexity.

### Option C: Keep progression, use for research (conservative)

Keep the shared-memory kernel as a prototype for future optimization research.
The progression kernel is already well-tuned for the production chunk size.

### Recommendation

**Option A** (shared as default) is the best trade-off. The 9% regression at
chunk_size=5000 is minor, and the gains at smaller batch sizes (2-10x) are
significant for interactive use cases. The code simplification (no chunk_size
tuning, no register pressure concerns) is also valuable.

## Relationship to Other Beads

- **FastSIMUS-8zp** (da absorption): Already implemented in the progression
  kernel. The shared kernel also absorbs da, but via direct computation at
  each frequency rather than into the geometric progression.
- **FastSIMUS-1yo** (db_thresh -60): Reducing frequencies from 812 to ~730
  would proportionally speed up all three approaches.
- **FastSIMUS-1yk** (multi-frame amortization): The shared-memory approach
  opens new possibilities for caching geometry across frames.
