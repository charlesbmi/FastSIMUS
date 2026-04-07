# Exp 14: SCAT_REDUCE Warp-Cooperative Scatterer Reduction

**Hypothesis**: Using `__shfl_xor_sync` to reduce across SCAT_REDUCE adjacent
threads handling different scatterers for the same frequency/element reduces
atomicAdd operations by SCAT_REDUCE factor while maintaining geometric progression.

**Result**: **Catastrophic failure.** Kernel does not complete even with 1K
scatterers (vs 100K target). Register spills from dynamic scatterer loop
prevent any useful performance.

## Design

v17 modifies v11's Phase 3 RX loop:

1. Thread mapping: `scat_lane = tid % SCAT_REDUCE`, `freq_lane = tid / SCAT_REDUCE`
2. Each thread handles `ceil(B_SCAT/SCAT_REDUCE)` scatterers with geometric progression
3. Frequency stride becomes `TG_SIZE/SCAT_REDUCE` (64 vs 128 for SR=2)
4. Per-frequency: accumulate across assigned scatterers, shuffle-reduce, lane 0 atomicAdd
5. Geometric progression step recomputed for FREQ_STRIDE

## Results

| Config | Regs | Local mem | Max blk/SM | 1K scat timing | Status |
|--------|------|-----------|-----------|----------------|--------|
| B=4 ET=8 SR=2 (MY_B_SCAT=2) | 96 | 336B | 2 | >90s (killed) | Failed |
| B=2 ET=4 SR=2 (MY_B_SCAT=1) | 80 | 128B | 5 | >60s (killed) | Failed |

For reference, v11 B=5 ET=8 completes 100K scatterers in 32.5ms.

## Root Cause Analysis

### 1. Dynamic loop prevents unrolling

In v11, the scatterer loop `for (int si = 0; si < B_SCAT; si++)` is a
compile-time constant loop that the compiler fully unrolls. All B_SCAT * ELEM_TILE
geometric progression states (cv/sv) are kept in registers (228 regs total).

In v17, the scatterer loop becomes `for (int si = scat_lane; si < actual_b; si += SCAT_REDUCE)`
where `actual_b` is runtime-variable (the batch may be partial). The compiler cannot
unroll this loop, so:
- `rx_cv[MY_B_SCAT * ELEM_TILE]` and `rx_sv[MY_B_SCAT * ELEM_TILE]` arrays spill to local memory
- `my_scat_idx[MY_B_SCAT]` also spills
- Every frequency iteration requires loading/storing spilled values from L1 cache

### 2. Register spills are catastrophic for geometric progression

The geometric progression inner loop (cv = cmul(cv, sv)) requires both cv and sv
in registers simultaneously. When they spill, each frequency iteration becomes:
- Load cv from L1 (2 floats x ET entries)
- Load sv from L1 (2 floats x ET entries)  
- Compute cmul
- Store cv back to L1
- Repeat for each of MY_B_SCAT scatterers

This converts a register-only operation (1 cycle) into a memory-bound operation
(~100 cycles per entry), making the kernel 50-100x slower.

### 3. Shuffle overhead adds per-frequency cost

In v11, the atomicAdd happens once per frequency per element tile. In v17,
the shuffle reduction adds `log2(SCAT_REDUCE)` shuffle operations per frequency
per element. With SR=2 ET=8: 8 shuffles per frequency (2 floats each = 16 shuffles).
This is a net addition with no corresponding reduction in atomics per iteration
(since the thread count per frequency is halved, the total atomics per frequency
are the same -- the benefit only comes from having more scatterers per batch).

### 4. SCAT_REDUCE does NOT reduce total atomics when B_SCAT stays constant

Key insight: with B_SCAT=4 and SR=2, each thread handles 2 scatterers. The
number of atomicAdd operations is unchanged (one per element per frequency per
block per batch). The only way SCAT_REDUCE helps is if it enables HIGHER B_SCAT
(by using fewer registers per thread). But the dynamic loop prevents this.

## Why Metal's SCAT_REDUCE Works

Metal's `simus_rx_simd.metal` uses SCAT_REDUCE=2 successfully because:
1. Two-pass architecture: RX kernel reads precomputed TX from global memory
2. No shared memory for TX buffer -> more threadgroup memory for states
3. Metal's register allocation is different (208 KB register file per EU)
4. SIMD group size is 32 (warp) with explicit shuffle
5. The RX kernel does NOT use shared memory for geometry -> simpler state

The critical difference: Metal's RX kernel has NO Phase 1+2 overhead in registers.
It only needs RX geometric progression states. CUDA's fused kernel carries
geometry + TX state across phases, consuming registers.

## Conclusion

**SCAT_REDUCE is not viable for the fused v11 architecture on CUDA.**

The approach would only work with:
1. A two-pass TX/RX architecture (separate TX kernel eliminates Phase 1+2 register pressure)
2. Compile-time constant scatterer counts (enables unrolling)
3. Sufficient registers for geometric progression states without spills

On RTX 4090 with a two-pass approach, SCAT_REDUCE could work because:
- RX kernel has no TX/geometry register overhead
- 72 MB L2 cache makes TX buffer accessible without DRAM penalty
- Combined with Grid-Y element partitioning, this could dramatically reduce atomics
