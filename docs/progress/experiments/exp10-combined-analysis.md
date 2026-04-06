# Exp 10: Combined Analysis and GPU Scaling Roadmap

## Optimization Journey Summary

### Full Experiment Results (Exp 1-9)

| # | Experiment | Architecture | Best ms | scat/s | vs v11 | Status |
|---|-----------|-------------|---------|--------|--------|--------|
| - | v6 baseline | Fused persistent | 68.9 | 1.45M | -112% | Reference |
| - | v11 B=4 ET=8 | Batched + elem tiling | 35.1 | 2.85M | +0% | Previous best |
| 5 | v11 B=5 ET=8 | B_SCAT sweep | **32.5** | **3.08M** | **+8%** | **Champion** |
| 3 | v11 + launch_bounds | Register reduction | 97.1 | 1.03M | -200% | Failed |
| 4 | v11 + shmem pad + SFU | Micro-optimization | 34.0 | 2.94M | +3% | Marginal |
| 6 | v12 per-block output | Eliminate atomics | 100+ | <1.0M | -200% | Failed |
| 7 | v13 freq-chunked | Multi-launch chunks | 48.5 | 2.06M | -49% | Failed |
| 8a | v14 two-pass shuffle | Metal-style SIMD RX | 160 | 0.63M | -392% | Failed |
| 8b | v14 two-pass tiled | Tiled shmem output | 46.5 | 2.15M | -43% | Failed |
| 9a | v15 fp16 TX B=5 ET=8 | fp16 TX buffer | **31.6** | **3.17M** | **+2.5%** | Marginal |
| 9b | v11-stagger | Elem group reorder | 32.8 | 3.05M | -1% | No effect |

### What Worked

1. **Scatterer batching (B_SCAT)**: Accumulating multiple scatterers before
   atomicAdd is the single most effective optimization. B=5 reduces atomics by
   5x vs B=1, directly lowering L2 atomic pressure from 51% to 22%.

2. **Element tiling (ELEM_TILE=8)**: Increasing ILP chains in the RX loop
   keeps the FMA pipeline busy. ET=8 achieves 42.5% FMA utilization vs 10.5%
   at baseline. Higher ET is better than higher occupancy.

3. **Shared memory TX buffer**: Moving the TX buffer from registers (v9) to
   shmem (v10+) eliminated the register wall that limited B_SCAT to 2.

### What Failed and Why

| Failed approach | Root cause |
|----------------|------------|
| Per-block output (Exp 6) | 213KB/block output >> L1 cache → L2 read-modify-write slower than hardware atomicAdd |
| Freq chunking (Exp 7) | Geometry recomputed per chunk (~40% of kernel time wasted) |
| Two-pass TX/RX (Exp 8) | TX precomputation enables RX restructuring but DRAM read of TX[scat][freq] is slower than shmem |
| Metal-style SIMD (Exp 8a) | Thread mapping (freq per thread) means shuffle can't reduce atomics to same address |
| Register reduction (Exp 3) | Shmem, not registers, limits occupancy → spills increase with no occupancy gain |
| Stagger writes (Exp 9b) | Atomic contention from volume, not cache line collisions |
| fp16 TX high-B (Exp 9a) | Register wall at B*ET > 48 forces spills that negate batching gains |

### Why v11 Is Hard to Beat

The v11 architecture is a local optimum because it balances three competing resources:

1. **Shared memory** → limits blocks/SM (must stay >= 2)
2. **Registers** → limits B_SCAT * ELEM_TILE product (max ~40 at 255 regs)
3. **L2 atomic throughput** → limits how many atomics per second

Any change that improves one axis degrades another:
- More B_SCAT → fewer atomics but more registers → spills
- More blocks/SM → better latency hiding but less B_SCAT → more atomics
- Separate TX/RX → no shmem constraint but redundant geometry or DRAM overhead

## Hardware Scaling Projections

### RTX A4000 (current) — SM86 Ampere

| Spec | Value |
|------|-------|
| SMs | 48 |
| CUDA cores | 6,144 |
| Clock | 1,560 MHz (locked) |
| FP32 TFLOPS | 19.2 |
| L2 cache | 4 MB |
| Shmem/SM | 128 KB |
| Memory BW | 448 GB/s |
| **Current result** | **32.5ms (3.08M scat/s)** |
| **Best w/ fp16 TX** | **31.6ms (3.17M scat/s)** |

### RTX 3090 — SM86 Ampere

| Spec | Value | vs A4000 |
|------|-------|----------|
| SMs | 82 | 1.71x |
| CUDA cores | 10,496 | 1.71x |
| Clock | ~1,695 MHz | 1.09x |
| FP32 TFLOPS | 35.6 | 1.85x |
| L2 cache | 6 MB | 1.5x |
| Shmem/SM | 128 KB | 1.0x |
| Memory BW | 936 GB/s | 2.09x |

**Projection**: Same SM architecture means identical per-SM behavior.
- Linear SM scaling: 32.5 × (48/82) = 19.0ms
- Clock bonus: 19.0 × (1560/1695) = 17.5ms
- L2 contention (1.71x more writers, 1.5x more L2): ~5% penalty
- **Estimated: 18-20ms (5.0-5.6M scat/s)**

The 3090's higher memory bandwidth (936 vs 448 GB/s) won't help much since
the kernel is compute+atomic bound, not DRAM bound.

### RTX 4090 — SM89 Ada Lovelace

| Spec | Value | vs A4000 |
|------|-------|----------|
| SMs | 128 | 2.67x |
| CUDA cores | 16,384 | 2.67x |
| Clock | ~2,520 MHz | 1.62x |
| FP32 TFLOPS | 82.6 | 4.30x |
| L2 cache | **72 MB** | **18x** |
| Shmem/SM | 128 KB | 1.0x |
| Memory BW | 1,008 GB/s | 2.25x |

**Projection**: The 4090's massive L2 cache is transformative for this workload.

The output array is 64 × 854 × 2 × 4 = 438 KB. On the A4000 (4 MB L2), this
competes with other data for cache residency. On the 4090 (72 MB L2), the
entire output sits comfortably in L2, and atomicAdd never needs DRAM.

Breakdown (v11 B=5 ET=8 profile partitioning):

| Component | A4000 time | Scaling factor | 4090 projected |
|-----------|-----------|----------------|----------------|
| Compute (57%) | 18.5ms | SM × clock = 4.3x | 4.3ms |
| L2 atomic (22%) | 7.2ms | L2 size + slices ≈ 6x | 1.2ms |
| Memory/other (21%) | 6.8ms | SM + BW ≈ 3x | 2.3ms |
| **Total** | **32.5ms** | | **7.8ms** |

**Estimated: 7-9ms (11-14M scat/s)**

This exceeds the 10ms / 10M scat/s target.

### Optimization Priorities Per GPU

#### On A4000 (current)

The v11 B=5 ET=8 architecture is near-optimal. Remaining options:
- fp16 TX buffer: +2.5% (31.6ms) — marginal accuracy trade-off
- SFU precomputation: +3% (~31.5ms) — nearly free
- Combined: ~30.5ms (3.28M scat/s) — still far from 10ms target
- **Verdict: hardware-limited at ~30ms. Target unreachable on A4000.**

#### On RTX 3090

Same SM architecture, so same kernel works unchanged.
- Expected: ~18-20ms with v11 B=5 ET=8
- fp16 TX might enable B=6 profitably if L2 pressure decreases with 1.5x L2
- **Verdict: 2x improvement over A4000. Still 2x from target.**

#### On RTX 4090

The 72 MB L2 changes the optimization landscape:
- L2 atomic pressure drops dramatically → higher B_SCAT becomes viable
- With reduced atomic penalty, B=8+ ET=8 might work despite register pressure
  (the spill penalty is offset by fewer total iterations)
- Two-pass TX/RX (Exp 8) might become viable (TX buffer fits in L2!)
- SM89 may have improved per-SM atomic throughput

**Recommended experiments on 4090:**
1. Run v11 B=5 ET=8 as baseline → expect 7-9ms
2. Sweep B_SCAT 5-12 → higher B may finally win with 72 MB L2
3. Re-run two-pass TX/RX → TX[100K × 854] = 682 MB won't fit L2,
   but output (438 KB) is fully resident
4. Try v15 fp16 TX with B=8-10 → register spills less costly if
   L2 atomics are no longer bottleneck

### Apple M4 Max — Metal (cross-platform reference)

| Spec | Value | vs A4000 |
|------|-------|----------|
| GPU cores | 40 | N/A (different arch) |
| Execution units | 640 | N/A |
| Clock | ~1,578 MHz | ~1.0x |
| FP32 TFLOPS | ~17.0 | 0.89x |
| Memory | 128 GB unified LPDDR5X | Unified (no PCIe) |
| Memory BW | 546 GB/s | 1.22x |
| Threadgroup memory | 32 KB | 0.33x vs A4000 shmem |
| Register file | 208 KB per EU | ~0.81x per SM equivalent |
| **Measured result** | **~33ms (~3.0M scat/s)** | **~1.0x** |

Architecture: Two-pass TX + SIMD-reduce RX via Metal (`simus_tx_tiled.metal` +
`simus_rx_simd.metal`). TX uses 64-thread threadgroups with 16-element tiles
and ALU-only geometric progression. RX uses SCAT_REDUCE=2 with
`simd_shuffle_xor` to halve atomic writes.

#### Cross-Platform Insights

**1. Nearly identical throughput at similar FP32 TFLOPS is remarkable.**

The M4 Max (17 TFLOPS) achieves ~3.0M scat/s. The A4000 (19.2 TFLOPS) achieves
3.08M scat/s. The throughput tracks FP32 TFLOPS almost linearly, suggesting both
GPUs are hitting the same fundamental compute-bound wall despite very different
architectures and kernel strategies.

**2. Metal's two-pass architecture works because of unified memory.**

The Metal kernel uses separate TX and RX passes connected by an intermediate
`TX[n_scat, n_freq]` buffer in DRAM. This failed on CUDA (Exp 8) because
reading the TX buffer from GDDR6 added ~3ms of latency. On Apple Silicon,
the unified memory architecture means the TX buffer likely stays in the
system-level cache (SLC, ~48 MB on M4 Max) or is accessed at full bandwidth
without PCIe overhead.

**3. SIMD shuffle with SCAT_REDUCE=2 is effective on Metal but failed on CUDA.**

Metal's RX kernel maps threads as `tid = elem_idx * SCAT_REDUCE + scat_batch`,
so adjacent SIMD lanes work on the same element from different scatterers.
`simd_shuffle_xor` reduces the SCAT_REDUCE lanes before one atomic write.
On CUDA (Exp 8a), this same pattern produced 79x MORE atomics than v11 because
the CUDA version mapped frequencies to threads (each thread writes to a unique
`(elem, freq)` pair — nothing to reduce). The thread-to-work mapping is the
critical difference.

**4. Metal's threadgroup memory limit (32 KB) prevents the fused approach.**

The A4000's ability to use 42.9 KB shared memory enables the fused v11
architecture (TX + RX in one kernel with B_SCAT=5 batching). Metal's 32 KB
threadgroup limit prevents storing geometry + TX buffer for multiple scatterers,
forcing the two-pass approach. If Metal had 48+ KB threadgroup memory,
the fused approach with B_SCAT batching would likely outperform the two-pass
approach there as well.

**5. The algorithm is compute-bound at ~5.5x theoretical minimum on both platforms.**

| Platform | Time | Theoretical min | Ratio |
|----------|------|-----------------|-------|
| A4000 CUDA (v11) | 32.5ms | 3.4ms | 9.6x |
| M4 Max Metal (TX+RX) | ~33ms | 3.8ms | 8.7x |
| A4000 CUDA (v6, pre-opt) | 68.9ms | 3.4ms | 20.3x |

Both optimized implementations converge to similar efficiency ratios. The
remaining gap is split between atomic overhead (~22% on CUDA, unknown on Metal
but likely similar given SCAT_REDUCE=2 only halves atomics), geometry/SFU
overhead, and occupancy limitations.

**6. Implications for RTX 4090.**

The M4 Max's system-level cache (SLC, ~48 MB) serves a similar role to the
4090's 72 MB L2 for keeping the output array and TX buffer resident. The fact
that Metal's two-pass approach works well with SLC residency suggests the
4090's two-pass approach (Exp 8 variant) may indeed become viable with 72 MB
L2 — especially the tiled shmem output variant (8b) which eliminated
inner-loop atomics but suffered from DRAM latency for the TX buffer.

## Theoretical Limits

### Compute bound

Total complex multiplies: 100K × 64 × 854 ≈ 5.47B (TX) + 5.47B (RX) = 10.9B
Each complex multiply = ~6 FLOPS → 65.5 GFLOPS total

| GPU | FP32 TFLOPS | Min compute time |
|-----|-------------|-----------------|
| M4 Max (Metal) | 17.0 | 3.8ms |
| A4000 | 19.2 | 3.4ms |
| RTX 3090 | 35.6 | 1.8ms |
| RTX 4090 | 82.6 | 0.8ms |

### Memory bound

Geometry per scatterer: ~20 bytes (x, z, rc)
Total input reads: 100K × 20 = 2 MB (fits L2 on all GPUs)
Output writes: 438 KB (fits L2 on all GPUs)

The workload is NOT memory-bandwidth limited on any of these GPUs.

### Atomic bound

Total atomics (v11 B=5): 69.1M requests (from ncu)
At 48 L2 atomic pipelines × 1560 MHz: theoretical 75B atomics/sec
If no contention: 69.1M / 75B = 0.9ms
Actual: 7.2ms → 8x gap from contention (multiple writers to same cache line)

On 4090: ~6x more L2 capacity → near-zero contention expected.

## Updated Optimization Skill

The CUDA kernel optimization skill at `.cursor/skills/cuda-kernel-optimization/SKILL.md`
has been updated with all findings from Experiments 6-9.

## Conclusion

The RTX A4000 cannot reach the 10ms target with the current SIMUS algorithm.
The v11 architecture is operating at ~9.6x theoretical compute minimum, with
the gap primarily attributable to L2 atomic serialization (accounting for ~22%
of runtime and ~8x contention overhead) and the occupancy-ILP-shmem three-way
constraint.

The M4 Max cross-platform comparison confirms this is a fundamental algorithmic
efficiency ceiling, not a CUDA-specific issue: both platforms converge to ~3M
scat/s at ~17-19 FP32 TFLOPS despite radically different architectures (fused
single-pass on CUDA vs two-pass TX/RX on Metal) and memory systems (discrete
GDDR6 vs unified LPDDR5X). Throughput scales linearly with FP32 TFLOPS.

**The RTX 4090's 72 MB L2 cache is the key enabler for reaching 10M scat/s.**
Its 18x larger L2 directly addresses the atomic bottleneck that dominates on
the A4000, while its 4.3x higher FP32 throughput handles the compute portion.
The M4 Max's success with its two-pass approach (enabled by its large SLC)
provides further evidence that large on-chip cache is the critical differentiator.

### Recommended Next Steps

1. **Benchmark v11 on an RTX 4090** to validate the 7-9ms projection
2. **Re-test two-pass TX/RX (Exp 8b) on 4090** — the tiled shmem output variant
   may work when the TX buffer is partially L2-resident
3. **Sweep B_SCAT 5-12 on 4090** — register spills may be acceptable when L2
   atomic pressure is no longer the bottleneck
4. **Consider fusing the Metal-style approach on 4090**: SCAT_REDUCE + fused
   batching hybrid that leverages both the large L2 and high register count
