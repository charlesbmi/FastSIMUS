# CUDA Kernel Optimization Progress

Target: 10M scatterers/second on RTX A4000 (100K scat in 10ms)
Hardware: NVIDIA RTX A4000, 48 SMs, clocks locked 1560 MHz

## Benchmark Config

P4-2v transducer, 100K scatterers, N_FREQ=854, N_ELEM=64, N_SUB=1, N_ES=64

## Architecture Timeline

| Version | Architecture                   | Best Config       | Time (ms) | Scat/s    | Speedup   | Notes                             |
| ------- | ------------------------------ | ----------------- | --------- | --------- | --------- | --------------------------------- |
| v2-v5   | Two-pass variants              | -                 | >300ms    | <0.5M     | <1x       | L2 cache thrashing from TX buffer |
| v6      | Fused persistent + register TX | 192 blk           | 68.9      | 1.45M     | 1.00x     | **Baseline**                      |
| v7      | Multi-scat accumulation        | -                 | ~113      | 0.88M     | 0.61x     | Geometry recomputation overhead   |
| v8      | Two-pass TX+RX                 | -                 | ~270      | 0.37M     | 0.26x     | DRAM bandwidth bound              |
| v9      | Batched scat (register TX)     | B=2, 192 blk      | 45.6      | 2.19M     | 1.51x     | Register spills at B>2            |
| v10     | Batched scat (shmem TX)        | B=4, 192 blk      | 43.1      | 2.32M     | 1.60x     | Eliminates register pressure      |
| v11     | Batched + element tiling       | B=4, ET=8, 192    | 35.1      | 2.85M     | 1.97x     | ILP from elem tiling              |
| **v11** | **Best config found**          | **B=5, ET=8, 192**| **32.5**  | **3.08M** | **2.12x** | **+8% from B_SCAT sweep**        |

## Experiment Results Summary

Full experiment docs in `docs/progress/experiments/`.

| # | Experiment                            | Result    | Key Finding                                   |
| - | ------------------------------------- | --------- | --------------------------------------------- |
| 1 | v6 baseline ncu profile               | Reference | L2 atomic bound (51%), 42% occupancy wasted   |
| 2 | v11 B=2 (higher occupancy)            | -10%      | 46% L2 atomic pressure kills the gain         |
| 3 | v11 + __launch_bounds__(128,4)        | **-200%** | Shmem is occupancy limiter, not regs; 6.7x spills |
| 4 | v11 + shmem padding + SFU precompute  | +3%       | SFU 21%->13%, but LSU 37%->50%; bank pad N/A  |
| 5 | v11 B_SCAT sweep (B=2,4,5,6,8)       | **+8%**   | B=5 optimal; 2->1 block/SM cliff at B=6       |
| 6 | v12 per-block output buffers          | **-200%** | Output >> L1; R-M-W slower than atomicAdd     |
| 7 | v13 freq-chunked multi-launch         | -49%      | Geometry recomputed per chunk dominates        |
| 8a| v14 two-pass Metal-style SIMD RX      | **-392%** | 79x more atomics than v11 (wrong thread map)  |
| 8b| v14 two-pass tiled shmem output       | -43%      | Geometry per tile + shmem overhead > savings   |
| 9a| v15 fp16 TX buffer                    | **+2.5%** | 31.6ms; register wall blocks higher B_SCAT    |
| 9b| v11-stagger element group order       | 0%        | Atomic contention from volume, not collisions  |
| 10| Combined analysis + GPU roadmap       | Analysis  | A4000 near-optimal; 4090 projected 7-9ms      |
| 11| Block count sweep (48-384)            | 0%        | Wave quantization matters; L2 contention disproved |
| 12| v16 Grid-Y element group partitioning | **-261%** | 8x redundant Phase 1+2 dominates atomic savings |
| 13| L2 persistence hints                  | 0%        | Output already L2-resident; no effect          |
| 14| v17 SCAT_REDUCE warp shuffle          | **DNF**   | Register spills from dynamic loop; catastrophic |

## Current Champion: v11 B=5 ET=8

### ncu Profile (B=5)

| Metric                  | B=5 (champion) | B=4 (previous) | v6 (baseline) |
| ----------------------- | -------------- | -------------- | ------------- |
| Time (real)             | **32.5ms**     | 35.1ms         | 68.9ms        |
| Registers/thread        | 228            | 168            | 96            |
| Shmem/block             | 42.9 KB        | 35.3 KB        | 3.1 KB        |
| Blocks/SM               | 2              | 2              | 5             |
| Achieved occupancy      | 16.7%          | 16.7%          | 41.7%         |
| SM throughput           | **57.1%**      | 53.7%          | 27.2%         |
| FMA pipe                | **42.5%**      | 39.7%          | 10.5%         |
| LSU pipe                | 39.2%          | 37.4%          | 22.1%         |
| SFU pipe                | 22.5%          | 20.9%          | 8.5%          |
| L2 atomic pressure      | **21.7%**      | 25.2%          | 50.6%         |
| Atomic requests         | **69.1M**      | 86.4M          | 345.6M        |
| IPC                     | **2.28**       | 2.15           | 1.09          |
| Wait stall              | **0.68**       | 0.90           | -             |

## Gap to Target

Current: 32.5ms (3.08M scat/s)
Target: 10ms (10M scat/s)
Gap: **3.25x improvement needed**

Theoretical compute minimum (100% FMA utilization): ~4.6ms

## Key Insights from Experiments

### 1. Occupancy is NOT the primary lever

v6 has 42% occupancy but is 2x slower than v11 at 17%. The dominant factor is
L2 atomic throughput. Reducing atomics (via batching) beats increasing occupancy.

### 2. The 2-block/SM cliff is absolute

Performance drops 45% when going from 2 to 1 block/SM (B=5 vs B=6). All
future optimizations MUST maintain >= 2 blocks/SM.

### 3. Register reduction backfires when shmem is the limiter

__launch_bounds__ reduced registers from 168 to 128 but spills exploded 6.7x
and occupancy didn't improve (shmem was the binding constraint). Register
optimization is only useful if paired with shmem reduction.

### 4. SFU -> LSU trade-off is nearly neutral

Precomputing trig values saves SFU cycles but adds shmem reads. Net: ~3%.
The kernel is equally bottlenecked on compute and memory, so shifting load
between them yields diminishing returns.

### 5. Bank conflict source is unknown

N_ES_PAD=65 padding did not reduce the 6.3M bank conflicts. The source is
not the geometry array stride alignment. May require source-level profiling
(ncu source counters) to identify.

## Optimization Strategy (Updated after Exp 6-10)

### On A4000: Near-optimal, diminishing returns

All high-confidence optimizations have been tested. Remaining gains are marginal:
- fp16 TX buffer: +2.5% (31.6ms) -- accuracy trade-off (max 2.7% magnitude error)
- SFU precomputation: +3% (~31.5ms)
- Combined: ~30.5ms (3.28M scat/s) -- still 3x from target

The M4 Max achieving ~3.0M scat/s at ~17 TFLOPS confirms this is a compute
efficiency ceiling, not a CUDA-specific issue. Both platforms operate at ~9x
theoretical minimum.

### On RTX 4090: Primary target for 10M scat/s

The 72 MB L2 cache changes the game:
1. **v11 B=5 ET=8 baseline**: projected 7-9ms (validates scaling hypothesis)
2. **B_SCAT sweep 5-12**: register spills acceptable when L2 atomic pressure drops
3. **Two-pass TX/RX re-test**: unified L2 residency for output; TX buffer partially cached
4. **fp16 TX with high B_SCAT**: register spill cost offset by atomic throughput gain

### Disproven approaches (do NOT retry on A4000)

- Per-block/per-SM output buffers (Exp 6): output >> L1, always loses to hardware atomicAdd
- Frequency chunking (Exp 7): geometry recomputation per chunk dominates
- Metal-style SIMD shuffle (Exp 8a): thread mapping creates MORE atomics on CUDA
- launch_bounds register reduction (Exp 3): shmem binds occupancy, not registers
- Element group staggering (Exp 9b): no measurable effect on L2 contention
- Block count reduction (Exp 11): 96 vs 192 identical; L2 contention not a factor at B=5
- Grid-Y element group partitioning (Exp 12): 8x redundant geo+TX dominates atomic savings
- L2 persistence hints (Exp 13): output already L2-resident; zero effect
- SCAT_REDUCE warp shuffle (Exp 14): dynamic scatterer loop prevents unrolling -> catastrophic spills
