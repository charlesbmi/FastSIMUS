# CUDA Kernel Optimization Progress

Target: 10M scatterers/second (100K scat in 10ms)
Hardware: NVIDIA RTX A4000 (48 SMs, 1560 MHz) -> RTX 4090 (128 SMs, 2520 MHz)

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
| **v11** | **A4000 champion**             | **B=5, ET=8, 192**| **32.5**  | **3.08M** | **2.12x** | **+8% from B_SCAT sweep**        |
| v11     | 4090 baseline (same kernel)    | B=5, ET=8, 256    | 8.5       | 11.76M    | 8.24x     | 3.82x scaling from A4000         |
| v15 | 4090 champion (fp16 TX) | B=9, ET=4, 256 | 6.88 | 14.55M | 10.0x | +23% vs 4090 v11; TARGET MET |
| v22 | 4090 idealized (nochain+ILP)    | B=5, ET=8, 256    | 2.86      | 34.97M    | 24.1x     | NOT correct -- per-element scatter degenerate |
| v23 | 4090 chain-split (correct fp32) | B=5, ET=8, 256    | 8.18      | 12.21M    | 8.55x     | Negative result -- exp20 ILP win bound to chain break |
| v15 | 4090 fp16 champion (prior)          | B=9, ET=4, 256    | 6.69      | 14.96M    | 10.5x     | Eclipsed by v25b on correctness AND speed |
| v25b | 4090 correct champion (prior) | B=7, ET=4, 256 | 5.66 | 17.68M | 12.4x | Eclipsed by v25c at lower ET / higher B |
| **v25c** | **4090 correct champion (fp32 regtx + sv shmem, ET=2)** | **B=9, ET=2, 256** | **5.39** | **18.56M** | **13.1x** | **+5% over v25b; smaller cv chain drops L2 78%->65%** |

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
| 15| RTX 4090 baseline (v11 B=5 ET=8)     | **3.82x** | 8.5ms; L2 atomic 47% is new bottleneck         |
| 16| Two-pass TX/RX on 4090 (v18 Grid-Y)  | **-310%** | 35ms; shmem TX >> L2; geometry same work        |
| 17| v15 fp16 TX B=9 ET=4 on 4090         | **+23%**  | **6.88ms = 14.55M scat/s; TARGET MET**          |
| 18| v21 idealized ceiling (B=5 ET=8)     | **+36%**  | Empirical ceiling 21.2M scat/s; 30M needs fp16 cmul |
| 19| v22 6-blocks/SM + Phase-3 ILP restructure | **+140%** | **v22_nochain_ilp B=5 ET=8: 2.86ms = 34.97M scat/s; exp19 ceiling broken (idealized only)** |
| 20| v23 chain-split / advance-split on correct fp32 (v11 base) | **0% / -7%** | exp20 ILP win does not transfer to correct kernel; cv[B*ET] live across freq blocks register headroom |
| 21| v25b register-resident TX (correct fp32) | **+66% / +18%** | **B=7 ET=4: 5.66 ms = 17.68M scat/s; per-thread tk arrays remove 60 KB shmem; new bottleneck = L2 throughput on local-mem spill** |
| 22| v26 freq-chunk + v25c sv-from-shmem + ET=2 sweep | **+5% / +24%** | **v25c B=9 ET=2: 5.39 ms = 18.56M scat/s; smaller cv chain drops L2 78->65%; v26 chunking shelved (cmul-init overhead > spill savings)** |

## Current correct-kernel champion: v25c register TX + sv-from-shmem (RTX 4090)

v22_nochain_ilp's 34.97 M scat/s is an *idealized-kernel ceiling*, not a
shippable correct result (per-element scatter behaviour is degenerate under
the `nochain` simplification — see exp20 + the v23 plan doc). Standing
correct-kernel champions, ordered by speed:

- **fp32 v25c (regtx + sv-shmem, ET=2)**: B=9 ET=2, 5.39 ms = **18.56 M scat/s**
- fp32 v25b (regtx + unrolled fi):       B=7 ET=4, 5.66 ms = 17.68 M scat/s
- fp16 TX v15:                            B=9 ET=4, 6.69 ms = 14.96 M scat/s
- fp32 v11:                               B=5 ET=4, 7.61 ms = 13.14 M scat/s

v25c is +24 % over the fp16 v15 baseline AND keeps full fp32 precision
(max mag rel err vs v11 = 4.3 × 10⁻⁵). Mechanism (vs v25b):

1. `sv_arr[B*ET]` is no longer cached in regs — `GEO_STP_RX_RE/IM` is
   re-read from shmem inside each `cmul`. Saves 56 floats/thread at
   B=7 ET=4.
2. Cutting `ELEM_TILE` from 4 to 2 shrinks the live `cv[B*ET]` chain
   from 28 to 18 elements (B=9), opening reg headroom for `tk` and
   raising the optimal `B_SCAT` from 7 to 9–10.

NCU @ B=9 ET=2: L2 throughput 64.9 % (was 76.5 %), Compute 63.4 %
(was 61.2 %), occupancy still 16 % (reg-cap = 255). Local mem 520 B
(`tk` mostly fits but still spills enough to dominate L1 traffic).

Next levers (FastSIMUS-9bj is highest leverage):

- **fp16 cv chain (FastSIMUS-9bj)**: pack 2 freqs per `__hfma2`,
  halving cv reg footprint. Should drop spill enough to either fit
  `tk` fully in regs or open 3 blk/SM (need ≤ 168 regs/thread).
- ~~Chunk freqs in Phase 2+3 to shrink live tk window~~: tried in
  v26 (FastSIMUS-frh). Reduces spill (400 B → 240 B) but cmul-chain
  init overhead exceeds the L2 savings. Shelved — see exp22.

## (Idealized) v22_nochain_ilp B=5 ET=8 (RTX 4090)

### Champion headline

- 2.86 ms = **34.97 M scat/s** at 100k scatterers, 256 blocks, boost clocks
- **2.25x** over v15 (6.44 ms / 15.53 M)
- **1.65x** over the exp19 v21_idealized "ceiling" (4.72 ms / 21.17 M)
- 64 regs/thread, 16.8 KB shmem/block, 144 B spills, 5 blocks/SM
- At ≥6 blocks/SM: `v22_nochain_ilp B=3 ET=8` at 3.03 ms / 33.02 M scat/s,
  48 regs, 10.1 KB shmem, 8 blocks/SM

### How the exp19 ceiling broke

Exp19 called 21.17 M a structural ceiling from the serial `cmul` rotation
chain in Phase 3. Exp20 found two deeper levers:

1. **Chain break (v22_nochain):** resynthesize `cv` via `__sincosf(ph_f)`
   per freq instead of carrying `cv = cmul(cv, sv)` live. Got us to 30.5 M
   at 3.28 ms (1.44x idealized). Modest — and `warps_active` did not rise.
2. **Two-stage Phase 3 (v22_nochain_ilp):** explicitly split the inner
   body into (a) batched SFU/TX-load prep across all `si`, then (b) ET
   FMA accumulation. Got us to 34.97 M at 2.86 ms (another 1.15x).

The mechanism is instruction-stream compression, **not** ILP latency
hiding: nochain_ilp executes 12.7% fewer instructions than nochain
(2.18 B vs 2.50 B), 49% fewer than v21_idealized (4.43 B), mostly on the
ALU pipe (redundant index / book-keeping ops nvcc could only remove
once the interleaved `for si -> for et` body was flattened). IPC fell
(2.26 → 2.02) because the remaining stream is denser and more wait-bound,
but the stream is so much shorter that wall time still dropped.

**`warps_active` is a structural 16.5% floor for this workload**, not a
knob that responds to occupancy or ILP. Stop targeting it. The next
lever is `half2` Phase 3 FMAs (projected ~45-55 M) and shmem-layout work
on the TX buffer (`long_scoreboard` has risen to 0.47, becoming the new
top stall).

## Historical Champion: v15 B=9 ET=4 (RTX 4090)

### ncu Profile (v15 B=9 ET=4 on RTX 4090)

| Metric                  | v11 B=5 (A4000) | v11 B=5 (4090)  | v15 B=9 (4090)  |
| ----------------------- | --------------- | --------------- | --------------- |
| Time (real)             | 32.5ms          | 8.5ms           | **6.88ms**      |
| Registers/thread        | 228             | 230             | 192             |
| Shmem/block             | 42.9 KB         | 42.9 KB         | 46.5 KB         |
| Blocks/SM               | 2               | 2               | 2               |
| Achieved occupancy      | 16.7%           | 16.7%           | 16.7%           |
| SM throughput           | 57.1%           | 48.0%           | **61.2%**       |
| FMA pipe                | 42.5%           | 36.5%           | **45.9%**       |
| L2 atomic pressure      | 21.7%           | 47.2%           | **32.8%**       |
| IPC                     | 2.28            | 1.92            | **2.45**        |

## Previous Champion: v11 B=5 ET=8 (A4000)

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

**TARGET MET.**

| Platform | Kernel | Time | Throughput | vs Target |
|----------|--------|------|-----------|-----------|
| A4000 | v11 B=5 ET=8 | 32.5ms | 3.08M scat/s | 0.31x |
| 4090 | v11 B=5 ET=8 | 8.5ms | 11.76M scat/s | 1.18x |
| **4090** | **v15 B=9 ET=4** | **6.88ms** | **14.55M scat/s** | **1.45x** |

Theoretical compute minimum (100% FMA utilization): ~4.6ms (A4000) / ~1.1ms (4090)

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

### On RTX 4090: TARGET ACHIEVED (14.55M scat/s)

The 72 MB L2 cache confirmed key predictions:
1. **v11 B=5 ET=8 baseline**: 8.5ms (projected 7-9ms, validated)
2. **B_SCAT sweep 5-12**: B>=6 still drops to 1 block/SM (shmem limit, not register)
3. **Two-pass TX/RX (Exp 16)**: FAILED, 4x slower -- shmem TX >> L2, geometry same work
4. **fp16 TX B=9 ET=4 (Exp 17)**: **SUCCESS, 6.88ms** -- fp16 enables B=9 with 2 blk/SM

### Idealized ceiling on RTX 4090 (Exp 18): 30M needs algorithmic change

Progressively relaxing shmem/register/atomic constraints while preserving
compute volume (v21_noatom, v21_constfreq, v21_singleelem, v21_idealized):

- Empirical ceiling: **21.17 M scat/s** (4.72 ms) at B=5 ET=8
- Headroom over champion: **1.36x** (champion 15.53 → idealized 21.17)
- 30M target = 3.33 ms is **1.43x beyond the idealized ceiling**

Conclusion: the shmem/occupancy/atomic design space is exhausted. The
`wait` stall (cmul chain) does not move (0.61 → 0.57) even under full
idealization, and `warps_active` stays at 16.5% because the register
cap at 255 holds firm. Reaching 30M requires **fp16 cmul** (halving the
serial chain per thread) or **cmul pipelining** (breaking the serial
dependency). See `exp19-idealized-ceiling.md` for the full ablation.

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
- Two-pass TX/RX on 4090 (Exp 16): shmem TX access >> L2 access; geometry recompute same total work
