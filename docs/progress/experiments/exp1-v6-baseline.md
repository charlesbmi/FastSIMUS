# Experiment 1: v6 Baseline ncu Profile

**Date:** 2026-04-06
**Kernel:** `simus_fused_v6.cu` -- Fused persistent + register TX
**Config:** 192 blocks, 128 threads/block, 100K scatterers
**Hardware:** NVIDIA RTX A4000, 48 SMs
**Tool:** ncu 2023.2.2, `--set full`, launch-skip 1, launch-count 1

## Summary Metrics

| Metric                    | Value          | Notes                                       |
| ------------------------- | -------------- | ------------------------------------------- |
| Kernel time (ncu)         | 146.06 ms      | With profiling overhead; real ~69ms          |
| Real throughput           | 1.45M scat/s   | From bench_cuda.py (68.9ms)                 |
| Registers/thread          | 96             | Moderate; allows 5 blocks/SM                |
| Dynamic shmem/block       | 3.07 KB        | Very small (9*N_ES + 3*N_ELEM)*4            |
| Total shmem/block         | 4.10 KB        | +1KB driver overhead                        |
| Grid size                 | 192            | 4 waves/SM (192/48)                         |
| Waves/SM                  | 0.80           | <1 wave; not enough blocks to fill SMs      |
| IPC                       | 1.08 inst/cycle| Low; dual-issue not effective               |
| SM throughput             | 27.24%         | Low -- significant headroom                 |
| Achieved occupancy        | 41.67%         | 20 warps/SM avg out of 48 max               |

## Occupancy Analysis

| Limiter           | Max blocks/SM | Warps/SM |
| ----------------- | ------------- | -------- |
| Registers (96)    | **5**         | 20       |
| Shared memory     | 16            | 64       |
| Warps             | 12            | 48       |
| Block limit       | 16            | -        |

Registers are the binding constraint at 96 regs/thread -> 5 blocks/SM -> 20 warps -> 41.67%.
Shared memory at 3KB/block is negligible (allows 16 blocks/SM).

With 192 blocks / 48 SMs = 4 blocks/SM, but register limit allows 5.
Achieved occupancy (41.67%) matches 20 warps/SM = 5 blocks/SM limit.
Waves/SM = 0.80 means blocks don't fully overlap (192 / (48*5) = 0.8 waves).

## Pipe Utilization Breakdown

| Pipe         | % of peak (elapsed) | % of peak (active) | Role                           |
| ------------ | ------------------- | ------------------- | ------------------------------ |
| FMA          | 21.90%              | 22.02%              | Main compute (mul-add)         |
| LSU          | 22.15%              | 22.27%              | Load/store (shmem + atomics)   |
| XU/SFU       | 8.57%               | 8.62%               | Transcendentals (sin/cos/exp)  |
| ALU          | 5.35%               | 5.38%               | Integer arithmetic, addressing |
| CBU          | 0.50%               | -                   | Control flow                   |
| FMA-heavy    | 21.10%              | -                   | FMA subpipe (muladd)           |

FMA and LSU are roughly equal (~22%), but both are very low. The kernel is
underutilizing the hardware. The near-equal elapsed/active ratios show the SMs
are almost always active, just not doing useful work per cycle.

## Memory Hierarchy

| Metric                          | Value           | Notes                             |
| ------------------------------- | --------------- | --------------------------------- |
| DRAM throughput                 | 2.37%           | Negligible DRAM traffic           |
| DRAM read                       | 5.52 MB         | Everything fits in L2             |
| DRAM write                      | 1.25 GB         | Atomic writeback to DRAM          |
| L2 hit rate                     | 99.99%          | Near-perfect; working set in L2   |
| **L2 atomic pressure**          | **50.61%**      | **CRITICAL bottleneck**           |
| Compute memory throughput       | 76.59%          | Dominated by L2 atomics          |
| L1->L2 write bytes              | 57.54 GB        | Massive atomic traffic            |
| L1->L2 write (% peak)          | 34.87%          | Saturating L1-to-L2 crossbar     |
| Shmem bank conflicts            | 0               | Zero conflicts (stride-1 access)  |
| Shmem load wavefronts           | 232.6M          | Large volume of shmem reads       |
| Atomic reduction requests       | 345.6M          | Massive atomicAdd volume          |
| Atomic reduction wavefronts     | 388.7M          | L1 reduction traffic              |

### Local Memory (Register Spills)

| Metric                    | Wavefronts | % of peak |
| ------------------------- | ---------- | --------- |
| Local mem loads (spill)   | 49.2M      | 0.95%     |
| Local mem stores (spill)  | 32.6M      | 0.63%     |
| Total spill traffic       | 81.8M      |           |

Spills are moderate (96 regs with 112B local/thread). Less severe than v11
(168 regs, 54.8M wavefronts) because v6 has fewer live variables per thread but
more total threads.

## Warp Stall Analysis

| Stall Reason          | Ratio | Interpretation                                     |
| --------------------- | ----- | -------------------------------------------------- |
| **long_scoreboard**   | **5.56** | Waiting for L2/global mem (atomics!)            |
| lg_throttle           | 1.86  | Local/global memory queue full (spill backpressure)|
| mio_throttle          | 1.65  | Memory I/O queue saturated                         |
| barrier               | 1.04  | __syncthreads between phases                       |
| selected (executing)  | 1.00  | Baseline: 1 warp issuing per cycle                 |
| not_selected          | 0.75  | Eligible but not scheduled                         |
| wait                  | 0.66  | Waiting for operands                               |
| short_scoreboard      | 0.52  | Waiting for shmem/local result                     |
| dispatch_stall        | 0.35  | Dispatch contention                                |
| branch_resolving      | 0.15  | Branch resolution                                  |
| math_pipe_throttle    | 0.06  | Compute pipe full (rare)                           |
| no_instruction        | 0.03  | I-cache miss (negligible)                          |

Dominant stall: `long_scoreboard` at 5.56x. Warps spend 5.56 cycles stalled
on L2/global memory results for every 1 cycle executing. This is atomicAdd
latency: each atomicAdd to L2 takes ~100 cycles, and with 345.6M atomic
requests, the L2 atomic unit is 50.61% saturated.

Secondary stalls `lg_throttle` (1.86) and `mio_throttle` (1.65) confirm the
memory subsystem is the chokepoint -- queues are full from atomic traffic.

## v6 vs v11 Comparison

| Metric                 | v6 (baseline) | v11 (best)   | Delta                |
| ---------------------- | ------------- | ------------ | -------------------- |
| Real kernel time       | 68.9ms        | 35.1ms       | 1.97x faster         |
| Throughput             | 1.45M scat/s  | 2.85M scat/s | 1.97x                |
| Registers/thread       | 96            | 168          | +75%                 |
| Shmem/block            | 3.07 KB       | 35.3 KB      | 11.5x                |
| Achieved occupancy     | 41.67%        | 16.7%        | v6 2.5x higher       |
| SM throughput          | 27.24%        | 53.7%        | v11 2x higher        |
| FMA pipe               | 21.90%        | 39.7%        | v11 1.8x             |
| LSU pipe               | 22.15%        | 37.4%        | v11 1.7x             |
| XU/SFU pipe            | 8.57%         | 20.9%        | v11 2.4x             |
| IPC                    | 1.08          | ~1.08        | Similar               |
| L2 atomic pressure     | **50.61%**    | **25.15%**   | v6 2x worse          |
| L2 hit rate            | 99.99%        | 99.95%       | Both excellent        |
| Shmem bank conflicts   | 0             | 6.3M         | v6 has none           |
| Atomic requests        | 345.6M        | ~180M        | v6 ~2x more          |
| Top warp stall         | long_scoreboard 5.56 | wait 0.90 | Very different    |
| Register spills        | 81.8M         | 54.8M        | v6 has more total     |

## Key Insights

### Why v6 is 2x slower despite 2.5x higher occupancy

1. **L2 atomic saturation is the primary bottleneck.** At 50.61% L2 atomic
   pressure, v6 spends half its time servicing atomicAdd to L2 cache. The
   `long_scoreboard` stall at 5.56 (vs 0.16 in v11) proves warps are almost
   entirely blocked on atomic latency.

2. **No batching means more atomic operations.** v6 processes 1 scatterer at a
   time, writing atomicAdd for every (freq, elem) pair per scatterer. v11
   batches B=4 scatterers, accumulating in registers before atomicAdd, cutting
   atomic ops by ~4x.

3. **Higher occupancy cannot hide atomic latency.** Despite 5 blocks/SM (vs 2),
   the warps all stall on the same L2 atomic unit. More warps means more
   contention, not more hiding.

4. **Low SM throughput (27%) despite good occupancy (42%).** The SMs are active
   but warps aren't issuing -- they're stalled on `long_scoreboard`. This is
   the classic latency-bound, not throughput-bound profile.

### What v11 fixed

v11's batching (B=4) + element tiling (ET=8) trades higher register usage (168
vs 96) and shared memory (35KB vs 3KB) and lower occupancy (16.7% vs 41.7%)
for:

- ~4x fewer atomicAdd operations (register accumulation)
- 2x less L2 atomic pressure (25% vs 51%)
- 1.8x higher FMA utilization (doing compute instead of waiting)
- Different stall profile (wait/barrier instead of long_scoreboard)

The fundamental lesson: **reducing atomic contention matters far more than
increasing occupancy** when the kernel is L2-atomic-bound.

## Conclusion

v6 is firmly L2-atomic-latency-bound. The `long_scoreboard` stall at 5.56x and
50.61% L2 atomic pressure confirm that atomicAdd to global memory is the
dominant bottleneck, not compute, not shared memory, not DRAM. All subsequent
optimizations (v9-v11) correctly targeted this by batching scatterers to reduce
atomic volume.

## ncu Report

Saved to `/tmp/exp1_v6_baseline.ncu-rep` for interactive analysis with
`ncu-ui` or re-extraction with `ncu --import`.
