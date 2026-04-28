# Exp 18: Architecture for 30M scat/s on RTX 4090

## Goal

2.06x speedup: from 14.55M scat/s (6.88ms) to 30M scat/s (3.33ms).

## Current Champion Deep Profile (v15 B=9 ET=4, RTX 4090 @ 2520 MHz)

### Key Numbers

| Metric | Value | Notes |
|--------|-------|-------|
| Kernel time (ncu) | 7.84 ms | Replay overhead; wall ~6.88ms |
| Grid | 256 blocks x 128 threads | 1 wave on 128 SMs |
| Registers/thread | 192 | Spills: 72 bytes (18 floats) |
| Shmem/block | 47.64 KB | 2 blocks/SM (both reg+shmem limited) |
| SM throughput | 61.2% | |
| FMA pipe | 45.9% (heavy: 41.8%) | |
| ALU pipe | 11.5% | |
| SFU/XU pipe | 23.1% | Headroom for transcendentals |
| LSU pipe | 40.7% | Shared loads + spills |
| IPC | 2.45 | Per cycle elapsed |
| L2 atomic pressure | 32.8% | Primary bottleneck |
| L2 hit rate | 99.99% | Everything in L2 |
| DRAM traffic | 1.84 MB read | Negligible |

### Instruction Breakdown (SMSP totals)

| Category | Warp Instructions | % of Total |
|----------|------------------|-----------|
| Total | 5,456M | 100% |
| Shared loads | 334M | 6.1% |
| Shared stores | 6.8M | 0.1% |
| Local loads (spills) | 31.9M | 0.6% |
| Local stores (spills) | 27.6M | 0.5% |
| Global loads | 12.6M | 0.2% |
| Global reductions (atomicAdd) | 38.4M | 0.7% |
| Branch | 97.4M | 1.8% |

### L2 Traffic (sectors = 32 bytes each)

| Category | Sectors | % of L2 Traffic |
|----------|---------|----------------|
| Total | 309M | 100% |
| Reductions (atomicAdd) | 180.6M | 58.5% |
| Writes (local spills) | 109.6M | 35.5% |
| Reads | 18.9M | 6.1% |

### Warp Stall Analysis

| Stall | Ratio | Root Cause |
|-------|-------|-----------|
| wait | 0.61 | cmul dependency chain (serial cv = cmul(cv, sv)) |
| not_selected | 0.55 | Scheduler has options (2 blocks = 8 warps) |
| dispatch_stall | 0.25 | Instruction cache / decode |
| barrier | 0.19 | __syncthreads between phases |
| short_scoreboard | 0.19 | Shmem load latency |
| long_scoreboard | 0.15 | L2/spill latency |
| math_pipe_throttle | 0.05 | FMA/SFU contention |
| mio_throttle | 0.05 | Memory I/O queue |

### Compute Analysis

| Metric | Value |
|--------|-------|
| FFMA throughput | 2993 thread inst/cycle (sum all SMSPs) |
| FMUL throughput | 3490 thread inst/cycle |
| FADD throughput | 630 thread inst/cycle |
| Total FP32 work | ~176 GFLOP |
| Achieved TFLOPS | 22.4 (at 2.22 GHz ncu clock) |
| Peak TFLOPS | 82.6 (at 2.52 GHz boost) |
| Compute utilization | 27.1% |
| Compute floor (same work) | ~2.1ms (at 100% FMA) |
| Target time | 3.33ms |
| Required utilization | ~64% of peak |

### Key Insight

The compute floor for the current work volume is ~2.1ms. The 3.33ms target
requires ~64% compute utilization. Currently at 27%. The gap comes from:
1. **cmul dependency chain** (`wait` = 0.61): serial phase rotation limits ILP
2. **L2 atomic overhead** (33% of L2 peak): 180M reduction sectors = 58% of L2 traffic
3. **Register spills** (59.5M local load/store): 72 bytes spill per thread
4. **Low occupancy** (16.7%): only 8 warps/SM to hide latency

## Hypotheses (ranked by expected impact)

### H1: fp16 Phase 3 Compute (2x FMA throughput)

**Idea**: RTX 4090 fp16 FMA throughput is 165 TFLOPS (2x fp32 82.6). The Phase 3
inner loop (cmul + accumulation) could run entirely in fp16, with fp32 accumulation.

**Mechanism**:
- TX buffer is already fp16 in shmem
- RX phase rotation (cv, sv) stored as fp16
- Inner cmul uses `__hmul2` / `__hadd2` for 2-wide fp16 operations
- Accumulation in fp32: `acc_re += __half2float(result)` per iteration

**Expected gain**: +40-60% on Phase 3 (which is ~50% of runtime) = +20-30% overall.
With reduced register pressure (fp16 = 2 regs per complex pair vs 4), could increase
B_SCAT further.

**Risk**: Accumulated fp16 error over 854 frequency iterations. Current fp16 TX
buffer gives 2.2e-4 relative error (within budget). Full fp16 compute may exceed
rtol=1e-4 validation threshold.

**Effort**: Medium. Modify Phase 3 inner loop. Need accuracy validation.

### H2: cmul Dependency Chain Pipelining (break serial bottleneck)

**Idea**: The `wait` stall (0.61) comes from the serial cmul chain:
`cv[n+1] = cmul(cv[n], sv)`. Break this into P independent chains.

**Mechanism**:
```
// 2-deep pipeline:
sv2 = cmul(sv, sv);           // double-step constant
cv0 = initial, cv1 = cmul(cv0, sv);
for (freq += 2*stride):
    acc0 += cv0; acc1 += cv1;
    cv0 = cmul(cv0, sv2);     // independent of cv1
    cv1 = cmul(cv1, sv2);     // independent of cv0
```

**Expected gain**: +15-25%. Reduces `wait` stall from 0.61 to ~0.3. IPC from 2.45
to ~3.0+.

**Risk**: 2x register usage for cv/sv arrays. With B=9 ET=4 already at 192 regs,
need to reduce B or ET to fit. Could combine with fp16 to offset.

**Effort**: Low-medium. Modify frequency sweep loop. No algorithmic change.

### H3: Per-SM Private Output (eliminate atomics)

**Idea**: Give each SM exclusive ownership of the output buffer. Eliminates all
180M L2 atomic sectors (58% of L2 traffic).

**Variant A**: 128 blocks (1/SM), 438 KB output each = 56 MB total (fits 72 MB L2).
Read-modify-write to L2 instead of atomicAdd. 1 block/SM = 8.3% occupancy.

**Variant B**: 256 blocks (2/SM), fp16 partial output = 219 KB each = 56 MB.
Reduction kernel afterward sums 256 partial buffers to fp32. Maintains 2 blocks/SM.

**Variant C**: Use cooperative groups to partition output across SMs. Each SM
accumulates its slice in shmem (3.4 KB for ~855 outputs) then writes once.

**Expected gain**: +15-30%. Removes L2 atomic bottleneck. But if compute-bound
(IPC limited by `wait`), the gain may be smaller.

**Risk**: Variant A loses occupancy (known 45% penalty from B=10 experiments).
Variant B adds fp16 accumulation error. Variant C requires cooperative launch.

**Effort**: Medium-high. New kernel architecture for B/C. Simple for A.

### H4: Warp-Cooperative Reduction (fixed SCAT_REDUCE)

**Idea**: v17 SCAT_REDUCE failed because the dynamic scat loop prevented unrolling.
Fix: remap threads so that groups of K threads within a warp process the SAME
(elem, freq) for K different scatterers. Use `__shfl_down_sync` to reduce,
producing 1 atomicAdd per K threads instead of K.

**Mechanism**:
- Thread mapping: `tid = scat_group * N_FREQ_PER_WARP + freq_local`
- Each group of K=4 threads handles 4 scatterers for 1 frequency
- After accumulation: `__shfl_down_sync` reduce → 4x fewer atomics
- L2 atomic pressure: 33% → ~8%

**Expected gain**: +15-25%. Fewer atomics, less `long_scoreboard`.

**Risk**: Completely different thread mapping breaks the existing shmem access
pattern. K must divide warp size (32). Memory coalescing may suffer.

**Effort**: High. Major kernel restructure.

### H5: Spatial Culling / Scatterer Sorting

**Idea**: Not all scatterers contribute meaningfully to all elements. For the
P4-2v probe (64 elements, ~2cm aperture), a scatterer at depth 8cm and x-offset
2cm has negligible contribution to elements on the opposite side.

**Mechanism**:
- Sort scatterers by x-coordinate
- For each element group, binary search to find the contributing scatterer range
- Skip scatterer-element pairs where distance > threshold

**Expected gain**: +50-100% for distributed scatterers (effectively halves N_SCAT
per element). Problem-dependent.

**Risk**: Diverges from PyMUST reference. Accuracy validation needed. Pre-processing
overhead (sorting + range computation). Benefits depend on scatterer distribution.

**Effort**: Medium. Pre-processing on CPU, kernel modifications for variable ranges.

## Recommended Experiment Plan

### Phase 1: Validate compute floor (quick)
Run a stripped-down kernel with atomics removed (write to /dev/null) to measure
pure compute time. This gives the hard floor for optimization.

### Phase 2: fp16 Phase 3 compute (H1, highest expected impact)
Implement fp16 inner loop in a new v19 kernel. Validate accuracy.

### Phase 3: cmul pipelining (H2, orthogonal to H1)
Implement 2-deep and 4-deep pipeline variants. Can combine with H1.

### Phase 4: Per-SM output, Variant B (H3, if still needed)
Only pursue if H1+H2 combined don't reach 30M.

### Phase 5: Profile and iterate
ncu profile after each experiment to verify bottleneck shifts.
