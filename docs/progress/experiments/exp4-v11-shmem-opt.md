# Experiment 4: v11 + Shmem Padding + SFU Reduction

## Hypothesis

1. N_ES_PAD=65 stride breaks bank-aligned access patterns, eliminating 6.3M conflicts
2. Precomputing delay trig values removes 2 of 3 __sincosf calls per tile element,
   reducing SFU pressure from 21% to ~7%

## Configuration

- Kernel: `simus_fused_v11_shmem.cu`
- B_SCAT=4, ELEM_TILE=8, 192 blocks x 128 threads
- Shared memory: 35.5 KB/block (vs 35.3 KB baseline)
- Registers/thread: 168 (unchanged)
- Changes: N_ES_PAD=65 stride, precomputed delay arrays in shmem

## Timing Result

**34.1ms (2.93M scat/s)** -- 3% faster than v11 baseline (35.1ms)

## ncu Profile Comparison

| Metric                  | v11_shmem  | v11 B=4    | Delta     |
| ----------------------- | ---------- | ---------- | --------- |
| Kernel time (real)      | 34.1ms     | 35.1ms     | **-2.8%** |
| Registers/thread        | 168        | 168        | same      |
| Shmem/block             | 35.5 KB    | 35.3 KB    | +0.6%     |
| Achieved occupancy      | 16.7%      | 16.7%      | same      |
| SM throughput           | 53.8%      | 53.7%      | same      |
| FMA pipe                | 39.0%      | 39.7%      | -1.8%     |
| **XU/SFU pipe**         | **13.4%**  | 20.9%      | **-36%**  |
| **LSU pipe**            | **49.6%**  | 37.4%      | **+33%**  |
| ALU pipe                | 12.2%      | 11.8%      | +3%       |
| L2 atomic pressure      | 26.0%      | 25.2%      | +3%       |
| Shmem bank conflicts    | 6.5M       | 6.3M       | +3%       |
| Register spills         | 54.8M      | 54.8M      | same      |
| IPC                     | 2.15       | 2.15       | same      |

### Warp Stall Comparison

| Stall              | v11_shmem | v11 B=4 | Delta    |
| ------------------ | --------- | ------- | -------- |
| wait               | 0.78      | 0.90    | **-13%** |
| barrier            | 0.52      | 0.48    | +8%      |
| short_scoreboard   | 0.23      | 0.24    | -4%      |
| mio_throttle       | 0.10      | 0.05    | +100%    |

## Analysis

### SFU Reduction Worked, But Shifted to LSU

Precomputing delay trig values reduced SFU from 20.9% to 13.4% (-36%). But
the precomputed values are now READ from shared memory instead of computed,
shifting load to LSU: 37.4% to 49.6% (+33%). The `wait` stall improved from
0.90 to 0.78, but `mio_throttle` doubled (0.05 to 0.10).

Net: ~3% faster. SFU savings partially offset by additional shmem traffic.

### Bank Conflict Padding Did NOT Help

6.5M conflicts (vs 6.3M baseline). The N_ES_PAD=65 stride did not eliminate
the bank conflicts. The source is NOT the geometry array layout with N_ES=64
stride. The conflicts likely come from TX array access patterns or from the
compiler's pipelined instruction scheduling of multiple shmem reads.

### LSU is Now the Dominant Bottleneck

With SFU reduced to 13%, the kernel is now more clearly **LSU-bound at 50%**.
The dual FMA+LSU bottleneck has shifted toward pure LSU. Further optimization
should target reducing shared memory traffic, not compute.

## Verdict

**Marginal improvement (3%).** The SFU precomputation is a valid micro-optimization
but doesn't change the fundamental bottleneck structure. Bank conflict padding
is a dead end for this kernel -- the conflicts come from a source we can't
identify statically.

### Implications

The kernel is now clearly LSU-bound. Further gains require:
1. Reducing shared memory traffic (fewer reads per scatterer)
2. Reducing atomic operations (higher B_SCAT or output buffering)
3. NOT shifting more work from compute to memory
