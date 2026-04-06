# Experiment 2: v11 B_SCAT=2 ET=8 (Higher Occupancy)

## Hypothesis

Halving B_SCAT from 4 to 2 reduces shared memory from 35KB to 18KB per block,
allowing 4 blocks/SM instead of 2. The 2x higher occupancy should improve
latency hiding, potentially offsetting the 2x more atomics.

## Configuration

- Kernel: `simus_fused_v11.cu` with B_SCAT=2, ELEM_TILE=8
- Grid: 192 blocks x 128 threads
- Shared memory: 18.0 KB/block
- Registers/thread: 128

## ncu Profile Summary

| Metric                  | v11 B=2    | v11 B=4    | v6 baseline |
| ----------------------- | ---------- | ---------- | ----------- |
| Kernel time (real)      | ~39.3ms    | ~35.1ms    | ~68.9ms     |
| Registers/thread        | **128**    | 168        | 96          |
| Shmem/block             | **18.0 KB**| 35.3 KB    | 3.1 KB      |
| Achieved occupancy      | **33.3%**  | 16.7%      | 41.7%       |
| SM throughput           | 51.5%      | **53.7%**  | 27.2%       |
| FMA pipe                | 38.0%      | **39.7%**  | 10.5%       |
| LSU pipe                | **40.6%**  | 37.4%      | 22.1%       |
| SFU pipe                | 19.1%      | **20.9%**  | 8.5%        |
| L2 hit rate             | 99.99%     | 99.95%     | 98.8%       |
| L2 atomic pressure      | **46.0%**  | 25.2%      | 50.6%       |
| Atomic requests         | 172.8M     | ~86.4M     | 345.6M      |
| Shmem bank conflicts    | 3.2M       | 6.3M       | 0           |
| IPC                     | 2.05       | **2.15**   | 1.09        |

## Warp Stall Analysis

| Stall              | v11 B=2 | v11 B=4 | Meaning                         |
| ------------------ | ------- | ------- | ------------------------------- |
| not_selected       | **1.13**| 0.42    | 2.7x more eligible-but-waiting  |
| barrier            | **1.06**| 0.48    | 2.2x more sync stalls           |
| long_scoreboard    | **0.96**| 0.16    | 6x more L2/global mem waits     |
| mio_throttle       | **0.86**| 0.05    | 17x more memory I/O throttling  |
| wait               | 0.79   | **0.90**| Less general waiting            |
| lg_throttle        | **0.22**| 0.00    | L2 write queue full             |

## Analysis

### Occupancy doubled but performance worsened

Despite 2x occupancy (33% vs 17%), B=2 is ~12% slower than B=4. The problem:
more warps means more atomic contention.

### L2 atomic pressure is the killer

L2 atomic pressure jumped from 25% to **46%** -- nearly double. With 2x more
warps issuing 2x more atomics (172M vs 86M requests), the L2 atomic unit is
nearly saturated. The `long_scoreboard` stall at 0.96 (vs 0.16) confirms
warps are blocked waiting for atomic results from L2.

### mio_throttle is new and severe

At 0.86 (vs 0.05 for B=4), the memory I/O queue is frequently full. This
means the LSU pipe (40.6%) is bottlenecked not by computation but by L2's
ability to service requests. `lg_throttle` (0.22 vs 0.00) confirms L2 write
queues are backing up.

### not_selected stall indicates scheduling pressure

With 4 blocks/SM, there are 16 warps competing for 4 warp schedulers. The
`not_selected` stall at 1.13 (vs 0.42) means eligible warps wait longer.

## Verdict

**B=4 is superior to B=2 for this kernel.** The atomic reduction factor
(batching more scatterers before atomicAdd) matters more than occupancy.
The 2x occupancy gain is entirely consumed by 2x more L2 atomic pressure.

### Key Insight

For this kernel, **occupancy is not the primary lever**. The v6 baseline has
42% occupancy but is 2x slower than v11 at 17% occupancy. The dominant factor
is L2 atomic throughput, and reducing atomic operations (via batching or
warp shuffle) is more valuable than increasing occupancy.

### Implications for Next Experiments

1. **Warp shuffle** (exp5) is the highest-priority optimization -- it reduces
   atomics without changing occupancy
2. **launch_bounds** (exp3) should be tested with B=4, not B=2
3. Any approach that increases atomics (higher occupancy, smaller batch) will
   lose to approaches that decrease atomics
