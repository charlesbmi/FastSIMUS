# Experiment 5: B_SCAT Batching Sweep (Atomics vs Occupancy)

## Hypothesis

Higher B_SCAT means more scatterers accumulated before atomicAdd, reducing L2
atomic pressure. But higher B_SCAT also means more shared memory, which can
drop occupancy. There's a sweet spot where the atomic reduction outweighs
the occupancy loss.

## Timing Sweep

| B_SCAT | Shmem (KB) | Blocks/SM | Regs | Time (ms) | Scat/s   | vs B=4   |
| ------ | ---------- | --------- | ---- | --------- | -------- | -------- |
| 2      | 17.6       | 4         | 128  | 39.3      | 2.55M    | -10%     |
| 4      | 35.3       | 2         | 168  | 35.1      | 2.85M    | baseline |
| **5**  | **42.9**   | **2**     | 228  | **32.5**  | **3.08M**| **+8%**  |
| 6      | 51.3       | 1         | 254  | 47.0      | 2.13M    | -25%     |
| 8      | 68.1       | 1         | 255  | 57.1      | 1.75M    | -39%     |

**B_SCAT=5 is the new best configuration: 32.5ms (3.08M scat/s)**

The cliff at B=6 (2→1 block/SM) confirms that 2 blocks/SM is the minimum
viable occupancy. B=5 is the maximum B_SCAT that fits in 2 blocks/SM.

## ncu Profile: v11 B=5 ET=8

| Metric                  | B=5        | B=4        | Delta     |
| ----------------------- | ---------- | ---------- | --------- |
| Registers/thread        | 228        | 168        | +36%      |
| Shmem/block             | 42.9 KB    | 35.3 KB    | +22%      |
| Achieved occupancy      | 16.7%      | 16.7%      | same      |
| **SM throughput**       | **57.1%**  | 53.7%      | **+6.3%** |
| **FMA pipe**            | **42.5%**  | 39.7%      | **+7.1%** |
| LSU pipe                | 39.2%      | 37.4%      | +4.8%     |
| SFU pipe                | 22.5%      | 20.9%      | +7.7%     |
| **L2 atomic pressure**  | **21.7%**  | 25.2%      | **-14%**  |
| **Atomic requests**     | **69.1M**  | ~86.4M     | **-20%**  |
| Shmem bank conflicts    | 6.4M       | 6.3M       | same      |
| Spills (load+store)     | 58.5M      | 54.8M      | +7%       |
| **IPC**                 | **2.28**   | 2.15       | **+6%**   |

### Warp Stall Comparison

| Stall              | B=5    | B=4    | Delta    |
| ------------------ | ------ | ------ | -------- |
| **wait**           | **0.68** | 0.90 | **-24%** |
| not_selected       | 0.53   | 0.42   | +26%     |
| barrier            | 0.48   | 0.48   | same     |
| short_scoreboard   | 0.18   | 0.24   | -25%     |
| long_scoreboard    | 0.14   | 0.16   | -12%     |

## Analysis

### Why B=5 Wins

1. **20% fewer atomics** (69M vs 86M): L2 atomic pressure dropped from 25% to
   22%. This directly reduces the `wait` stall from 0.90 to 0.68 (-24%).

2. **Same occupancy** (2 blocks/SM): B=5 fits in 42.9 KB, still under the
   50 KB threshold for 2 blocks/SM with 100 KB carveout.

3. **Higher FMA utilization** (42.5% vs 39.7%): With less time waiting for
   atomics, the FMA pipe gets used more efficiently. IPC rose 6% to 2.28.

4. **Register pressure acceptable**: 228 regs (vs 168) with only 7% more spills.
   The compiler handles the 5-scatterer unroll without catastrophic spilling.

### Why B=6+ Loses

B=6 (51.3 KB) exceeds the 2-block/SM threshold, dropping to 1 block/SM (8.3%
occupancy). The 16.7% fewer atomics cannot compensate for halved warp scheduling
capacity. The SM throughput would drop below 30%.

### Key Insight: Occupancy Cliff

The performance landscape has a sharp cliff at the 2→1 block/SM boundary:
- B=5 (2 blocks): 32.5ms
- B=6 (1 block): 47.0ms (45% slower)

This means optimization must stay within 2 blocks/SM. The next speedup requires
reducing shmem PER SCATTERER to fit more batched scatterers in the same 50 KB.

## Verdict

**B_SCAT=5 is the new optimal configuration.** +8% over B=4, reaching 3.08M scat/s.

### Next Steps from This Experiment

1. **Combine B=5 with exp4's SFU optimization** -- apply precomputed delays to
   the B=5 config for another ~3% (projected: ~31.5ms, 3.17M scat/s)
2. **Reduce per-scatterer shmem** to fit B=6 or B=7 in 2 blocks:
   - Current: 7 float arrays × N_ES = 7 × 64 = 448 floats per scatterer
   - If we can reduce to 5 arrays: B=7 needs 5×7×64 + 2×7×854 + 192 = 14,432
     = 57.7 KB → 1 block. Still too much.
   - The TX buffer (2 × B_SCAT × N_FREQ) dominates at B>=5:
     2×5×854 = 8,540 vs 7×5×64 = 2,240 geometry
3. **Compress TX buffer**: Store TX in half precision (fp16) in shmem, cutting
   TX buffer from 8,540 to 4,270 floats. B=7 would then need
   7×7×64 + 7×854 + 192 = 3,136 + 5,978 + 192 = 9,306 × 4 = 37.2 KB → 2 blocks!
