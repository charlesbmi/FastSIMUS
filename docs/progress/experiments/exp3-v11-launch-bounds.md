# Experiment 3: v11 + __launch_bounds__(128, 4)

**Status:** DONE

## Hypothesis

Adding `__launch_bounds__(128, 4)` forces the compiler to reduce register usage
from 168 to <=128 per thread. With 128 regs, the register-based occupancy limit
rises from 2 to 4 blocks/SM, potentially doubling occupancy and improving
latency hiding.

## Configuration

- Kernel: `simus_fused_v11_lb.cu` (v11 with `__launch_bounds__(128, 4)`)
- Grid: 192 blocks x 128 threads
- B_SCAT=4, ELEM_TILE=8, N_SCAT=100,000

## Result: Occupancy Did NOT Improve

The compiler reduced registers from 168 to 128, but occupancy stayed at 2
blocks/SM because **shared memory is the binding constraint**, not registers:

- Shmem/block = 35.3 KB
- 3 blocks x 35.3 KB = 106 KB > 100 KB available
- Therefore max 2 blocks/SM regardless of register count

The register reduction was pointless. Worse, the compiler spilled massively to
hit the 128-register target.

## ncu Profile Comparison

| Metric                  | v11_lb (this) | v11 B=4    | Delta       |
| ----------------------- | ------------- | ---------- | ----------- |
| Registers/thread        | **128**       | 168        | -40 regs    |
| Local memory/thread     | **232B**      | 64B        | +168B (3.6x)|
| Achieved occupancy      | 16.7%         | 16.7%      | same        |
| SM throughput           | 43.8%         | **53.7%**  | -18%        |
| FMA pipe                | 28.0%         | **39.7%**  | -29%        |
| LSU pipe                | **43.8%**     | 37.4%      | +17% (spills)|
| SFU pipe                | 14.7%         | **20.9%**  | -30%        |
| L2 atomic pressure      | 17.7%         | **25.2%**  | -29%        |
| Shmem bank conflicts    | 7.7M          | 6.3M       | +22%        |
| IPC                     | 1.58          | **2.15**   | -26%        |
| Kernel time (ncu)       | ~105ms        | ~35ms      | **3x slower**|

### Spill Comparison

| Spill metric       | v11_lb (this) | v11 B=4 | Ratio  |
| ------------------- | ------------- | ------- | ------ |
| Local mem loads     | 234.3M        | 27.2M   | **8.6x** |
| Local mem stores    | 131.8M        | 27.6M   | **4.8x** |
| Total spills        | 366.1M        | 54.8M   | **6.7x** |

## Warp Stall Comparison

| Stall              | v11_lb  | v11 B=4 | Interpretation                       |
| ------------------ | ------- | ------- | ------------------------------------ |
| long_scoreboard    | **1.40**| 0.16    | 8.8x -- spill loads stalling warps   |
| wait               | 0.84   | **0.90**| Similar                              |
| barrier            | **0.60**| 0.48    | Slightly worse                       |
| not_selected       | 0.34   | **0.42**| Slightly better (fewer eligible)     |
| short_scoreboard   | **0.28**| 0.24    | Slightly worse                       |
| dispatch_stall     | **0.19**| --      | New stall from instruction pressure  |
| mio_throttle       | **0.07**| 0.05    | LSU queue starting to fill           |
| lg_throttle        | **0.06**| 0.00    | L2 write queue pressure from spills  |

The dominant stall changed from `wait` (0.90) to `long_scoreboard` (1.40).
Long scoreboard means warps stall waiting for L1/L2 results -- exactly what
happens when registers spill to local memory (L1 cache backed by L2).

## Verdict

**Negative result.** `__launch_bounds__(128, 4)` made the kernel **3x slower**.

### Why It Failed

1. **Wrong bottleneck targeted.** Shared memory (35.3 KB/block) limits occupancy
   to 2 blocks/SM. Reducing registers from 168 to 128 changed nothing about the
   occupancy bound.

2. **Spills destroyed performance.** To fit in 128 registers, the compiler
   spilled 6.7x more data through L1 cache. This introduced `long_scoreboard`
   as the top stall (1.40) and shifted the kernel from compute-bound (FMA 40%)
   to memory-bound (LSU 44%).

3. **IPC collapsed.** From 2.15 to 1.58 (-26%) because warps now spend their
   time waiting for spill loads instead of executing FMA.

### Key Insight

`__launch_bounds__` only helps when register pressure is the **binding**
occupancy constraint. For v11 B=4, shared memory binds first. To increase
occupancy beyond 2 blocks/SM, shared memory per block must drop below ~33 KB
(100 KB / 3 blocks). That requires reducing B_SCAT (exp2 already showed B=2
gives 4 blocks but loses on atomics) or reducing the shmem layout.

## Files Changed

- `src/fast_simus/kernels/simus_fused_v11_lb.cu` -- v11 with `__launch_bounds__(128, 4)`
- `/tmp/exp3_v11_lb.ncu-rep` -- ncu profile report
