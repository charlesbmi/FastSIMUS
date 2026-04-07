# Exp 11: Block Count Reduction Sweep

**Hypothesis**: Reducing block count below 192 (current champion) reduces concurrent
atomic writers competing for L2 cache lines, improving L2 throughput and overall
kernel time.

**Result**: L2 contention is **not the bottleneck**. Performance is dominated by
**wave quantization** -- block counts must be exact multiples of 96 (2 active
blocks/SM * 48 SMs) to avoid tail effects.

## Setup

- GPU: RTX A4000 (48 SMs, Ampere GA104), clocks locked at 1560 MHz
- Kernel: `simus_fused_v11.cu` (batched accum + shmem TX + element tiling)
- Config: B_SCAT=5, ELEM_TILE=8, TG_SIZE=128, N_SCAT=100,000, N_FREQ=854, N_ELEM=64
- Shared memory: 43,888 bytes (42.9 KB) per block
- Registers/thread: 228, Local mem: 64B, Max blocks/SM: 2
- 3 runs per configuration, median reported

## Results

| Blocks | Blk/SM | Waves | Run 1 (ms) | Run 2 (ms) | Run 3 (ms) | Median (ms) | Scat/s | vs 96 |
|-------:|-------:|------:|-----------:|-----------:|-----------:|------------:|-------:|------:|
| 48 | 1.0 | 0.5 | 46.9 | 46.9 | 47.1 | **46.9** | 2.13M | -44% |
| **96** | **2.0** | **1.0** | **32.5** | **32.5** | **32.7** | **32.5** | **3.08M** | **0%** |
| 128 | 2.7 | 1.33 | 48.4 | 48.2 | 48.2 | **48.2** | 2.07M | -48% |
| **192** | **4.0** | **2.0** | **32.5** | **32.5** | **32.5** | **32.5** | **3.08M** | **0%** |
| 256 | 5.3 | 2.67 | 36.5 | 36.5 | 36.5 | **36.5** | 2.74M | -12% |
| **384** | **8.0** | **4.0** | **32.6** | **32.6** | **32.7** | **32.6** | **3.07M** | **0%** |

"Waves" = blocks / 96, where 96 = max_active_blocks = 2 blocks/SM * 48 SMs.

## Analysis

### Wave quantization is the dominant effect

The results form a clear pattern based on whether the block count is an integer
multiple of 96:

- **Exact multiples (96, 192, 384)**: 32.5--32.6 ms, essentially identical
- **Non-multiples (48, 128, 256)**: 36.5--48.2 ms, proportional to tail waste

This is textbook wave quantization. With max_blocks/SM = 2 and 48 SMs, the GPU
can execute exactly 96 blocks concurrently. A persistent kernel with grid-stride
loops assigns equal work to every block, so the last "partial wave" leaves SMs
idle while the remaining blocks finish.

### Why 128 blocks is the worst

128 blocks = 1.33 waves. The partial wave has only 32 blocks running on 48 SMs
(at most 1 block/SM on 32 SMs, 16 SMs fully idle). This wastes 67% of the SM
capacity in the tail. Measured overhead: 48%, which is slightly less than the
theoretical 33% tail waste because the partial wave also suffers reduced latency
hiding (1 vs 2 blocks/SM).

### Why 48 blocks underperforms despite no tail

48 = exactly 1 block/SM across all 48 SMs -- no tail effect. But it only
achieves half the occupancy (128 threads/SM vs 256 threads/SM with 2 blocks).
With 228 registers/thread and a compute-heavy inner loop, the second concurrent
block provides meaningful latency hiding. Measured: 46.9 ms vs 32.5 ms = 1.44x
slowdown, confirming that 2 blocks/SM is needed for full throughput.

### L2 contention: not a factor

The original hypothesis was that fewer blocks would reduce L2 atomic contention.
The data disproves this: 96 blocks (half the atomic writers of 192) and 192
blocks produce identical timing at 32.5 ms. Reducing from 192 to 96 concurrent
atomic warps has zero measurable L2 benefit.

This is consistent with Exp 10 analysis: at B_SCAT=5, atomic pressure is already
low enough (~22% of L2 bandwidth) that it is not the bottleneck.

### Optimal block count

96 blocks is the minimum optimal config -- same performance as 192 with half the
blocks. However, since there is zero performance difference between 96, 192, and
384, the choice is cosmetic. Using 96 blocks is slightly preferable because:
1. Same performance as 192
2. Half the blocks means scheduler manages fewer CTAs
3. Each block does 2x more iterations, better amortizing block setup overhead

## Key Takeaway

**Block count must be a multiple of `max_blocks/SM * n_SMs`.** For this kernel on
A4000: multiples of 96. Any deviation incurs tail effects up to 48%.

This also means the kernel scales well across GPUs -- as long as the block count
is tuned to `2 * n_SMs` for the target GPU.

## Recommendation

Update the default block count from 192 to `2 * n_SMs` (auto-detected), or keep
192 since it works for A4000 (and happens to also be 2*96). No code change
needed for the kernel itself.
