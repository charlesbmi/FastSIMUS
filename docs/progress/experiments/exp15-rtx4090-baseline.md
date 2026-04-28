# Exp 15: RTX 4090 Baseline Validation

## Setup

- GPU: RTX 4090 (128 SMs, sm_89, 72MB L2, 82.6 TFLOPS FP32)
- Clocks: Locked at 2520 MHz
- Kernel: v11 B=5 ET=8 (A4000 champion), blocks=256 (2x128 SMs)

## Changes

- `cuda_runtime.py`: Default arch `sm_86` -> `sm_89`
- `test_fused_v11.py`: NUM_SMS 48 -> 128, block configs updated

## Results

### v11 B=5 ET=8 Timing

| Metric | A4000 | RTX 4090 | Ratio |
|--------|-------|----------|-------|
| Best time | 32.5ms | 8.5ms | **3.82x** |
| Throughput | 3.08M scat/s | 11.76M scat/s | 3.82x |

**Target: 10M scat/s -- EXCEEDED.**

### ncu Profile (v11 B=5 ET=8)

| Metric | A4000 | RTX 4090 | Analysis |
|--------|-------|----------|----------|
| SM throughput | 57.1% | 48.0% | Lower utilization |
| FMA pipe | 42.5% | 36.5% | Under-utilized |
| L2 hit rate | - | 100% | Output fully L2-resident |
| DRAM throughput | - | 0.02% | Zero DRAM traffic |
| **L2 atomic pressure** | **21.7%** | **47.2%** | **New bottleneck** |
| IPC | 2.28 | 1.92 | -16% from atomic stalls |
| Achieved occupancy | 16.7% | 16.7% | Same (shmem-limited) |

Key finding: L2 atomic pressure doubled because 128 SMs generate 2.67x more
atomic traffic on the same output cache lines. The 72MB L2 gives 100% hit rate
but contention on L2 slices is the bottleneck.

### B_SCAT Sweep

| B_SCAT | Time | Throughput | Regs | Local | Blocks/SM |
|--------|------|-----------|------|-------|-----------|
| **5** | **8.5ms** | **11.7M** | 230 | 64B | **2** |
| 6 | 9.4ms | 10.6M | 254 | 64B | 1 |
| 7 | 10.4ms | 9.7M | 255 | 72B | 1 |
| 8 | 12.2ms | 8.2M | 255 | 224B | 1 |
| 10 | 17.4ms | 5.8M | 255 | 480B | 1 |
| 12 | DNF | - | - | - | shmem > max |

B>=6 drops to 1 block/SM. At B=6, shmem=52.5KB; two blocks need 105KB > 100KB
max shmem/SM on Ada Lovelace. The 2-block/SM cliff from A4000 persists.

## Conclusion

v11 B=5 ET=8 achieves 8.5ms (11.76M scat/s) on RTX 4090, exceeding the 10M
target. L2 atomic pressure at 47% is the primary bottleneck. Higher B_SCAT is
blocked by shmem limits -- motivates fp16 TX (v15).
