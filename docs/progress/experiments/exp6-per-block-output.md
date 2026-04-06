# Exp 6: Per-Block Private Output Buffers

**Beads**: FastSIMUS-nib
**Hypothesis**: Eliminate ALL atomics by giving each block a private output buffer.
**Result**: FAILED -- 3-5x slower than v11

## Configuration

Kernel: v12 (v11 + per-block output, no atomics)
Same shmem layout as v11, identical TX/RX computation.
Only change: `atomicAdd` replaced with regular `+=` to per-block output region.
Separate reduction kernel sums per-block outputs at the end.

## Results

| Config | Per-block Output | Total Output | Time (ms) | vs v11 B=5 |
| --- | --- | --- | --- | --- |
| 96 blocks | 213 KB | 20 MB (> L2) | 105.0 | -3.2x |
| 48 blocks | 213 KB | 10 MB (< L2) | 179.0 | -5.5x |
| v11 B=5 (ref) | N/A (atomics) | 213 KB | 32.5 | baseline |

Reduction kernel: 0.1ms (negligible).

## Root Cause Analysis

### Why per-block output is slower than atomicAdd

1. **Output doesn't fit in L1**: Per-block output = N_ELEM x N_FREQ x 2 x 4 = 213 KB.
   L1 data cache = 48 KB. Every read-modify-write hits L2, not L1.

2. **L2 atomicAdd has hardware optimization**: The L2 atomic unit pipelines
   read-modify-write operations efficiently. Regular R+M+W through L2 has
   separate read and write transactions with higher overhead.

3. **Shared output is more cache-friendly**: v11's 213 KB shared output stays
   hot in L2 because ALL blocks access the same data. v12's per-block outputs
   spread across 10-20 MB, causing L2 contention.

4. **Fewer blocks = less parallelism**: 48 blocks = 1 block/SM. Not enough
   warps to hide the memory latency of the read-modify-write pattern.

## Key Insight

**Per-block output only wins when the output fits in L1 cache.** For FastSIMUS,
the output (N_ELEM * N_FREQ * 8 = 438 KB per complex pair) is 9x too large
for L1. This approach would require frequency chunking to reduce per-launch
output to ~27-54 KB.

The atomicAdd hardware in L2 is specifically optimized for scatter-gather patterns.
Regular memory access is NOT faster for large scattered outputs.

## Implication for Future Experiments

To eliminate atomics, we need either:
- **Two-pass separation**: TX to global, RX reads TX with no cross-block write contention
- **Output tiling + freq chunking**: Reduce per-launch output to fit in L1
- **SIMD reduction**: Reduce atomic frequency via warp shuffles (not elimination)
