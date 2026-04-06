# Exp 8: Two-Pass TX/RX with L2-Tiled Scatterer Access

**Beads**: FastSIMUS-b95
**Hypothesis**: Separate TX/RX, tile output in shmem to eliminate inner-loop atomics.
**Result**: SLOWER than v11 across all configurations.

## Results

### TX Kernel (v14): Consistently 8.8-9.0ms

Excellent standalone performance: 94 regs, 5 blocks/SM, 2.5 KB shmem.
This proves TX can be computed efficiently when isolated.

### RX Kernel Variants

**Variant A: Metal-style per-frequency atomicAdd (simus_rx_v14.cu)**

| SCAT_REDUCE | Blocks | Regs | RX Time | Total | Issue |
| --- | --- | --- | --- | --- | --- |
| 2 | 50,000 | 32 | 151.5ms | 160.5ms | 5.47B atomics (79x more than v11) |

Root cause: each (scat, elem, freq) triple writes one atomic. Even with SR=2,
100K/2 × 64 × 854 × 2 = 5.47B atomics vs v11's 69M. The Metal approach works
on Metal because SIMD groups are larger and L2 atomics have different characteristics.

**Variant B: Tiled shmem output (simus_rx_tiled_v14.cu)**

| FREQ_TILE | BPT | Blocks/SM | RX Time | Total | vs v11 |
| --- | --- | --- | --- | --- | --- |
| 32 | 96 | 5 | 37.7ms | 46.5ms | -43% |
| 64 | 96 | 3 | 56.9ms | 65.7ms | -102% |
| 128 | 96 | 1 | 155.3ms | 164.0ms | -405% |

Root cause: geometry + SFU overhead is repeated per tile. With 27 tiles (FT=32),
geometry is computed 27x per scatterer vs 1x in v11. Also, shmem R-M-W accumulation
is slower than register accumulation in v11.

## Key Findings

### 1. TX buffer DRAM overhead is acceptable but adds ~3ms

TX buffer: 652 MB for 100K scatterers. Write: 1.5ms. Read: varies by access pattern.
On 4090 (72 MB L2), the TX buffer would partially fit in L2.

### 2. Metal's SIMD RX pattern doesn't translate to CUDA

Metal's simd_shuffle_xor reduces atomics per output element. But on CUDA,
the total atomic COUNT increases because each (scat, elem) pair writes
all N_FREQ atomics. In v11, B_SCAT batching accumulates frequencies in
registers before writing.

### 3. Shmem-tiled output eliminates inner-loop atomics but adds overhead

The output tile (64 × FREQ_TILE × 8 bytes) fits in shmem. Zero atomics during
scatterer processing. But geometry recomputation per tile (27x for FT=32) and
shmem R-M-W latency outweigh the atomic savings.

### 4. The blocks/SM cliff dominates

FT=128: 1 block/SM → 155ms. FT=32: 5 blocks/SM → 37ms. 4x difference for
4x occupancy reduction.

## Architecture-Level Conclusion

The v11 fused approach is near-optimal for the A4000 because:
1. **Shmem TX buffer eliminates DRAM traffic** -- two-pass adds 3ms+
2. **Register accumulation is faster than shmem accumulation**
3. **Geometric progression avoids SFU in inner loop** -- any approach that breaks
   frequency-sequential processing loses this
4. **B_SCAT batching reduces atomics efficiently** -- the 5x reduction is near-optimal
   given shmem constraints

## Hardware-Dependent Assessment

| Hardware | TX in L2? | Two-pass viable? | Reason |
| --- | --- | --- | --- |
| A4000 (16MB L2) | No (652MB >> 16MB) | No | DRAM overhead + geometry recomputation |
| 3090 (6MB L2) | No | No | Same issues, slightly more SMs |
| 4090 (72MB L2) | Partially (~11%) | Maybe | Needs scatterer tiling for L2 reuse |
| 4090 + fp16 TX | Partially (~22%) | Possible | 326MB buffer, more L2 friendly |

The two-pass architecture becomes viable when L2 can hold a meaningful fraction of
the TX buffer, AND there are enough SMs to offset the geometry overhead per tile.
