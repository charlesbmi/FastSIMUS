# Exp 12: Grid-Y Element Group Partitioning

**Hypothesis**: Using `gridDim.y` to distribute element groups across blocks
(instead of iterating all groups per block) reduces L2 atomic contention by 8x.
Each block writes to only 1/8th of the output, reducing atomics per output entry
from ~19,968 to ~2,496.

**Result**: **-261% regression** (117ms vs 32.5ms). The 8x redundant geometry and
TX computation completely dominates the atomic savings.

## Design

v16 modifies v11 so that:
- Grid: `(blocks_x, N_ELEM_GROUPS=8, 1)` instead of `(blocks, 1, 1)`
- Phase 1+2 (geometry + TX): identical, computed by every block
- Phase 3 (RX): each block processes ONLY `blockIdx.y`-th element group

This means 8x more blocks execute Phase 1+2 for the same scatterer data,
but each block writes atomics to only 8*854*2 output entries instead of 64*854*2.

## Results

| Config | Grid | Total blocks | Time (ms) | Scat/s | vs v11 |
|--------|------|-------------|-----------|--------|--------|
| v11 B=5 ET=8 | (192, 1, 1) | 192 | **32.5** | **3.08M** | baseline |
| v16 blocks_x=6 | (6, 8, 1) | 48 | 151.5 | 0.66M | -366% |
| v16 blocks_x=12 | (12, 8, 1) | 96 | **117.4** | **0.85M** | **-261%** |
| v16 blocks_x=24 | (24, 8, 1) | 192 | 117.2 | 0.85M | -261% |

All configs: 225 regs/thread, 64B local mem, 2 max blocks/SM.

## Analysis

### Phase 1+2 dominance

From ncu profiling of v11, Phase 1+2 (geometry + TX sweep) accounts for ~60%
of kernel time (~19.5ms). Phase 3 (RX + atomics) accounts for ~40% (~13ms).

In v16, each scatterer batch requires 8 blocks to compute Phase 1+2 (one per
element group) instead of 1. With 96 concurrent blocks (max on A4000), the
scatterer iteration count per block changes:
- v11: 100K / (192 * 5) = 104 batches per block
- v16: 100K / (12 * 5) = 1,667 batches per block (since only blocks_x participates)

Each batch still requires full Phase 1+2, so the total Phase 1+2 work is 8x.

### Wave quantization

- blocks_x=6 (48 total): only 48/96 = 0.5 waves active, 50% SM idle
- blocks_x=12 (96 total): 1.0 wave, all SMs active
- blocks_x=24 (192 total): 2.0 waves, no tail effect, same speed as 12

### Register savings

v16 uses slightly fewer registers (225 vs 228) because the compiler can
optimize the single-element-group Phase 3. Not meaningful.

## Key Insight

The fundamental problem with Grid-Y partitioning is that **Phase 1+2 and
Phase 3 share the same scatterer iteration loop**. You cannot distribute
Phase 3 work across Y-dimension blocks without also duplicating Phase 1+2.

For this approach to work, Phase 1+2 (geometry + TX) must be separated from
Phase 3 (RX) -- which is the two-pass architecture (Exp 8). The two-pass
approach failed on A4000 due to TX buffer DRAM latency, but on RTX 4090
with 72 MB L2 cache, it would enable Grid-Y RX without redundant TX compute.

## Relevance to RTX 4090

On the 4090, a hybrid approach could work:
1. **Pass 1**: Compute TX[scat][freq] to global memory (fits in 72 MB L2)
2. **Pass 2**: Grid-Y RX kernel with `(blocks_x, N_ELEM_GROUPS, 1)`,
   each block reads TX from L2 and writes to 1/8th of output

This eliminates both the redundant Phase 1+2 AND reduces atomic contention.
The A4000's 4 MB L2 makes this impossible, but the 4090's 72 MB L2 enables it.
