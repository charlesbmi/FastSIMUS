# Exp 7: Frequency-Chunked Multi-Launch

**Beads**: FastSIMUS-xj6
**Hypothesis**: Split frequencies into chunks to reduce shmem, enabling higher B_SCAT.
**Result**: FAILED -- geometry recomputation per chunk dominates.

## Results

| Config | Chunks | B_SCAT | ET | Regs | Shmem | Blk/SM | Per-chunk | Total | vs v11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v13 | 2 | 8 | 8 | 255 | 41 KB | 2 | 29.5ms | 59.1ms | -82% |
| v13 | 2 | 8 | 4 | 168 | 41 KB | 2 | 25.3ms | 50.6ms | -56% |
| v13 | 2 | 8 | 2 | 96 | 41 KB | 2 | 28.1ms | 56.2ms | -73% |
| v13 | 2 | 10 | 8 | 255 | 52 KB | 1 | 44.8ms | 89.7ms | -176% |
| v13 | 3 | 8 | 8 | 255 | 33 KB | 2 | 24.9ms | 74.4ms | -129% |
| v13 | 4 | 10 | 8 | 255 | 35 KB | 2 | 26.6ms | 106.3ms | -227% |
| v11 ref | 1 | 5 | 8 | 228 | 43 KB | 2 | 32.5ms | 32.5ms | baseline |

## Analysis

### Geometry is too expensive to repeat

Each frequency chunk re-runs the full geometry computation (Phase 1) for all
100K scatterers. Geometry includes: distance, obliquity, sinc, amplitude,
phase init, attenuation init -- ~20 FLOPs per (scatterer, sub-element).

Estimated geometry fraction: ~40% of per-chunk time (based on per-chunk time
being ~80% of full-freq time despite half the frequency work).

### Best case: 25.3ms per chunk (2 chunks, B=8 ET=4)

Each chunk processes 427 frequencies with B_SCAT=8 (1.6x more batching than
v11's B=5). Per-chunk is 22% faster than v11. But 2 chunks = 50.6ms total.
The atomic reduction (fewer per chunk) doesn't offset the repeated geometry.

### Register pressure with high B_SCAT and high ET

B=8 ET=8 hits 255 registers with 240B local mem (spills). B=8 ET=4 uses 168
registers cleanly. B=10 ET=8 spills 464B and 1 block/SM at 2 chunks.

## Key Insight

**Frequency chunking only helps when geometry is a small fraction of total work.**
For FastSIMUS, geometry + TX setup is ~40%, making multi-launch infeasible.

To make chunking work, geometry would need to be cached in global memory between
launches (175 MB for 100K scatterers -- impractical).

## Alternative

The two-pass approach (Exp 8) solves this differently: TX is computed ONCE into
global memory, then RX reads from it. No geometry recomputation.
