# Architecture Exploration: Eliminating the Atomic Bottleneck

## Root Cause: Why Are There So Many Atomics?

### The Algorithm Structure

The SIMUS algorithm computes:
```
output[elem][freq] = sum_over_scatterers(
    TX[scat][freq] * RX_phasor[scat][elem][freq]
) * probe[freq]
```

Where:
- TX[scat][freq] = sum_over_sub_elements(phasor(scat, sub_elem, freq))
- RX_phasor = geometric_progression(distance, kw_step) per frequency

The output is a reduction over 100K scatterers into 64 x 854 = 54,656 complex values.
**Every scatterer contributes to every output element.** This is the source of all atomics.

### Why the Current Kernel Uses Atomics

The v11 kernel uses a **persistent block** pattern:
- 192 blocks, each processing batches of B_SCAT=5 scatterers in a loop
- Multiple blocks run simultaneously, all accumulating into the SAME output
- Without atomics, their writes would race

The batching (B_SCAT) reduces atomic frequency by accumulating multiple scatterers
before writing, but the fundamental problem remains: **cross-block reduction requires
either atomics or explicit synchronization.**

### Why Atomics Are Expensive Here

With 192 blocks x ~521 batches x 64 elements x 854 freqs x 2 (re/im):
- ~69M atomicAdd operations (B=5)
- Each atomicAdd is a read-modify-write to L2 cache (~100 cycle latency)
- L2 atomic unit is 22% saturated
- Wait stall = 0.68 (warps blocked on atomic results)

The kernel spends roughly **25-30% of its time on atomic overhead** that contributes
zero useful compute.

### What the Metal Kernels Did Differently

The Metal RX kernel (`simus_rx_simd.metal`) uses a fundamentally different approach:
- Thread mapping: `tid = elem_idx * SCAT_REDUCE + scat_batch`
- Adjacent threads handle DIFFERENT scatterers for the SAME element
- SIMD shuffle (`simd_shuffle_xor`) reduces across scatterers within a SIMD group
- Only 1 in SCAT_REDUCE threads writes an atomic (2-16x fewer atomics)

The Metal TX kernel (`simus_tx_tiled.metal`) is fully separate:
- One threadgroup per scatterer, writes TX[scat][freq] to global memory
- NO atomics in the TX pass at all
- Geometric progression inner loop is purely ALU (0 SFU calls)

Key insight from Metal: **separating TX (no atomics) from RX (few atomics via SIMD
reduce) is architecturally cleaner than fusing them with massive shmem.**

## Architectural Hypotheses

### Hypothesis A: Per-Block Private Output Buffers [HIGHEST PRIORITY]

**Concept**: Eliminate ALL atomics by giving each block its own output buffer.

Launch N_BLOCKS persistent blocks. Each block accumulates scatterer contributions
into `output[blockIdx.x][elem][freq]` using REGULAR writes (no atomics). A final
reduction kernel sums the per-block outputs.

**Memory**: N_BLOCKS x 64 x 854 x 2 x 4 bytes:
- 48 blocks: 21 MB (1 per SM)
- 96 blocks: 42 MB (2 per SM)

**Why this should work**:
- Eliminates 22% L2 atomic pressure entirely
- Regular write + L2 caching is 3-5x cheaper than atomicAdd
- Final reduction: 21-42 MB at 448 GB/s = 0.05-0.1ms (negligible)
- The output buffer (438 KB/block) fits entirely in L2 per SM

**Risks**:
- 48 blocks = 1 block/SM = low warp count. Need 256 threads/block for 8 warps.
- 42 MB extra memory (acceptable for 100K scatterers)

**GPU scaling**: On RTX 3090 (82 SMs) and 4090 (128 SMs), more SMs means more
blocks needed but also more L2 cache. The approach scales linearly.

### Hypothesis B: Frequency-Chunked Multi-Launch [HIGH PRIORITY]

**Concept**: Split the 854 frequencies into F chunks. Each launch has a smaller TX
buffer, enabling higher B_SCAT and/or more blocks/SM.

With 4 chunks of 214 frequencies:
- TX buffer per batch: 2 x B_SCAT x 214 (vs 2 x B_SCAT x 854)
- B_SCAT=10 fits in 35 KB → 2 blocks/SM with 100KB carveout
- 50% fewer atomics than B=5 (more batching)

**Why this should work**:
- The geometry computation is frequency-independent (done once per scatterer)
- Only the TX/RX frequency sweep repeats, which is the fast ALU-only inner loop
- 4 launches vs 1 launch: ~0.1ms overhead (negligible)

**Geometry recomputation cost**: Phase 1 geometry is ~5% of total time. Repeating
it 4x adds ~15% overhead. But the 50% atomic reduction should more than compensate.

Actually, we can CACHE geometry in global memory across launches to avoid recomputation.

**GPU scaling**: Identical benefit on 3090/4090. Higher SM count means more blocks.

### Hypothesis C: Two-Pass TX/RX with L2-Tiled Scatterer Access [HIGH PRIORITY]

**Concept**: Separate TX and RX into two kernel launches, like the Metal approach.

Pass 1 (TX): One block per scatterer batch. Compute TX[scat][freq] → global memory.
No atomics. Very high parallelism (100K/batch blocks).

Pass 2 (RX): One block per element (or element group). For each element, iterate
over scatterers in L2-friendly tiles, reading TX[scat][freq] from L2. Accumulate
RX contribution in registers. Write output ONCE (no atomics).

**Key innovation**: Tile the scatterer dimension for L2 reuse.
- L2 on A4000: 4 MB. TX tile of 585 scatterers fits entirely.
- All 64 elements reuse the same TX tile from L2 → 64x amplification
- Total DRAM reads: 684 MB (TX data read once). At 448 GB/s: 1.5ms.

**Why v8 (previous two-pass) was slow**: It read the full TX buffer for each element
without L2 tiling, causing DRAM bandwidth saturation.

**Why this should work now**:
- Zero atomics in RX (each block owns its output element)
- TX data stays in L2 across element blocks
- RX inner loop is pure FMA (geometric progression)
- Very high parallelism in both passes

**TX buffer size**: 100K x 854 x 2 x 4 = 684 MB. This is the main concern.
Could use fp16 TX to halve to 342 MB.

**GPU scaling**: RTX 3090 has 6 MB L2 (larger tiles), 4090 has 72 MB L2 (!).
On 4090, the ENTIRE TX buffer fits in L2. This architecture would be transformative.

### Hypothesis D: Warp-Specialized Producer-Consumer [MEDIUM PRIORITY]

**Concept**: Within each block, dedicate some warps to TX production and others
to RX consumption, connected via shared memory ring buffer.

- TX warps: compute TX for scatterers, write to shmem ring buffer
- RX warps: read TX from ring buffer, compute RX, accumulate
- Overlap TX computation with RX consumption (pipeline parallelism)

**Why this might work**:
- Hides TX latency behind RX computation
- Reduces shmem pressure (ring buffer is smaller than full TX buffer)
- Warps alternate between compute-heavy (TX) and memory-heavy (RX)

**Risks**: Complex synchronization. Warp scheduling is not deterministic on NVIDIA.
Probably not worth the complexity vs. simpler approaches.

### Hypothesis E: Cooperative Groups Grid Reduction [SPECULATIVE]

**Concept**: Use CUDA cooperative kernel launch for grid-wide synchronization.
Process scatterers in waves, with grid-wide barrier between waves.

- All blocks process one scatterer batch, write to shared output (no atomics
  within a wave since each block handles unique elements)
- Grid-wide barrier
- Next wave

**Why this is risky**: Cooperative launch requires occupancy == 1.0 (all blocks
must be co-resident). With our shmem usage, this limits grid size severely.

## Comparison Matrix

| Approach | Atomics Eliminated | Memory Cost | Complexity | Expected Speedup | 3090/4090 Benefit |
| -------- | ------------------ | ----------- | ---------- | ---------------- | ----------------- |
| A: Per-block output | 100% | 21-42 MB | Low | 1.5-2x | Linear with SMs |
| B: Freq-chunk | 50% (more batching) | 0 | Low | 1.3-1.5x | Same |
| C: Two-pass tiled | 100% (RX) | 684 MB TX buf | Medium | 2-3x | Huge on 4090 (L2) |
| D: Warp specialization | ~50% | 0 | High | 1.2-1.5x | Moderate |
| E: Cooperative groups | 100% | 0 | Medium | 1.5-2x | Better with more SMs |

## Recommended Experiment Order

1. **Exp 6**: Per-block output (A) -- simplest path to eliminating atomics
2. **Exp 7**: Frequency chunking (B) -- quick win on existing architecture
3. **Exp 8**: Two-pass L2-tiled (C) -- highest theoretical performance
4. **Exp 9**: Metal-inspired thread mapping -- different perspective

Each experiment gets its own beads epic and ncu profile.

## Hardware-Specific Opportunities

### RTX 3090 (GA102)
- 82 SMs (vs 48 on A4000) → 1.7x more parallelism
- 10 GB DRAM, 936 GB/s bandwidth
- 6 MB L2 cache → larger tiles for approach C
- fp16 tensor cores could accelerate inner loop

### RTX 4090 (AD102)
- 128 SMs → 2.7x more parallelism
- 24 GB DRAM, 1008 GB/s bandwidth
- **72 MB L2 cache** → entire TX buffer fits in L2 for approach C
- fp16 tensor cores + larger register file
- Approach C becomes trivial: TX to L2, RX reads from L2, zero DRAM pressure

### Implications
- Design for the L2-tiled approach (C) first -- it scales best across hardware
- Per-block output (A) is the simplest win on current hardware
- On 4090, the "correct" architecture is almost certainly two-pass with full L2 residency
