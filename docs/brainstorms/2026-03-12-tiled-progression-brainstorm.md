# Element-Tiled Progression: Closing the Shared-Memory Gap

Date: 2026-03-12

## What We're Building

A hybrid TX kernel combining the geometric progression's SFU-free inner loop with
the shared-memory kernel's low register pressure. Goal: match or beat the
progression kernel at large N (>=5000 scatterers) while retaining the shared
kernel's advantages at small N.

## Why This Approach

The shared-memory kernel (`simus_tx_shared.metal`) is 9% slower than the
geometric progression at chunk_size=5000. Analysis of the bottleneck:

### SFU vs ALU Decomposition

**Progression kernel** (per sub-element per frequency, inner loop):
- Complex multiply advance: 6 ALU, **0 SFU**
- Accumulation: 2 ALU
- Total: 8 ALU, 0 SFU

**Shared-memory kernel** (per sub-element per frequency, inner loop):
- Phase computation: 2 ALU
- exp(-alpha): 1 SFU + 2 ALU
- cos(phase): 1 SFU + 1 ALU
- sin(phase): 1 SFU + 1 ALU
- da cos/sin: 2 SFU + 4 ALU (amortized per element boundary)
- Accumulation: 2 ALU
- Total: ~12 ALU, 4-5 SFU

**Key insight**: On both Apple Silicon and NVIDIA GPUs, SFU throughput is ~1/8th
of ALU throughput. The progression's inner loop is 100% ALU-bound; the shared
kernel is SFU-bound. This explains the 9% gap despite higher parallelism.

### Register Pressure Recap

| Approach | Thread-local state | Registers (float32) | Spill risk |
|----------|-------------------|---------------------|------------|
| Progression | cur[64] + stp[64] float2 | 256 | Severe on Apple, marginal on CUDA |
| Shared | ~20 scalars | ~20 | None |
| Tiled (T=16) | cur[16] + stp[16] float2 + 8 accumulators | ~92 | None |
| Tiled (T=32) | cur[32] + stp[32] float2 + 8 accumulators | ~156 | None on CUDA, possible on Apple |

## Design: Element-Tiled Progression

### Architecture

- One threadgroup (256 threads) per scatterer
- Shared memory: geometry + da-absorbed stride steps (~2.5 KB)
- Process sub-elements in tiles of TILE_SE (default 16)
- Each thread handles ceil(N_FREQ/256) strided frequencies
- Per tile: init cur/stp from shared memory at thread's starting frequency,
  then advance via geometric progression (ALU-only inner loop)
- stp precomputed in shared memory (same for all threads in threadgroup)

### Shared Memory Layout

```
shared_amp[N_ES]         amplitude (freq-independent)           256 B
shared_kw_r[N_ES]        base phase (kw_init * r)               256 B
shared_kr_step[N_ES]     phase step (kw_step * r)               256 B
shared_alpha_r[N_ES]     base attenuation                       256 B
shared_ar_step[N_ES]     attenuation step                       256 B
shared_stp[N_ES]         float2 stride step (da-absorbed)       512 B
shared_da_init_re[N_ELEM] da init real                          256 B
shared_da_init_im[N_ELEM] da init imag                          256 B
shared_dps[N_ELEM]       delay_phase_step per element           256 B
Total: 2560 bytes (fits easily in 32KB Metal / 100KB+ CUDA)
```

### Key Optimization: Precompute stp Cooperatively

The stride step `stp[se]` is identical for all threads in the threadgroup
(depends only on stride=TG and geometry, not on thread's starting frequency).
By precomputing stp (including da absorption) in shared memory during the
cooperative init phase, we save 3 SFU calls per sub-element per thread.

Per-thread tile init drops from 8 SFU to **5 SFU per sub-element**:
- exp(-alpha), cos(phase), sin(phase) for cur: 3 SFU
- cos(da_phase), sin(da_phase) for da at starting freq: 2 SFU

### FLOP Comparison

For P4-2v (64 elements, 812 frequencies, 256 threads/threadgroup):

| Metric | Progression | Shared | Tiled (T=16) |
|--------|-------------|--------|--------------|
| SFU per scatterer (init) | ~640 | 0 | 82K (cooperative + per-thread) |
| SFU per scatterer (sweep) | 0 | 328K | **0** |
| ALU per scatterer (sweep) | 417K | 786K | 525K |
| Total SFU | 640 | 328K | 82K |
| Inner loop SFU | 0 | 5/elem/freq | **0** |

### Cross-Platform Compatibility (Metal / CUDA)

| Feature | Metal | CUDA |
|---------|-------|------|
| Shared memory | threadgroup (32KB) | __shared__ (100-228KB) |
| Fast sincos | metal::fast::cos/sin | __sincosf() |
| Stride step | Same | Same |
| Tile size | TILE_SE=16 recommended | TILE_SE=32 feasible (255 regs) |
| Atomic outputs | atomic_fetch_add_explicit | atomicAdd |

The algorithmic pattern is identical. Only the language syntax differs.

## Benchmark Results

Tested on Apple Silicon (M-series). P4-2v transducer (64 elements, 1 sub-element,
812 frequencies). TG=64 threadgroup size for tiled kernel (empirically optimal).

| N_SCAT | Progression | Shared | Tiled-16 | Tiled vs Prog | Tiled vs Shared |
|--------|-------------|--------|----------|---------------|-----------------|
| 100 | 41K/s | 421K/s | 142K/s | 3.5x | 0.34x |
| 500 | 207K/s | 1072K/s | 2209K/s | 10.7x | 2.1x |
| 1000 | 409K/s | 1301K/s | 3278K/s | 8.0x | 2.5x |
| 2000 | 822K/s | 1519K/s | 4740K/s | 5.8x | 3.1x |
| 5000 | 1832K/s | 1668K/s | **6051K/s** | **3.3x** | **3.6x** |
| 10000 | 1958K/s | 1747K/s | **7043K/s** | **3.6x** | **4.0x** |

### Tile Size Sweep (N=5000)

| TILE_SE | Throughput | Notes |
|---------|-----------|-------|
| 8 | 4878K/s | More tiles = more init overhead |
| **16** | **6051K/s** | Optimal: 256 bytes registers, 4 tiles |
| 32 | 1754K/s | Register spills start (~512 bytes) |
| 64 | 724K/s | Full spill (same as progression) |

### Threadgroup Size Sweep (N=5000, TILE_SE=16)

| TG_SIZE | Throughput | Notes |
|---------|-----------|-------|
| **64** | **6020K/s** | Optimal: 13 freq/thread, minimal SFU |
| 128 | 5590K/s | Competitive |
| 256 | 4235K/s | Too much SFU in init |
| 512 | 3334K/s | SFU bottleneck |

### Accuracy

Max relative error vs progression reference: 8e-4 to 2e-3. Slightly higher
at large N (expected: fast math accumulation). Still well within the 1e-2
tolerance for ultrasound simulation.

### Why TG=64 Beats TG=256

Fewer threads per threadgroup = fewer tile inits per scatterer:
- TG=64: 64 threads * 64 sub-elements * 5 SFU = 20,480 SFU per scatterer
- TG=256: 256 * 64 * 5 = 81,920 SFU per scatterer (4x more!)

The inner loop (ALU-only) handles more frequencies per thread (13 vs 4),
which is fine since ALU throughput vastly exceeds SFU throughput.

## Resolved Questions

1. **Optimal TILE_SE**: **16**. Fits in 256 bytes of registers (92 float32
   total). TILE_SE=32 causes spills; TILE_SE=8 has too much init overhead.
2. **Accuracy**: 8e-4 to 2e-3 relative error. Acceptable.
3. **Occupancy**: TG=64 with 92 registers/thread gives excellent occupancy.
   5000 threadgroups saturates the GPU at all tested sizes.

## Production Recommendation

Replace the progression kernel with tiled-16 (TG=64) as the default TX kernel.
The tiled approach is faster at all production-relevant sizes (N >= 500) and
uses the same algorithmic pattern on both Metal and CUDA. For N < 300,
the shared-direct kernel could be used as a fallback, but in practice the
chunking system always uses N >= 5000.

### CUDA Port Notes

The tiled kernel maps directly to CUDA:
- `threadgroup` -> `__shared__`
- `thread_position_in_threadgroup` -> `threadIdx`
- `metal::fast::cos/sin` -> `__sincosf()`
- `threadgroup_barrier` -> `__syncthreads()`
- TG=64 = 2 warps per block (good occupancy on NVIDIA)
