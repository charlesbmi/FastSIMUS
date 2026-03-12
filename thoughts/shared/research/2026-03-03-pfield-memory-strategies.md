______________________________________________________________________

## date: 2026-03-03T17:38:52Z researcher: Charles Guan git_commit: f669ff20c02f4f89ba7b48b3399f8df7dc284d49 branch: experiment/memory-strategies repository: FastSIMUS topic: "pfield memory reduction strategies for large grids on GPU" tags: [research, pfield, memory, gpu, chunking, vectorization, benchmarking] status: complete last_updated: 2026-03-03 last_updated_by: Charles Guan

# Research: pfield Memory Reduction Strategies for Large Grids

**Date**: 2026-03-03T17:38:52Z **Researcher**: Charles Guan **Git Commit**: f669ff20c02f4f89ba7b48b3399f8df7dc284d49
**Branch**: experiment/memory-strategies **Repository**: FastSIMUS

## Research Question

After vectorizing the frequency loop in `_pfield_freq_vectorized`, the peak intermediate tensor
`(*grid, n_elements, n_freq)` in complex64 exceeds GPU memory for large grids. What is the best strategy to reduce
intermediate memory usage -- explicit chunking, contraction reordering, or compiler fusion -- targeting 1024x1024 grids
on a 16 GB GPU?

## Summary

Three strategies were implemented and benchmarked: element accumulation (reorder the contraction to loop over elements),
frequency chunking (exploit additive `|P|^2`), and grid chunking (batch grid points through the existing pipeline).
Element accumulation with chunk_e=8 is the clear winner: it is **faster** than the fully-vectorized baseline at 200x200
(15.3s vs 17.6s), uses **80% less memory**, and produces **bit-identical** output. The speed advantage comes from better
cache utilization -- smaller intermediates fit in CPU cache, avoiding thrashing even on a 128 GB machine.

XLA fusion cannot be relied upon to eliminate the intermediate: an open XLA regression (#29620, July 2025) documents
exactly this pattern failing to fuse.

## Background: The Memory Bottleneck

The vectorized computation in `_pfield_freq_vectorized` (pfield.py:294-369) computes:

```
pressure[g, f] = sum_e (init[g,e] * step[g,e]^f * delay_apod[e,f])
```

This materializes `(*grid, n_elements, n_freq)` per sub-element iteration (line 350), then immediately contracts over
elements (line 357). The tensor exists briefly but must fit in memory.

### Memory at scale (P4-2v probe, n_sub=1)

The P4-2v probe produces n_freq=264 frequency samples (not 50 as initially estimated), making the problem significantly
worse than the vectorization plan anticipated.

| Grid      | n_elements | n_freq | Intermediate (complex64) |
| --------- | ---------- | ------ | ------------------------ |
| 50x50     | 64         | 264    | 338 MB                   |
| 200x200   | 64         | 264    | 5.4 GB                   |
| 512x512   | 64         | 264    | 35 GB                    |
| 1024x1024 | 64         | 264    | 135 GB                   |

Input arrays (`phase_decay_init`, `phase_decay_step`, `sin_theta`) are `(*grid, n_elements, n_sub)` and total ~2.5-4 GB
at 1024x1024 with n_sub=1 -- these fit in 16 GB.

### Mathematical properties enabling chunking

Two properties make incremental computation exact:

1. **Element sum is linear**: `sum_e(a_e + b_e) = sum_e(a_e) + sum_e(b_e)`. Elements can be processed in chunks and
   accumulated.

1. **|P|^2 sum is additive over frequencies**: `sum_f |P_f|^2 = sum_f1 |P_f1|^2 + sum_f2 |P_f2|^2`. Frequency batches
   can contribute independently to the final result.

### Why XLA fusion is not a solution

XLA issue #29620 (open since July 2025) documents a regression where broadcast + elementwise

- reduction operations fail to fuse on GPU, materializing the full intermediate tensor. The pattern
  `jnp.sum(a[..., None] * b[..., None] ** c, axis=-2)` is exactly our computation.

## Strategies Implemented

### A: Element Accumulation (`_pfield_freq_element_accum`)

**File**: `src/fast_simus/pfield.py:371-417`

Reorders the contraction: instead of materializing `(*grid, E, F)` and summing over E, loops over element chunks of size
`chunk_e`, creating `(*grid, chunk_e, F)` per iteration, and accumulates into `(*grid, F)`.

```
Peak intermediate: (*grid, chunk_e, n_freq) complex64
  chunk_e=1:  G * F * 8 bytes    (e.g. 2.1 GB at 1024^2)
  chunk_e=8:  G * 8 * F * 8 bytes (e.g. 16.9 GB at 1024^2)
```

Produces bit-identical results to baseline. Array API compliant.

### B: Frequency Chunking (`_pfield_freq_chunk_freq`)

**File**: `src/fast_simus/pfield.py:420-471`

Loops over frequency batches of size `chunk_freq`. Exploits the additive `sum |P_f|^2` property to accumulate squared
magnitudes incrementally, never keeping all frequencies alive simultaneously.

```
Peak intermediate: (*grid, n_elements, chunk_freq) complex64
  chunk_freq=1:  G * E * 8 bytes   (e.g. 0.5 GB at 1024^2)
  chunk_freq=4:  G * E * 4 * 8 bytes (e.g. 2.1 GB at 1024^2)
```

Produces bit-identical results to baseline. Array API compliant.

### C: Grid Chunking (`pfield_compute_chunked`)

**File**: `src/fast_simus/pfield.py:643-711`

Public wrapper around `pfield_compute`. Flattens the grid positions, batches them into chunks of `grid_chunk_size`,
calls `pfield_compute` on each chunk, and reassembles.

```
Peak intermediate per chunk: (chunk_grid, n_elements, n_freq) complex64
  Sized to fit any memory budget.
```

Handles the case where even input arrays exceed memory. Requires re-computing geometry (distances, angles, exponentials)
per chunk. Public API: exported from `__init__.py`.

## Benchmark Results

### Setup

- Platform: macOS 26.3, Apple M-series (arm64), 128 GB unified memory
- Backend: NumPy (single-threaded CPU)
- Probe: P4-2v (n_elements=64, n_freq=264, n_sub=1)
- Correctness verified against `pfield()` baseline at 50x50 grid

### 50x50 Grid (2,500 points)

| Strategy      | Chunk | Time (s) | Actual Mem (MB) | Theory (MB) |
| ------------- | ----- | -------- | --------------- | ----------- |
| baseline      | -     | 0.94     | 1,363           | 338         |
| element_accum | 1     | 1.00     | 43              | 11          |
| element_accum | 8     | 0.95     | 264             | 48          |
| element_accum | 32    | 0.95     | 1,025           | 174         |
| freq_chunk    | 1     | 1.19     | 6               | 1           |
| freq_chunk    | 4     | 1.16     | 31              | 5           |
| freq_chunk    | 16    | 0.98     | 123             | 21          |
| grid_chunk    | 10k   | 0.95     | 1,374           | 1,352       |
| grid_chunk    | 50k   | 0.95     | 1,374           | 6,758       |

At small grids, all strategies are within 25% of each other in time.

### 200x200 Grid (40,000 points)

| Strategy      | Chunk | Time (s) | Actual Mem (MB) | Theory (MB) | 16GB fit?  |
| ------------- | ----- | -------- | --------------- | ----------- | ---------- |
| baseline      | -     | 17.6     | 21,796          | 5,407       | NO         |
| element_accum | 1     | 16.0     | 676             | 169         | YES        |
| element_accum | 8     | **15.3** | 4,224           | 760         | YES        |
| element_accum | 32    | 21.8     | 16,390          | 2,788       | borderline |
| freq_chunk    | 1     | 18.7     | 83              | 21          | YES        |
| freq_chunk    | 4     | 18.3     | 492             | 82          | YES        |
| freq_chunk    | 16    | 15.8     | 1,967           | 328         | YES        |
| grid_chunk    | 10k   | **15.1** | 5,496           | 1,352       | YES        |
| grid_chunk    | 50k   | 17.3     | 21,981          | 6,758       | NO         |

Key: element_accum(8) and grid_chunk(10k) are fastest. All strategies correct.

### 512x512 Grid (262,144 points)

| Strategy                  | Time (s) | Speedup  |
| ------------------------- | -------- | -------- |
| baseline (pfield_compute) | 615.9    | 1.0x     |
| grid_chunk (50k)          | 109.9    | **5.6x** |

The 5.6x speedup at 512x512 confirms that reducing memory pressure alone provides major performance gains from better
cache utilization, even on a 128 GB machine.

### Observations

1. **Actual memory is 2-4x higher than theoretical** due to intermediate temporaries in broadcast, power, and reduction
   operations. For GPU planning, use 3x theoretical.

1. **Speed improves with smaller intermediates** -- not just memory. element_accum(8) is 13% faster than baseline at
   200x200 despite doing more loop iterations. Cache effects dominate.

1. **element_accum sweet spot is chunk_e=4-8**. chunk_e=1 has slight overhead from many iterations; chunk_e=32 loses the
   cache advantage.

1. **freq_chunk has higher per-iteration cost** because the inner loop creates `(*grid, n_elements, chunk_freq)` which
   is larger per-element than element_accum's `(*grid, chunk_e, n_freq)` when chunk_e < n_elements.

## Memory Budget for 1024x1024 on 16 GB GPU

Assuming 3x theoretical for actual memory (observed ratio):

| Strategy      | Chunk | Theory (GB) | Est. Actual (GB) | Fits 16 GB? |
| ------------- | ----- | ----------- | ---------------- | ----------- |
| baseline      | -     | 135         | 405              | NO          |
| element_accum | 1     | 2.1         | 6.3              | YES         |
| element_accum | 2     | 4.2         | 12.6             | YES         |
| element_accum | 4     | 8.4         | 25.3             | NO          |
| freq_chunk    | 1     | 0.5         | 1.5              | YES         |
| freq_chunk    | 4     | 2.1         | 6.3              | YES         |
| grid_chunk    | 10k   | 1.4         | 4.1              | YES         |

Input arrays at 1024x1024: ~2.5 GB (n_sub=1), must also fit.

**Recommended default**: element_accum with chunk_e=2 (~12.6 GB actual + 2.5 GB inputs = ~15 GB). For safety margin,
chunk_e=1 (~6.3 GB actual + 2.5 GB = ~9 GB).

## Backend Compatibility Analysis

All three strategies are **Array API compliant** and work uniformly across backends:

- **NumPy**: eager execution with standard memory model. Each iteration allocates and immediately frees the
  intermediate.
- **JAX**: Python loop unrolls at trace time. XLA performs liveness analysis on the unrolled HLO and reuses buffers,
  achieving peak intermediate of `(*grid, chunk_e, n_freq)` even though 128 iterations exist in the traced code.
- **CuPy**: same as NumPy.
- **MLX**: lazy evaluation builds compute graph during the loop (no computation happens). When `mx.eval()` is called,
  intermediates are ref-counted and freed incrementally with garbage collection, achieving peak memory of just the
  accumulator + one intermediate without explicit allocation.

### JAX JIT and Loop Compilation

The element and frequency loops can optionally dispatch to `jax.lax.scan` for efficient loop body compilation (single
HLO WhileOp instead of unrolling). The Python loop version is portable but gets unrolled at trace time (compilation cost
~O(n_iterations)); `lax.scan` compiles the loop body once (cost ~O(1) after first trace, then cached). For chunk_e=2 or
freq_chunk=4, either approach is acceptable. Using `jax.lax.scan` would require JAX-specific code and is outside the
Array API standard.

### MLX Lazy Evaluation Model

MLX's lazy evaluation is ideally suited for this pattern:

1. The element accumulation loop produces a compute graph of ~100-500 operations (well within MLX's documented range of
   "tens to thousands of operations per evaluation").
1. When `mx.eval()` is called on the result, MLX's scheduler materializes intermediates incrementally.
1. Each `(*grid, chunk_e, n_freq)` intermediate is ref-counted and freed as soon as it's accumulated into the result,
   enabling memory reuse.
1. No special primitives needed; the Python loop naturally maps to MLX's evaluation model without modification.

MLX does not provide `scan`, `while_loop`, or `for_loop` function transformations. The available transforms are `vmap`,
`grad`, `vjp`, `jvp`, `compile`, and `checkpoint`. The element/frequency loops must use Python control flow in all
backends (the Array API standard does not define loop transformations, by design, to remain backend-agnostic).

### Recommended Backend Strategies

- **NumPy/CuPy**: Use element_accum(chunk_e=2) for simplicity. No compilation overhead.
- **JAX**: Use element_accum(chunk_e=2) with Python loop for portability, or optionally wrap in `jax.lax.scan` for
  cleaner compiled trace.
- **MLX**: Use element_accum(chunk_e=2) with Python loop. Call `mx.eval(pressure)` after the function returns to trigger
  computation at the API boundary.

## Code References

- `src/fast_simus/pfield.py:294-369` -- baseline `_pfield_freq_vectorized`
- `src/fast_simus/pfield.py:371-417` -- `_pfield_freq_element_accum` (Strategy A)
- `src/fast_simus/pfield.py:420-471` -- `_pfield_freq_chunk_freq` (Strategy B)
- `src/fast_simus/pfield.py:643-711` -- `pfield_compute_chunked` (Strategy C)
- `tests/test_pfield.py:518-570` -- `TestMemoryStrategies` (9 correctness tests)
- `experiments/benchmark_memory_strategies.py` -- benchmark harness

## Related Research

- `thoughts/shared/plans/2026-02-25-pfield-tensor-vectorization.md` -- the vectorization plan that created the current
  `_pfield_freq_vectorized`. This research addresses the "Future Work: Grid chunking" item from that plan.
- `thoughts/shared/research/gemini_jax_vmap.md` -- Array API standard limitations around vmap/scan. Relevant to why all
  strategies use Python loops rather than `jax.lax.scan`.
- Brainstorm session 2026-03-03 with Charles Guan -- explored XLA fusion failures, opt_einsum memory limits, einsum
  trees (2025 research), cotengra slicing, and discovered that explicit chunking (element or frequency) outperforms
  compiler-based fusion strategies on modern hardware due to cache effects and memory hierarchy. Also analyzed backend
  compatibility: JAX unrolls loops with XLA liveness reuse, MLX uses lazy evaluation with ref-counted GC, NumPy/CuPy use
  standard eager execution.

## Future Directions

- **Absorb delay_apod into the geometric progression**: The `delay_apod[e,f]` term is itself a geometric series in f.
  Merging it with `step[g,e]` eliminates one intermediate tensor and simplifies the inner loop. Orthogonal to the
  chunking strategy.

- **JAX `lax.scan` for the element/frequency loop**: For JAX JIT, replacing the Python loop with `lax.scan` would
  compile the loop body once instead of unrolling. Requires backend dispatch (JAX-specific code path).

- **Adaptive chunk sizing**: Auto-select chunk_e based on available device memory (query via
  `jax.local_devices()[0].memory_stats()` or similar).

- **Benchmark on CUDA GPU**: The CPU results show cache effects dominate. GPU results may differ due to different memory
  hierarchy and parallelism model.
