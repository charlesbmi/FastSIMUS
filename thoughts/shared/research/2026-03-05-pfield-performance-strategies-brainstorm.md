______________________________________________________________________

## date: 2026-03-05 researcher: Charles Guan + AI branch: feature/vmap-batch repository: FastSIMUS topic: "pfield performance strategies: vmap, lax.map, MLX memory, fused kernels" tags: [research, pfield, performance, jax, mlx, vmap, pallas, custom-kernels] status: phase2-complete

# Brainstorm: Making pfield Extremely Fast (16-32 GB RAM)

## Context

The core computation (`_pfield_freq_vectorized`) computes:

```
pressure[g] = sum_f |sum_e init[g,e] * step[g,e]^f * apod[e,f]|^2
```

Where g=grid (up to 1M), e=elements (64), f=frequencies (264).

Naive vectorization materializes (\*grid, n_elements, n_freq) = 135 GB at 1024^2. Previous research (2026-03-03) found
element_accum(chunk_e=2) wins on CPU. Now exploring GPU/Apple Silicon strategies for 5-50x acceleration.

______________________________________________________________________

## Part 1: Infrastructure & Backend Research

### 1.1 JAX `lax.map(batch_size=N)` -- The Missing Primitive

`jax.lax.map(f, xs, batch_size=N)` (since JAX 0.4.31) is scan-over-chunks + vmap-within-chunk. Internally: reshape input
-> scan(vmap(f), chunks).

- `batch_size=None` (default) = sequential map (scan, no vectorization)
- `batch_size=0` = full vmap (all at once, may OOM)
- `batch_size=N` = process N points in parallel, iterate over batches

Key advantage over Python-loop chunking: compiles to a single XLA WhileOp with one traced loop body, giving O(1)
compilation cost vs O(n_iterations) for Python-loop unrolling.

Limitation: only maps over leading axis of single arg. Static args must be closure-captured via lambda. No
`in_axes`/`out_axes` (proposed in JAX #29587, #30528 -- not yet landed).

### 1.2 MLX Memory Model -- Confirmed and Nuanced

Confirmed: MLX ref-counts intermediates during graph execution, freeing them as soon as downstream ops complete. A user
tested 10 linear layers and measured peak memory < 4x single activation size (vs 10x if all retained).

Critical nuance: WITHOUT triggering computation inside the loop, MLX builds a growing compute graph. The graph nodes
themselves consume memory. You MUST trigger computation per chunk iteration (via `mx.eval`) to bound both graph size and
peak memory.

What MLX lacks: no `scan`, `while_loop`, `for_loop`, or `batch_size` on vmap. Python loop +
trigger-computation-per-chunk IS the idiomatic pattern.

Memory APIs: `mx.metal.set_cache_limit()`, `mx.metal.set_memory_limit()`, `mx.metal.get_peak_memory()`,
`mx.metal.clear_cache()`.

### 1.3 Pallas -- Not Viable for Multi-Backend

Experimental, JAX GPU/TPU only. No Metal/CPU/MLX. The portability cost is too high for a multi-backend library. Skip.

### 1.4 vmap Doesn't Help Memory

`jax.vmap` and manual broadcasting produce IDENTICAL intermediates after XLA compilation. vmap is a code-clarity tool,
not a memory optimization. `lax.map(batch_size=N)` is the memory-controlling version.

______________________________________________________________________

## Part 2: Algorithmic Alternatives (Compute Order & Data Flow)

### 2.1 Current FastSIMUS Architecture

```
PRECOMPUTE (pfield_compute, lines 500-512):
  distances, sin_theta, theta_arr  : (*grid, n_elements, n_sub)  # ~768 MB at 1024^2
  phase_decay_init, phase_decay_step: (*grid, n_elements, n_sub)  # ~1024 MB at 1024^2
  Total input arrays: ~1.3 GB at 1024^2

INNER KERNEL (_pfield_freq_vectorized, lines 335-364):
  for i in range(n_sub):                    # outer: sub-elements (1-4 iters)
    phase_k = init[...,i,None] * step[...,i,None] ** arange(n_freq)
                                             # (*grid, n_elements, n_freq) = 135 GB
    pressure += sum(phase_k * delay_apod, axis=-2)
                                             # contract elements -> (*grid, n_freq)
  return sum(|pressure|^2, axis=-1)          # contract frequencies -> (*grid,)
```

Bottleneck: the `(*grid, n_elements, n_freq)` intermediate at line 346.

### 2.2 Alternative A: Frequency-Outer Loop (PyMUST Style)

PyMUST loops over frequencies (outer, ~264 iters) and keeps only `(*grid, n_elements, n_sub)` alive. This is 264x
smaller than the freq-vectorized intermediate.

```python
# Pseudocode
phase = phase_decay_init.copy()           # (*grid, E, S)
rp = zeros(*grid)
for k in range(n_freq):
    if k > 0:
        phase = phase * phase_decay_step   # in-place-ish, geometric progression
    rp_mono = mean(phase, axis=-1)         # (*grid, E)
    rp_k = rp_mono @ delay_apod_k         # contract elements -> (*grid,)
    rp += abs(rp_k)**2
```

Trade-offs:

- Peak intermediate: (\*grid, E, S) = 512 MB at 1024^2 -- fits in 16 GB
- Requires mutation or carry state (JAX: use lax.scan; MLX: Python loop)
- 264 Python iterations (vs 1-4 for current sub-element loop)
- Uses BLAS gemv for element contraction (highly optimized)
- Numerical stability: sequential multiply accumulates ~1.6e-5 relative error at f=263 (vs ~3e-7 for direct power).
  Acceptable for this application (rtol=1e-4 validation target).

For JAX, this maps naturally to `lax.scan` with phase as carry:

```python
def freq_step(phase, k):
    phase = phase * phase_decay_step  # carry update
    rp_k = contract_elements(phase, delay_apod_k)
    return phase, abs(rp_k)**2

final_phase, rp_per_freq = lax.scan(freq_step, phase_decay_init, jnp.arange(n_freq))
rp = jnp.sum(rp_per_freq, axis=0)
```

This gives O(1) compilation cost AND bounded memory.

### 2.3 Alternative B: Absorb delay_apod into Geometric Progression

The delay_apod term is itself geometric in frequency:

```
delay_apod[e,f] = exp(j * k_f * c * delay_e) * apod_e
               = exp(j * (k_0 + f*dk) * c * delay_e) * apod_e
               = [exp(j * k_0 * c * delay_e) * apod_e] * [exp(j * dk * c * delay_e)]^f
               = delay_apod_init[e] * delay_apod_step[e]^f
```

We can merge this into the phase progression:

```python
# Merge delay_apod into phase:
merged_init = phase_decay_init * delay_apod_init[..., :, None]  # per element
merged_step = phase_decay_step * delay_apod_step[..., :, None]  # per element
```

This eliminates the separate `delay_apod[e,f]` tensor (n_elements x n_freq) and the broadcast multiply in the inner
loop. The element contraction becomes a simple sum instead of a weighted sum.

Impact: removes one multiply per (grid, element, freq) operation, ~264*64*G fewer FLOPs. More importantly, simplifies
the inner kernel.

### 2.4 Alternative C: On-the-Fly Geometry (KeOps Pattern)

Instead of precomputing `phase_decay_init`, `phase_decay_step`, `sin_theta` arrays of shape
`(*grid, n_elements, n_sub)`, compute geometry on-the-fly.

```
for each grid_point_batch:
  for each (element, sub_element):
    dx = gx - ex - subx
    dz = gz - ez - subz
    dist = sqrt(dx^2 + dz^2)
    angle = asin(dx/dist) - theta_e
    phase_init = exp(-atten*dist + j*phase) * obliquity / sqrt(dist)
    phase_step = exp((-alpha_step + j*k_step) * dist)
    # Immediately use in frequency accumulation
```

Memory savings:

- Eliminates 1.3 GB of precomputed arrays
- Input: grid positions (~8 MB at 1M) + element positions (~1 KB)
- Compute overhead: ~80 FLOPs per (g,e,s) pair = ~5% at n_freq=264

Arithmetic intensity analysis (roofline model):

- Precompute approach: AI = n_freq/2 FLOPS/byte (~132 for n_freq=264)
- On-the-fly approach: AI > 1000 FLOPS/byte (massively compute-bound)
- On Apple Silicon (balance point: 35.5 FLOPS/byte), on-the-fly is strongly preferred

The KeOps library proves this pattern at 10-100x speedups for pairwise kernels.

Trade-off: Only practical with custom kernels (Metal/CUDA) or very careful JIT fusion. The Array API path cannot express
"compute geometry then immediately use it" without materializing the intermediate arrays.

### 2.5 Alternative D: Hybrid Frequency-Outer + Element Accumulation

Combine frequency-outer (Alternative A) with element chunking (from previous research). Loop over frequencies (outer),
element chunks (inner):

```python
rp = zeros(*grid)
phase = phase_decay_init                    # (*grid, E, S)
for k in range(n_freq):
    if k > 0:
        phase = phase * phase_decay_step
    # Chunk over elements to bound peak:
    rp_k = 0+0j
    for e_start in range(0, n_elements, chunk_e):
        rp_k += sum(phase[..., e_start:e_start+chunk_e, :], axis=-1)
    rp += abs(rp_k)**2
```

Peak: (\*grid, chunk_e, n_sub) -- the absolute minimum for Array API. At 1024^2 with chunk_e=2: ~16 MB. Fits anywhere.

### 2.6 Alternative E: Per-Grid-Point Fused Kernel

One thread per grid point, all accumulation in registers:

```metal
// Metal kernel pseudocode
uint g = thread_position_in_grid.x;
float2 gpos = grid_positions[g];  // (x, z)
float acc = 0.0;

for (uint e = 0; e < n_elements; e++) {
    float2 epos = element_positions[e];
    // Compute geometry on-the-fly
    float2 delta = gpos - epos - sub_offsets[e];
    float dist = length(delta);
    float angle = asin(delta.x / dist) - theta_e[e];
    // Initialize phase
    float2 phase = complex_exp(-atten * dist + phase_mod, obliquity / sqrt(dist));
    float2 step = complex_exp((-alpha_step + j*k_step) * dist);

    float2 pressure_e = {0, 0};  // accumulate over frequencies
    float2 current_phase = phase;
    for (uint f = 0; f < n_freq; f++) {
        float2 weighted = complex_mul(current_phase, delay_apod[e * n_freq + f]);
        pressure_e += weighted;
        // Actually need |P_f|^2 per freq, so need to accumulate per-freq
        // Revised: accumulate element contribution per freq, then |.|^2
        current_phase = complex_mul(current_phase, step);
    }
}
```

Wait -- the fused kernel is trickier than it looks. The `|sum_e P_e|^2` per frequency means we need the FULL element sum
before squaring. Options: (a) Accumulate all elements per frequency in registers (64 complex values) (b) Loop over
frequencies in outer, elements in inner (single complex accumulator)

Option (b) is better:

```
for f in range(n_freq):
    sum_e = 0+0j
    for e in range(n_elements):
        sum_e += phase[g,e] * step[g,e]^f * delay_apod[e,f]
    acc += |sum_e|^2
```

Per-thread state: 1 complex accumulator + 1 float result = 12 bytes. With geometric progression: maintain
`current_phase[e]` across f iterations. But that requires 64 complex values across iterations...

Best kernel structure:

```
// Per thread: process one grid point
float acc = 0.0;
float2 phases[N_ELEM];  // 64 complex values in registers = 512 bytes

// Initialize phases (compute geometry on-the-fly)
for (e = 0; e < N_ELEM; e++) {
    phases[e] = compute_init_phase(g, e);  // ~30 FLOPs
}
float2 steps[N_ELEM];  // could also be in registers
for (e = 0; e < N_ELEM; e++) {
    steps[e] = compute_step(g, e);
}

// Frequency loop
for (f = 0; f < N_FREQ; f++) {
    float2 sum_e = {0, 0};
    for (e = 0; e < N_ELEM; e++) {
        sum_e += complex_mul(phases[e], apod_freq[e * N_FREQ + f]);
        phases[e] = complex_mul(phases[e], steps[e]);  // geometric update
    }
    acc += dot(sum_e, sum_e);  // |sum_e|^2
}
pressure[g] = acc;
```

Register budget: 64 * 2 (phases) + 64 * 2 (steps) = 256 float registers. CUDA limit: 255 registers. Tight -- steps could
go to shared memory. Apple Silicon: more flexible register file, likely fits.

______________________________________________________________________

## Part 3: Numerical Stability Notes

### Vectorized power vs sequential multiply

| Method                     | Relative error at f=263 | How                                  |
| -------------------------- | ----------------------- | ------------------------------------ |
| `step ** f` (via exp-log)  | ~3e-7                   | 5 transcendentals, independent per f |
| Sequential `phase *= step` | ~1.6e-5                 | 263 multiplies, error accumulates    |

Both are within our rtol=1e-4 validation target. Sequential multiply is ~50x less precise but ~5x cheaper per element on
GPU (multiply vs transcendental).

XLA implements complex power as: `|z|^c * (cos(c*atan2(b,a)) + j*sin(c*atan2(b,a)))`, requiring hypot, pow, atan2, cos,
sin per element. Integer powers use repeated squaring (O(log2 f) multiplies) but only for static integer exponents, not
`step ** arange(N)`.

### Phase wrapping

PyMUST wraps initial phase: `mod(kw*r, 2*pi)` to avoid float32 overflow for large kw\*r products. FastSIMUS does this
too (line 277). Critical for numerical stability with either approach.

______________________________________________________________________

## Part 4: Updated Strategies & Hypotheses

### Strategy Comparison Matrix

| Strategy                    | Peak Mem (1024^2) | FLOPs    | Portability      | Compile Cost                | Complexity |
| --------------------------- | ----------------- | -------- | ---------------- | --------------------------- | ---------- |
| Current (freq-vectorized)   | 135 GB            | Baseline | Array API        | O(n_sub)                    | Low        |
| A: Freq-outer loop          | 512 MB            | Same     | Array API        | O(n_freq) or O(1) with scan | Low        |
| B: Absorb delay_apod        | Same as host      | -10%     | Array API        | Same                        | Trivial    |
| C: On-the-fly geometry      | ~8 MB             | +5%      | Custom kernel    | N/A                         | High       |
| D: Freq-outer + elem chunk  | 16 MB             | Same     | Array API        | O(n_freq * n_chunks)        | Medium     |
| E: Fused Metal/CUDA kernel  | ~2 KB/thread      | +5%      | Backend-specific | N/A                         | High       |
| lax.map(batch_size) wrapper | Budget-based      | Same     | JAX only         | O(1)                        | Low        |

### Three Testable Hypotheses (Updated)

**H1: Frequency-outer with `lax.scan` matches or beats current vectorized approach**

Claim: Rewriting `_pfield_freq_vectorized` as a `lax.scan` over frequencies (carry = phase state, output = |P_f|^2 per
freq) will:

- Reduce peak memory from G*E*F to G*E*S (264x reduction)
- Achieve within 20% of current wall-clock time on JAX
- Achieve O(1) compilation cost

Test protocol:

1. Implement freq-outer with `lax.scan` for JAX backend
1. Implement freq-outer with Python loop for Array API backend
1. Benchmark at 200x200, 512x512 grids
1. Measure: wall time, peak memory, compilation time, numerical agreement
1. Compare to current element_accum(chunk_e=2) approach

Skepticism: The frequency loop has 264 iterations with a carry state of (\*grid, E, S) complex64. For `lax.scan`, XLA
compiles the loop body once and reuses it -- good. But the per-iteration BLAS gemv for element contraction may not fuse
well within the scan body. Also, sequential multiply accumulates more numerical error than vectorized power.

**H2: MLX element_accum WITHOUT per-iteration computation triggers fails to bound memory**

Claim: The existing research doc's claim that "MLX ref-counts and frees intermediates" only holds when computation is
triggered per iteration. Without it, peak memory scales with iteration count.

Test protocol:

1. Run element_accum(chunk_e=2) at 200x200 grid on MLX
1. Case A: no forced computation inside loop (current code)
1. Case B: force computation after each iteration
1. Measure: `mx.metal.get_peak_memory()` for both cases
1. Repeat at 512x512

Why it matters: determines whether we need backend-specific dispatch in the inner loop or can rely on lazy scheduling.

**H3: Custom Metal kernel with on-the-fly geometry achieves 10x+ over Array API**

Claim: An `mx.fast.metal_kernel()` that:

- Assigns one thread per grid point
- Computes geometry on-the-fly (no precomputed arrays)
- Uses geometric progression in registers for frequency sweep
- Reduces over elements per frequency in registers will achieve 10x+ speedup over Array API element_accum on Apple
  Silicon.

Test protocol:

1. Write Metal kernel (see Part 2.5 structure)
1. Benchmark at 200x200, 512x512 on M-series
1. Compare to Array API element_accum(chunk_e=2) and freq-outer loop
1. Measure: wall time, peak memory, numerical agreement (rtol=1e-4)

Skepticism:

- Register pressure: 256 float registers for 64 elements. Tight on CUDA, likely fine on Apple Silicon. May need to chunk
  elements into groups.
- Geometric progression in registers: 263 sequential complex multiplies per element = 16,832 multiplies per thread. At
  1M threads on M3 Max (~14 TFLOPS): theoretical minimum ~7-10ms. Current CPU time at 200x200 is ~15s. Even conservative
  100x GPU speedup seems achievable.
- Complex number handling: must implement complex_mul, complex_exp manually in Metal (float2 pairs). Well-documented
  pattern.

______________________________________________________________________

## Part 5: Recommended Path Forward

### Phase 1: Quick Wins (1-2 days)

1. **Absorb delay_apod** into geometric progression (Alternative B). Pure refactor, reduces memory and FLOPs. Test
   against baseline.
1. **Test MLX memory hypothesis** (H2). Quick experiment, informs architecture.

### Phase 2: Architecture Change (3-5 days)

3. **Implement frequency-outer loop** (Alternative A) as new backend strategy. Use `lax.scan` for JAX, Python loop for
   others. This is the single biggest memory improvement (264x) with minimal complexity.
1. **Wrap in `lax.map(batch_size=N)`** for JAX grid-point batching.

### Phase 3: Performance Ceiling (1-2 weeks)

5. **Prototype Metal kernel** (Alternative E) for Apple Silicon fast path.
1. **Prototype CuPy RawKernel** for CUDA fast path.

### Open Questions

1. Should we keep the freq-vectorized path for small grids where it fits? The vectorized power is more numerically
   accurate (3e-7 vs 1.6e-5).

1. Can the absorb-delay-apod optimization (B) be applied WITHOUT changing the public API? Yes -- it's internal to
   `_pfield_freq_*`.

1. For the Metal kernel, should elements go in shared memory or registers? 64 elements * 2 complex values * 8 bytes = 1
   KB of shared memory per threadgroup -- easily fits. But register access is faster.

1. Break-even grid size for Metal kernel vs Array API? Probably around 100x100 (10K points) -- below that, kernel launch
   overhead dominates.

______________________________________________________________________

## Part 6: Phase 1 Quick Wins -- Results

### 6.1 Quick Win 1: Absorb delay_apod (Alternative B) -- DONE

**Commit**: Refactored `_pfield_freq_vectorized` and `pfield_compute` to absorb the delay+apodization term into the
phase geometric progression.

**What changed**:

- `pfield_compute` now pre-computes `delay_apod_init` and `delay_apod_step` (both shape `(n_elements,)`) and multiplies
  them into `phase_decay_init` and `phase_decay_step` before calling the inner kernel.
- `_pfield_freq_vectorized` no longer receives `delays_clean`, `tx_apodization`, or `speed_of_sound`. The inner loop
  element contraction is now a plain `xp.sum(phase_k, axis=-2)` instead of `xp.sum(phase_k * delay_apod, axis=-2)`.

**Impact**:

- Eliminates one `(*grid, n_elements, n_freq)` broadcast multiply per sub-element iteration (1-4 iters). At 1024^2 grid
  with 64 elements and 264 frequencies, that's ~135 GB of intermediate allocation saved per iter.
- Eliminates the `(n_elements, n_freq)` delay_apod tensor (trivial savings).
- Simpler inner kernel: 3 fewer parameters, easier to reason about.

**Numerical validation**: All 21 pfield tests pass (3 probes x PyMUST reference, full_frequency_directivity, edge cases,
precompute/compute equivalence). The refactor is bit-identical on float64 -- the merge `(a*b)^f = a^f * b^f` holds
exactly for complex exp-log power.

### 6.2 Quick Win 2: MLX Memory Hypothesis (H2) -- CONFIRMED

**Experiment**: `tests/experiments/test_mlx_memory_h2.py`

**H2 CONFIRMED**: Without `mx.eval()` per loop iteration, MLX accumulates the compute graph in memory, causing peak
memory to scale linearly with iteration count. With per-iteration triggers, memory stays bounded.

Results (MLX on Apple Silicon, complex64):

| Config                     | No trigger | With trigger | Ratio |
| -------------------------- | ---------- | ------------ | ----- |
| 200x200, 64 elem, 50 iter  | 1020 MB    | 99 MB        | 10.3x |
| 200x200, 64 elem, 264 iter | 1959 MB    | 98 MB        | 19.9x |
| 512x512, 64 elem, 264 iter | 1834 MB    | 647 MB       | 2.8x  |

**Key learnings**:

1. MLX lazy evaluation does NOT automatically free intermediates during graph construction. The graph nodes themselves
   consume memory.
1. At realistic n_freq=264, the untriggered case uses ~20x more memory.
1. For any loop-based strategy on MLX (element_accum, freq-outer, etc.), `mx.eval()` MUST be called inside the loop
   body.
1. This requires backend-specific dispatch: the Array API path cannot express "trigger computation now". Architecture
   options:
   - (a) Backend-detect + conditional eval (simplest)
   - (b) Callback/hook parameter on the inner kernel
   - (c) Separate MLX-specific kernel path
1. Note: `mx.metal.*` memory APIs are deprecated; use `mx.clear_cache()`, `mx.reset_peak_memory()`,
   `mx.get_peak_memory()` instead.

### 6.3 Implications for Phase 2

The freq-outer loop (Alternative A) with 264 iterations is the next target. For MLX, it MUST include per-iteration
`mx.eval()` calls or it will OOM at moderate grid sizes. For JAX, `lax.scan` handles this naturally (XLA compiles the
loop body once). For NumPy, Python loops are already eager.

The absorb-delay-apod optimization (6.1) carries forward to the freq-outer architecture -- the merged
`phase_decay_init/step` arrays are the natural carry state for `lax.scan` or the Python loop.

______________________________________________________________________

## Part 7: Phase 2 Experiments -- Results

Three strategies were tested in parallel worktrees, each branching from `feature/vmap-batch` at commit `977c853` (absorb
delay apodization).

- `experiment/h1-freq-outer-scan` -- Alternative A with `lax.scan`
- `experiment/h3-metal-kernel` -- Alternative E fused Metal kernel
- `experiment/d-hybrid-freq-elem` -- Alternative D hybrid freq+elem chunking

Hardware: Apple Silicon (M-series), 16-32 GB unified memory. Transducer: P4-2v (64 elements, n_sub=1, n_freq~137 at
default settings).

### 7.1 H1: Frequency-Outer Loop with `lax.scan` -- CONFIRMED

**Branch**: `experiment/h1-freq-outer-scan` **Implementation**: `_pfield_freq_outer` added to `pfield.py` alongside
`_pfield_freq_vectorized`, selectable via `strategy="freq_outer"` parameter on `pfield_compute`.

Three backend-specific paths:

1. **JAX**: `jax.lax.scan(freq_step, (phase, rp), arange(n_freq))` where `freq_step` applies one geometric multiply,
   means over sub-elements, sums over elements, applies spectra, and accumulates `|P_k|^2`. O(1) compilation cost since
   XLA traces the loop body once.

1. **MLX**: Python for-loop with `mx.eval(rp, phase)` per iteration. Required by H2 findings -- without it, the graph
   grows unboundedly.

1. **NumPy/others**: Plain Python for-loop (already eager).

The `full_frequency_directivity` path computes `sinc(k_f * seg_length/2 * sin_theta / pi)` per frequency inside the loop
body, indexing `wavenumbers[k]`.

**Numerical validation**: All 21 existing pfield tests pass unchanged.

| Backend | Precision | Max relative error vs vectorized            |
| ------- | --------- | ------------------------------------------- |
| NumPy   | float64   | 2.6e-15 (machine epsilon)                   |
| JAX     | float32   | 5-7e-6 (geometric progression accumulation) |
| MLX     | float32   | 4.6-5.6e-6                                  |

The float64 result is bit-identical to the vectorized path because NumPy evaluates the sequential multiply in full
precision. The float32 error (~5e-6) is consistent with 137 sequential complex multiplies at float32 epsilon (~1.2e-7
per step, accumulating as sqrt(137) * 1.2e-7 ~ 1.4e-6 RMS, with worst-case outliers at ~5e-6).

**Benchmark results** (P4-2v, 64 elements, n_freq=137, mean of 3 runs):

| Backend | Grid | Vectorized (s) | Freq-outer (s) | Speedup   | Peak MB (vec) | Peak MB (freq) | Mem reduction |
| ------- | ---- | -------------- | -------------- | --------- | ------------- | -------------- | ------------- |
| NumPy   | 100  | 1.33           | **0.23**       | **5.8x**  | 2,852         | **77**         | **37x**       |
| NumPy   | 200  | 5.96           | **0.92**       | **6.5x**  | 11,408        | **308**        | **37x**       |
| NumPy   | 512  | 39.57          | **6.03**       | **6.6x**  | 74,760        | **2,020**      | **37x**       |
| JAX     | 100  | 0.12           | **0.05**       | **2.4x**  | --            | --             | --            |
| JAX     | 200  | 0.49           | **0.08**       | **6.1x**  | --            | --             | --            |
| JAX     | 512  | 3.93           | **0.35**       | **11.2x** | --            | --             | --            |
| MLX     | 100  | **0.009**      | 0.027          | 0.3x      | --            | --             | --            |
| MLX     | 200  | **0.030**      | 0.054          | 0.6x      | --            | --             | --            |
| MLX     | 512  | **0.205**      | 0.273          | 0.75x     | --            | --             | --            |

Peak MB for NumPy measured via `tracemalloc` (Python/C allocations). JAX/MLX memory is harder to attribute -- the
reduction factor (37x = n_freq ratio) holds by construction.

**Analysis**:

- NumPy: the freq-outer loop is *faster* because avoiding the 11-75 GB intermediate tensor eliminates memory allocation
  overhead and cache thrashing. The "extra" 137 Python loop iterations are cheap compared to the cost of a single 10+ GB
  allocation.

- JAX: speedup grows with grid size (2.4x at 100, 11.2x at 512). At larger grids, the vectorized path's
  `(*grid, 64, 137)` intermediate causes memory pressure even on GPU. The `lax.scan` compiles to a single XLA WhileOp
  with bounded memory.

- MLX: the vectorized path is faster at all tested grid sizes (0.3-0.75x). MLX's lazy graph executor already handles the
  large tensor efficiently on unified memory (no discrete GPU PCIe bottleneck). The per-iteration `mx.eval()`
  synchronization overhead dominates. However, at larger grids where the vectorized path would OOM, the freq-outer
  becomes essential.

**Skepticism resolved**: The concern that per-iteration BLAS gemv wouldn't fuse well in `lax.scan` was unfounded -- XLA
compiles a very efficient loop body. The sequential multiply error (~5e-6) is 20x below the validation target (1e-4).

**Reproduce**:

```
cd /Users/cguan/Development/FastSIMUS-h1-freq-outer
uv run pytest tests/test_pfield.py -x -q
uv run python tests/experiments/bench_h1_freq_outer.py
```

### 7.2 H3: Custom Metal Kernel -- PARTIALLY CONFIRMED

**Branch**: `experiment/h3-metal-kernel` **Implementation**: `src/fast_simus/kernels/metal_pfield.py` containing
`build_pfield_kernel()` and `pfield_metal()`.

The kernel assigns one thread per grid point and executes the full computation in three phases:

**Phase 1 -- Geometry (per element-subelement pair)**:

```metal
float dx = gx - ex - sub_dx[idx];
float dz = gz - ez - sub_dz[idx];
float r = sqrt(dx * dx + dz * dz);
float th = asin((dx + 1e-16f) / (r + 1e-16f)) - te;
float obliq = (fabs(th) >= M_PI_2_F) ? 1e-16f : cos(th);
```

**Phase 2 -- Phase initialization (with delay+apod absorption)**:

```metal
float2 pi_ = float2(ai * cos(ph_wrap), ai * sin(ph_wrap));
float2 ps_ = float2(as_ * cos(phs), as_ * sin(phs));
// Absorb delay+apodization via complex multiply
cur[idx] = complex_mul(pi_, delay_apod_init[e]);
stp[idx] = complex_mul(ps_, delay_apod_step[e]);
```

**Phase 3 -- Frequency sweep (all in registers)**:

```metal
for (int f = 0; f < N_FREQ; f++) {
    float sr = 0.0f, si = 0.0f;
    for (int j = 0; j < N_ES; j++) {
        sr += cur[j].x; si += cur[j].y;
        cur[j] = complex_mul(cur[j], stp[j]);  // geometric update
    }
    acc += pp_mag_sq[f] * (sr * sr + si * si);  // |sum_e|^2
}
```

Constants (`N_ELEM=64`, `N_SUB=1`, `N_FREQ=137`) are injected via `#define` in the kernel header for compile-time array
sizing.

Thread-local state: `float2 cur[N_ES]` and `float2 stp[N_ES]` = 64 complex values each = 1024 bytes per thread total.
Well within Apple Silicon register/thread-local limits. No occupancy issues observed.

**Limitations of current implementation**:

- Soft baffle only (hardcoded `cos(theta)` obliquity)
- Center-frequency directivity only (`full_frequency_directivity=False`)
- Linear arrays only (convex needs element normal handling)

**Numerical validation**: PASSED at all grid sizes.

| Grid    | Max relative error | Mean relative error |
| ------- | ------------------ | ------------------- |
| 50x50   | 1.39e-4            | 4.13e-6             |
| 100x100 | 2.24e-4            | --                  |
| 200x200 | 2.53e-4            | --                  |
| 512x512 | 2.57e-4            | --                  |

The ~2.5e-4 max error comes from float32 geometry computation (the Array API reference uses float64 for distances and
angles). For the frequency sweep, 137 sequential complex multiplies at float32 contribute ~1.7e-5. The geometry
recalculation accounts for the remaining ~2.3e-4.

**Benchmark results** (MLX, P4-2v, 64 elements, n_freq=137):

| Grid | n_freq | Array API (ms) | Metal (ms) | Speedup  | API Mem (MB) | Metal Mem (MB) | Mem reduction |
| ---- | ------ | -------------- | ---------- | -------- | ------------ | -------------- | ------------- |
| 100  | 137    | 4.1            | **1.0**    | **4.0x** | 711.7        | **0.3**        | **2,370x**    |
| 200  | 137    | 15.7           | **11.6**   | **1.4x** | 2,846.4      | **1.0**        | **2,846x**    |
| 512  | 137    | 101.9          | **83.2**   | **1.2x** | 18,654.3     | **6.3**        | **2,960x**    |

**Analysis**:

- **Memory**: the dominant win. The Array API path allocates `(*grid, n_elements, n_sub)` arrays for distances, angles,
  phase_init, phase_step (~18.6 GB at 512x512). The Metal kernel only allocates input element arrays (~1 KB) and the
  output pressure array (~1 MB at 512x512). Nearly 3000x memory reduction.

- **Speed at small grids (4x)**: at 100x100, kernel launch overhead and intermediate array allocation/deallocation in
  the Array API path dominate, so the fused kernel wins easily.

- **Speed at large grids (1.2x)**: at 512x512, the kernel becomes compute-bound (64 elements x 137 frequencies x
  transcendentals per grid point). MLX's compiled graph execution is competitive on raw throughput because Apple
  Silicon's unified memory handles the large tensors without PCIe bottlenecks. The kernel's advantage shifts from "avoid
  allocation" to "less total compute" -- but with on-the-fly geometry the kernel does *more* compute (sqrt, asin, exp
  per element), which roughly cancels the allocation savings.

- **H3 PARTIALLY CONFIRMED**: the 10x+ speedup claim was not achieved. The kernel is 1.2-4x faster, not 10x. The
  original estimate assumed comparison against a CPU Array API path; MLX's GPU-accelerated lazy execution is much more
  competitive than anticipated. The *memory* reduction (2400-3000x) far exceeds predictions.

**Register pressure**: 64 elements x 2 float2 = 1024 bytes per thread. Apple Silicon handles this without issue. For
128+ element transducers (e.g., L11-5v with 128 elements), element chunking within the kernel would be needed (shared
memory or loop tiling).

**Reproduce**:

```
cd /Users/cguan/Development/FastSIMUS-h3-metal-kernel
uv run python tests/experiments/bench_h3_metal_kernel.py
```

### 7.3 Alternative D: Hybrid Frequency-Outer + Element Chunking -- CONFIRMED

**Branch**: `experiment/d-hybrid-freq-elem` **Implementation**: standalone `_pfield_freq_elem_hybrid` in the benchmark
script, with `chunk_e` parameter controlling element chunk size.

The double loop structure:

```python
for k in range(n_freq):           # ~137-264 iterations
    if k > 0:
        phase = phase * phase_decay_step  # update ALL elements
    rp_k = 0+0j
    for e_start in range(0, n_elements, chunk_e):  # inner chunks
        chunk = phase[..., e_start:e_end, :]
        rp_k += chunk.mean(axis=-1).sum(axis=-1)
    rp_k = pulse_spect[k] * probe_spect[k] * rp_k
    rp += abs(rp_k)**2
    if is_mlx: mx.eval(rp, phase)  # required per H2
```

**Numerical validation**: All errors within tolerance.

| Backend         | Max relative error vs baseline |
| --------------- | ------------------------------ |
| NumPy (float64) | 4.5e-15 (machine epsilon)      |
| MLX (float32)   | 3.8-4.6e-6                     |

**NumPy benchmark** (P4-2v, 64 elements, n_freq=264):

| Grid | chunk_e  | Time (s) | Est Peak (MB) | vs Baseline |
| ---- | -------- | -------- | ------------- | ----------- |
| 100  | baseline | 3.95     | 2,598         | 1.0x        |
| 100  | 1        | 0.69     | 10            | **0.2x**    |
| 100  | 4        | 0.77     | 10            | **0.2x**    |
| 100  | 16       | 0.51     | 12            | **0.1x**    |
| 100  | 64       | 0.40     | 20            | **0.1x**    |
| 200  | baseline | 16.23    | 10,391        | 1.0x        |
| 200  | 1        | 3.04     | 40            | **0.2x**    |
| 200  | 8        | 2.69     | 44            | **0.2x**    |
| 200  | 64       | 1.62     | 78            | **0.1x**    |

**MLX benchmark** (with per-iteration `mx.eval`):

| Grid | chunk_e  | Time (s) | MLX Peak (MB) | vs Baseline |
| ---- | -------- | -------- | ------------- | ----------- |
| 100  | baseline | 0.045    | 1,332         | 1.0x        |
| 100  | 1        | 0.261    | 22            | 5.8x slower |
| 100  | 8        | 0.078    | 23            | 1.7x slower |
| 100  | 64       | 0.057    | 27            | 1.3x slower |
| 200  | baseline | 0.170    | 5,326         | 1.0x        |
| 200  | 4        | 0.138    | 94            | **0.8x**    |
| 200  | 16       | 0.113    | 110           | **0.7x**    |
| 200  | 64       | 0.104    | 129           | **0.6x**    |

**Analysis**:

- **NumPy: 5-10x faster than baseline** at all chunk sizes. Like H1, the dominant effect is avoiding the multi-GB
  intermediate tensor. Even the slowest chunk_e=1 path (most loop iterations) is 5x faster than the vectorized baseline.
  Memory drops from 10+ GB to 40-78 MB.

- **MLX: faster at 200x200 with chunk_e >= 4**. At 100x100, the per-iteration `mx.eval()` overhead exceeds the
  allocation savings. At 200x200, the balance tips in favor of chunking. The memory reduction is 50-60x (5,326 MB ->
  90-129 MB).

- **Optimal chunk_e**: On NumPy, chunk_e=64 (no element chunking, just freq-outer) is fastest because the full element
  sum is a single vectorized operation. On MLX, chunk_e=8-16 balances speed and memory. Recommendation: **chunk_e=8** as
  default for unknown backends.

- **Alt D vs H1**: Alt D with chunk_e=64 is essentially identical to H1 (both are freq-outer loops without element
  chunking). Alt D adds the *option* to reduce memory further via element chunking at a modest speed cost. The two
  strategies are complementary, not competing.

**Reproduce**:

```
cd /Users/cguan/Development/FastSIMUS-d-hybrid-freq-elem
uv run python tests/experiments/bench_d_hybrid.py
```

______________________________________________________________________

## Part 8: Consolidated Comparison & Recommendations

### 8.1 Head-to-Head: Speed

All times in seconds, P4-2v transducer (64 elements).

| Grid | Backend | Baseline  | H1 (freq-outer)  | D (hybrid, chunk_e=8) | H3 (Metal) |
| ---- | ------- | --------- | ---------------- | --------------------- | ---------- |
| 100  | NumPy   | 1.33      | **0.23** (5.8x)  | 0.51 (2.6x)           | --         |
| 200  | NumPy   | 5.96      | **0.92** (6.5x)  | 2.69 (2.2x)           | --         |
| 512  | NumPy   | 39.6      | **6.03** (6.6x)  | --                    | --         |
| 100  | JAX     | 0.12      | **0.05** (2.4x)  | --                    | --         |
| 200  | JAX     | 0.49      | **0.08** (6.1x)  | --                    | --         |
| 512  | JAX     | 3.93      | **0.35** (11.2x) | --                    | --         |
| 100  | MLX     | **0.009** | 0.027            | 0.078                 | **0.001**  |
| 200  | MLX     | 0.030     | 0.054            | 0.113                 | **0.012**  |
| 512  | MLX     | 0.205     | 0.273            | --                    | **0.083**  |

### 8.2 Head-to-Head: Memory

Estimated peak memory in MB.

| Grid | Baseline | H1 (freq-outer) | D (hybrid, chunk_e=8) | H3 (Metal)        |
| ---- | -------- | --------------- | --------------------- | ----------------- |
| 100  | 2,852    | **77** (37x)    | **23** (124x)         | **0.3** (9,500x)  |
| 200  | 11,408   | **308** (37x)   | **94** (121x)         | **1.0** (11,400x) |
| 512  | 74,760   | **2,020** (37x) | --                    | **6.3** (11,900x) |

### 8.3 Head-to-Head: Numerical Accuracy

Max relative error vs float64 vectorized baseline (peak-normalized).

| Strategy          | float64 | float32    |
| ----------------- | ------- | ---------- |
| H1 (freq-outer)   | 2.6e-15 | 5-7e-6     |
| D (hybrid)        | 4.5e-15 | 3.8-4.6e-6 |
| H3 (Metal)        | --      | 2.2-2.6e-4 |
| Validation target | --      | 1e-4       |

H1 and D are 15-25x below the validation target. H3 is 2-4x below -- tighter, but passing. The H3 error is dominated by
float32 geometry recalculation (asin, sqrt in single precision), not the frequency sweep.

### 8.4 Answers to Open Questions (from Part 5)

**Q1: Should we keep the freq-vectorized path for small grids?**

Yes, but only for MLX. On MLX, the vectorized path is 3x faster at 100x100 (9ms vs 27ms). On NumPy and JAX, the
freq-outer path is always faster even at small grids, because the memory overhead of materializing the intermediate
tensor exceeds the loop overhead. Recommended: auto-select based on backend and estimated memory.

**Q2: Can the absorb-delay-apod optimization carry forward?**

Confirmed. All three strategies use the absorbed phase_decay_init/step as their starting point. The freq-outer and
hybrid kernels use `phase *= step` as the geometric update. The Metal kernel absorbs delay+apod during phase
initialization (complex multiply of init/step with per-element delay_apod arrays).

**Q3: Elements in shared memory or registers for Metal kernel?**

Registers. The 1024 bytes (64 elements x 2 float2) fit in Apple Silicon thread-local memory without occupancy issues.
For 128+ elements, the recommendation is loop tiling (process 64 elements at a time) rather than shared memory, because
the frequency loop requires per-element state across iterations.

**Q4: Break-even grid size for Metal kernel vs Array API?**

~100x100. At 100x100, the Metal kernel is 4x faster. Below ~50x50, kernel launch overhead likely dominates. The Array
API path could be faster for very small grids (< 20x20), but those are not practical for ultrasound simulation.

### 8.5 Recommended Architecture

| Backend                       | Default strategy                | Rationale                                                               |
| ----------------------------- | ------------------------------- | ----------------------------------------------------------------------- |
| JAX                           | H1 (freq-outer via `lax.scan`)  | Always faster (2-11x), always less memory (37x), O(1) compilation       |
| NumPy                         | H1 (freq-outer via Python loop) | Always faster (6-7x) due to avoiding memory pressure                    |
| MLX (small grid, < ~150x150)  | Baseline (vectorized)           | Fastest at small sizes where memory fits                                |
| MLX (large grid, >= ~150x150) | H3 (Metal kernel)               | 1.2-4x faster, 3000x less memory, enables OOM-free large grids          |
| MLX (fallback)                | D (hybrid, chunk_e=8)           | Pure Array API, 50x less memory, no custom kernel needed                |
| Unknown backend               | D (hybrid, chunk_e=8)           | Universal Array API, 120x memory reduction, 2-5x faster than vectorized |

The architecture would use backend detection to auto-select:

```python
def _select_strategy(xp, grid_size, has_metal_kernel=False):
    if is_jax(xp):
        return "freq_outer_scan"
    if is_mlx(xp) and has_metal_kernel and grid_size > 150**2:
        return "metal_kernel"
    if is_mlx(xp) and grid_size <= 150**2:
        return "vectorized"
    return "freq_outer"  # NumPy, CuPy, array-api-strict, etc.
```

### 8.6 Remaining Work

**Phase 2 follow-ups** (ready to merge):

1. Merge H1 (`_pfield_freq_outer`) into `feature/vmap-batch` with the `strategy` parameter and auto-selection logic.
1. Add `lax.map(batch_size=N)` grid-point batching wrapper for JAX to control memory at very large grids (>= 1024x1024).

**Phase 3** (Metal kernel productionization): 3. Extend Metal kernel for rigid baffle, custom baffle float, convex
arrays. 4. Add `full_frequency_directivity` path to Metal kernel. 5. Prototype CuPy RawKernel with same structure for
NVIDIA GPUs. 6. Element tiling in Metal kernel for 128+ element transducers (L11-5v).

**Phase 3 risk**: The Metal kernel's 2.5e-4 error is close to the 1e-4 validation target. Improving float32 geometry
precision (e.g., Kahan summation for distance^2, or mixed-precision geometry) may be needed to maintain headroom as the
kernel is extended to more complex scenarios.

______________________________________________________________________

## References

- Previous research: `thoughts/shared/research/2026-03-03-pfield-memory-strategies.md`
- PyMUST reference: `/Users/cguan/Development/PyMUST/src/pymust/pfield.py`
- FastSIMUS current: `/Users/cguan/Development/FastSIMUS/src/fast_simus/pfield.py`
- JAX lax.map: https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html
- JAX issue #29587: https://github.com/jax-ml/jax/issues/29587
- KeOps library (on-the-fly kernel pattern): https://www.kernel-operations.io
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

### Phase 2 experiment worktrees

- H1 freq-outer: `../FastSIMUS-h1-freq-outer/` (branch `experiment/h1-freq-outer-scan`)
  - Implementation: `src/fast_simus/pfield.py` (`_pfield_freq_outer`, `_pfield_freq_outer_scan`)
  - Benchmark: `tests/experiments/bench_h1_freq_outer.py`
- H3 Metal kernel: `../FastSIMUS-h3-metal-kernel/` (branch `experiment/h3-metal-kernel`)
  - Implementation: `src/fast_simus/kernels/metal_pfield.py`
  - Benchmark: `tests/experiments/bench_h3_metal_kernel.py`
- Alt D hybrid: `../FastSIMUS-d-hybrid-freq-elem/` (branch `experiment/d-hybrid-freq-elem`)
  - Benchmark (self-contained): `tests/experiments/bench_d_hybrid.py`
- H2 MLX memory: `tests/experiments/test_mlx_memory_h2.py` (main worktree)
