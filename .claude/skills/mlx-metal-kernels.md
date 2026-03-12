---
name: mlx-metal-kernels
description: Guide for writing custom Metal kernels using MLX's mx.fast.metal_kernel() API on Apple Silicon, plus comprehensive reference for MLX's internal kernel architecture patterns. Use when implementing GPU-accelerated compute kernels, fused operations, or performance-critical numerical code in MLX. Covers the complete API, complex number handling, threading patterns, memory strategies, and debugging. Triggers on "metal kernel", "custom kernel", "mx.fast", "GPU kernel", "apple silicon kernel", or performance-critical MLX code.
---

# Writing Metal Kernels for MLX

Two paths for Metal kernels in MLX:
1. **Custom kernels** via `mx.fast.metal_kernel()` — Python API for one-off fused ops
2. **Internal kernels** in `mlx/backend/metal/kernels/` — C++ Metal templates for framework ops

---

## Part 1: Custom Kernels via `mx.fast.metal_kernel()`

### When to Write a Custom Kernel

Write a custom Metal kernel when:
- You need to **fuse multiple operations** to eliminate intermediate array allocations
- An operation is **memory-bandwidth-bound** and can benefit from on-the-fly computation
- You need **thread-local accumulation** across a large reduction (e.g., frequency sweep)
- The Array API path materializes arrays that exceed available memory

Do NOT write a custom kernel when:
- `mx.compile()` already fuses the operation graph well enough
- The operation is simple elementwise/reduction (use MLX built-in ops)
- You need portability across NumPy/JAX/CuPy backends

### API Reference

#### Construction (build once, reuse many times)

```python
kernel = mx.fast.metal_kernel(
    name="my_kernel",           # Cache key; prefix "custom_kernel_" added internally
    input_names=["a", "b"],     # Metal parameter names for inputs
    output_names=["out"],       # Metal parameter names for outputs (>= 1 required)
    source="...",               # Kernel BODY only (signature auto-generated)
    header="",                  # Prepended before signature: #defines, helpers, structs
    ensure_row_contiguous=True, # Copy non-contiguous inputs to row-major before launch
    atomic_outputs=False,       # If True: `device atomic<T>*` for concurrent writes
)
```

#### Invocation (all keyword-only)

```python
outputs = kernel(
    inputs=[arr_a, arr_b],          # Must match len(input_names); scalars auto-wrapped
    output_shapes=[(N,)],           # One shape per output_name
    output_dtypes=[mx.float32],     # One dtype per output_name
    grid=(N, 1, 1),                 # Total threads (dispatchThreads, NOT threadgroups)
    threadgroup=(256, 1, 1),        # Threads per threadgroup (product <= 1024)
    template=[("T", mx.float32)],   # Optional: typename/int/bool template args
    init_value=0.0,                 # Optional: pre-fill outputs before kernel runs
    verbose=False,                  # Print generated Metal source for debugging
)
```

#### Auto-Generated Signature Rules

MLX scans `source` for names and attributes, then generates the full Metal function:

**Inputs** (by array properties):
- Normal array: `const device float* a [[buffer(0)]]`
- Scalar (0-d array): `const constant float& a [[buffer(0)]]` (access as `a`, not `a[0]`)
- Small array (<8 elements): `const constant float* a [[buffer(0)]]`

**Shape/stride helpers** (only if name found in source):
- `a_shape` in source -> adds `const constant int* a_shape`
- `a_strides` in source -> adds `const constant int64_t* a_strides`
- `a_ndim` in source -> adds `const constant int& a_ndim`

**Metal attributes** (auto-detected by string search in source):
- `thread_position_in_grid` (uint3), `thread_position_in_threadgroup` (uint3)
- `threadgroup_position_in_grid` (uint3), `threads_per_threadgroup` (uint3)
- `simdgroup_index_in_threadgroup` (uint), `thread_index_in_simdgroup` (uint)
- `thread_execution_width` (uint), and all other Metal table attributes

**Gotcha**: Detection is simple `string.find()` — even attribute names in *comments* or *variable names* trigger generation. Avoid naming variables like `my_thread_position_in_grid_offset`.

**Outputs**: Always `device T*`, always row-contiguous, no shape/stride helpers.

### Custom Kernel Patterns

#### Pattern 1: One Thread Per Output Element

```metal
uint g = thread_position_in_grid.x;
if (g >= N_GRID) return;  // bounds check
out[g] = result;
```

```python
grid=(output_size, 1, 1), threadgroup=(256, 1, 1)
```

Use `#define` constants via `header` for compile-time loop bounds:
```python
header = f"#define N_FREQ {n_freq}\n#define N_ELEM {n_elem}\n"
```

#### Pattern 2: Fused Compute (pfield example)

Fuse geometry computation + frequency sweep + reduction into one kernel:

```metal
uint g = thread_position_in_grid.x;
float gx = grid_x[g], gz = grid_z[g];

float2 cur[N_ES];  // thread-local register arrays
float2 stp[N_ES];

// Phase 1: Compute geometry on-the-fly, init phases
for (int e = 0; e < N_ELEM; e++) {
    float dx = gx - elem_x[e], dz = gz - elem_z[e];
    float r = sqrt(dx*dx + dz*dz);
    // ... init cur[e] and stp[e] from geometry ...
}

// Phase 2: Frequency sweep with geometric progression
float acc = 0.0f;
for (int f = 0; f < N_FREQ; f++) {
    float sr = 0.0f, si = 0.0f;
    for (int j = 0; j < N_ES; j++) {
        sr += cur[j].x; si += cur[j].y;
        // Advance phase: cur *= stp (complex multiply)
        float cr = cur[j].x, ci = cur[j].y;
        cur[j] = float2(cr*stp[j].x - ci*stp[j].y, cr*stp[j].y + ci*stp[j].x);
    }
    acc += weight[f] * (sr*sr + si*si);
}
out[g] = sqrt(acc);
```

#### Pattern 3: Reduction with Shared Memory

For row-wise reductions (one threadgroup per row):

```metal
threadgroup float shared_vals[32];  // one per SIMD group

float partial = 0.0f;
for (uint i = lid; i < row_size; i += threads_per_threadgroup.x)
    partial += data[row * row_size + i];

partial = simd_sum(partial);  // SIMD-level reduction (free)

if (thread_index_in_simdgroup == 0)
    shared_vals[simdgroup_index_in_threadgroup] = partial;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simdgroup_index_in_threadgroup == 0) {
    float val = (thread_index_in_simdgroup < simdgroups_per_threadgroup)
                ? shared_vals[thread_index_in_simdgroup] : 0.0f;
    val = simd_sum(val);
    if (thread_index_in_simdgroup == 0) out[row] = val;
}
```

#### Pattern 4: Atomic Scatter

```python
kernel = mx.fast.metal_kernel(
    name="scatter_grad", ...,
    atomic_outputs=True,  # outputs become device atomic<float>*
)
outputs = kernel(inputs=[grad, idx], ..., init_value=0.0)
```

```metal
uint i = thread_position_in_grid.x;
atomic_fetch_add_explicit(&out[indices[i]], grad[i], memory_order_relaxed);
```

### Complex Number Handling

For custom kernels, prefer `float2` over `complex64_t` — it's a native Metal SIMD type:

```metal
float2 cmul(float2 a, float2 b) {
    return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
float2 cexp(float r, float theta) {
    float mag = exp(r);
    return float2(mag * cos(theta), mag * sin(theta));
}
float cabs2(float2 z) { return z.x*z.x + z.y*z.y; }
```

Phase wrapping for numerical stability with large `k*r`:
```metal
float TWO_PI = 2.0f * M_PI_F;
float wrapped = kwr - TWO_PI * floor(kwr / TWO_PI);
float2 phase = float2(cos(wrapped), sin(wrapped));
```

### Performance Guidelines

**Thread-local arrays**: Keep total state < 2 KB (256 floats). Apple Silicon has a generous register file; 64-128 `float2` values typically fit. Beyond ~64 elements, consider chunking.

**Grid/threadgroup sizing**: 256 threads is a good default. Product must be <= 1024. For memory-bound kernels, 128 or 64 may improve cache behavior.

**Kernel caching**: Build kernel objects once; cache by dimension key:
```python
_cache: dict[tuple, object] = {}
def get_kernel(n_elem, n_freq):
    key = (n_elem, n_freq)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"my_kernel_{n_elem}_{n_freq}",
            header=f"#define N {n_elem}\n#define F {n_freq}\n", ...
        )
    return _cache[key]
```

**Memory measurement**:
```python
mx.eval(mx.zeros(1)); mx.reset_peak_memory()
result = kernel(...); mx.eval(result)
peak_bytes = mx.get_peak_memory()
```

### Debugging

Use `verbose=True` to print the complete generated Metal source including auto-generated signature.

| Error | Cause | Fix |
|-------|-------|-----|
| "Wrong number of inputs" | `len(inputs) != len(input_names)` | Match input count |
| Kernel produces NaN | Uninitialized thread-local arrays | Initialize all values |
| Silent wrong results | Off-by-one in thread index | Add bounds check: `if (g >= N) return;` |
| Compilation error | Syntax error in Metal code | Use `verbose=True` |
| Outputs all zero | Grid size mismatch | Verify `grid` matches output count |
| `bool` input issues | MLX bool -> Metal mismatch | Cast to `float32`, compare `> 0.5f` |

### Open Questions for Experimentation

1. **Register spill threshold**: At what N does `float2 arr[N]` spill? Likely ~128-256 for M-series. Profile with Instruments (Metal System Trace).
2. **Threadgroup shared memory for element data**: When many threads read the same small arrays, does `threadgroup` memory help when arrays are already in `constant` address space?
3. **`metal::fast::` vs `metal::precise::` math**: MLX's FFT uses `fast::cos/sin`. Default `cos()/sin()` maps to `metal::precise::`.
4. **`mx.compile()` interaction**: Custom kernels are already compiled Metal; wrapping in `mx.compile()` may fuse surrounding array prep but won't affect the kernel itself.
5. **Template parameters vs `#define`**: Both work for compile-time constants. No observed performance difference.

---

## Part 2: MLX Internal Kernel Architecture

Reference for MLX's built-in kernel patterns in `mlx/backend/metal/kernels/` (~18,700 lines).

### File Organization

Every kernel family follows a three-file split:

| File | Role | Example |
|---|---|---|
| `*_ops.h` | Small functor structs with `operator()` | `unary_ops.h`, `binary_ops.h` |
| `*.h` | Templated kernel function bodies | `unary.h`, `binary.h`, `reduce.h` |
| `*.metal` | Instantiation macros → named kernel entry points | `unary.metal`, `binary.metal` |

### Core Instantiation Mechanism

[`defines.h:22-24`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/defines.h#L22-L24) — Every kernel registered via:
```metal
#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
```
Uses `[[host_name(...)]]` to assign a stable string name for runtime lookup. Naming convention: `variant + op + type` (e.g., `"v_AbsFloat32"`, `"gn4large_AbsFloat32"`).

### Two Build Paths

**Precompiled metallib**: `.metal` → `.air` via `xcrun metal` → linked into `mlx.metallib`.

**JIT**: Kernel source strings concatenated at runtime; `get_template_definition()` generates explicit instantiations for needed specializations only.

### Foundational Infrastructure

| Header | Key Contents |
|---|---|
| `defines.h` | `REDUCE_N_READS=4`, `SOFTMAX_N_READS=4`, `instantiate_kernel` |
| `utils.h` | `WorkPerThread<T>`, `Limits<T>`, `elem_to_loc`, SIMD shuffle wrappers, `ceildiv` |
| `bf16.h`/`bf16_math.h` | BFloat16 type + ~35 math overloads routed through `float` |
| `complex.h` | `struct complex64_t { float real, imag; }` + full arithmetic |
| `fp8.h`/`fp4.h` | FP8 E4M3/E8M0 and FP4 E2M1 types |
| `atomic.h` | CAS-loop emulation extending atomics to all types |

**[`WorkPerThread<T>`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/utils.h#L17-L21)**: `n = 8/sizeof(T)` → `uint8→8`, `half→4`, `float→2`, `int64→1`. Default template parameter for per-thread work count.

**[`elem_to_loc` family](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/utils.h#L97-L299)**: Converts flat/grid indices to strided memory offsets. Variants: single array, two arrays simultaneously (`vec<IdxT,2>`), three arrays (`vec<IdxT,3>`), fixed 1D/2D/3D fast paths, and `LoopedElemToLoc<DIM>` stateful iterator.

**[`Limits<T>`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/utils.h#L28-L86)**: `max`, `min`, `finite_max`, `finite_min` for every type. Float: `max=+inf`, `min=-inf`.

**[SIMD shuffle wrappers](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/utils.h#L357-L434)**: 64-bit overloads via `uint2` bit-cast for `uint64_t`, `int64_t`, `bool`, `complex64_t`.

### Element-wise Kernels

**Operation functor pattern**: Small structs with `operator()` passed as template parameter:
```metal
struct Abs {
    template <typename T> T operator()(T x) { return metal::abs(x); }
    uint8_t operator()(uint8_t x) { return x; }
    complex64_t operator()(complex64_t x) { ... }
};
```

**Three kernel variants**: contiguous 1D (`_v`), contiguous 2D for large tensors (`_v2`), general strided (`_g`). Each parameterized by `WorkPerThread<T>::n`, `IdxT` (32/64-bit indices).

**Broadcasting**: Contiguous path uses separate `ss`/`sv`/`vs`/`vv` instantiations. Strided path uses zero strides (broadcast dim has stride 0 → always reads position 0).

**Macro dispatch hierarchy**: `instantiate_*_types(op)` → `instantiate_*_all(op, type)` → `instantiate_*_base(...)` generating all layout × type combinations.

### Reduction Patterns

**Universal two-level reduction**:
```
Level 1: simd_sum/simd_max (hardware, 32 threads)
Level 2: shared_vals[simd_group_id] → threadgroup_barrier → simd_reduce again
```
Threadgroup memory always `simd_size` (32) slots.

**N_READS = 4**: Each thread sequentially accumulates 4 elements before SIMD reduction, amortizing latency.

**Six reduction ops** ([`reduction/ops.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/reduction/ops.h)): `And`, `Or`, `Sum`, `Prod`, `Min`, `Max` — each with `init`, `operator()`, `atomic_update`, `simd_reduce`. 64-bit types use manual `simd_shuffle_down` tree.

**Row reductions**: `row_reduce_small` (1 thread/SIMD per output), `row_reduce_simple` (1 threadgroup per N_WRITES=4 rows), `row_reduce_looped` (arbitrary strides).

**Column reductions**: `col_reduce_small`, `col_reduce_longcolumn`, `col_reduce_looped` (BM×BN=32×32 tiles), `col_reduce_2pass`.

**Softmax**: Three-phase (max → exp+sum → normalize). Large rows use single-pass online algorithm with `exp(old_max - new_max)` rescaling.

**Layer/RMS Norm**: `threadgroup_sum<N>` helper reduces N scalars simultaneously. RMS norm broadcasts via single-slot `threadgroup float local_inv_mean[1]`.

### GEMM (Matrix Multiply)

**steel/ library**: `BlockLoader` (cooperative device→threadgroup), `BaseMMAFrag` (8×8 `simdgroup_matrix`), `MMATile`, `BlockMMA`, `GEMMKernel`.

**Template params**: `BM, BN, BK` (tile dims), `WM, WN` (simdgroup count), `transpose_a/b`, `MN_aligned, K_aligned`. Tile configs: `(64,64,16,2,2)`, `(64,64,16,1,2)`, `(64,32,32,2,2)`, `(32,64,16,1,2)`, `(32,32,16,2,2)`, `(64,32,8,4,1)`.

**K-loop**: `barrier → load_unsafe → barrier → mma → next`, with safe/unsafe variants selected by alignment flags.

**Bank conflict avoidance**: `tgp_padding = 16/sizeof(T)` columns added to threadgroup tiles.

**Block swizzle**: Z-order remapping of threadgroup indices for L2 reuse.

**GEMV**: Separate path — thread-level dot product with `TM×TN` register blocking + `simd_shuffle_down` tree. No `simdgroup_matrix`.

**Quantized GEMM**: `QuantizedBlockLoader` dequantizes packed weights (2-8 bits) into threadgroup memory. Pre-scaled activations avoid bit shifts in dot product.

**NAX path**: `mpp::tensor_ops::matmul2d` with 16×16 fragments (Metal 4.0+, macOS SDK 26.2+).

### Specialized Kernels

| Kernel | Pattern | Key Technique |
|--------|---------|--------------|
| Sort (`sort.h`) | 3-kernel merge sort | Merge-path partitioning, NaN→end |
| FFT (`fft/`) | Mixed-radix Stockham | Hand-optimized codelets (2-13), Rader/Bluestein for primes |
| Hadamard (`hadamard.h`) | Mixed-radix butterfly | JIT-generated `hadamard_radix_m()` for non-power-of-2 |
| SDPA (`sdpa_vector.h`) | Online softmax attention | Transposed shared-memory scratchpad, 2-pass for long seqs |
| Conv (`conv.metal`) | 3 strategies | Naive unfold, depthwise with `function_constant`, Winograd F(6×6,3×3) |
| RoPE (`rope.metal`) | Forward/inverse rotation | `function_constant` for direction/layout, N=4 heads/thread |
| Scan (`scan.h`) | Parallel prefix scan | SIMD exclusive scan + threadgroup coordination |

### Key Performance Techniques

| Technique | Where | Purpose |
|---|---|---|
| `MLX_MTL_PRAGMA_UNROLL` | All inner loops | Full loop unrolling |
| `WorkPerThread<T>::n` | Element-wise | Amortize dispatch per type size |
| `N_READS = 4` | Reductions, norms | Sequential accumulation before SIMD |
| Threadgroup padding | GEMM tiles | Bank conflict avoidance |
| Block swizzle | GEMM | Z-order for L2 reuse |
| Pre-scaled activations | Quantized dot product | Avoid bit shifts |
| Online max-normalization | Softmax, logsumexp | Single-pass stable computation |
| `function_constant` | RoPE, SDPA, conv, FFT | Dead branch elimination at compile time |
| Safe/unsafe load dichotomy | All loaders | Alignment-selected bounds checking |

### Thread Organization Summary

| Pattern | Example | Grid |
|---|---|---|
| 1D flat | `unary_v`, `arange` | `uint index [[thread_position_in_grid]]` |
| 2D large | `unary_v2`, `copy_v2` | `uint2 index`, `int64_t` offset |
| 3D batch | `unary_g`, `binary_g` | `uint3` — x=inner, y=middle, z=outer |
| Threadgroup-local | Reductions, sort | `lid`, `simd_lane_id`, `simd_group_id` |
| Tiled 2D | GEMM, col_reduce | `gid` selects tile, `lid` within tile |

### Checklist: Writing a New Internal Kernel

1. **Define op functors** in `*_ops.h` — small structs with templated `operator()` + per-type specializations
2. **Write kernel templates** in `*.h` — contiguous (`_v`), 2D (`_v2`), general strided (`_g`), fixed-dim (`_nd1/2/3`)
3. **Write instantiation macros** in `*.metal` — `instantiate_kernel(name, func, ...)` with macro hierarchy for all type/layout combos
4. **Register with build** — add to `CMakeLists.txt` (non-JIT) + JIT source accessor
5. **Use infrastructure** — `utils.h`, `MLX_MTL_PRAGMA_UNROLL`, `WorkPerThread`, `Limits`, `threadgroup_barrier`
6. **For reductions** — two-level SIMD → threadgroup → SIMD with `shared_vals[32]`
7. **For tiled compute** — `BlockLoader` with `16/sizeof(T)` padding

### Advanced Patterns: Performance, Efficiency, Maintainability

Generally prefer readability, but if absolute performance is necessary, here are some ideas.

#### Double-Barrier Pattern in Tiled Kernels

The GEMM K-loop uses **two barriers per iteration**, not one. Each protects a different hazard:

```metal
for (int k = 0; k < K_iters; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);  // Barrier 1: wait for prior MMA reads
    loader_a.load_unsafe();  // overwrite As
    loader_b.load_unsafe();  // overwrite Bs
    threadgroup_barrier(mem_flags::mem_threadgroup);  // Barrier 2: wait for loads to finish
    mma_op.mma(As, Bs);     // read As, Bs
    loader_a.next();
    loader_b.next();
}
```

**Barrier 1** prevents loaders from overwriting `As`/`Bs` while lagging threads still read them from the previous MMA. **Barrier 2** prevents the MMA from reading partially-written tiles. Missing either causes silent data races. No kernel in MLX uses true double-buffering (two alternating physical buffers) — the single-buffer two-barrier pattern is universal.

#### `dispatch_bool` — Runtime Bools to Compile-Time Specialization

[`steel/utils/integral_constant.h:97-104`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/steel/utils/integral_constant.h#L97-L104): Converts a runtime `bool` into a compile-time type parameter, enabling dead-code elimination:

```metal
dispatch_bool(align_K, [&](auto kAlignedK) {
  dispatch_bool(align_M, [&](auto kAlignedM) {
    dispatch_bool(align_N, [&](auto kAlignedN) {
      // kAlignedM.value is constexpr — compiler eliminates dead branches
      gemm_loop<..., kAlignedM.value, kAlignedN.value, kAlignedK.value>(...)
```

Three nested `dispatch_bool` calls create 8 specializations. Each inner lambda receives a `true_type` or `false_type` whose `.value` is `constexpr`, so the compiler fully specializes each path. This is the standard pattern for alignment flags, transpose flags, and mask-type flags throughout the codebase.

#### `make_uniform` — Annotating Threadgroup-Uniform Values

[`gemv.metal:153-157`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/gemv.metal#L153-L157): `metal::make_uniform()` tells the GPU compiler a value is the same across all threads in the threadgroup, enabling better loop pipelining:

```metal
const uniform<int> in_size = make_uniform(in_vec_size);
const uniform<int> n_iter = in_size / loop_stride;
```

Arithmetic on `uniform<T>` propagates uniformity. Use for loop bounds derived from buffer sizes or kernel parameters.

#### K-Tail-First Trick in GEMM

[`steel_gemm_fused.h:133-163`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L133-L163): When K is not tile-aligned, the remainder is processed **before** the main loop:

```metal
if (!align_K) {
    loader_a.src += k_jump_a;  // jump to tail
    loader_b.src += k_jump_b;
    loader_a.load_safe(tile_dims);  // bounds-checked load
    loader_b.load_safe(tile_dims);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);            // accumulate tail
    loader_a.src -= k_jump_a;      // reset to start
    loader_b.src -= k_jump_b;
}
// Main loop now runs pure load_unsafe() — no bounds checks
```

The accumulator is not reset between tail and main loop — both contribute to the same result. This keeps the main loop body uniform and fully unrollable.

#### Index Type Parameterization (`_large` suffix)

Host-side code (`reduce.cpp:406`): `bool large = in.size() > INT32_MAX`. When true, appends `"_large"` to the kernel name, selecting an `int64_t`/`size_t` `IdxT` instantiation. The 32-bit path avoids 64-bit integer arithmetic overhead for the common case of small-to-medium tensors. All strided kernels support both via the `IdxT` template parameter.

#### Karatsuba Complex GEMM

[`mma.h:840-897`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/steel/gemm/mma.h#L840-L897): The `complex64_t` `BlockMMA` specialization uses 3 real MMAs instead of 4 via the Karatsuba identity:

```
P = Ar*Br        (MMA 1)
Q = Ai*Bi        (MMA 2)
R = (Ar+Ai)*(Br+Bi)  (MMA 3)
C_real += P - Q
C_imag += R - P - Q
```

Saves 25% MMA operations at the cost of extra additions. Two separate accumulator tiles (`Ctile_r`, `Ctile_i`) are maintained and recombined in `store_result`.

#### NaN Handling Conventions

Two distinct strategies:

**Sort** ([`sort.h:22-46`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/sort.h#L22-L46)): NaN sorts **last**. Padding uses `quiet_NaN()`. Comparator: `(!isnan(a)) & isnan(b)` — non-NaN always less than NaN.

**Reduce** ([`reduction/ops.h:193`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/reduction/ops.h#L193)): NaN **propagates**. `Min`/`Max` check `isnan(a) || isnan(b)` and return `NAN`. The `simd_reduce` path checks `simd_any(val != val)` (self-inequality = NaN) before calling `simd_min/max`, returning NaN immediately if any lane has it.

#### GEMV Register Blocking (TM/TN)

[`gemv.metal:28-36`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/gemv.metal#L28-L36): The 6-level hierarchy `BM×BN×SM×SN×TM×TN` controls register pressure directly:

- Each thread holds `TM` accumulator registers + `2*TN` input registers
- Inner loop is `TM × TN` multiply-accumulates, fully unrolled
- Typical configs: `TM=4,TN=4` (16 MACs/iteration, 12 registers)
- `TM=1` selected when `out_vec_size < 4` to reduce per-thread work
- Threadgroup memory only allocated when `BN > 1` (cross-SIMD-group reduction needed)

#### Epilogue Transforms in GEMM

[`transforms.h:14-54`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/steel/gemm/transforms.h#L14-L54): Three composable epilogues selected via `function_constant`:

| Transform | Formula | State |
|---|---|---|
| `TransformNone` | `D = cast(accumulator)` | Stateless |
| `TransformAdd` | `D = cast(acc) + C` | Stateless |
| `TransformAxpby` | `D = alpha*acc + beta*C` | `alpha`, `beta` |

Applied per-element after the K-loop completes, before writing to device memory. The `function_constant` `do_axpby` (index 110) selects between `TransformAdd` and `TransformAxpby` at pipeline compile time.

#### `LoopedElemToLoc` — Odometer-Style Strided Iterator

[`utils.h:201-299`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/utils.h#L201-L299): A recursive template that converts sequential element indices to strided offsets without recomputing from scratch. Maintains per-dimension `index` and `offset` state. `next()` increments the innermost dimension; on overflow, carries to the next (odometer pattern). The `DIM` template parameter fixes recursion depth at compile time for register allocation. Used in `reduce_row_looped` where the SIMD stride is 32:

```metal
loop.next(simd_size, reduce_shape, reduce_strides);  // skip 32 elements
```

#### Host-Side Dispatch Heuristics

The host C++ code selects kernel variants through string-name construction and size-based heuristics:

**Element-wise** (`unary.cpp:39-55`): `work_per_thread = 1` for arrays < 2^16 elements, else `8/itemsize`. `_large` suffix when > `UINT32_MAX`.

**Reductions** (`reduce.cpp:331-563`):
- `all_reduce`: Single threadgroup if `size <= N_READS * 1024`, else two-pass
- `row_reduce_small`: `row_size <= 64`
- `row_reduce_simple`: Contiguous, no non-row reductions, `rows >= 32`
- Threadgroup size: 32 for row ≤ 512; 128 for ≤ 1024; `ceil(row/N_READS)` otherwise

**GEMM** (`matmul.cpp:88-169`): Tile sizes selected by architecture suffix char (`'g'`/`'p'` = small GPU, `'d'` = ultra, default = medium) and transpose/dtype/size combinations. NAX path uses fixed 128×128×512 tiles.

**Quantized MV** (`quantized.cpp:1293-1352`): Routes through `qmv_quad` (K ≤ 128, power-of-2 bits), `qmv_fast` (K % 512 == 0 and N % 8 == 0), or `qmv` (general case).

#### JIT Caching Architecture

Two-level cache keyed by strings:

1. **Library cache** (`device.cpp:686-704`): `lib_name` → `MTL::Library*`. Source assembled by concatenating header strings + `get_template_definition()`. Double-checked locking with `std::shared_mutex`.
2. **Pipeline state cache** (`device.cpp:772-799`): `(lib, hash_name)` → `MTL::ComputePipelineState*`. The `hash_name` includes function constant values (e.g., alignment flags encoded as `'t'`/`'n'` chars).

`function_constant` values are set via `MTLFunctionConstantValues::setConstantValue(value, type, index)` at pipeline creation time (`device.cpp:576-616`). Each unique combination of function constants creates a separate PSO entry. GEMM uses 6 function constants (indices 10, 100, 110, 200, 201, 202).

#### Threadgroup Memory Sizing Conventions

| Kernel Family | Size Formula | Notes |
|---|---|---|
| GEMM | `BM * (BK + 16/sizeof(T))` | Padding avoids bank conflicts |
| Reduction (all) | `shared_vals[32]` | One slot per SIMD group (fixed) |
| Row reduce (multi-row) | `shared_vals[32 * N_WRITES]` | N_WRITES=4 simultaneous outputs |
| Col reduce (small) | `32 * 8 * n_reads` | 32 x-threads × 8 y-threads × n_reads |
| Col reduce (looped) | `BN * BM` (typically 32×32) | Full tile |
| GEMV | `BN > 1 ? BN*(blockM + TM) : 0` | Zero when no cross-SIMD reduction |

### Key Code References

All links are to [ml-explore/mlx](https://github.com/ml-explore/mlx) at commit `be872ebd`. Base path: `mlx/backend/metal/kernels/`.

**Infrastructure**:
- [`defines.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/defines.h) — `instantiate_kernel` (L22), constants
- [`utils.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/utils.h) — `WorkPerThread` (L17), `Limits` (L28), `elem_to_loc` (L97), SIMD wrappers (L357)
- [`atomic.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/atomic.h) — CAS-loop atomics for all types
- [`complex.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/complex.h) — `complex64_t` struct
- [`bf16.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/bf16.h) / [`bf16_math.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/bf16_math.h) — BFloat16 support
- [`fp8.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/fp8.h) / [`fp4.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/fp4.h) — Low-precision types

**Element-wise**:
- [`unary.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/unary.h) / [`unary_ops.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/unary_ops.h) / [`unary.metal`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/unary.metal)
- [`binary.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/binary.h) / [`binary_ops.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/binary_ops.h) / [`binary.metal`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/binary.metal)
- [`copy.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/copy.h) — Richest variant set (13×13 type cross-product)

**Reductions**:
- [`reduction/ops.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/reduction/ops.h) — Op structs + SIMD reduce
- [`reduction/reduce_all.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/reduction/reduce_all.h) / [`reduce_row.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/reduction/reduce_row.h) / [`reduce_col.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/reduction/reduce_col.h)
- [`softmax.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/softmax.h) — Online softmax
- [`layer_norm.metal`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/layer_norm.metal) / [`rms_norm.metal`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/rms_norm.metal)
- [`scan.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/scan.h) — Parallel prefix scan

**GEMM**:
- [`steel/gemm/gemm.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/steel/gemm/gemm.h) — GEMMKernel
- [`steel/gemm/loader.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/steel/gemm/loader.h) — BlockLoader
- [`steel/gemm/mma.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/steel/gemm/mma.h) — BaseMMAFrag, BlockMMA
- [`steel/gemm/transforms.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/steel/gemm/transforms.h) — BlockSwizzle
- [`gemv.metal`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/gemv.metal) — Matrix-vector multiply
- [`quantized.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/quantized.h) — Quantized GEMM/GEMV

**Specialized**:
- [`sort.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/sort.h) — GPU merge sort
- [`fft/radix.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/fft/radix.h) — FFT radix codelets
- [`hadamard.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/hadamard.h) — Walsh-Hadamard transform
- [`sdpa_vector.h`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/sdpa_vector.h) — Attention
- [`conv.metal`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/conv.metal) — Convolution
- [`fence.metal`](https://github.com/ml-explore/mlx/blob/be872ebd/mlx/backend/metal/kernels/fence.metal) — System-scope sync
