---
name: jax-pallas-kernels
description: Write JAX Pallas GPU kernels for custom CUDA/Triton computation. Use when writing Pallas kernels, optimizing JAX GPU code beyond what XLA auto-generates, or when the user mentions Pallas, custom kernels, Triton, or Mosaic GPU. Covers kernel patterns, BlockSpec, grid design, memory management, and backend constraints.
---

# JAX Pallas Kernels

## When to Use Pallas (and When NOT To)

Pallas writes custom GPU kernels in JAX. But XLA already generates excellent code for most patterns.

**Use Pallas when:**
- XLA generates suboptimal kernels (verify with profiler first)
- You need explicit shared memory (SMEM) control
- Custom memory access patterns (tiling, swizzling)
- Hardware-specific features (Hopper WGMMA, Blackwell TMEM)
- Fusing operations XLA keeps as separate kernels
- Non-standard reductions or scan patterns

**Do NOT use Pallas when:**
- `jax.vmap` + `lax.scan` already performs well (profile first!)
- The workload is embarrassingly parallel and XLA handles it
- You need complex number support on Triton backend (decompose manually)
- The problem is memory-bound and tiling won't help

**Practical benchmark result (RTX A4000, SM 86):** For a frequency-sweep loop (sequential scan over 256-512 freqs, reducing 128-256 sources per grid point), `jax.vmap` + `lax.scan` was 1.6-3x FASTER than an equivalent Pallas/Triton kernel. XLA's scan optimization is hard to beat for this pattern.

## Backend Selection

| Backend | GPU Requirement | Env Var | Strengths |
|---------|----------------|---------|-----------|
| **Mosaic GPU** | SM 90+ (Hopper/Blackwell) | Default in JAX 0.9+ | WGMMA, SMEM control, pipelining |
| **Triton** | SM 70+ (Volta+) | `JAX_PALLAS_USE_MOSAIC_GPU=0` | Wider GPU support, auto-tuning |
| **interpret** | CPU (debug only) | `interpret=True` in pallas_call | Correctness validation |

**Critical:** Check your GPU's compute capability first:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

## Core API

```python
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu  # Hopper+ only
import jax
import jax.numpy as jnp
```

### Minimal Kernel

```python
def add_kernel(x_ref, y_ref, o_ref):
    """Refs are mutable memory buffers, not arrays."""
    o_ref[...] = x_ref[...] + y_ref[...]

result = pl.pallas_call(
    add_kernel,
    out_shape=jax.ShapeDtypeStruct((N,), jnp.float32),
    grid=(N // BLOCK,),
    in_specs=[
        pl.BlockSpec((BLOCK,), lambda i: (i,)),
        pl.BlockSpec((BLOCK,), lambda i: (i,)),
    ],
    out_specs=pl.BlockSpec((BLOCK,), lambda i: (i,)),
)(x, y)
```

### BlockSpec

Maps grid indices to data tiles. Each program processes one tile.

```python
pl.BlockSpec(
    block_shape=(TILE_M, TILE_K),    # tile dimensions
    index_map=lambda i, j: (i, 0),   # grid_idx -> block_idx
)
```

- `block_shape=None` -> full array
- `index_map=None` -> always index (0, 0, ...)
- Block index is multiplied by block_shape to get element offset
- Out-of-bounds reads are padded; out-of-bounds writes are discarded

### Grid

```python
grid=(M // TILE_M, N // TILE_N)
```

Inside kernel: `pl.program_id(axis=0)` returns current grid index.

**Triton:** All grid dims are parallel (CUDA grid).
**Mosaic GPU:** Use `compiler_params` to mark sequential dims for pipelining.

## Dtype Constraints

### Triton Backend

- **No complex dtype support.** Decompose complex64/128 into real/imaginary pairs:

```python
def kernel_real(ph_re_ref, ph_im_ref, step_re_ref, step_im_ref, out_ref):
    """Manual complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
    a, b = ph_re_ref[...], ph_im_ref[...]
    c, d = step_re_ref[...], step_im_ref[...]
    # phase *= step
    new_re = a * c - b * d
    new_im = a * d + b * c
    out_ref[...] = new_re * new_re + new_im * new_im  # |phase|^2
```

- Block shapes: each dim should be power-of-2
- Supported: float16, bfloat16, float32, int32, int64

### Mosaic GPU Backend

- Minormost block dimension must be multiple of 16 bytes
- e.g., 8 elements for float16, 4 for float32
- Complex types may work depending on JAX version

## Block Size Alignment

| Backend | Constraint |
|---------|-----------|
| Triton | Power-of-2 per dimension |
| Mosaic GPU | Minormost dim: multiple of 16 bytes (128 bytes for async copies) |

Minimum transfer size for Mosaic GPU async copies: **128 bytes** (warpgroup size).

## Patterns

### Parallel Elementwise

```python
def relu_kernel(x_ref, o_ref):
    o_ref[...] = jnp.maximum(x_ref[...], 0.0)

pl.pallas_call(relu_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid=(x.shape[0] // BLOCK,),
    in_specs=[pl.BlockSpec((BLOCK,), lambda i: (i,))],
    out_specs=pl.BlockSpec((BLOCK,), lambda i: (i,)),
)(x)
```

### Tiled Matmul

```python
def matmul_kernel(a_ref, b_ref, c_ref):
    c_ref[...] = a_ref[...] @ b_ref[...]

pl.pallas_call(matmul_kernel,
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    grid=(M // TM, N // TN),
    in_specs=[
        pl.BlockSpec((TM, K), lambda i, j: (i, 0)),
        pl.BlockSpec((K, TN), lambda i, j: (0, j)),
    ],
    out_specs=pl.BlockSpec((TM, TN), lambda i, j: (i, j)),
)(a, b)
```

### Sequential Reduction (fori_loop inside kernel)

When one dimension must be sequential (e.g., accumulating weighted sums):

```python
def weighted_sum_kernel(x_ref, weights_ref, out_ref):
    """Accumulate weighted L2 norms across the weights dimension."""
    x = x_ref[...]           # [TILE, D]
    n = weights_ref.shape[0]

    def body(i, acc):
        weighted = weights_ref[i] * jnp.sum(x, axis=-1)
        return acc + weighted * weighted

    out_ref[...] = jax.lax.fori_loop(0, n, body, jnp.zeros(x.shape[0]))
```

**Note:** For this pattern, compare against `jax.vmap` + `lax.scan` -- XLA often wins.

### Scratch Buffers (Mosaic GPU)

For kernels needing temporary SMEM or accumulators beyond input/output tiles:

```python
plgpu.kernel(
    kernel_fn, out_shape=..., grid=...,
    scratch_shapes=dict(
        smem_buf=plgpu.SMEM((128, 128), jnp.float16),
        acc=plgpu.ACC((128, 128), jnp.float32),   # Hopper WGMMA accumulator
    ),
)(inputs)
```

### Reduction Initialization (Pipelining)

When using grid-based reduction with Mosaic GPU pipelining:

```python
def sum_kernel(x_ref, o_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)
    o_ref[...] += x_ref[...]
```

Reduction must be on the **innermost (last)** grid dimension.

## Mosaic GPU Specifics (SM 90+)

### Memory Spaces

```python
plgpu.GMEM   # Global/HBM memory
plgpu.SMEM   # Shared memory per SM
plgpu.ACC    # WGMMA accumulator (Hopper)
plgpu.TMEM   # Tensor memory (Blackwell)
```

### Pipelining

```python
pipeline = plgpu.emit_pipeline(
    body_fn,
    grid=(grid_k,),
    in_specs=[plgpu.BlockSpec(...)],
    out_specs=[...],
    max_concurrent_steps=2,  # 2 for compute-heavy, 4-6 for memory-heavy
)
```

### Async Copies

```python
plgpu.copy_gmem_to_smem(gmem_ref, smem_ref, barrier)
plgpu.barrier_wait(barrier)
# ... compute ...
plgpu.commit_smem()  # Required before async ops after SMEM writes
plgpu.copy_smem_to_gmem(smem_ref, gmem_ref)
plgpu.wait_smem_to_gmem(0)
```

### L2 Cache Optimization (Grid Tiling)

Rearrange grid for L2-friendly access:
```python
grid = (m_iters // mt, n_iters, mt)
# In kernel: pid_m = pl.program_id(0) * mt + pl.program_id(2)
```

## Debugging

```python
# CPU emulation mode
pl.pallas_call(..., interpret=True)(inputs)

# Print from inside kernel
pl.debug_print("value: {}", some_ref[0])
```

### Environment Variables

```bash
MOSAIC_GPU_DUMP_PTXAS=1    # Register counts, spills (critical!)
MOSAIC_GPU_DUMP_PTX=1      # PTX assembly
MOSAIC_GPU_DUMP_SASS=1     # Final SASS
JAX_PALLAS_USE_MOSAIC_GPU=0  # Force Triton backend
```

## SFU Bottleneck: Avoid sin/cos/exp in Inner Loops

Special Function Units (SFU) have 1/8th the throughput of ALU on both NVIDIA and Apple Silicon. Kernels with sin/cos/exp in the inner loop are SFU-bound.

**Anti-pattern:** Direct phase computation per element per frequency:
```python
# BAD: 4 SFU calls per element per frequency
phase = jnp.exp(-attenuation * dist) * jnp.cos(wavenumber * dist)
```

**Pattern:** Geometric progression (0 SFU in inner loop):
```python
# GOOD: Init once (SFU cost), then ALU-only complex multiply
cur_re, cur_im = init_re, init_im  # SFU cost paid once
for freq in range(n_freq):
    # Complex multiply = 4 ALU ops, 0 SFU
    new_re = cur_re * stp_re - cur_im * stp_im
    new_im = cur_re * stp_im + cur_im * stp_re
    cur_re, cur_im = new_re, new_im
```

This pattern gave a **3.6x speedup** in the FastSIMUS Metal tiled TX kernel.

## Element Tiling for Register Pressure

Process sub-elements in tiles to stay within register limits. The FastSIMUS Metal kernel tiles 64 sub-elements into groups of 16 (256 bytes of registers per tile -- no spills).

**Register budget per architecture:**
| GPU | Max Registers/Thread | Bytes | Safe budget (50% occupancy) |
|-----|---------------------|-------|----------------------------|
| NVIDIA Ampere (SM 86) | 255 x 32-bit | 1020B | ~512B |
| NVIDIA Hopper (SM 90) | 255 x 32-bit | 1020B | ~512B |
| Apple Silicon | 256 x 16-bit | 512B | ~256B |

**Pattern:** Tile elements, init phase per tile (SFU cost), sweep via ALU-only multiply:
```python
TILE_SE = 16  # sub-elements per tile
for tile_start in range(0, n_sub, TILE_SE):
    # Init cur/stp for this tile (SFU cost, paid once per tile)
    # Sweep frequencies via complex multiply (0 SFU)
    # Accumulate result
```

## Common Pitfalls

1. **Complex dtypes on Triton** -- decompose into real/imag pairs
2. **Small blocks on Mosaic GPU** -- minimum 128 bytes per async copy
3. **Register spills** -- check `MOSAIC_GPU_DUMP_PTXAS=1`, reduce per-thread state
4. **SFU bottleneck** -- sin/cos/exp in inner loop = 1/8th ALU throughput; use geometric progression
5. **Missing `commit_smem()`** -- silent data races between SMEM writes and async ops
6. **Barrier double-completion** -- never complete a barrier twice without waiting
7. **Layout conflicts** -- arrays with different register layouts can't be added directly
8. **Reduction on wrong axis** -- must be innermost grid dim for pipelining
9. **Not profiling first** -- always benchmark vs XLA baseline before writing Pallas
10. **`delay_release` omission** -- buffers overwritten while WGMMA still reading

## Cross-Platform Design (Metal + CUDA/Pallas)

The element-tiled progression algorithm maps directly across platforms:

| Concept | Metal | CUDA/Pallas |
|---------|-------|-------------|
| Shared memory | `threadgroup float*` | `__shared__` / `plgpu.SMEM` |
| Fast trig init | `metal::fast::cos/sin` | `__sincosf()` / `jnp.cos/sin` |
| Barrier | `threadgroup_barrier()` | `__syncthreads()` / `plgpu.barrier_wait` |
| SIMD shuffle | `simd_shuffle_xor` | `__shfl_xor_sync` |
| Atomic add | `atomic_fetch_add_explicit` | `atomicAdd` / `pl.atomic_add` |

**Production-tested results** (FastSIMUS, P4-2v transducer, 812 frequencies):
- Metal tiled TX: 7M scat/s at N=10K (3.6x over progression)
- Metal SIMD-reduce RX: 2.49M scat/s end-to-end (25% over baseline)
- Per-TFLOP: within 1.3x of FieldGPU (time-domain CUDA simulator)

## Autotuning Block Sizes

Block size is the single most important performance parameter. Always sweep:

```python
import itertools

best_time = float('inf')
for tile_g in [64, 128, 256, 512]:
    for tile_src in [32, 64, 128]:
        t = benchmark(tile_g, tile_src)
        if t < best_time:
            best_time = t
            best_config = (tile_g, tile_src)
```

## Composability

Pallas kernels compose with JAX transformations:
- `jax.jit` -- JIT-compile the outer function containing `pallas_call`
- `jax.vmap` -- batch over an additional dimension
- `jax.grad` -- differentiate through Pallas kernels (limited support)

## Environment Compatibility

### cuDNN + Driver Version

JAX's XLA GPU path requires cuDNN, which requires specific NVIDIA driver versions. The Pallas/Triton path bypasses cuDNN, so Pallas kernels may work even when standard JAX GPU ops fail:

```
# cuDNN error on jnp.ones() but Pallas/Triton works:
Could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED
Possibly insufficient driver version: 535.288.1
```

**Workaround:** Pallas/Triton custom kernels don't use cuDNN for their ops, so they may work when standard JAX GPU ops fail. The real fix is matching your JAX version to your driver (see [JAX CUDA compatibility](https://github.com/jax-ml/jax#pip-installation-gpu-cuda-installed-via-pip-easier)).

### Compute Capability Requirements

| Feature | Minimum CC |
|---------|-----------|
| Triton backend | SM 70 (Volta) |
| Mosaic GPU backend | SM 90 (Hopper) |
| WGMMA TensorCore | SM 90 (Hopper) |
| TMEM, tcgen05 | SM 100 (Blackwell) |
| Cluster thread blocks | SM 90 (Hopper) |
