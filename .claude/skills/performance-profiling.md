# Performance Profiling (FastSIMUS-specific)

## When to Use

- Profiling ultrasound simulation performance in FastSIMUS
- Optimizing JAX/CuPy code for GPU acceleration
- Benchmarking against PyMUST NumPy baseline
- Working toward the 50-100x speedup target

## Overview

FastSIMUS targets 50-100x speedup over NumPy baseline via JAX/CuPy acceleration. This guide covers profiling tools and
patterns specific to the FastSIMUS project.

## Quick Profiling

### Time a Function

```python
import time

start = time.perf_counter()
result = simus(x, z, rc, delays, params)
elapsed = time.perf_counter() - start
print(f"simus: {elapsed:.3f}s")
```

### Use pytest-benchmark

```python
def test_pfield_benchmark(benchmark):
    """Benchmark pfield computation."""
    result = benchmark(pfield, x, z, delays, params)
    assert result.shape[0] > 0
```

Run benchmarks:

```bash
uv run poe benchmark                    # Run all benchmarks
uv run pytest --benchmark-only          # Just benchmarks
uv run pytest --benchmark-autosave      # Save results
uv run pytest --benchmark-compare       # Compare with saved
```

## Detailed Profiling

### cProfile (Overall)

```bash
python -m cProfile -s cumulative -m fast_simus.examples.benchmark
```

Or in code:

```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    result = simus(x, z, rc, delays, params)

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### line_profiler (Line-by-Line)

Install: `uv add line_profiler --dev`

Decorate functions:

```python
@profile  # Add this decorator
def frequency_loop(exp, exp_df, n_freq):
    ...
```

Run:

```bash
kernprof -l -v script.py
```

### memory_profiler

Install: `uv add memory_profiler --dev`

```python
from memory_profiler import profile

@profile
def simus(x, z, rc, delays, params):
    ...
```

Run:

```bash
python -m memory_profiler script.py
```

## JAX-Specific Profiling

### JIT Compilation Time

```python
import jax
from jax import jit

@jit
def pfield_jax(x, z, delays, params):
    ...

# First call includes compilation
start = time.perf_counter()
result = pfield_jax(x, z, delays, params).block_until_ready()
compile_time = time.perf_counter() - start
print(f"First call (with compile): {compile_time:.3f}s")

# Subsequent calls are faster
start = time.perf_counter()
result = pfield_jax(x, z, delays, params).block_until_ready()
run_time = time.perf_counter() - start
print(f"Second call (compiled): {run_time:.3f}s")
```

**Important:** Always use `.block_until_ready()` for accurate timing with JAX.

### JAX Profiler (GPU)

```python
import jax

# Profile to TensorBoard format
with jax.profiler.trace("/tmp/jax-trace"):
    result = pfield_jax(x, z, delays, params).block_until_ready()

# View in TensorBoard
# tensorboard --logdir=/tmp/jax-trace
```

### Check Device Placement

```python
import jax

print(f"Default backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# Check where array lives
x_jax = jax.numpy.array(x)
print(f"Array device: {x_jax.device()}")
```

## CuPy-Specific Profiling

### CUDA Events

```python
import cupy as cp

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()
result = pfield_cupy(x, z, delays, params)
end.record()
end.synchronize()

elapsed_ms = cp.cuda.get_elapsed_time(start, end)
print(f"GPU time: {elapsed_ms:.2f}ms")
```

### nvprof / Nsight

```bash
nvprof python script.py
# or
nsys profile python script.py
```

## Benchmark Patterns

### Compare Backends

```python
import numpy as np
import jax.numpy as jnp

def benchmark_backends(x_np, z_np, rc_np, delays_np, params, n_runs=5):
    """Compare performance across backends."""

    results = {}

    # NumPy baseline
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = simus(x_np, z_np, rc_np, delays_np, params)
        times.append(time.perf_counter() - start)
    results['numpy'] = np.median(times)

    # JAX
    x_jax = jnp.array(x_np)
    z_jax = jnp.array(z_np)
    rc_jax = jnp.array(rc_np)
    delays_jax = jnp.array(delays_np)

    # Warmup (compile)
    _ = simus(x_jax, z_jax, rc_jax, delays_jax, params)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = simus(x_jax, z_jax, rc_jax, delays_jax, params)
        result.block_until_ready()
        times.append(time.perf_counter() - start)
    results['jax'] = np.median(times)

    # Print summary
    baseline = results['numpy']
    for backend, t in results.items():
        speedup = baseline / t
        print(f"{backend:10s}: {t:.3f}s ({speedup:.1f}x)")

    return results
```

### Standard Benchmark Configuration

Use consistent problem sizes for comparison:

```python
# Small (unit tests)
SMALL = dict(n_points=100, n_elements=64, n_freq=100)

# Medium (benchmarks)
MEDIUM = dict(n_points=10_000, n_elements=128, n_freq=555)

# Large (stress tests)
LARGE = dict(n_points=100_000, n_elements=256, n_freq=555)
```

## Performance Targets

| Configuration    | NumPy | JAX CPU   | JAX GPU | Speedup |
| ---------------- | ----- | --------- | ------- | ------- |
| MEDIUM (10k pts) | 1.5s  | 100-150ms | 15-30ms | 50-100x |

## Common Bottlenecks

### 1. Python Loop Overhead

**Symptom:** Similar time for NumPy and JAX CPU **Fix:** Use `jax.lax.scan` or `jax.lax.fori_loop`

### 2. Repeated Compilation

**Symptom:** JAX slow on every call **Fix:** Ensure inputs have consistent shapes/dtypes

### 3. Host-Device Transfer

**Symptom:** GPU not faster than CPU **Fix:** Keep data on GPU, minimize `.block_until_ready()` calls mid-computation

### 4. Memory Bandwidth

**Symptom:** GPU utilization low **Fix:** Batch operations, use `jax.vmap` for multiple simulations

## Profiling Checklist

1. [ ] Establish NumPy baseline with `pytest-benchmark`
1. [ ] Profile with `cProfile` to find bottlenecks
1. [ ] Use `line_profiler` on hot functions
1. [ ] Measure JAX compile time separately from runtime
1. [ ] Use `.block_until_ready()` for accurate JAX timing
1. [ ] Check GPU utilization with `nvidia-smi` or Nsight
1. [ ] Compare against target speedups (10x CPU, 50-100x GPU)
