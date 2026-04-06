---
name: gpu-profiling
description: Profile and optimize CUDA/JAX/Pallas GPU code using Nsight Systems, Nsight Compute, and JAX profiler. Use when profiling GPU performance, identifying bottlenecks, optimizing kernels, or when the user mentions nsight, nvprof, profiling, roofline, occupancy, or GPU optimization.
---

# GPU Profiling for JAX/Pallas Code

## Profiling Workflow (Always Follow This Order)

1. **JAX profiler** -- understand XLA graph structure, kernel launches
2. **nsys** (Nsight Systems) -- system timeline, find which kernels are slow
3. **ncu** (Nsight Compute) -- deep dive into the slow kernel
4. **Iterate** -- optimize, re-profile, compare

Never start with ncu (too slow for full app). Never skip step 1 (JAX async dispatch hides real timings).

## Step 0: Accurate Timing

```python
import time
import jax

def benchmark_gpu(fn, *args, n_warmup=5, n_runs=20):
    for _ in range(n_warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)  # CRITICAL: JAX dispatch is async
        times.append(time.perf_counter() - t0)

    times.sort()
    return times[len(times) // 2] * 1000  # median ms
```

**Rules:**
- Always `.block_until_ready()` before timing
- Always warmup (first calls include JIT compilation)
- Use consistent input shapes (varying shapes = recompilation)

## Step 1: JAX Profiler

```python
import jax

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    result = fn(inputs)
    result.block_until_ready()
```

View with XProf (recommended over TensorBoard; check [JAX profiling docs](https://docs.jax.dev/en/latest/profiling.html) for current package names):
```bash
pip install xprof  # or tensorboard-plugin-profile
xprof --port 8791 /tmp/jax-trace
```

XProf tools: trace_viewer, **Roofline Model**, HLO Op Stats, Memory Profile.

### NVTX Annotations

```python
with jax.profiler.TraceAnnotation("pfield_sweep"):
    result = pfield_compute(...)
```

### Useful JAX Debug Variables

```bash
JAX_LOG_COMPILES=1              # Print when JIT compiles
JAX_DEBUG_NANS=True             # Check for NaNs (slow)
XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/xla_dump"
```

## Step 2: Nsight Systems (nsys)

System-wide timeline -- find WHICH kernels are slow.

```bash
nsys profile --stats=true \
  --trace=cuda,nvtx \
  --python-backtrace=cuda \
  -o /tmp/nsys_output \
  python benchmark.py

nsys stats /tmp/nsys_output.nsys-rep
```

**What to look for:**
- Kernel launch gaps (CPU overhead between kernels)
- Host-device memory copies (`cudaMemcpy`)
- Kernel durations (longest kernels = optimization targets)
- CPU-GPU overlap (or lack thereof)

## Step 3: Nsight Compute (ncu)

Per-kernel deep profile -- find WHY a kernel is slow.

```bash
# List all kernels
ncu --list-kernels python benchmark.py

# Quick classification (is it compute-bound or memory-bound?)
ncu --set basic -o /tmp/ncu_basic python benchmark.py

# Full analysis of specific kernel
ncu --kernel-name "KERNEL_NAME" --set full -o /tmp/ncu_full python benchmark.py

# Triton/Pallas kernel pattern matching
ncu --kernel-name-base "triton_" --set full python benchmark.py
```

### Section Sets

| Set | Overhead | Use Case |
|-----|----------|----------|
| `--set basic` | Low | Quick: compute vs memory bound |
| `--set detailed` | Medium | Standard analysis with roofline |
| `--set full` | High | Deep investigation with source correlation |

### Key Metrics

```bash
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
  l1tex__data_bank_conflicts_pipe_lsu.sum,\
  l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum \
  python benchmark.py
```

## Bottleneck Classification

| SM Throughput | DRAM Throughput | Type | Action |
|--------------|----------------|------|--------|
| High (>60%) | Low | **Compute-bound** | Reduce SFU ops, use FMA, check instruction mix |
| Low | High (>60%) | **Memory-bound** | Optimize access patterns, reduce transfers, tile for cache |
| Low | Low | **Latency-bound** | Increase occupancy, reduce stalls, overlap compute/memory |
| High | High | **Balanced** | Near peak -- check roofline for remaining gaps |

**Confirm with warp stall reasons:**
- `stalled_long_scoreboard` -> memory-bound (waiting for DRAM/L2)
- `stalled_math_pipe_throttle` -> compute-bound (ALU saturated)
- `stalled_wait` -> synchronization-bound
- `stalled_not_selected` -> scheduler contention

## Register Spill Detection

Register spills kill performance (store to device memory instead of registers).

```bash
# Pallas/Mosaic GPU
MOSAIC_GPU_DUMP_PTXAS=1 python benchmark.py 2>&1 | grep -i spill

# Triton
PTXAS_OPTIONS="-v" python benchmark.py 2>&1 | grep -i "spill\|Used.*registers"
```

Look for:
```
ptxas info: Used 64 registers, 0 bytes smem
ptxas info: Spill stores: 16 bytes, Spill loads: 16 bytes  # BAD
```

**Fixes:**
1. Reduce per-thread state (use tiling with smaller tiles)
2. Reduce loop unroll factor
3. Trade occupancy for fewer spills (fewer warps = more registers/warp)

## Roofline Analysis

```bash
# Via ncu
ncu --section SpeedOfLight_RooflineChart -o /tmp/roofline python benchmark.py
# Open .ncu-rep in Nsight Compute GUI

# Via JAX profiler
# Use xprof -> Tools -> Roofline Model
```

**Interpretation:**
- Below sloped line (memory roof) -> memory-bound
- Below flat line (compute roof) -> compute-bound
- At ridge point -> balanced utilization

## Common Anti-Patterns

| Anti-Pattern | Detection | Fix |
|-------------|-----------|-----|
| SFU bottleneck (sin/cos/exp) | High `inst_executed_op_generic` in ncu; count trig calls in source | Geometric progression (ALU-only multiply). **3.6x speedup** in FastSIMUS tiled kernel |
| Uncoalesced global access | Low DRAM efficiency in ncu | Adjacent threads -> adjacent addresses |
| Register spilling | `MOSAIC_GPU_DUMP_PTXAS=1` shows spills; >512B per thread | Tile elements (TILE=16), reduce per-thread state |
| Atomic write contention | High `l2_atomic_throughput`, RX-dominant profile | SIMD shuffle to reduce before atomic (SR=2 halved atomics, 25% speedup) |
| Small kernel launches | Many short kernels in nsys | Batch into larger kernels, CUDA graphs |
| Host-device transfers | `cudaMemcpy` in nsys timeline | Keep data on device |
| Missing block_until_ready | Timing measures dispatch, not execution | Always sync before timing |
| Shared memory bank conflicts | `l1tex__data_bank_conflicts` > 0 | Pad arrays (+1 element per row) |
| Repeated JIT compilation | Multiple compile events in JAX log | Consistent input shapes/dtypes |
| Low occupancy | Few active warps | Reduce registers/thread, reduce smem/block |

### Register Pressure Estimation

Before profiling, estimate register usage from your kernel source:

```
bytes_per_thread = num_live_variables * sizeof(dtype)
# float32: 4 bytes, float2 (complex): 8 bytes
# Example: float2[64] = 1024 bytes per thread -> guaranteed spills on NVIDIA (max ~1020B)
```

**Rule of thumb:** Keep per-thread state under 512 bytes for good occupancy on NVIDIA, 256 bytes on Apple Silicon. Use element tiling (TILE_SE=16) to process arrays in register-sized chunks.

## Pallas-Specific Profiling

### Triton Backend (SM 70+)

```bash
# Force Triton backend
JAX_PALLAS_USE_MOSAIC_GPU=0 python benchmark.py

# Triton compilation output
TRITON_PRINT_AUTOTUNING=1 python benchmark.py

# Inspect generated PTX
TRITON_CACHE_DIR=/tmp/triton_cache python benchmark.py
# Then inspect /tmp/triton_cache/*/*.ptx
```

### Mosaic GPU Backend (SM 90+)

```bash
MOSAIC_GPU_DUMP_PTXAS=1 python benchmark.py  # Register counts
MOSAIC_GPU_DUMP_PTX=1 python benchmark.py    # PTX code
MOSAIC_GPU_DUMP_SASS=1 python benchmark.py   # SASS assembly
```

### Debugging with interpret mode

```python
result = pl.pallas_call(..., interpret=True)(inputs)
```

Runs on CPU -- validates kernel logic without GPU compilation.

## Profiling Checklist

1. [ ] Establish baseline timing with `block_until_ready()` + warmup
2. [ ] JAX profiler trace to understand XLA graph
3. [ ] nsys timeline to find slow kernels and CPU-GPU gaps
4. [ ] ncu `--set basic` to classify compute vs memory bound
5. [ ] Check register spills (`MOSAIC_GPU_DUMP_PTXAS=1`)
6. [ ] ncu roofline to see distance from peak
7. [ ] If memory-bound: check access patterns, L2 hit rate, bank conflicts
8. [ ] If compute-bound: check instruction mix (FMA vs SFU), warp divergence
9. [ ] Optimize, re-profile, compare against baseline
10. [ ] Verify correctness after optimization (rtol=1e-4 vs reference)

## Quick Reference: ncu Sections

```bash
ncu --section SpeedOfLight           # Compute vs memory bound
ncu --section Occupancy              # Occupancy limiters
ncu --section MemoryWorkloadAnalysis # Memory bottleneck details
ncu --section WarpStateStats         # Stall reasons
ncu --section InstructionStats       # Instruction mix
ncu --section SourceCounters         # Per-line metrics, branch efficiency
ncu --section SpeedOfLight_RooflineChart  # Roofline
```

## Troubleshooting

### cuDNN Initialization Failure

```
Could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED
Possibly insufficient driver version: 535.288.1
```

**Cause:** JAX's bundled cuDNN requires a newer NVIDIA driver than installed.

**Fix:** Match JAX version to your driver (see [JAX CUDA compatibility](https://github.com/jax-ml/jax#pip-installation-gpu-cuda-installed-via-pip-easier)).

**Workarounds:**
1. `JAX_PLATFORMS=cpu` -- force CPU backend (loses GPU)
2. Pallas/Triton custom kernels don't use cuDNN for their ops (`JAX_PALLAS_USE_MOSAIC_GPU=0`)
3. Downgrade JAX to match driver version

### nsight Tools Not Installed

If nsys/ncu aren't available, use JAX's built-in profiler:
```python
with jax.profiler.trace("/tmp/trace"):
    result.block_until_ready()
# View with: pip install xprof && xprof --port 8791 /tmp/trace
```

## FastSIMUS-Specific Notes

- **pfield frequency sweep**: `lax.scan` outperforms Pallas/Triton for sequential loops (tested on RTX A4000). Focus Pallas on embarrassingly parallel kernels.
- **Complex numbers**: Triton backend doesn't support complex64. Decompose into real/imaginary pairs.
- **SFU bottleneck**: The pfield inner loop uses `exp`, `sin`, `cos` heavily. Replace with geometric progression (multiply instead of exp) to avoid SFU throughput limits.
- **Benchmark command**: `uv run poe benchmark`
- **Target**: 50-100x speedup over NumPy baseline
