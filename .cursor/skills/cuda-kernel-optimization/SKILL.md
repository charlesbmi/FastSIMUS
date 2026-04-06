---
name: cuda-kernel-optimization
description: Systematic CUDA kernel optimization using Nsight Compute profiling. Use when writing, profiling, or optimizing CUDA kernels compiled with NVRTC and launched via the CUDA Driver API. Covers the full cycle from profiling to bottleneck analysis to architecture decisions.
---

# CUDA Kernel Optimization for FastSIMUS

## Profiling Setup

### Nsight Compute (ncu)

Binary: `/usr/local/cuda-12.2/bin/ncu` (version 2023.2.2)
Requires `sudo` for GPU performance counters on this system.

```bash
sudo /usr/local/cuda-12.2/bin/ncu \
  --target-processes all \
  --launch-skip 1 --launch-count 1 \
  --set full \
  -f -o /path/to/report \
  $(which uv) run python tools/ncu_profile.py <kernel.cu> [options]
```

- `--target-processes all`: profile kernels from child processes (ctypes loads libcuda directly)
- `--launch-skip 1 --launch-count 1`: skip warmup, profile second launch only
- `--set full`: collect all metrics (34 replay passes, ~10-30s per kernel)
- Report extraction: `sudo ncu --import report.ncu-rep --page raw 2>&1 | uv run python tools/ncu_parse.py`

### Profiling Harness

`tools/ncu_profile.py` -- parameterized kernel launcher:
```bash
uv run python tools/ncu_profile.py <kernel.cu> --b-scat N --elem-tile N --blocks N --n-scat N
```

`tools/ncu_parse.py` -- extracts key metrics from ncu raw output into markdown tables.

### GPU Clock Locking (for consistent benchmarks)

```bash
sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1560
```
Must re-run after reboot. Verify: `nvidia-smi -q -d CLOCK | grep -A3 "Max Clocks"`.

## Key Metrics to Extract

### Primary Bottleneck Indicators

| Metric | What it tells you | Healthy range |
| --- | --- | --- |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM utilization | >70% for well-optimized |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` | FMA compute saturation | >60% for compute-bound |
| `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed` | Memory pipe pressure | <30% for compute-bound |
| `lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed` | L2 atomic contention | <10% is good, >25% is problem |
| `sm__inst_executed.avg.per_cycle_elapsed` | Instructions per cycle | >2.0 on Ampere is good |

### Occupancy Limiters

Check ALL three:
- `launch__occupancy_limit_registers` -- register-limited blocks/SM
- `launch__occupancy_limit_shared_mem` -- shmem-limited blocks/SM
- `launch__occupancy_limit_warps` -- warp-limited blocks/SM

The MINIMUM of these is the binding constraint.

### Warp Stall Reasons (most informative metric)

| Stall | Root cause | Fix |
| --- | --- | --- |
| `wait` >0.5 | Data dependency on shmem/register | Increase ILP, overlap compute |
| `long_scoreboard` >0.5 | Waiting for L2/global memory | Reduce atomics, reduce spills |
| `barrier` >0.5 | __syncthreads overhead | Reduce sync points |
| `mio_throttle` >0.1 | Memory I/O queue full | Reduce memory traffic |
| `lg_throttle` >0.1 | L2 write queue full | Reduce atomic/store volume |
| `not_selected` >1.0 | Too many eligible warps | Normal at high occupancy |

## Proven Optimization Principles (from experiments 1-10)

### 1. Occupancy is NOT the primary lever for scatter-gather kernels

v6 had 42% occupancy but was 2x slower than v11 at 17%. More warps hitting the
same L2 atomic unit creates MORE contention, not less. Occupancy matters only
when the kernel is latency-bound with insufficient warps to hide latency.

### 2. The blocks-per-SM cliff is absolute

Going from 2 to 1 block/SM (the "occupancy cliff") drops performance 45%. All
optimizations MUST maintain the current blocks/SM or improve it.

### 3. Register reduction backfires when shmem binds occupancy

`__launch_bounds__` forces the compiler to spill registers to local memory (L1 cache).
If shared memory is the occupancy limiter (not registers), the spills add latency
with zero occupancy benefit. Only reduce registers when registers are the binding limiter.

### 4. Atomic batching scales sub-linearly

B_SCAT=5 batches 5 scatterers before atomicAdd, reducing atomics by 5x vs v6.
But L2 atomic pressure only dropped 2.3x (from 51% to 22%). The remaining pressure
is cross-block contention -- no amount of per-thread batching can eliminate it.

### 5. SFU ↔ LSU trade-offs are near-neutral

Precomputing trig values saves SFU cycles but adds shmem reads (LSU). When the
kernel is dual-bottlenecked on compute and memory, shifting work between them
yields diminishing returns (~3%).

### 6. Per-block output buffers lose to hardware atomicAdd (Exp 6)

Giving each block a private output region eliminates atomics but the output
(N_ELEM * N_FREQ * 8 bytes ≈ 438 KB) vastly exceeds L1 cache. The resulting L2
read-modify-write traffic is slower than hardware-optimized atomicAdd. Only viable
if the output fits in L1 (< 48 KB).

### 7. Frequency chunking wastes geometry compute (Exp 7)

Splitting frequencies across multiple launches reduces shmem per launch but repeats
the geometry computation (~40% of kernel time) per chunk. Only viable if geometry
is negligible compared to frequency sweep work.

### 8. Metal-style SIMD shuffle fails on CUDA due to thread mapping (Exp 8)

On Metal, `tid = elem * SCAT_REDUCE + scat` means shuffle reduces same-element
contributions from different scatterers → fewer atomics. On CUDA with v11's
thread-per-frequency mapping, each thread writes to a unique (elem, freq) pair,
so shuffle has nothing to reduce. The thread mapping dictates whether shuffle helps.

### 9. fp16 TX buffer gives marginal gains; register wall caps B_SCAT (Exp 9)

Halving the TX buffer (78% of shmem) to fp16 saves 17 KB at B=5 but doesn't enable
higher B_SCAT: the register pressure from B_SCAT * ELEM_TILE cv/sv arrays (4 regs
per pair) hits the 255 register limit at B=7 ET=8 or B=9 ET=4. The shmem savings
is real but the register wall is the binding constraint for larger batches.

### 10. Cross-platform convergence at ~3M scat/s per ~17 TFLOPS

M4 Max Metal (17 TFLOPS) and A4000 CUDA (19.2 TFLOPS) both achieve ~3M scat/s
despite different architectures. This confirms the ~9x gap from theoretical is
algorithmic, not implementation-specific. On the 4090 (82.6 TFLOPS), linear
FP32 scaling projects ~13M scat/s, plus the 72 MB L2 may further reduce the
atomic overhead.

## Architecture Decision Framework

### When to use atomics

- Small output space (< 1000 elements) where contention is low
- When per-block output buffers would exceed available memory (output > L1 cache)
- When launch count must be minimized (latency-sensitive)
- When hardware atomicAdd throughput exceeds software read-modify-write (always
  true on Ampere+ when output doesn't fit L1)

### When to eliminate atomics

- Output fits in L1 cache (< 48 KB) AND per-block private buffers are feasible
- GPU has large L2 (>= 64 MB) that can absorb atomic traffic efficiently
- `long_scoreboard` or `mio_throttle` stalls > 0.5

### Strategies to reduce/eliminate atomics (ranked by effectiveness)

1. **Scatterer batching (B_SCAT)**: Accumulate B scatterers in registers before
   atomicAdd. Reduces atomics by Bx. Limited by register pressure (B*ET*4 regs
   for cv/sv arrays). Best bang-for-buck on the A4000.

2. **Two-pass with L2-resident intermediate**: Separate TX/RX kernels with TX
   output in DRAM. Only viable when GPU has large L2 (4090's 72 MB) or unified
   memory (Apple Silicon SLC). Failed on A4000 (4 MB L2).

3. **SIMD shuffle reduction**: Only effective when thread mapping ensures adjacent
   threads contribute to the SAME output element. Requires careful thread layout.

4. **Per-block output buffers**: Viable only when output < L1 cache (~48 KB).
   Otherwise L2 R-M-W overhead exceeds atomicAdd hardware throughput.

## RTX A4000 Hardware Specs

- 48 SMs, 128 CUDA cores/SM = 6144 total
- 16 MB L2 cache (4 MB per L2 slice, 4 slices)
- 256 KB register file per SM
- 100 KB configurable shared memory per SM (max with 0 KB L1 preference)
- 48 warps per SM max
- 19.2 TFLOPS fp32 peak at 1560 MHz
- 448 GB/s DRAM bandwidth (GDDR6X, 256-bit)
- Compute capability 8.6 (Ampere)

## File References

- Kernels: `src/fast_simus/kernels/simus_fused_v*.cu`
- Profiling tools: `tools/ncu_profile.py`, `tools/ncu_parse.py`
- Progress: `docs/progress/cuda-kernel-optimization.md`
- Experiments: `docs/progress/experiments/exp*.md`
- Architecture analysis: `docs/progress/architecture-exploration.md`
