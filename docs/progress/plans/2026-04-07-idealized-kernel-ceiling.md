# Idealized Kernel Ceiling Experiment -- Implementation Plan

## Overview

Measure the empirical performance ceiling on the RTX 4090 by progressively
removing occupancy / shmem / atomic bottlenecks from the champion kernel while
preserving the same total compute volume. The result is an **empirical upper
bound** on what a full redesign (different threading, regrouping, shmem layout)
could achieve -- bounding the 30M scat/s goal from below (realistic) and above
(idealized).

This is a **diagnostic** experiment, not a production path. The idealized
kernels are numerically incorrect by design; we only measure timing.

## Prerequisites & Environment

### Hardware / software state

- GPU: NVIDIA RTX 4090 (CC 8.9, 128 SMs). Verify: `nvidia-smi --query-gpu=name,compute_cap --format=csv`
- CUDA 12.2 toolchain at `/usr/local/cuda-12.2/` (ncu binary: `/usr/local/cuda-12.2/bin/ncu`, version 2023.2.2)
- NVRTC runtime compilation uses CC target `sm_89` (set in `src/fast_simus/kernels/cuda_runtime.py`; verify before starting)
- Package manager: `uv` (all Python runs use `uv run python ...`)
- Working dir: `/home/user/FastSIMUS`
- Branch: `feature/simus-cuda-pallas`
- Shared memory config: dynamic shmem > 48 KB requires `set_max_dynamic_shared_mem(func, nbytes)` before launch (already handled by `ncu_profile_v15.py` and `bench_compute_floor.py`)

### Clock policy

Measured champion numbers in exp18 were taken at GPU **boost** clocks (~2520-2595 MHz), *not* locked. `nvidia-smi -lgc 1560` from the general skill doc is A4000-specific; on the 4090 we leave clocks at boost for throughput measurements. For `ncu` profiling the tool auto-locks to base clock (`--clock-control base` by default), which is why the ncu IPC is measured at ~2.22 GHz while wall-clock benchmarks run at ~2.52 GHz. When comparing variants, use **wall-clock timing** from the same sweep so the clock state is identical across variants.

### Prerequisite reading (do these first, in order)

Before touching any kernel code, read these **fully** (no offset/limit):

1. `.cursor/skills/cuda-kernel-optimization/SKILL.md` -- ncu setup, metric interpretation, 10 proven optimization principles from prior experiments. This is the most important background.
2. `docs/progress/cuda-kernel-optimization.md` -- experiment timeline, champion history, proven dead-ends.
3. `docs/progress/experiments/exp18-30M-architecture.md` -- the deep profile this plan builds on (ncu data for the champion, stall breakdown, instruction mix).
4. `src/fast_simus/kernels/simus_fused_v15.cu` -- the champion kernel. You will be deriving from this.
5. `src/fast_simus/kernels/simus_fused_v15_noatomic.cu` -- example of the dead-code-sink pattern you will reuse for all four variants.
6. `tools/bench_compute_floor.py` -- the timing harness pattern (warmup, reset, 5-repeat timing) you will extend for the sweep.
7. `tools/ncu_profile_v15.py` -- NVRTC compile + launch harness; thin wrapper is the Phase 3 deliverable.
8. `tools/ncu_parse.py` -- consumes `ncu --page raw` stdout and prints a markdown summary. Know what metrics it extracts.
9. `src/fast_simus/kernels/cuda_runtime.py` -- `compile_module`, `get_function`, `launch_kernel`, `set_max_dynamic_shared_mem`, plus `_get_cuda()` for direct `cuFuncGetAttribute` / `cuOccupancyMaxActiveBlocksPerMultiprocessor` calls.

### Known traps (from prior experiments — do not re-discover these)

Inherited from `cuda-kernel-optimization` skill + experiment docs 1-17. These apply even in idealized mode:

- **Dead-code elimination**: if the compiler can prove the inner loop has no externally-visible side effect, it deletes everything and you get sub-millisecond timings with ~0% SM throughput. Detection: `smsp__inst_executed.sum` drops >>20% vs champion's 5.46 B. Fix: the existing sink pattern `if (acc_re[et] == -1e30f && acc_im[et] == -1e30f) spect_re[...] = acc_re[et] + acc_im[et];` -- works because NaN/inf cannot be proven away statically.
- **2-block/SM cliff**: performance drops ~45% when occupancy falls from 2 to 1 block/SM (exp 5). The idealized variants are specifically designed to lift this; verify via `cuOccupancyMaxActiveBlocksPerMultiprocessor` reporting ≥3.
- **Register-spill cliff**: shmem relaxation can *increase* register pressure by pulling GEO_* into registers. At B=9 ET=4 in v15 champion, local memory (spill) is 72 B; at B=8 ET=8 it jumps to 232 B and kills performance (-35%, exp 17). Watch the `local` column in the sweep.
- **`__launch_bounds__` backfires when shmem binds occupancy** (exp 3): do not add it to any variant. If occupancy limiter is `shared_mem`, reducing registers via launch_bounds only spills registers, never adding blocks.
- **Per-block output buffers lose to hardware `atomicAdd`** when output > L1 (exp 6). Not relevant for the idealized experiment (we kill atomics entirely), but do not resurrect that approach.
- **Grid sizing**: 256 blocks (2×NUM_SMS on the 4090) for persistent-style kernels. Fewer → wave-quantization penalty; more → redundant scat_base passes.

## Current State Analysis

### Champion (v15 B=9 ET=4, RTX 4090)

- Time: 6.88 ms, 14.55 M scat/s
- Registers/thread: 192 (72 B spills to L1)
- Shmem/block: 47.6 KB
- Occupancy limiters: `regs=2`, `shmem=2`, `warps=12` → **2 blocks/SM** (16.7%)
- SM throughput 61.2%, FMA pipe 45.9%
- L2 atomic pressure 32.8%, but overall compute-latency-bound
- `wait` stall 0.61 (cmul dependency chain)

### What we already know (exp 18, Phase 1)

Removing atomicAdd (replacing with dead-code-eliminated sink) gave **1.3%**
speedup (6.93 → 6.84 ms). So atomic contention is fully hidden behind the
cmul wait stall. The shmem + register pressure is the dominant constraint via
occupancy.

### Shmem breakdown at B=9 ET=4 (47.6 KB)

| Region | Size | % | Eliminable by |
| --- | --- | --- | --- |
| Geo fp32 (7 × B × N_ES × 4) | 16.1 KB | 34% | CONSTFREQ + SINGLEELEM |
| TX fp16 (2 × B × N_FREQ × 2) | 30.0 KB | 63% | (keep -- inter-phase handoff) |
| Delay (3 × N_ELEM × 4) | 0.77 KB | 1.6% | SINGLEELEM |

### Why this diagnostic matters

If idealized kernel reaches 25-30M → 30M is achievable through redesign.
If idealized kernel caps at ~18-20M → 30M requires algorithmic change
(fp16 compute, half2 packing, or reduced work).

## Desired End State

Four new benchmark kernels exist in `src/fast_simus/kernels/`:

1. `simus_fused_v21_noatom.cu` -- atomics replaced with sinks (already exists as v15_noatomic)
2. `simus_fused_v21_constfreq.cu` -- hardcoded single (kw_step, alpha_step) constant
3. `simus_fused_v21_singleelem.cu` -- all elements use elem[0] geometry
4. `simus_fused_v21_idealized.cu` -- all three applied

A benchmark harness (`tools/bench_idealized_ceiling.py`) runs each variant
across B_SCAT/ELEM_TILE sweeps and reports timing + blocks/SM + registers.

An ncu profile of the winning idealized config identifies the remaining
bottleneck.

An experiment document (`docs/progress/experiments/exp19-idealized-ceiling.md`)
records results and conclusions about 30M feasibility.

### Success criteria (how to verify)

- Each variant compiles for sm_89 with B=9 ET=4 at minimum
- Each variant runs through all 100K scatterers without crashing
- Wall timing reproducible within 5% across 5 runs
- ncu report captured for the winning idealized config
- Document includes explicit answer: "30M achievable via redesign? Y/N"

### Key Discoveries

- `src/fast_simus/kernels/simus_fused_v15.cu:108-111` -- B_SCAT unrolled
  `out_flag[B_SCAT]` + `cv[B_SCAT * ELEM_TILE]` drives register pressure
- `src/fast_simus/kernels/simus_fused_v15.cu:28-34` -- 7 geometry arrays per
  scatterer in shmem; dominant shmem at low B, subsidiary at high B
- `tools/bench_compute_floor.py` -- pattern for paired champion/variant timing
  runs with dead-code-sink
- `tools/ncu_profile_v15.py` -- parameterized profiling harness to fork from

## What We're NOT Doing

- NOT validating correctness against PyMUST (idealized kernels are wrong by design)
- NOT shipping these kernels or integrating with `cuda_simus.py`
- NOT exploring Tensor Cores (`wmma`) -- separate future experiment
- NOT touching `N_FREQ`, `N_ELEM`, `N_SUB`, `N_ES` -- same compute volume
- NOT eliminating the Phase 2 → Phase 3 shmem handoff -- keeps kernel structurally
  similar to the champion so results transfer
- NOT a production fp16-compute kernel -- that's v19 in the parallel track

## Implementation Approach

Derive each variant from `simus_fused_v15.cu` with minimal diffs. Keep the
outer structure (persistent blocks, scat_base loop, Phase 1/2/3) identical so
total compute work is preserved -- we only gate **per-instance state** (shmem
arrays, per-element registers).

Each variant uses a different CUDA macro gate so we can share most of the
source (or alternately keep as copies; copies preferred for clarity of the
diagnostic). Copies are preferred: each file stands alone for ncu analysis.

All variants use the same dead-code-sink pattern as the existing
`simus_fused_v15_noatomic.cu` to prevent the compiler from eliminating the
inner loop.

Benchmark sweep: for each variant, test (B=5 ET=4), (B=9 ET=4), (B=13 ET=4),
(B=17 ET=4), (B=21 ET=4). Higher B values only become viable under idealized
constraints (reduced shmem unlocks them).

## Bring-up & Debugging Playbook

Expect to hit most of these during Phase 1 implementation. Each entry has a
**symptom** and the **fix** from prior experiments.

### Compile failures

- **`NVRTC error: identifier "blah" is undefined`** → likely a missing
  `#define` in the `defines=` tuple passed to `compile_module`. The v15 kernel
  needs exactly: `N_ELEM, N_SUB, N_FREQ, N_ES, TILE_SE, TG_SIZE, MAX_FPT, B_SCAT, ELEM_TILE`.
  Derive `MAX_FPT = (n_freq + TG_SIZE - 1) // TG_SIZE` -- forgetting this is common.
- **`error: too much shared memory used`** → exceeds 100 KB static limit. You
  must call `set_max_dynamic_shared_mem(func, nbytes)` for anything > 48 KB
  *and* the requested shmem must be ≤ 100 KB per block. Sanity-check: print
  the computed shmem size before the launch.
- **`CUDA_ERROR_INVALID_VALUE` at launch** → usually grid dim 0 (division by
  zero somewhere computing grid) or a null pointer in `kernel_args`. Print
  every arg type and value before the first launch.

### Runtime failures

- **`CUDA_ERROR_ILLEGAL_ADDRESS`** mid-run → out-of-bounds write. Most likely
  place in the idealized variants: accidentally indexing `GEO_*(si)[se]` after
  you changed the layout to scalars. Rebuild the macros carefully.
- **Hang / TDR timeout** → infinite loop, usually `for (int scat_base = ...; scat_base < n_scat; scat_base += gridDim.x * B_SCAT)` with `gridDim.x * B_SCAT == 0`. Verify B_SCAT ≥ 1 and grid ≥ 1 at launch.
- **Kernel returns but output is zero** → you hit DCE. See "DCE detection" below.

### DCE detection (most common trap in this experiment)

The idealized kernels deliberately skip atomic writes. If the compiler figures
out that the inner cmul loop has no side effect, it deletes everything.

**Detection signals (in order of cost):**

1. Wall-clock benchmark: idealized is 100×+ faster than champion (e.g. <0.1 ms). Way too fast.
2. Register count near 0 or 32 when you expect ≥150. The compiler optimized away the live values.
3. `ncu` on the variant shows `smsp__inst_executed.sum` orders of magnitude below champion's 5.46 B.

**Fix:** use the dead-code-sink from `simus_fused_v15_noatomic.cu:228-232`:
```c
if (acc_re[et] == -1e30f && acc_im[et] == -1e30f)
    spect_re[elem * N_FREQ + f] = acc_re[et] + acc_im[et];
```
The comparison with a sentinel value cannot be proven false at compile time,
so the accumulation chain must execute. The branch is never taken at runtime,
so no actual stores happen.

### Register-spill debugging

- Check `launch__registers_per_thread` (from `cuFuncGetAttribute(CU_FUNC_ATTR_NUM_REGS)`) and `CU_FUNC_ATTR_LOCAL_SIZE_BYTES` (local = spills).
- 4090 hard cap: **255 registers/thread**. At that cap, the compiler must spill.
- Spills go to L1 cache first, then L2. At 72 B/thread (the champion) they are absorbed fine; at >200 B/thread they stall `long_scoreboard`.
- If you see unexpected spills in a variant: try a smaller `ELEM_TILE` (cuts `cv[B_SCAT*ELEM_TILE]` array size in half). `ELEM_TILE=4` is the v15 champion for this reason.

### Occupancy verification

Call `cuOccupancyMaxActiveBlocksPerMultiprocessor(func, TG_SIZE, shmem)` before
the launch (pattern already in `ncu_profile_v15.py:192`). Expected values:

- v15 champion: 2 blocks/SM
- IDEALIZED at B=9 ET=4: ≥3 (target)
- IDEALIZED at B=5 ET=4: 5-6 (if regs permit)

If the reported occupancy is lower than expected, use the three `launch__occupancy_limit_*` metrics from ncu to find the binding constraint (regs vs shmem vs warps).

### Correctness red-flag: the sink must not itself execute

If your benchmark reports both "fast" *and* `spect_re` has NaN/inf values at
some indices, the sink's `-1e30f` test triggered. That cannot happen with
initialized-to-zero accumulators, but if you ever add randomized init code
for debugging, remove it before measurement.

## Phase 1: Variant Kernels

### Overview

Create three new idealized kernel variants alongside the existing no-atomic
baseline.

### Changes Required

#### 1. Rename existing no-atom kernel for consistency

**File**: `src/fast_simus/kernels/simus_fused_v21_noatom.cu`

Copy from `src/fast_simus/kernels/simus_fused_v15_noatomic.cu` unchanged.
Keep the existing file too (zero cost) so the compute-floor script keeps
working.

#### 2. CONSTFREQ variant

**File**: `src/fast_simus/kernels/simus_fused_v21_constfreq.cu`

Derive from `simus_fused_v21_noatom.cu`. Replace per-sub-element frequency
step storage with constants. Effect:

- Remove `GEO_KW_R`, `GEO_KR_STEP`, `GEO_ALPHA_R`, `GEO_AR_STEP` arrays from shmem
- Use scalar registers computed once per scatterer (captured from elem 0)
- Shmem drops by `4 * B_SCAT * N_ES * 4` bytes = 9.2 KB at B=9

Specifically, in the Phase 1 geometry write-out:

```c
// Before (v15):
GEO_KW_R(si)[se]    = kw_init * rc_;
GEO_KR_STEP(si)[se] = kw_step * rc_;
GEO_ALPHA_R(si)[se] = alpha_init * rc_;
GEO_AR_STEP(si)[se] = alpha_step * rc_;

// After (CONSTFREQ): only write to shmem the first (se=0 per scatterer) and
// read that value uniformly in Phase 2/3. Use a single pair of constant
// registers per scatterer:
float ph_init_s[B_SCAT];       // captured from se=0 per scatterer
float ph_step_s[B_SCAT];       // captured from se=0 per scatterer
float amp_init_s[B_SCAT];      // captured from se=0 per scatterer
float amp_step_s[B_SCAT];      // captured from se=0 per scatterer
```

Phase 2/3 reads these per-scatterer scalars instead of per-element shmem
arrays. Same number of FMA instructions in the cmul loops.

#### 3. SINGLEELEM variant

**File**: `src/fast_simus/kernels/simus_fused_v21_singleelem.cu`

Derive from `simus_fused_v21_noatom.cu`. Recompute geometry as if every
element sat at `elem[0]`. Effect:

- `GEO_AMP` and `GEO_STP_RX_*` become identical across `se`
- Keep the full loop over `se` (same compute), just read the same scalar
- Shmem `GEO_*` arrays collapse to `7 × B_SCAT` scalars (e.g. 252 bytes at B=9)
- Delay arrays collapse to 3 scalars (12 B)

Specific change in Phase 1 inner loop:

```c
// Before: uses elem = se / N_SUB to index elem_x, elem_z, cos_te, etc.
// After: always use elem = 0
int elem = 0;  // SINGLEELEM idealization
float ex_ = elem_x[0], ez_ = elem_z[0];
// ... rest unchanged, still iterates over all se
```

In Phases 2/3, `GEO_*(si)[se]` → `GEO_SCALAR(si)` (indexed by si only).

#### 4. IDEALIZED variant

**File**: `src/fast_simus/kernels/simus_fused_v21_idealized.cu`

Combines CONSTFREQ + SINGLEELEM + no-atom sink. Shmem reduces to just the
TX fp16 handoff buffer (~30 KB at B=9) plus a tiny constant region.

At B=9 ET=4: shmem ≈ 30.7 KB → **3 blocks/SM possible** (25% occupancy, 1.5×
current).
At B=5 ET=4: shmem ≈ 17.1 KB → **5-6 blocks/SM possible** (if regs permit).

The register impact is also significant: with `GEO_*` arrays gone, the inner
cmul loop loads from registers instead of shmem, freeing the LSU pipe but
raising register pressure in the other direction. Net effect TBD by
measurement.

### Success Criteria

#### Automated Verification

- [x] All four variants compile: `uv run python -c "from pathlib import Path; ..."` via the benchmark harness
- [x] At B=9 ET=4 each variant runs to completion without crash: harness reports a time
- [ ] IDEALIZED variant at B=5 ET=4 achieves `occ >= 4 blocks/SM` (reported by `cuOccupancyMaxActiveBlocksPerMultiprocessor`) -- *measured 3 blocks/SM (reg-capped at 168 regs); plan estimate was optimistic*
- [x] No runtime errors in CUDA API calls during 5-repeat timing runs

#### Manual Verification

- [x] Inspect generated PTX (optional) to confirm the compiler did not DCE the
      inner loop in any variant. Signal: `smsp__inst_executed.sum` for
      IDEALIZED is within 20% of champion's 5.46B (same compute volume).
      *Measured: 4.43 B vs champion 5.46 B = -18.9% (within 20% tolerance).*

## Phase 2: Benchmark Sweep Harness

### Overview

A single harness that runs all four variants across B_SCAT ∈ {5, 9, 13, 17,
21} and ELEM_TILE ∈ {4, 8}, timing each with 5 repeats and reporting
best/median timing, occupancy, registers, and shmem.

### Changes Required

#### 1. Benchmark script

**File**: `tools/bench_idealized_ceiling.py`

Derive from `tools/bench_compute_floor.py`. Key differences:

- Accepts a `--variants` flag (default: all four)
- Runs B_SCAT sweep {5, 9, 13, 17, 21}
- ELEM_TILE sweep {4, 8}
- For each (variant, B, ET) config, computes `compute_shmem_v21(variant, B, ET)`
  with variant-specific formula
- Outputs Markdown table: variant | B | ET | shmem_KB | regs | spill | blk/SM | ms | scat/s
- Highlights the best per variant

```python
def compute_shmem_v21(variant, b_scat, n_es, n_freq, n_elem):
    tx_half_bytes = 2 * b_scat * n_freq * 2
    tx_half_floats = (tx_half_bytes + 3) // 4
    if variant == "noatom":
        geo_floats = 7 * b_scat * n_es
        delay_floats = 3 * n_elem
    elif variant == "constfreq":
        geo_floats = 3 * b_scat * n_es  # AMP + STP_RX_RE + STP_RX_IM only
        delay_floats = 3 * n_elem
    elif variant == "singleelem":
        geo_floats = 7 * b_scat       # scalar per scatterer, not per se
        delay_floats = 3              # scalar
    elif variant == "idealized":
        geo_floats = 3 * b_scat
        delay_floats = 3
    return (geo_floats + tx_half_floats + delay_floats) * 4
```

#### 2. Output format

```
Idealized Ceiling | N_SCAT=100,000, N_FREQ=854, N_ELEM=64, N_ES=64
================================================================================
variant      B  ET  shmem  regs  spill blk/SM    ms  M scat/s   vs champion
v15 champ.   9   4  47.6KB 192    72B     2    6.88   14.55      1.00x
noatom       9   4  47.6KB 237    72B     2    6.84   14.63      1.01x
constfreq    9   4  38.5KB xxx   xxxB     x    x.xx   xx.xx      x.xxx
singleelem   9   4  30.8KB xxx   xxxB     x    x.xx   xx.xx      x.xxx
idealized    9   4  30.7KB xxx   xxxB     x    x.xx   xx.xx      x.xxx
idealized    5   4  17.1KB xxx   xxxB     x    x.xx   xx.xx      x.xxx   <- likely winner
idealized   13   4  40.4KB xxx   xxxB     x    x.xx   xx.xx      x.xxx
...
================================================================================
Winning idealized config: B=<b> ET=<et> | <ms>ms = <M> scat/s
Headroom vs champion: <factor>x
```

### Success Criteria

#### Automated Verification

- [x] Harness runs to completion for all 4 variants × 5 B values × 2 ET values = 40 configs *(B=21 configs correctly rejected as shmem > 100 KB; 38/40 produce a timing)*
- [x] At least 1 config per variant produces a non-error timing
- [x] Table prints with all columns populated
- [x] Best idealized timing is < champion timing (headroom > 1x); if not, investigate DCE *(1.36x headroom, 4.72 ms vs 6.44 ms)*

#### Manual Verification

- [x] Best idealized config's ncu report shows SM throughput > champion's 61.2% *(65.59% measured)*
- [ ] Best idealized config's `launch__occupancy_per_block_size` > champion's 2 blocks/SM *(FAILED: hard register cap at 255 keeps it at 2 blocks/SM; the reg-relaxed B=5 ET=4 variant does reach 3 blocks/SM but is slower due to less ILP.)*

## Phase 3: NCU Profile of Winning Config

### Overview

Profile the winning idealized configuration with `--set full` and extract
bottleneck metrics to compare against the champion.

### Changes Required

#### 1. Profile invocation

Use existing `tools/ncu_profile_v15.py` pattern but pointing to the winning
idealized kernel:

```bash
sudo /usr/local/cuda-12.2/bin/ncu \
  --target-processes all \
  --launch-skip 1 --launch-count 1 \
  --set full \
  -f -o 4090_v21_idealized.ncu-rep \
  $(which uv) run python tools/ncu_profile_v21.py \
    --variant idealized --b-scat <winner> --elem-tile <winner> --blocks 256
```

This requires a thin wrapper `tools/ncu_profile_v21.py` -- essentially a
copy of `ncu_profile_v15.py` that takes `--variant` and picks the right
kernel path.

#### 2. Comparison table generation

Extract and tabulate:

- Time, IPC, SM throughput, FMA%, SFU%, LSU%
- `wait`, `long_scoreboard`, `barrier`, `mio_throttle` stalls
- `sm__warps_active` (occupancy)
- `lts__d_atomic_input_cycles_active` (should be 0 for idealized)
- Total instruction count (to verify compute volume preserved)

Record in `docs/progress/experiments/exp19-idealized-ceiling.md`.

### Success Criteria

#### Automated Verification

- [x] ncu report file exists: `ls 4090_v21_idealized.ncu-rep`
- [x] `tools/ncu_parse.py` produces a clean metric table from the report
- [x] Total instruction count within 20% of champion's 5.46B (compute volume
      sanity check; if way lower, DCE removed work and results are invalid) *(4.43 B = -18.9%, within tolerance)*

#### Manual Verification

- [x] Comparison table in experiment doc includes all 4 variants side by side
- [x] Conclusion section answers the `30M achievable Y/N` question *(N without algorithmic change; idealized ceiling 21.17 M = 1.43x below 30M)*

## Phase 4: Analysis and Experiment Document

### Overview

Synthesize findings into `docs/progress/experiments/exp19-idealized-ceiling.md`.

### Changes Required

#### 1. Experiment document

**File**: `docs/progress/experiments/exp19-idealized-ceiling.md`

Sections:

- Hypothesis (what ceiling we expect per theoretical analysis)
- Results table (all 4 variants × best B/ET)
- Ablation: which constraint release gave which speedup
  - champion → noatom: X% (atomic release)
  - noatom → constfreq: X% (per-freq state release)
  - constfreq → idealized: X% (per-elem state release)
- NCU comparison table (champion vs IDEALIZED winner)
- Answer: 30M achievable via redesign? With how much headroom?
- Recommendations for v19/v20/future kernels based on where the real
  bottleneck sits in the idealized case

#### 2. Main progress tracker update

**File**: `docs/progress/cuda-kernel-optimization.md`

Add row 18 to experiment summary:
```
| 18| v21 idealized ceiling | <result> | Empirical ceiling is <X>M scat/s |
```

Update "Optimization Strategy" section with the ceiling finding.

### Success Criteria

#### Automated Verification

- [x] `exp19-idealized-ceiling.md` exists with all sections populated
- [x] `docs/progress/cuda-kernel-optimization.md` has row 18
- [x] All result numbers in the doc cross-check against the raw
      `bench_idealized_ceiling.py` output captured in the doc

#### Manual Verification

- [x] The ablation section identifies the single biggest contributor to the
      ceiling (one of: atomic / per-freq state / per-elem state / occupancy)
      *(per-elem state / shmem collapse via singleelem: +12.4% standalone, biggest single step)*
- [x] The recommendation clearly states which realistic redesign avenue
      (if any) is worth pursuing to reach 30M *(fp16 cmul in Phase 3 + cmul chain pipelining)*

## Testing Strategy

### Unit Tests

N/A -- these are diagnostic kernels that intentionally produce wrong answers.

### Integration Tests

N/A.

### Manual Testing Steps

1. Run `uv run python tools/bench_idealized_ceiling.py` and verify the table
   populates for all 40 configs.
2. Confirm the idealized kernel at best config has at least 3 blocks/SM
   (reported in the `blk/SM` column).
3. Run the NCU profile on the winning config.
4. Compare `smsp__inst_executed.sum` for idealized vs champion. They should
   be within 20% (compute volume preserved).
5. Cross-check the ablation deltas: no step should be negative (each relaxation
   should monotonically improve or be neutral).

## Performance Considerations

- If the CONSTFREQ/SINGLEELEM simplifications also trigger register reduction
  (no shmem loads → fewer live registers), that's a second-order win beyond
  shmem; report it.
- Conversely, if moving GEO_* from shmem to registers *increases* register
  pressure (because now they live in the register file per thread), the
  variant may spill more. Watch the `local` column.
- **Clock policy**: run wall-clock benchmarks at boost (no manual lock) and
  take the `best` of 5 repeats; `ncu` auto-locks to base clock so its timing
  is lower than wall-clock by ~13% (see "Clock policy" in Prerequisites).
  Compare variants only against measurements taken under the same policy.
  The A4000 skill's `nvidia-smi -lgc 1560` does not apply to the 4090 (its
  base clock is higher; locking there crushes throughput).

## Migration Notes

None -- these files are additions in `src/fast_simus/kernels/v21_*` and
`tools/bench_idealized_ceiling.py`; nothing mutates existing kernels or
the dispatch path in `cuda_simus.py`.

## Expected Result Ranges (Sanity Checks)

Use these to detect when something is wrong before interpreting results.

### Timing (wall clock, B=9 ET=4, blocks=256, 100K scatterers)

| Variant | Expected range | Red flag |
| --- | --- | --- |
| v15 champion | 6.85-7.05 ms | <6.5 or >7.5 → bad measurement |
| v21 noatom | 6.75-6.95 ms | Same as champion ±1.5% (exp 18 Phase 1) |
| v21 constfreq | 5.0-6.5 ms | <3 ms → DCE; >6.8 ms → no win (check regs/occ) |
| v21 singleelem | 4.5-6.0 ms | <2 ms → DCE; ≈champion → shmem wasn't the limiter |
| v21 idealized | 3.0-5.5 ms | <1 ms → DCE; >6.5 ms → variants didn't stack |

The 30M target corresponds to 3.33 ms. If `idealized` at its best B/ET reaches
~3-4 ms, redesign can plausibly reach 30M. If it stalls at ~5 ms, reaching 30M
requires fp16 compute or algorithmic change.

### NCU metrics for the winning IDEALIZED config

| Metric | Champion | Expected idealized | Signal |
| --- | --- | --- | --- |
| `sm__throughput` | 61.2% | ≥75% | Higher → win real |
| `sm__pipe_fma` | 45.9% | ≥60% | Higher → compute dominating |
| `launch__occupancy_limit_shared_mem` | 2 | ≥3 | Shmem no longer binding |
| `sm__warps_active` | 16.5% | ≥25% | More warps covering latency |
| `lts__d_atomic_input_cycles_active` | 32.8% | 0% | Atomics gone |
| `smsp__inst_executed.sum` | 5.46 B | 4.5-5.5 B | Compute volume preserved |
| `wait` stall | 0.61 | Should drop if cmul chain shortens with lower ELEM_TILE | |

If `smsp__inst_executed.sum` falls below 4 B, DCE eliminated work and the
whole measurement is invalid -- strengthen the sink and re-run.

## Command Reference (Copy-Paste)

All commands assume `cwd = /home/user/FastSIMUS`.

### 0. Sanity check environment
```bash
nvidia-smi --query-gpu=name,compute_cap,clocks.current.sm --format=csv
ls /usr/local/cuda-12.2/bin/ncu
grep 'arch: str' src/fast_simus/kernels/cuda_runtime.py | head -2    # expect sm_89
```

### 1. Reproduce the champion baseline
```bash
uv run python tools/ncu_profile_v15.py --b-scat 9 --elem-tile 4 --blocks 256
# Expected: ~6.9 ms, ~14.5 M scat/s, blk/SM=2, regs=192, local=72B
```

### 2. Reproduce the compute-floor number (exp 18 Phase 1)
```bash
uv run python tools/bench_compute_floor.py
# Expected: noatom ≈ champion - 0.09ms (1.3% atomic overhead)
```

### 3. Run the idealized sweep (Phase 2 of this plan)
```bash
uv run python tools/bench_idealized_ceiling.py
# Prints the 40-row table (4 variants × 5 B × 2 ET).
# Use `--variants idealized` to restrict; `--b-scat N --elem-tile M` for single run.
```

### 4. NCU profile of winning idealized config (Phase 3)
Replace `<B>` and `<ET>` with the winning config from step 3:
```bash
sudo /usr/local/cuda-12.2/bin/ncu \
  --target-processes all \
  --launch-skip 1 --launch-count 1 \
  --set full \
  -f -o 4090_v21_idealized.ncu-rep \
  $(which uv) run python tools/ncu_profile_v21.py \
    --variant idealized --b-scat <B> --elem-tile <ET> --blocks 256
```
Runtime: ~30-60 s (ncu replays the kernel ~34 times for `--set full`). If ncu
complains about permissions, confirm `sudo` and that `/proc/driver/nvidia/params` has `NVreg_RestrictProfilingToAdminUsers=0` (already set on this VM per exp 15).

### 5. Parse the NCU report
```bash
sudo /usr/local/cuda-12.2/bin/ncu --import 4090_v21_idealized.ncu-rep --page raw 2>&1 \
  | uv run python tools/ncu_parse.py
```
Produces the markdown summary table (kernel time, registers, shmem, occupancy limits, pipe utilization, L2 atomic pressure, IPC, warp stall reasons).

### 6. Raw metric extraction for custom analysis
To pull specific metrics not surfaced by `ncu_parse.py`:
```bash
sudo /usr/local/cuda-12.2/bin/ncu --import 4090_v21_idealized.ncu-rep --page raw 2>&1 \
  | grep -E "(smsp__inst_executed\.sum|launch__occupancy|sm__warps_active|lts__t_sectors)"
```

### 7. Project health check
```bash
uv run poe lint    # ruff + mypy
uv run poe test    # pytest (kernel tests require CUDA-capable GPU)
```

### 8. Beads workflow (optional, for tracking)
```bash
bd create --title="exp19 idealized ceiling" --type=task --priority=1
bd update <id> --claim                 # start
bd close  <id>                         # finish
bd remember "idealized ceiling = X M scat/s at B=Y ET=Z; 30M achievable = Y/N"
```

## References

- Champion kernel: `src/fast_simus/kernels/simus_fused_v15.cu`
- Existing no-atom variant: `src/fast_simus/kernels/simus_fused_v15_noatomic.cu`
- Compute floor result: exp18, Phase 1 (atomic overhead 1.3%)
- Architecture context: `docs/progress/experiments/exp18-30M-architecture.md`
- Profiling harness: `tools/ncu_profile_v15.py`, `tools/ncu_parse.py`, `tools/bench_compute_floor.py`
- NCU data for champion: `4090_v15_b9et4.ncu-rep`
- Primary skill: `.cursor/skills/cuda-kernel-optimization/SKILL.md`
- Supporting skill: `.cursor/skills/gpu-profiling/SKILL.md`
- NVRTC runtime helpers: `src/fast_simus/kernels/cuda_runtime.py` (`compile_module`, `get_function`, `launch_kernel`, `set_max_dynamic_shared_mem`)
- Progress tracker: `docs/progress/cuda-kernel-optimization.md`
- Theoretical limit analysis (parallel track): `tools/theoretical_limits.py`
