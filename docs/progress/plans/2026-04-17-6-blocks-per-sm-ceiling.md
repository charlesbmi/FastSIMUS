# 6 Blocks/SM Ceiling Implementation Plan

## Overview

Follow-up to exp19 (idealized ceiling). Exp19 proved the "shmem / atomic /
per-se state" design space is exhausted at 1.36x headroom (21.17 M scat/s,
4.72 ms, **2 blocks/SM**). Critically, exp19 also showed that the register
hard-cap (255 regs/thread) held occupancy at 2 blocks/SM even after the
shmem limiter was fully relaxed -- so we never actually measured the
kernel at high occupancy. The reg-relaxed B=5 ET=4 config reached only
3 blocks/SM and did not beat ET=8, but that is not a representative
high-occupancy point either (168 regs is still tight).

This experiment forces the kernel to **6 blocks/SM** by applying four
relaxations in combination, and answers two questions:

1. Can 6 blocks/SM be reached at all on this kernel shape?
2. Does 6 blocks/SM beat the 21.17 M scat/s ceiling, or does some other
   constraint (FMA pipe saturation, scheduler, shmem bandwidth) bind next?

Purely diagnostic -- numerical correctness is NOT required, but compute
volume must stay within 20% of the champion's `smsp__inst_executed.sum =
5.46 B` (same sanity check as exp19) to rule out DCE artifacts.

## Current State Analysis

From `docs/progress/experiments/exp19-idealized-ceiling.md`:

- Best idealized config: **B=5 ET=8**, 255 regs, 17.15 KB shmem, **2 blocks/SM**,
  4.72 ms, 21.17 M scat/s, SM throughput 65.59%, `wait` stall 0.57.
- Best 3-blocks/SM config: idealized **B=5 ET=4**, 168 regs, 16.8 KB shmem,
  3 blocks/SM, 4.97 ms, 20.13 M scat/s. `sm__warps_active` unchanged vs ET=8,
  so extra blocks did not add active warps at this occupancy level.
- **Register pressure source**: Phase 3 carries `f2 cv[B*ET]` + `f2 sv_arr[B*ET]`
  live across the frequency sweep = 2 * 2 * B * ET scalar regs. At B=5 ET=8
  that is 160 scalar regs plus temporaries -> 255 cap.
- **Chain structure**: Phase 3 steps `cv[idx] = cmul(cv[idx], sv_arr[idx])`
  each frequency iteration. This chain both (a) forces cv/sv to be live and
  (b) creates the `wait` stall (0.57) that dominates the remaining latency.
- Shmem already fits 6 blocks/SM (17.15 KB * 6 = 103 KB vs 100 KB shmem/SM cap
  -- narrowly blown; trimming to <=16 KB/block clears it).

### RTX 4090 (SM89) occupancy math at 6 blocks/SM

| Resource | Per-SM total | Per-block budget at 6 blk/SM, TG=128 |
| --- | --- | --- |
| Threads | 1536 | 128 (6*128 = 768, fine) |
| Registers | 65536 | 65536/(6*128) = **85 regs/thread max** |
| Dynamic shmem | 100 KB (opt-in) | 100/6 ~ **16.6 KB/block max** |

So the two hard requirements are `regs <= 85` and `shmem <= 16384 B`. The
current idealized kernel is at 255 regs and 17.15 KB; neither budget is met.

### Key Discoveries

- `__launch_bounds__(128, 6)` is the canonical NVCC/NVRTC way to force the
  compiler to size the allocation to 6 blocks/SM, at the cost of spills
  when the working set doesn't fit.
  See `src/fast_simus/kernels/simus_fused_v21_idealized.cu:78` (no launch
  bound today).
- The register-pressure hotspot is the Phase 3 cv/sv live arrays at
  `simus_fused_v21_idealized.cu:254-268` (Phase 3 preamble) and `:271-291`
  (per-frequency cmul step).
- TG_SIZE is a define at `tools/bench_idealized_ceiling.py:42` and
  `simus_fused_v21_idealized.cu:81` (via `TG_SIZE` macro). Halving to 64
  doubles the per-thread reg budget (170 regs at 6 blk/SM) but halves
  theoretical warps/SM.
- The DCE sink pattern at `simus_fused_v21_idealized.cu:298-304` must be
  preserved in every variant or timing becomes meaningless.

## Desired End State

- Four new kernel variants `v22_{lb6,nochain,tg64,floor}.cu` exist,
  all compile and run through the benchmark harness.
- At least **one variant demonstrably runs at >=6 blocks/SM** as reported
  by `cuOccupancyMaxActiveBlocksPerMultiprocessor`.
- Benchmark sweep table lists time, regs, spill, shmem, blocks/SM, and
  M scat/s for every (variant, B, ET) config run.
- NCU profile of the fastest 6-blocks/SM config exists and is parsed.
- `docs/progress/experiments/exp20-6-blocks-per-sm.md` presents the sweep,
  the ablation (which of the four levers contributed what), and states
  plainly whether 6 blocks/SM beats the 21.17 M / 4.72 ms ceiling, and if
  not, what binds next (FMA pipe, shmem BW, scheduler, etc.).
- Progress tracker `docs/progress/cuda-kernel-optimization.md` has a new
  row for exp20.

### Verification
- `ls src/fast_simus/kernels/simus_fused_v22_*.cu` returns 4 files.
- `uv run python tools/bench_6blk_ceiling.py` runs to completion, prints
  a full results table, and at least one row shows `blk/SM >= 6`.
- `ls 4090_v22_6blk.ncu-rep` exists.
- `docs/progress/experiments/exp20-6-blocks-per-sm.md` exists with all
  sections populated.

## What We're NOT Doing

- **Not pursuing numerical correctness.** All v22 variants are idealized
  further than v21; outputs are deliberately wrong. Compute volume is
  preserved only to validate that results are not DCE artifacts.
- **Not shipping any v22 variant to the runtime** -- these are diagnostic
  only. v15 remains the production kernel.
- **Not implementing fp16 cmul or half2 packing** (H1 from exp18). That is
  the next algorithmic direction, tracked separately.
- **Not touching the TX buffer layout** (still fp16 ushort, same shmem
  footprint as idealized).
- **Not sweeping `__launch_bounds__(T, N)` for N in {3,4,5}.** We only
  target N=6 and compare to v21 N=2/3 (already measured).
- **Not re-running the v21 sweep.** exp19 numbers are the baseline.

## Implementation Approach

Four independent levers for 6 blocks/SM; one variant per lever plus one
combined-floor variant.

| Variant | Lever | Expected reg/thread | Expected blk/SM | Risk |
| --- | --- | --- | --- | --- |
| **v22_lb6** | Add `__launch_bounds__(128, 6)` to idealized | <=85 (compiler-enforced) | 6 | heavy spill -> slow |
| **v22_nochain** | Eliminate Phase 3 cv/sv running state (recompute per freq) | ~120-150 | 3-4 | may bring us to 6 when combined with lb6 |
| **v22_tg64** | TG_SIZE=64 + idealized | ~170 | 6 | halves warp pool, may worsen `wait` |
| **v22_floor** | B=1, ET=1, TILE_SE=1, lb6, nochain (minimum possible work per block) | <=60 | 6-8 | ILP floor; likely slowest but cleanest occupancy measurement |

The `nochain` transform is the only subtle one:
- v21 Phase 3 per-freq step: `acc += h2f(tx) * cv; cv = cmul(cv, sv);`
  (cv/sv live across all freqs, chain length = N_FREQ).
- v22 nochain equivalent: compute `cv_f = init_cv(ph + f * stride * step)`
  freshly per freq via `__sincosf`, so cv is a scalar in the freq loop
  body and not carried across iterations. Compute volume goes UP (extra
  sincosf calls per freq) which preserves the DCE sanity check while
  shrinking live register footprint. Same DCE sink is kept.

## Phase 1: Create v22 Variant Kernels

### Overview
Copy `simus_fused_v21_idealized.cu` as the base and apply one lever per variant.

### Changes Required

#### 1. v22_lb6 -- launch-bounds only
**File**: `src/fast_simus/kernels/simus_fused_v22_lb6.cu`
**Changes**: Copy of `simus_fused_v21_idealized.cu` with a single one-line
addition to the kernel declaration.

```c
extern "C" __global__
__launch_bounds__(128, 6)
void simus_fused_kernel(
    ...
```

The B_SCAT/ELEM_TILE defines are left unchanged -- the compiler will
spill whatever doesn't fit in 85 regs.

#### 2. v22_nochain -- Phase 3 chain break
**File**: `src/fast_simus/kernels/simus_fused_v22_nochain.cu`
**Changes**: Base is v21_idealized. Phase 3 preamble no longer builds
`cv[B*ET]` / `sv_arr[B*ET]` arrays; instead the per-freq loop recomputes
the rotating `cv` on the fly.

```c
// Phase 3 (replaces lines 254-291 of v21_idealized)
for (int fi = 0; fi < my_n_freq; fi++) {
    int f = lid + fi * TG_SIZE;
    if (f >= N_FREQ) break;
    float pf = probe[f];

    float acc_re[ELEM_TILE];
    float acc_im[ELEM_TILE];
    #pragma unroll
    for (int et = 0; et < ELEM_TILE; et++) { acc_re[et]=0.0f; acc_im[et]=0.0f; }

    #pragma unroll
    for (int si = 0; si < B_SCAT; si++) {
        if (si >= actual_b || out_flag[si]) continue;
        float tk_re = h2f(sh_tx_re[si * N_FREQ + f]);
        float tk_im = h2f(sh_tx_im[si * N_FREQ + f]);
        // Freshly synthesize rotation at frequency f -- NO live chain.
        float ph_f = kw_r_s[si] + (lid_f + (float)f * stride_f) * kr_step_s[si];
        float av_f = alpha_r_s[si] + (lid_f + (float)f * stride_f) * ar_step_s[si];
        float ai = GEO_AMP(si)[0] * expf(-av_f);

        #pragma unroll
        for (int et = 0; et < ELEM_TILE; et++) {
            if (et >= etl) break;
            float vr, vi;
            __sincosf(ph_f, &vi, &vr);
            float rr = vr * ai * inv_nsub;
            float ri = vi * ai * inv_nsub;
            acc_re[et] += (tk_re*rr - tk_im*ri) * pf;
            acc_im[et] += (tk_re*ri + tk_im*rr) * pf;
        }
    }

    #pragma unroll
    for (int et = 0; et < ELEM_TILE; et++) {
        if (et >= etl) break;
        int elem = (se_base + et) / N_SUB;
        if (acc_re[et] == -1e30f && acc_im[et] == -1e30f)
            spect_re[elem * N_FREQ + f] = acc_re[et] + acc_im[et];
    }
}
```

Note: this increases compute volume (one `__sincosf` per B*ET per freq
instead of amortized across the chain). That keeps `smsp__inst_executed.sum`
above the DCE threshold, which is useful, not a problem.

#### 3. v22_tg64 -- smaller threadgroup
**File**: `src/fast_simus/kernels/simus_fused_v22_tg64.cu`
**Changes**: Copy of v21_idealized with no kernel-source changes. The
harness will compile it with `TG_SIZE=64` and `MAX_FPT=ceil(N_FREQ/64)`.
The only edit vs v21 is the file header comment identifying the variant.
Also adds `__launch_bounds__(64, 6)` for clarity.

```c
extern "C" __global__
__launch_bounds__(64, 6)
void simus_fused_kernel(...)
```

#### 4. v22_floor -- minimum-work floor
**File**: `src/fast_simus/kernels/simus_fused_v22_floor.cu`
**Changes**: Base v21_idealized + v22_nochain Phase 3 + lb6. Compiled at
`B_SCAT=1`, `ELEM_TILE=1`, `TILE_SE=1`. This is the smallest register
footprint we can realistically get while still preserving the outer loop
structure.

```c
extern "C" __global__
__launch_bounds__(128, 6)
void simus_fused_kernel(...)
```

### Success Criteria

#### Automated Verification
- [ ] Four files exist:
      `ls src/fast_simus/kernels/simus_fused_v22_{lb6,nochain,tg64,floor}.cu`
- [ ] Each compiles via NVRTC in the benchmark harness (no `COMPILE:` errors
      in the table).
- [ ] At least one variant reports `blk/SM >= 6` from
      `cuOccupancyMaxActiveBlocksPerMultiprocessor`.
- [ ] Register count at best config for each variant:
      - `v22_lb6`: regs <= 85
      - `v22_nochain`: regs < 255 (chain-break removed live state)
      - `v22_tg64`: regs <= 170 (with TG=64 + lb(64,6))
      - `v22_floor`: regs <= 85 with B=1 ET=1 TILE_SE=1
- [ ] DCE sanity: for the fastest 6-blocks/SM variant,
      `smsp__inst_executed.sum` is within 20% of the champion's 5.46 B
      (or higher; nochain is expected to be higher).

#### Manual Verification
- [ ] Compare PTX of `v22_lb6` vs `v21_idealized` at B=5 ET=8 and confirm
      the compiler inserted spill stores (`STL` / `LDL` opcodes) proportional
      to the spill-bytes delta.

**Implementation Note**: After completing Phase 1, pause and confirm that
at least one variant achieves 6 blocks/SM in the table before proceeding
to Phase 2 analysis. If none do, revisit the levers.

---

## Phase 2: Benchmark Sweep Harness

### Overview
Extend the existing `tools/bench_idealized_ceiling.py` harness with v22
variants, per-variant shmem calculations (same as v21 idealized for lb6 /
nochain / floor; same as v21 idealized for tg64), and TG_SIZE override.

### Changes Required

#### 1. New harness
**File**: `tools/bench_6blk_ceiling.py`
**Changes**: Copy `tools/bench_idealized_ceiling.py` and:

- Add the four v22 paths to `VARIANT_PATHS`.
- Add a `TG_SIZE_BY_VARIANT` dict (128 for lb6/nochain/floor, 64 for tg64).
- Recompute `MAX_FPT = ceil(N_FREQ / TG_SIZE)` per variant.
- Default sweep: `--b-scat 1 3 5 9`, `--elem-tile 1 2 4 8`. For `v22_floor`
  only B=1 ET=1 is evaluated (hardcoded).
- Print a single summary line per variant identifying the fastest 6-blocks/SM
  config and the fastest overall.

Example table row format (unchanged from exp19):
```
    variant   B  ET   shmem  regs  spill  blk/SM   best_ms   med_ms   M scat/s  vs champ  vs ideal
```
with two ratio columns: vs v15 champion (6.44 ms) and vs v21 idealized
best (4.72 ms). Both baselines are hardcoded constants; we do NOT re-run
them here.

#### 2. NCU profile harness
**File**: `tools/ncu_profile_v22.py`
**Changes**: Copy `tools/ncu_profile_v21.py`, add v22 variants to
`VARIANT_PATHS`, support `--tg-size` flag (default 128; 64 for tg64).

### Success Criteria

#### Automated Verification
- [ ] `uv run python tools/bench_6blk_ceiling.py` runs to completion and
      prints a table.
- [ ] Every non-error row has all columns populated.
- [ ] At least one row with `blk/SM >= 6` exists in the table.
- [ ] Summary line identifies `Best at >=6 blk/SM: <variant> B=<b> ET=<et>
      | <ms> ms = <sps> M scat/s` and `Fastest overall: <same or different>`.

#### Manual Verification
- [ ] At least one v22 variant beats the 21.17 M ceiling OR the experiment
      doc explains clearly why not (with evidence from NCU in Phase 3).

---

## Phase 3: NCU Profile at 6 Blocks/SM

### Overview
Profile the fastest 6-blocks/SM config and (if different) the fastest
overall config. Compare to the two exp19 NCU reports (idealized B=5 ET=8
and idealized B=5 ET=4).

### Commands

```bash
# Fastest 6-blocks/SM config (placeholder: assume v22_lb6 B=5 ET=8 wins)
sudo /usr/local/cuda/bin/ncu --target-processes all \
    --launch-skip 1 --launch-count 1 --set full \
    -f -o 4090_v22_6blk.ncu-rep \
    $(which uv) run python tools/ncu_profile_v22.py \
        --variant lb6 --b-scat 5 --elem-tile 8 --blocks 256

# Fastest overall if different
sudo /usr/local/cuda/bin/ncu --target-processes all \
    --launch-skip 1 --launch-count 1 --set full \
    -f -o 4090_v22_fastest.ncu-rep \
    $(which uv) run python tools/ncu_profile_v22.py \
        --variant <winner> --b-scat <b> --elem-tile <et> --blocks 256

# Parse both
sudo /usr/local/cuda/bin/ncu --import 4090_v22_6blk.ncu-rep --page raw \
    | uv run python tools/ncu_parse.py
```

### Success Criteria

#### Automated Verification
- [ ] `ls 4090_v22_6blk.ncu-rep` exists.
- [ ] `tools/ncu_parse.py` produces a clean metric table from the report.
- [ ] `smsp__inst_executed.sum` within 20% of 5.46 B (DCE check).
- [ ] `launch__occupancy_per_block_size.bounds_lower` reports 6 for the
      6-blocks/SM config.

#### Manual Verification
- [ ] Metric deltas vs v21 idealized B=5 ET=8 extracted for:
      `sm__warps_active.avg.pct_of_peak_sustained_active`,
      `smsp__pcsamp_warps_issue_stalled_long_scoreboard.pct`,
      `smsp__pcsamp_warps_issue_stalled_wait.pct`,
      `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active`,
      `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` (IPC proxy).
- [ ] The dominant stall at 6 blocks/SM is identified and reported.

---

## Phase 4: Analysis and Experiment Document

### Overview
Write `exp20-6-blocks-per-sm.md` mirroring the structure of
`exp19-idealized-ceiling.md`, and add the row to the progress tracker.

### Changes Required

#### 1. Experiment document
**File**: `docs/progress/experiments/exp20-6-blocks-per-sm.md`
**Changes**: New file with sections:

- Goal: measure the empirical ceiling at 6 blocks/SM.
- Hypothesis: state one of {"warps_active will finally rise and `wait`
  stall will drop", "warps_active will stay at 16.5% because active warps
  are scheduler-limited, not pool-limited", "spill from lb6 will erase
  the occupancy win"}. Pick the one most consistent with exp19 evidence.
- Results: full sweep table, fastest-per-variant table, ablation table
  (lever by lever).
- NCU comparison(s): v22 best-6blk vs v21 idealized B=5 ET=8.
- Key findings.
- Answer to: "does 6 blocks/SM beat the 21.17 M ceiling?"
  If yes, by how much and what changed. If no, what constraint binds next.
- Recommendations for the next kernel direction (likely: confirm exp19's
  "algorithmic change required" conclusion or revise it).
- Sanity checks (DCE, shmem allocation, launch-bound honored).

#### 2. Progress tracker update
**File**: `docs/progress/cuda-kernel-optimization.md`
**Changes**: Add one row after the exp19 row with the exp20 headline
result and update the Optimization Strategy section if the conclusion
materially changes.

### Success Criteria

#### Automated Verification
- [ ] `exp20-6-blocks-per-sm.md` exists with all stated sections
      (goal, hypothesis, results, ncu, findings, answer, recommendations,
      sanity).
- [ ] `cuda-kernel-optimization.md` has a new row referencing exp20.
- [ ] All numbers in the doc cross-check against the raw
      `bench_6blk_ceiling.py` output captured inline.

#### Manual Verification
- [ ] Ablation section identifies the single biggest contributor to the
      6-blocks/SM time (one of: lb6, nochain, tg64, floor).
- [ ] Document states plainly whether the exp19 "30M requires algorithmic
      change" conclusion stands or needs revision.

---

## Testing Strategy

### Automated checks
- Ruff lint on new tools scripts: `uv run ruff check tools/bench_6blk_ceiling.py
  tools/ncu_profile_v22.py`.
- Compile check for each variant is implicit in the harness (a `COMPILE:`
  error fails the row).
- DCE sanity: `smsp__inst_executed.sum` within 20% of 5.46 B.

### Manual checks
- Visual inspection of the first `__launch_bounds__` PTX to confirm reg
  allocation ceiling is honored.
- Confirm the DCE sink is present in all four variant files.
- Confirm harness printed `blk/SM >= 6` for at least one config.

## Performance Considerations

- `__launch_bounds__(128, 6)` will cause heavy spills on the current
  idealized body (255 -> 85 regs means ~170 scalars must be spilled per
  thread). Expect spill bytes well over 1 KB per thread -> L1 pressure.
  This is part of what the experiment measures.
- `TG_SIZE=64` halves the theoretical warps/SM from 48 to 24 even at
  12 blocks/SM. If the kernel is warp-issue-bound (which exp19 suggests
  via `wait` stall dominance), this may regress.
- Phase 3 chain break in `nochain` adds roughly `B*ET*my_n_freq` extra
  `__sincosf` calls per scatterer. For B=5 ET=8 my_n_freq<=1 this is
  ~40 extra sincosf per scatterer; acceptable, and it keeps the SFU pipe
  from collapsing to idle (which would be a DCE signal).

## Migration Notes

None. All v22 variants are diagnostic. No runtime code path changes.
Variants should be deleted after exp20 is written if they end up being
dead weight, but for now keep them so the numbers are reproducible.

## References

- Prior experiment: `docs/progress/experiments/exp19-idealized-ceiling.md`
- Prior plan: `docs/progress/plans/2026-04-07-idealized-kernel-ceiling.md`
- Base kernel: `src/fast_simus/kernels/simus_fused_v21_idealized.cu`
- Base harness: `tools/bench_idealized_ceiling.py`
- Base NCU harness: `tools/ncu_profile_v21.py`
- NCU parse: `tools/ncu_parse.py`
- Progress tracker: `docs/progress/cuda-kernel-optimization.md`
- Related prior profile: `docs/progress/experiments/exp18-30M-architecture.md`
