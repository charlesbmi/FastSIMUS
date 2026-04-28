# Exp 19: Idealized Kernel Ceiling on RTX 4090

## Goal

Measure the empirical performance ceiling by progressively relaxing the
champion kernel (v15 B=9 ET=4, 14.55 M scat/s @ 6.88 ms) toward an
**idealized** kernel that preserves compute volume but removes per-se
shmem state, per-se register pressure, and atomic contention. The result
bounds what a realistic redesign can achieve toward the 30 M scat/s goal.

## Hypothesis

Based on the exp18 profile (cmul `wait` stall 0.61, 2 blocks/SM, shmem
47.6 KB split 34% geo / 63% TX / 3% delay, L2 atomic pressure 32.8%), the
expected ranges per variant were:

| Variant | Expected range | Reasoning |
| --- | --- | --- |
| v15 noatom | 6.75-6.95 ms | Atomic 32.8% of L2, but fully hidden behind cmul wait (exp18 Phase 1) |
| v21 constfreq | 5.0-6.5 ms | -4 geo arrays (9.2 KB at B=9), plus sincosf hoisting |
| v21 singleelem | 4.5-6.0 ms | Geo collapses to 7 per-si scalars, -15.8 KB shmem at B=9 |
| v21 idealized | 3.0-5.5 ms | All three, -17 KB shmem, occupancy unlocked |

If idealized reached ~3-4 ms (22-30 M scat/s), 30M would be plausible
through realistic redesign. If it capped ~5 ms (20 M), 30M requires
algorithmic change (fp16 compute, half2 packing, or reduced work).

## Results

### Sweep (B_SCAT x ELEM_TILE, best of 5 wall-clock reps, 100K scatterers, 256 blocks, boost clocks)

```
    variant   B  ET   shmem  regs  spill  blk/SM   best_ms   med_ms   M scat/s  vs champ
 v15 champ.   9   4   46.5K   192    72B       2      6.44     6.50      15.53    1.000x
     noatom   5   4   26.2K   127    64B       3      6.60     6.61      15.16    0.976x
     noatom   5   8   26.2K   255    64B       2      6.33     6.34      15.81    1.018x
     noatom   9   4   46.5K   237    72B       2      6.66     6.68      15.02    0.967x
     noatom   9   8   46.5K   255   384B       2     10.31    10.46       9.70    0.625x
     noatom  13   4   66.9K   254    72B       1      9.56     9.56      10.46    0.674x
     noatom  17   4   87.2K   255   248B       1     11.95    11.96       8.37    0.539x
     noatom  21   4+   (exceeds 100 KB per-block shmem cap, excluded)
  constfreq   5   4   21.2K   168   144B       3      6.18     6.18      16.18    1.042x
  constfreq   5   8   21.2K   255   144B       2      5.67     5.68      17.64    1.135x
  constfreq   9   4   37.5K   255   216B       2      5.92     5.93      16.89    1.088x
  constfreq   9   8   37.5K   255   536B       2      9.22     9.22      10.85    0.699x
  constfreq  13   4   53.9K   255   560B       1      9.31     9.32      10.74    0.691x
 singleelem   5   4   16.8K   128    64B       4      5.55     5.57      18.01    1.159x
 singleelem   5   8   16.8K   255    64B       2      5.13     5.13      19.51    1.256x
 singleelem   9   4   30.3K   247    72B       2      5.13     5.13      19.50    1.256x
 singleelem  13   4   43.7K   254    72B       2      5.69     5.70      17.57    1.131x
  idealized   5   4   16.8K   168   144B       3      4.97     4.97      20.13    1.296x
  idealized   5   8   16.8K   255   144B       2      4.72     4.73      21.17    1.363x
  idealized   9   4   30.1K   255   216B       2      4.80     4.81      20.85    1.342x
  idealized  13   4   43.5K   255   552B       2      7.13     7.16      14.02    0.903x
```

Per-variant best configs:

| Variant | Best config | Time | Throughput | vs champ |
| --- | --- | --- | --- | --- |
| v15 champion | B=9 ET=4 | 6.44 ms | 15.53 M | 1.00x |
| v21 noatom | B=5 ET=8 | 6.33 ms | 15.81 M | 1.02x |
| v21 constfreq | B=5 ET=8 | 5.67 ms | 17.64 M | 1.14x |
| v21 singleelem | B=5 ET=8 | 5.13 ms | 19.51 M | 1.26x |
| v21 idealized | B=5 ET=8 | **4.72 ms** | **21.17 M** | **1.36x** |

### Ablation (best config per variant, sorted by constraint released)

| Step | Best config | ms | M scat/s | Delta vs prev | Cumulative |
| --- | --- | --- | --- | --- | --- |
| v15 champion | B=9 ET=4 | 6.44 | 15.53 | — | 1.000x |
| + no atomic (noatom) | B=5 ET=8 | 6.33 | 15.81 | **+1.7%** | 1.018x |
| + constfreq | B=5 ET=8 | 5.67 | 17.64 | **+11.7%** | 1.135x |
| + singleelem (alone, not combined) | B=5 ET=8 | 5.13 | 19.51 | — | 1.256x |
| + idealized (all three) | B=5 ET=8 | 4.72 | 21.17 | **+9.3% vs singleelem alone** | **1.363x** |

The biggest contributor is **shmem collapse via singleelem** (+12.4% by
itself), followed by **constfreq** (+11.7%), with atomic removal a distant
third (+1.7%). This confirms exp18's finding that the L2 atomic pressure
(32.8% of peak) is fully hidden behind the cmul `wait` stall.

### NCU comparison (idealized B=5 ET=8 vs v15 champion)

| Metric | Champion (B=9 ET=4) | Idealized (B=5 ET=8) | Delta |
| --- | --- | --- | --- |
| Kernel time (ncu base clock) | 7.84 ms | 5.90 ms | **-25%** |
| Registers/thread | 192 | 255 (hard cap) | +33% |
| Local spill | 72 B | 144 B | +100% |
| Dynamic shmem/block | 47.64 KB | 17.15 KB | **-64%** |
| Occupancy limit (regs) | 2 blocks/SM | **2 blocks/SM** | same |
| Occupancy limit (shmem) | 2 blocks/SM | 3 blocks/SM | shmem no longer binds |
| Occupancy (achieved) | 16.67% | 16.67% | **unchanged** |
| SM throughput | 61.2% | 65.59% | +4.4 pts |
| FMA pipe | 45.9% | 49.42% | +3.5 pts |
| LSU pipe | 40.7% | **27.45%** | -13.3 pts |
| XU/SFU pipe | 23.1% | 9.67% | -13.4 pts |
| ALU pipe | 11.5% | 19.47% | +8.0 pts |
| IPC | 2.45 | 2.62 | +7% |
| L2 atomic pressure | 32.8% | **0%** | as designed |
| DRAM throughput | minor | 0.02% | L1/L2 resident |
| L2 hit rate | 99.99% | 99.94% | same |
| `wait` stall | 0.61 | 0.57 | -6.6% |
| `short_scoreboard` stall | 0.19 | 0.11 | -42% (shmem reads) |
| `long_scoreboard` stall | 0.15 | 0.06 | -60% (spill/L2 reads) |
| `smsp__inst_executed.sum` | 5.46 B | **4.43 B** | -18.9% (within 20% tolerance) |

### B=5 ET=4 idealized (3 blocks/SM) ncu comparison

A second idealized config reached 3 blocks/SM via lower ELEM_TILE:

| Metric | idealized B=5 ET=8 | idealized B=5 ET=4 |
| --- | --- | --- |
| Time | 5.90 ms | 6.23 ms |
| Registers/thread | 255 | 168 |
| Occupancy (achieved) | 16.67% | 25% |
| Warps active | 16.45% | 16.52% |
| SM throughput | 65.59% | 64.88% |
| FMA pipe | 49.42% | 47.82% |
| IPC | 2.62 | 2.59 |

Surprisingly, **more blocks/SM did not help** — `sm__warps_active` is
identical (16.5%). Because Phase 3 dominates and each thread already
holds its own B*ET partial accumulators, adding more concurrent blocks
doesn't add more *active* warps, it just adds ready-to-issue backup warps
that are already covered by the existing 8 warps/SM. The ET=8 unrolling
of the Phase 3 inner loop gives more ILP than the occupancy delta, so
ET=8 wins despite lower blocks/SM.

## Key Findings

1. **Empirical ceiling is 21.17 M scat/s** (4.72 ms) at B=5 ET=8 with all
   three relaxations applied. That is a 1.36x headroom over the v15
   champion's 15.53 M / 6.44 ms.

2. **Occupancy does not rise** beyond 16.67% even when shmem limiter is
   fully relaxed — register pressure caps at 255 regs/thread and the
   compiler aggressively converts formerly-shmem state into registers.
   Lower-register configs (B=5 ET=4, 168 regs, 3 blocks/SM) do not beat
   higher-register higher-ILP configs.

3. **The `wait` stall does not move much** (0.61 → 0.57). The cmul
   dependency chain in Phase 3's frequency sweep remains the dominant
   latency source. This is a structural algorithmic property, not a
   shmem/occupancy issue.

4. **Shmem-vs-register tradeoff**: collapsing shmem arrays to per-si
   scalars shifts load from LSU (40.7% → 27.45%) to ALU (11.5% → 19.47%)
   and FMA (45.9% → 49.42%). The SM throughput lift is real (+4.4 pts)
   but small, because the kernel is already compute-latency bound, not
   compute-throughput bound.

5. **Atomic removal is a rounding error** (+1.7%) at the current
   operating point. The 32.8% L2 atomic pressure metric is misleading
   in isolation — the atomics themselves fit in the latency already
   covered by the cmul `wait` stall.

## Answer: 30 M scat/s achievable via redesign?

**Not without algorithmic change.** The idealized ceiling is **21.17 M
scat/s = 4.72 ms**. The 30 M target is 3.33 ms, i.e. **1.43x beyond the
idealized ceiling**. No realistic redesign that preserves the current
fp32 cmul chain and per-frequency accumulation can close that gap,
because:

- Even after eliminating all per-se shmem pressure, atomic traffic, and
  maximizing shmem-freed occupancy headroom, warps_active does not rise.
- The `wait` stall stays near 0.6 because the cmul chain is structurally
  serial.
- The SM throughput at ceiling is 65.6%, ~1.5x away from theoretical
  100%. FMA pipe at 49.4% has headroom on paper, but ALU/LSU/XU pipes
  cap the schedulable issue rate.

**Required algorithmic changes to reach 30 M:**

1. **fp16 / half2 compute in Phase 3 cmul chain** (H1 from exp18): each
   cmul becomes `__hfma2` acting on 2 frequencies at once, halving the
   serial chain length per thread. Expected: 1.5-1.8x speedup on top of
   the 21 M ceiling → **32-38 M projected**.

2. **cmul chain pipelining** (H2): break `cv[n+1] = cmul(cv[n], sv)`
   into 2 independent chains sharing `sv2 = cmul(sv, sv)`. Expected
   IPC gain from 2.6 → 3.2, worth ~20% on top.

3. Sparse / culled scatterer-element pairs (H5): not algorithmically
   equivalent, but if accuracy permits, effectively halves the work.

These are NOT idealizations — they preserve (sufficient) numerical
correctness and map to the real kernel. The idealized-ceiling result
says that the "shmem layout + occupancy + atomic" design space is
exhausted; the next wins must come from the arithmetic itself.

## Recommendations

- **Ship H1 (fp16 Phase 3 compute)** as the next kernel (v22). The v19
  partial fp16 kernel in the parallel track already has the plumbing
  (fp16 TX buffer); the missing piece is the fp16 cmul chain.
- **Do NOT pursue further shmem tricks** on the fp32 kernel — the
  idealized ceiling proves the gains top out at 1.36x and cannot close
  the 30M gap.
- **Drop atomicAdd elimination as a standalone effort** — 1.7% is not
  worth the architectural disruption (per-SM output buffers, reduction
  kernel, etc.). Only remove atomics as a side-effect of a broader
  restructure.
- **Keep B=5 ET=8 as a candidate config** for any future kernel. Low
  B reduces register pressure enough to keep ET=8 viable, and ET=8
  provides the ILP for Phase 3.

## Sanity Checks

- `smsp__inst_executed.sum`: champion 5.46 B, idealized 4.43 B. Delta
  -18.9%, within the 20% DCE-detection tolerance. Compute volume is
  preserved (compiler hoisted sincosf/expf out of uniform inner loops
  as expected, no dead-code-elimination blowout).
- Sink pattern `if (acc_re[et] == -1e30f && acc_im[et] == -1e30f) ...`
  present in all four variants; `lts__d_atomic_input_cycles_active = 0%`
  confirms no atomic traffic in idealized.
- B=21 configs fail shmem allocation (>100 KB) — correctly rejected
  by `cuFuncSetAttribute`.

## Files & Commands

- Variants: `src/fast_simus/kernels/simus_fused_v21_{noatom,constfreq,singleelem,idealized}.cu`
- Sweep harness: `tools/bench_idealized_ceiling.py`
- NCU harness: `tools/ncu_profile_v21.py`
- NCU reports: `4090_v21_idealized.ncu-rep` (winner, B=5 ET=8),
  `4090_v21_idealized_b5et4.ncu-rep` (occupancy-3 comparison)

```bash
# Reproduce the sweep
uv run python tools/bench_idealized_ceiling.py

# Reproduce the winner NCU profile
sudo /usr/local/cuda/bin/ncu --target-processes all \
    --launch-skip 1 --launch-count 1 --set full \
    -f -o 4090_v21_idealized.ncu-rep \
    $(which uv) run python tools/ncu_profile_v21.py \
        --variant idealized --b-scat 5 --elem-tile 8 --blocks 256

# Parse the profile
sudo /usr/local/cuda/bin/ncu --import 4090_v21_idealized.ncu-rep --page raw \
    | uv run python tools/ncu_parse.py
```

## References

- Plan: `docs/progress/plans/2026-04-07-idealized-kernel-ceiling.md`
- Prior profile: `docs/progress/experiments/exp18-30M-architecture.md`
- Compute-floor predecessor: exp18 Phase 1 (atomic overhead 1.3%,
  matches the 1.7% measured here at a different operating point)
- Champion kernel: `src/fast_simus/kernels/simus_fused_v15.cu`
- Main progress tracker: `docs/progress/cuda-kernel-optimization.md`
