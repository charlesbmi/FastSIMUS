# Plan: v23 Chain-Split — Bring exp20 ILP Restructure to a Correct Kernel

Date: 2026-04-27
Status: **complete — negative result**
Bd: FastSIMUS-3fm

## Result summary (run 2026-04-27)

Base switched from v15 (fp16 TX) to v11 (fp32 TX) per direction to
preserve precision. Two restructure variants implemented:

- `v23_chainsplit`: full prep + FMA + advance three-stage body
- `v23b_advsplit`: lighter — keep cv reads inline, only split the
  chain advance into its own pass

**Numerical equivalence vs v11 (fp32):**
both PASS — max mag err 3.6e-5, p99 2.4e-6, mean 4.0e-7
(pure fp32 FMA-reorder noise).

**Wall-clock (RTX 4090, N_SCAT=100k, blocks=256):**

| Config       | v11 (M scat/s) | v23 chain-split | v23b adv-split | v15 fp16 ref |
| ---          | ---            | ---             | ---            | ---          |
| B=5 ET=4     | 13.13          | 12.25 (-7%)     | 13.04 (~0%)    | 13.12        |
| B=5 ET=8     | 13.07          | 12.21 (-7%)     | 13.11 (~0%)    | 13.26        |
| B=9 ET=4     | 11.37          | 11.37 (0%)      | 11.37 (0%)     | **14.96**    |
| B=9 ET=8     |  6.69 *(spill)*|  7.18 (+7%)     |  7.18 (+7%)    |  8.06        |

The exp20 ILP-restructure win **does not translate** to the correct
fp32 kernel at the regimes that actually perform. Both restructures
are neutral or negative on fp32; the +7% appears only in the
register-spilled B=9 ET=8 corner that we don't ship from anyway.

**Diagnosis.** Exp20's win was *bound to the chain break*. Removing
`cv = cmul(cv, sv)` freed B\*ET registers worth of state, opening
headroom for nvcc to materialise the explicit two-stage scheduling.
On a correct kernel `cv[B*ET]` is permanent live state across the
freq loop, so:

- v23 (full split) adds `rr_arr[B*ET] + ri_arr[B*ET]` scratch on top
  of cv, wins no register dance, and pays a small slowdown.
- v23b (light split) avoids the scratch but the FMA stage still has
  to read cv inline — same RAW hazard the original loop has — and
  splitting only the chain advance turns out to be ~zero benefit.

**Standing fp32-correct champion: v11 B=5 ET=8 at 13.07 M scat/s
(7.65 ms).** v15's 14.96 M is achievable only by accepting fp16 TX.

## Forward implications

The lever that *did* work in exp20 (chain break) is unavailable
without precision loss. The lever that translated cleanly to a
correct kernel (explicit stage decomposition) is unavailable as a
standalone win — it requires register headroom that the correct fp32
kernel doesn't have.

Real next moves (re-prioritized):

1. **TX shmem footprint reduction without precision loss**
   (FastSIMUS-99r). v15's fp16 TX gets 2 blk/SM at B=9 ET=4 vs v11's
   1 blk/SM. *That* is the explanation for v15's 14.96 M. A bf16/tf32
   compromise on the TX buffer that preserves Phase 3 fp32 math is
   probably the single largest correct-kernel win available.

2. **half2 cv chain** (FastSIMUS-9bj). Halves cv register footprint;
   could open the door for v23-style explicit staging to actually win.
   Validate sub-1e-3 accuracy first.

3. **Periodic chain reset.** Every K freqs reset cv via `__sincosf`.
   K=N_FREQ/8 = ~107 trades 8\*B\*ET sincosf for shorter cumulative
   rounding. Numerically lossless if K small enough; opens chain-break
   territory at controlled SFU cost.

(Original 12% from v23 was mis-projected — actual is 0%. v24 half2
  estimate of 1.5-1.8x must be re-validated under the same
  "no register headroom" reality v23 just exposed.)

---

## Original plan (kept for reference)


## Why this plan exists

Exp19 / exp20 produced four idealized kernels (`v21_idealized`,
`v22_lb6`, `v22_nochain`, `v22_nochain_ilp`) that were *not numerically
equivalent* to the correct kernel:

- `noatom`: drops atomicAdd, replaces with DCE-resistant sink
- `constfreq`: derives `kw_r/kr_step/alpha_r/ar_step` from `elem[0]/sub[0]` only; per-element variation is *erased*
- `singleelem`: collapses `GEO_AMP / STP_RX_RE / STP_RX_IM` from per-(scatterer,element) to per-scatterer; per-element variation again erased
- `nochain`: removes the `cv = cmul(cv, sv)` running rotation chain in Phase 3 entirely; `__sincosf(ph_f)` recomputed per freq with element-invariant `ph_f` formula

The combined effect is that v22_nochain_ilp's 34.97 M scat/s is on a
kernel where Phase 3's per-element accumulator is degenerate (all ET
entries at fixed (si, f) compute the same value). The headline result is
real *as a profiling/ceiling experiment* but does not represent a correct
kernel.

Exp20's mechanistic finding — explicit two-stage Phase 3 layout cut ALU
instructions ~40% by removing compiler-emitted index/book-keeping in the
interleaved `for si -> for et -> chain-advance` body — applies independent
of correctness. v23 brings that single learning to the correct v15 kernel.

## What v23 keeps from v22_nochain_ilp

Single change: explicit multi-stage Phase 3 inner body.

```
for fi:                                  // unchanged
  // Stage 1 (prep): snapshot current cv state and load TX
  for si: load tk_re/tk_im, for et: rr/ri = cv[idx] * inv_nsub
  // Stage 2 (FMA): pure accumulation, no chain interleave
  for si: for et: acc_re/im[et] += FMA(tk, rr, ri, pf)
  // Stage 3 (advance): cmul chain, independent across (si, et)
  for si: for et: cv[idx] = cmul(cv[idx], sv_arr[idx])
  // Stage 4 (sink): atomicAdd output (unchanged)
```

The chain `cv = cmul(cv, sv)` is preserved with full per-(si, et) state.
Decoupling its advance from the FMA-using-cv read lets nvcc:

1. Issue all FMAs without a RAW hazard on the cmul of the *previous* (si, et)
   from the same freq.
2. Issue all cmuls back-to-back (B*ET independent multiplies per freq) —
   the same kind of inter-lane ILP that gave v22_nochain_ilp its win on the
   sincosf path.

## What v23 explicitly does NOT do

- **Chain break.** Resynth via `__sincosf(ph_f)` per freq with per-(si,et)
  phase needs B*ET sincosf per freq → SFU pipe saturation at the v15
  champion config (B=9 ET=4 = 36 sincosf/freq vs current ~9). Defer to
  v24+ if at all.
- **`__launch_bounds__` in isolation.** Exp20 confirmed lb6 alone is a
  net loss (best `v22_lb6` reached 6.22 ms vs v15 6.88 ms but with
  unstable register/spill regression at higher B/ET).
- **TG_SIZE=64.** Exp20: 5.78 ms best, weaker latency hiding in smaller TG.
- **No-atom output buffers.** Exp19/exp20: atomic removal is a 1.7%
  rounding error at the current operating point.
- **Half-precision Phase-3 math.** Defer to v24 (see FastSIMUS-9bj).

## Success criteria

- Numerical correctness: `v23` output vs `v15` output
  `max relative error < 1e-5` (same fp16 TX buffer, same arithmetic;
  reordering should be near bit-identical mod fp summation order).
- Wall-clock at v15 champion config (B=9 ET=4): ≤ v15's 6.88 ms (no regression).
- At least one config reaches < 5.5 ms (closing on the 4.72 ms exp19
  ceiling but with full correctness).
- Regs/thread within 220 (vs v15's 192). > 220 risks 1 blk/SM regression.

## Validation plan

1. Build `v23_chainsplit` at the v15 champion sweep configs (B=5/9, ET=4/8).
2. For each config, run against `v15` on identical input, compute
   element-wise `max relative error` over non-zero output mags.
   Threshold: 1e-5 (within fp32 rounding from FMA reorder).
3. Cross-check with `accuracy_compare.py`-style v11 (fp32 reference)
   comparison to confirm v23 is within v15's fp16 quantization envelope
   (~1e-2 vs v11 fp32).
4. NCU profile the winning v23 config at full mode; compare instruction
   count, ALU pipe %, FMA pipe %, wait stall, long_scoreboard stall vs
   v15's exp17 baseline profile.

## Forward roadmap (after v23)

| Order | Issue | Lever | Expected gain |
| --- | --- | --- | --- |
| 1 | FastSIMUS-3fm (v23, this plan) | Chain-split inner body | 10-15% (instruction compression on correct base) |
| 2 | FastSIMUS-9bj (v24 half2) | half2 cmul chain | 1.5-1.8x on top of v23 |
| 3 | FastSIMUS-99r (TX shmem) | Bank-conflict / layout audit | 5-10% (tackles new top stall after v23) |
| 4 | future | Phase 2/3 fusion to drop barrier | 5% |

The combined projection (multiplicative on time): if v23 gains 12% and
v24 gains 60%, time goes 6.88 -> 6.05 -> 3.78 ms = ~26 M scat/s on a
fully correct kernel. That's still short of the 30 M target on a correct
kernel; reaching 30 M correct likely also needs FastSIMUS-99r or a
slightly more invasive Phase 3 rewrite.

## References

- Exp19 (idealized ceiling): `docs/progress/experiments/exp19-idealized-ceiling.md`
- Exp20 (6-blk/SM + ILP): `docs/progress/experiments/exp20-6-blocks-per-sm.md`
- Champion: `src/fast_simus/kernels/simus_fused_v15.cu`
- Validation harness: `tools/accuracy_compare.py`
