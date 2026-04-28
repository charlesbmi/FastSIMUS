# Exp 22: sv_arr → shmem and ET=2 sweep → 18.56 M scat/s fp32

## Goal

Exp21 left v25b at 16.71 M scat/s (B=7 ET=4) bottlenecked by L2 throughput at 76.5 % from 400 B/thread of `tk` spill.
Block-limit was reg-cap (2 blk/SM, 16 % occupancy).

Two follow-ups were filed:

- **FastSIMUS-frh**: cut `tk` live window via freq-chunked Phase 2/3 (v26 freqchunk).
- **FastSIMUS-99r addendum**: free per-thread regs by spilling sv_arr back to shmem (v25c).

## Variants

### v25c -- sv_arr from shmem

`v25b` cached `sv_arr[B*ELEM_TILE]` in registers (56 floats at B=7 ET=4) for the Phase-3 `cmul` chain. v25c drops that
cache and re-reads `GEO_STP_RX_RE/IM` directly from shmem inside the cmul:

```c
f2 sv_local = {GEO_STP_RX_RE(si)[se], GEO_STP_RX_IM(si)[se]};
cv[idx] = cmul(cv[idx], sv_local);
```

Cost: one extra shmem read per `(si, et, fi)` advance. Bank-conflict free since each thread reads its own `(si, se)`
row.

Numerical: identical to v25b modulo float reordering.

### v26 -- freq-chunked Phase 2/3 (negative result)

Wraps Phase 2 + Phase 3 in an outer `for chunk in N_CHUNKS` loop. Shrinks `tk` live array from `B*MAX_FPT*2` to
`B*CHUNK_FPT*2`.

**Numerical bug**: initial implementation re-init'd the Phase-3 `cv` chain at chunk_start via direct
`__sincosf(GEO_KW_R + chunk_lid_f * GEO_KR_STEP, ...)`. For far scatterers (`rc ≈ 0.2 m`), the phase argument reaches
**5–8 K rad**, where `__sincosf`'s argument reduction introduces ~10⁻³ absolute error. This propagated to ~1.6 × 10⁻³
max mag-rel-err vs v11.

**Fix**: init at `lid_f` (small arg, accurate) and chain-advance `chunk * CHUNK_FPT` cmul steps to reach chunk_start.
Each cmul adds ~1 ulp; the chain matches v25b's drift profile (max err 3 × 10⁻⁵).

**Cost**: each chunk except the first replays the cv-chain advances, roughly +70 % cmul ops in Phase 2 and Phase 3.
Spill drops as expected (B=7 ET=4: 400 B → 240 B), and B=5 even reaches 3 blk/SM, but compute overhead exceeds the L2
savings:

```
v25c B=7 ET=4: 16.98 M scat/s, 5.89 ms, 400 B spill, 2 blk/SM
v26  B=7 ET=4: 11.36 M scat/s, 8.84 ms, 240 B spill, 2 blk/SM (-33 %)
v26  B=5 ET=4: 11.32 M scat/s, 8.83 ms, 176 B spill, 3 blk/SM (-19 %)
```

Conclusion: chunking is the right architectural idea (smaller live window), wrong cost profile (cmul advance + extra
\_\_sincosf in Phase 2 inits dominate the L2 savings).

### ET=2 sweep -- the actual breakthrough

While debugging v26, sweeping `ELEM_TILE` revealed v25b/c performance peaks at **lower ET, higher B** rather than the
prior B=7 ET=4 sweet spot. Smaller `ET` → smaller `cv[B*ET]` chain → more reg headroom for `tk_re/tk_im`.

| variant    | B   | ET  | shmem  | regs | spill | blk/SM | best ms  | M scat/s  |
| ---------- | --- | --- | ------ | ---- | ----- | ------ | -------- | --------- |
| v25b       | 7   | 4   | 13.0 K | 255  | 400 B | 2      | 5.98     | 16.71     |
| v25b       | 10  | 2   | 18.2 K | 255  | 576 B | 2      | 5.40     | **18.52** |
| **v25c**   | 9   | 2   | 16.5 K | 254  | 520 B | 2      | **5.39** | **18.56** |
| v25c       | 8   | 2   | 14.8 K | 255  | 464 B | 2      | 5.50     | 18.19     |
| v15 (fp16) | 9   | 4   | 46.5 K | 192  | 72 B  | 2      | 6.69     | 14.94     |

v25c at B=9 ET=2 is the new fp32 champion (5.39 ms = 18.56 M scat/s), **+5 % over the prior B=7 ET=4 champion** and
**+24 % over v15 fp16**.

## NCU profile of v25c @ B=9 ET=2 vs v25b @ B=7 ET=4

| Metric                  | v25b (B=7,ET=4) | v25c (B=9,ET=2) | Δ                 |
| ----------------------- | --------------- | --------------- | ----------------- |
| Compute (SM) Throughput | 61.21 %         | 63.39 %         | +2.2 pp           |
| L2 Cache Throughput     | 76.50 %         | 64.87 %         | -11.6 pp          |
| L1 Hit Rate             | 18.26 %         | 22.94 %         | +4.7 pp           |
| L2 Hit Rate             | 99.98 %         | 99.97 %         | (still all spill) |
| Eligible Warps / Sched  | 1.00            | 1.03            | +3 %              |
| Achieved Occupancy      | 16.29 %         | 16.45 %         | (reg-bound)       |
| Block Limit Registers   | 2               | 2               | (255 regs)        |
| Local memory            | 400 B/thread    | 520 B/thread    | (more `tk`)       |

Mechanism: smaller `cv[B*ET]` chain (B=9 ET=2 → 18 elements vs B=7 ET=4 → 28) lets the Phase-3 inner loop issue cmul +
FMA at higher ILP without saturating the spill traffic. L2 throughput drops because each cmul advance touches fewer
cv-spill addresses per cycle.

## Forward implications

1. **L2 spill is no longer the only bottleneck.** With L2 at 65 % the kernel is closer to balanced; any further wins
   likely come from reducing local-mem traffic (FastSIMUS-9bj fp16 cv chain) or from raising occupancy beyond 2 blk/SM.

1. **Reg-cap is still the active limiter.** `Block Limit Registers = 2`. To get to 3 blk/SM at this kernel structure
   we'd need ≤ 168 regs/thread (RTX 4090: 65 K regs / SM / 3 blk / 128 threads).

1. **fp16 cv (FastSIMUS-9bj)** is the next high-leverage lever: `__hfma2` packs 2 freqs per advance, halving cv reg
   footprint. Should drop spill enough to either fit `tk` fully in regs or open 3 blk/SM.

1. v26 chunked structure is shelved. Re-evaluate only if a way is found to avoid the cmul-chain init overhead.
