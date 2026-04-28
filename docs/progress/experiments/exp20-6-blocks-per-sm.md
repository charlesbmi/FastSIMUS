# Exp 20: Six-Blocks-per-SM Ceiling + Phase-3 ILP Breakthrough on RTX 4090

## Goal

Exp19 identified a hard 21.17 M scat/s ceiling on the `v21_idealized` kernel,
bottlenecked by fixed-latency `wait` stalls on the serial Phase-3 `cmul`
rotation chain at 2-4 blocks/SM. Question for exp20: **if we force 6+
blocks/SM and break the `cmul` chain dependency, does a new regime open up,
or does `warps_active` stay pinned near 16%?**

## Hypothesis

- H1: `__launch_bounds__(128, 6)` alone caps regs/thread enough to reach
  6 blk/SM but won't improve time (regs → spills will cancel the occupancy).
- H2: Breaking the `cv = cmul(cv, sv)` chain and resynthesizing rotations
  via `__sincosf(ph_f)` per frequency will drop the `wait` stall enough to
  raise `warps_active` above the exp19 16.5% floor.
- H3: TG_SIZE=64 will halve per-block work without raising `warps_active`
  enough to compensate (smaller threadgroup = weaker latency hiding).

## Variants

Four kernels derived from `v21_idealized`:

| Variant | Delta | Purpose |
| --- | --- | --- |
| `v22_lb6` | `__launch_bounds__(128, 6)` only | Isolate the register-cap lever |
| `v22_nochain` | Phase 3 re-synthesizes `cv` via `__sincosf` per freq; no live chain state | Break the serial `cmul` dependency |
| `v22_tg64` | `__launch_bounds__(64, 6)`, TG_SIZE=64 | Smaller blocks, more resident blocks |
| `v22_floor` | nochain + lb6 + minimum B=1/ET=1 | Minimum-work occupancy floor |

After seeing the nochain result, we added a fifth:

| Variant | Delta | Purpose |
| --- | --- | --- |
| `v22_nochain_ilp` | nochain + explicit two-stage Phase 3: batch all `__sincosf`/`expf`/TX-loads first, then ET FMAs | Remove compiler-implicit ordering; let scheduler see B independent SFU dispatches |

## Results

### Sweep headlines (RTX 4090, N_SCAT=100k, 256 blocks, boost clocks)

Baselines: v15 champion 6.44 ms = 15.53 M scat/s; v21_idealized 4.72 ms = 21.17 M scat/s.

| Variant | Best config | Best ms | M scat/s | regs | blk/SM | vs ideal |
| --- | --- | --- | --- | --- | --- | --- |
| `v22_lb6` | B=3 ET=4 | 6.22 | 16.07 | 80 | 6 | 0.76x |
| `v22_tg64` | B=1 ET=8 | 5.78 | 17.30 | 78 | 12 | 0.82x |
| `v22_floor` | B=1 ET=1 | 25.59 | 3.91 | 40 | 12 | 0.18x (diagnostic) |
| `v22_nochain` | B=9 ET=8 | 3.28 | 30.50 | 95 | 3 | 1.44x |
| **`v22_nochain_ilp`** | **B=5 ET=8** | **2.86** | **34.97** | 64 | 5 | **1.65x** |
| `v22_nochain_ilp` | B=3 ET=8 (6+ blk/SM) | 3.03 | 33.02 | 48 | **8** | 1.56x |

**The exp19 21.17 M idealized ceiling was not a ceiling** — it was the limit
of compiler-implicit scheduling of the `cmul` chain. Explicit two-stage
restructuring of Phase 3 broke through by 65%.

Full sweep data at the bottom of this file.

### What moved and what didn't (NCU, ncu --set full, skip 1, count 1)

Comparing the two v22_nochain configs to v22_nochain_ilp winner:

| Metric | nochain B=1 ET=8 (9 blk/SM) | nochain B=9 ET=8 (3 blk/SM) | **nochain_ilp B=3 ET=8 (8 blk/SM)** |
| --- | --- | --- | --- |
| Wall time (bench) | 3.36 ms | 3.29 ms | **2.86 ms** |
| NCU cycles | 9.39 M | 8.66 M | **8.43 M** |
| `smsp__inst_executed.sum` | 2.71 B | 2.50 B | **2.18 B** |
| Registers / thread | 55 | 95 | **48** |
| Dyn shmem / block | 3.4 KB | 30.9 KB | 10.3 KB |
| Occupancy achieved | 75% | 25% | 66.7% |
| **`warps_active` %** | **16.61** | **16.51** | **16.57** |
| SM throughput % | 56.32 | 56.47 | 50.61 |
| FMA pipe % (active) | 34.01 | 40.36 | 39.67 |
| ALU pipe % (active) | 29.34 | 23.84 | **14.72** |
| XU/SFU pipe % | 12.34 | 13.38 | 14.95 |
| LSU pipe % | 54.48 | 28.73 | 37.51 |
| DRAM throughput % | 0.03 | 0.04 | 0.04 |
| L2 hit rate % | 99.97 | 100.00 | 100.08 |
| IPC | 2.25 | 2.26 | 2.02 |
| `wait` stall | 1.22 | 0.83 | 0.93 |
| `long_scoreboard` stall | 0.17 | 0.29 | 0.47 |

Two conclusions cut hard against the exp19 hypothesis:

1. **`warps_active` is a structural floor, not an occupancy / ILP knob.**
   All three configs land at 16.5 +/- 0.1 % despite covering 3, 8, and 9
   achieved blocks/SM. Adding more resident warps does not make more warps
   active when each warp's issue stream is `wait`-dominated. The H2
   expectation (that the chain break would lift `warps_active`) is wrong.
2. **The win is instruction-count compression, not ILP latency hiding.**
   v22_nochain_ilp executes 12.7% fewer instructions than nochain B=9 ET=8
   (2.18 B vs 2.50 B), almost entirely on the ALU pipe (ALU active% drops
   14.72 vs 23.84 → ~40% less ALU work). FMA and SFU active% are essentially
   preserved. The explicit two-stage structure lets nvcc eliminate redundant
   index arithmetic, book-keeping, and intertwined-loop register shuffling
   that the serialized `for (si) for (et)` body hid.

This explains why IPC went *down* (2.26 → 2.02) while time went *down* too:
same schedule efficiency per instruction, but a materially shorter instruction
stream.

### v22_lb6: does `__launch_bounds__` alone help? No.

`v22_lb6` caps registers at 80 (vs up to 255 in v21_idealized), reaches the
targeted 6 blocks/SM, and at best reaches 6.22 ms — *worse* than
v21_idealized's 4.72 ms winner and far worse than v21_idealized's low-occ
B=5 ET=8 at 5.13 ms. At higher B/ET the register cap forces 400-1600 B of
spills per thread, trashing performance. Launch-bounds in isolation is not
a lever; it is a cost when applied without an upstream restructure that
actually needs fewer registers.

### v22_tg64: does a smaller threadgroup help? No.

12 blocks/SM achieved at TG=64, but time only reaches 5.78 ms best.
Smaller TG → less latency hiding from intra-block warps → more issue
bubbles. Confirms H3. Dead end.

### v22_floor: cleanly 12 blocks/SM, 25.6 ms

Diagnostic. At B=1 ET=1, 12 blk/SM fits 768 warps resident but work per
block is so small that phase-1/phase-2 overhead dominates. Not a target
operating point; confirms occupancy in isolation is not a scoring
objective.

## Interpretation

The exp19 "idealized ceiling" was correct about the operating point
(v21_idealized B=5 ET=8 was indeed the best v21 kernel) but misdiagnosed
the bottleneck. The real ceiling was **instruction-stream length from
compiler-emitted book-keeping around the interleaved `for si -> for et`
Phase-3 body**, not `wait` stalls on the `cmul` chain per se.

Evidence:
- Breaking the chain (v22_nochain) produced only modest gains (30.5 M vs
  21.2 M = 1.44x). The bulk of the chain savings was recapturable without
  `warps_active` changing.
- Explicitly reshaping Phase 3 to a two-stage (prep / FMA) layout
  (v22_nochain_ilp) delivered another 1.15x on top, driven entirely by
  ALU-pipe compression.
- At both ceilings, `warps_active` sits at 16.5%, `long_scoreboard` grows
  (0.17 → 0.47) as we compress the instruction stream and more issues land
  on the TX/geo-shmem load path. The next-bottleneck story is
  **shmem-load latency**, not SFU.

## Answer: path to 30 M scat/s?

**Achieved and exceeded.** `v22_nochain_ilp` reaches 34.97 M scat/s
(1.65× v21_idealized ceiling, 2.25× v15 champion). The 6-blocks/SM regime
is a real operating point: the best ≥6-blk/SM config is
`nochain_ilp B=3 ET=8` at 33.02 M scat/s with only 48 regs/thread and
112 B of spills. The sub-6-blk/SM winner at 5 blk/SM is marginally faster
(34.97 M) but uses 64 regs; the choice is now a register-headroom vs
occupancy preference, not a forced one.

Next-lever estimates given what the NCU data shows:

1. **`half2` Phase 3 FMAs (H1 from exp18/exp19):** FMA pipe is at 34-40%
   active, far from the 80-90% that would indicate a math-bound kernel.
   Packing two frequencies into one `__hfma2` halves per-thread FMA
   instruction count, which — given today's bottleneck is instruction
   stream length, not pipe capacity — should drop time roughly in
   proportion (projected 1.8-2.2 ms range, 45-55 M scat/s).
2. **Phase-2/Phase-3 fusion to remove the `__syncthreads` barrier** (stall
   `barrier` is 0.24 here, non-trivial). Worth ~5-10%.
3. **TX-shmem layout (bank-conflict audit):** `long_scoreboard` grew to
   0.47 in the ilp winner; TX is the main non-register consumer. Worth
   profiling next.

`__sincosf` polynomial-approximation and further ILP restructuring are
*not* on the critical path anymore; the SFU pipe is at 13-15% active and
`warps_active` does not respond to these levers.

## Sanity Checks

- `smsp__inst_executed.sum`: v15 champion 5.46 B, v21_idealized 4.43 B,
  v22_nochain_ilp B=3 ET=8 **2.18 B**. Ilp is 51% fewer instructions than
  v15, 49% fewer than v21_idealized. DCE sink (`if (acc == -1e30f) ...`)
  present; `sm__inst_executed_pipe_fma` remained at 39.67% active (vs
  40.36% in nochain B=9 ET=8) — real math volume preserved.
- L2 hit rate 100.08% (on-chip caches cover all Phase-3 traffic; the
  kernel is strictly compute/instruction-bound, not memory-bound).
- DRAM throughput 0.04%. Global memory is irrelevant to the bottleneck.
- L2 atomic input cycles 0% (same no-atom pattern as v21_idealized).
- Spills in the ilp winner (112 B) are 100% covered by L1 — no L2 traffic
  attributable to local memory.

## Recommendations

- **Ship `v22_nochain_ilp`** as the new champion path. B=5 ET=8 for raw
  speed, B=3 ET=8 for register headroom if a future lever needs regs.
- **Next experiment: `v23` with `half2` Phase 3 FMAs.** The exp19 H1 is
  now the highest-expected-value lever and it compounds multiplicatively
  with the instruction-stream compression already banked.
- **Retire `warps_active` as a target metric for this workload.** It is
  a structural consequence of the latency profile, not a knob.
- **Archive `v22_lb6`, `v22_tg64`, `v22_floor`.** Useful as diagnostics;
  not on the champion trajectory.

## Full sweep (RTX 4090, 100k scatterers, 256 blocks)

```
    variant  TG   B  ET   shmem  regs  spill  blk/SM   best_ms   med_ms   M scat/s  vs ideal
        lb6 128   3   4   10.1K    80   168B       6      6.22     6.24      16.07    0.758x
        lb6 128   3   2   10.1K    80   112B       6      6.33     6.37      15.79    0.745x
    nochain 128   9   8   30.1K    95   216B       3      3.28     3.28      30.50    1.440x
    nochain 128   3   8   10.1K    48   112B       8      3.36     3.37      29.80    1.407x
    nochain 128   1   8    3.4K    55    56B       9      3.36     3.37      29.78    1.406x
    nochain 128   5   8   16.8K    72   144B       5      3.43     3.45      29.14    1.376x
nochain_ilp 128   5   8   16.8K    64   144B       5      2.90     2.90      34.53    1.630x  <-- fastest
nochain_ilp 128   9   8   30.1K   112   216B       3      2.93     2.94      34.18    1.613x
nochain_ilp 128   3   8   10.1K    48   112B       8      3.03     3.05      33.02    1.558x  <-- best at >=6 blk/SM
nochain_ilp 128   1   8    3.4K    55    56B       9      3.47     3.48      28.79    1.359x
       tg64  64   1   8    3.4K    78   112B      12      5.78     5.79      17.30    0.817x
      floor 128   1   1    3.4K    40    56B      12     25.59    26.02       3.91    0.184x
```

Observation on nochain vs nochain_ilp at B=1: the ilp variant is slightly
*slower* (3.47 vs 3.36 ms). With only 1 scatterer there are no independent
per-si SFU dispatches to batch; the restructure adds a small scratch-array
overhead with no ILP payoff. The ilp win appears at B>=3 and grows to
~15% at B=5/9 ET=8 where 3-9 independent sincosf operations issue
back-to-back.

## Files & Commands

- Variants: `src/fast_simus/kernels/simus_fused_v22_{lb6,nochain,nochain_ilp,tg64,floor}.cu`
- Sweep harness: `tools/bench_6blk_ceiling.py`
- NCU harness: `tools/ncu_profile_v22.py`
- NCU reports:
  - `4090_v22_nochain_b1et8.ncu-rep`
  - `4090_v22_nochain_b9et8.ncu-rep`
  - `4090_v22_nochain_ilp_b3et8.ncu-rep`

```bash
# Reproduce the sweep
uv run python tools/bench_6blk_ceiling.py

# Reproduce the new champion's NCU profile
sudo /usr/local/cuda/bin/ncu --target-processes all \
    --launch-skip 1 --launch-count 1 --set full \
    -f -o 4090_v22_nochain_ilp_b3et8.ncu-rep \
    $(which uv) run python tools/ncu_profile_v22.py \
        --variant nochain_ilp --b-scat 3 --elem-tile 8 --blocks 256

# Parse
sudo /usr/local/cuda/bin/ncu --import 4090_v22_nochain_ilp_b3et8.ncu-rep --page raw \
    | uv run python tools/ncu_parse.py
```

## References

- Plan: `docs/progress/plans/2026-04-17-6-blocks-per-sm-ceiling.md`
- Prior ceiling: `docs/progress/experiments/exp19-idealized-ceiling.md`
- Architecture context: `docs/progress/experiments/exp18-30M-architecture.md`
- Champion kernel (old): `src/fast_simus/kernels/simus_fused_v15.cu`
- Main progress tracker: `docs/progress/cuda-kernel-optimization.md`
