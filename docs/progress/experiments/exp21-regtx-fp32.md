# Exp 21: Register-resident TX → +66% fp32 champion on RTX 4090

## Goal

Exp20 left two artifacts:

1. `v22_nochain_ilp` (idealized, broken Phase-3 chain) reached
   34.97 M scat/s, but is not numerically valid.
2. `v23_chainsplit` / `v23b_advsplit` (correct fp32 builds with the same
   exp20 restructure) returned negative results because the persistent
   `cv[B*ET]` live state across the freq loop kept register pressure
   pinned at 255 — there was no headroom to translate the ILP win.

The standing **correct** champion was `v15` (fp16 TX, 14.96 M scat/s),
limited at higher `B_SCAT` by shmem footprint (1 blk/SM at B=9 ET=4 in
the fp32 v11 path).

Goal: **shrink shmem footprint without dropping precision** so fp32
kernels can match v15's occupancy, then beat it.

## Insight

In v11, `sh_tx_re` / `sh_tx_im` is `2 * B_SCAT * N_FREQ` floats
(60 KB at B=9 / N_FREQ=854 — the dominant shmem term). Closely
reading Phase 2 + Phase 3:

- Phase 2 line 196: thread `lid` writes `sh_tx[si*N_FREQ + f]` for
  `f ∈ {lid, lid+TG_SIZE, lid+2*TG_SIZE, ...}` only.
- Phase 3 line 240: thread `lid` reads `sh_tx[si*N_FREQ + f]` for the
  **same** set of `f`s.

There is no cross-thread sharing of TX. The shmem allocation is purely
a per-thread temporary. **Move it to per-thread registers.** Cost:
`B*MAX_FPT*2` fp32 = 98 floats at B=7 N_FREQ=854 TG=128 (MAX_FPT=7).

## Variants

| Variant | Delta | Purpose |
| --- | --- | --- |
| `v25_regtx` | TX in `tk_re/tk_im[B*MAX_FPT]` reg arrays, drop sh_tx + post-Phase-2 sync | Test the lever |
| `v25b_regtx_unroll` | v25 + `for fi in 0..MAX_FPT` `#pragma unroll` with predicated valid checks (so `fi` is statically known and tk doesn't fall to local memory) | Make tk truly register-resident |

Both are direct numerical-equivalents of v11 (only storage location of
TX changes; arithmetic is identical).

## Results (RTX 4090, N_SCAT=100k, 256 blocks)

### v25 (dynamic fi → tk spilled to local mem)

| Config | Shmem | Regs | Spill | blk/SM | M scat/s | vs v11 |
| --- | --- | --- | --- | --- | --- | --- |
| B=5 ET=4 | 9.5K | 128 | 344 B | 4 | 11.34 | 0.87× |
| B=9 ET=4 | 16.5K | 208 | 576 B | 2 | 13.14 | 1.16× |
| B=9 ET=8 | 16.5K | 255 | 872 B | 2 | 7.81 | 1.16× |

Shmem dropped 76 K → 16 K at B=9 (1 → 2 blk/SM unlocked), but the
entire tk array spilled to local memory because the freq loop bound
(`my_n_freq`, dynamic) prevented unrolling. Net: marginal at B=9,
regression at B=5 (extra L1/L2 traffic from spill outweighed the
1 → 4 blk/SM jump).

### v25b (unrolled fi → tk register-resident)

| Config | Shmem | Regs | Spill | blk/SM | M scat/s | vs v11 same-config | vs v11 best (13.0) | vs v15 fp16 best (14.94) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B=5 ET=4 | 9.5K | 255 | 288 B | 2 | 15.03 | 1.16× | 1.16× | 1.01× |
| B=6 ET=4 | 11.2K | 255 | 344 B | 2 | 16.00 | 1.46× | 1.23× | 1.07× |
| **B=7 ET=4** | **13.0K** | **255** | **400 B** | **2** | **17.68** | **1.66×** | **1.36×** | **1.18×** |
| B=8 ET=4 | 14.8K | 255 | 512 B | 2 | 15.23 | — | — | — |
| B=9 ET=4 | 16.5K | 255 | 744 B | 2 | 15.55 | 1.37× | 1.20× | 1.04× |

**New correct fp32 champion: `v25b @ B=7 ET=4 = 17.68 M scat/s
(5.66 ms)`** — also beats v15 fp16 by +18.4%, on a numerically
correct fp32 kernel.

### Numerical validation (B=7 ET=4, n=54528)

```
v25b vs v11:  max=4.09e-05  p99=2.59e-06  mean=4.38e-07  PASS
```

Pure FMA-reorder noise (no precision loss).

## NCU profile of v25b @ B=7 ET=4

```
Memory Throughput          76.57 %       (L2 bound)
L2 Cache Throughput        76.57 %
L2 Hit Rate                99.97 %
L1/TEX Hit Rate            18.38 %
DRAM Throughput             0.20 %
Compute (SM) Throughput    61.29 %
Achieved Occupancy         16.22 %       (8 warps/SM)
Block Limit Registers           2        (limit on blk/SM)
Block Limit Shmem               4        (shmem could host more)
Warp Cycles / Issued Inst    3.11        (low stall)
Active Threads / Warp       31.85
```

The kernel has shifted from compute/wait-bound (v11 / v22 nochain
regime) to **L2 bandwidth bound on local-memory spill traffic**.
99.97 % L2 hit on the spill keeps it fast, but L2 is saturated.

## Why this works (mechanism)

- v11 burned 60 KB shmem at B=9 storing TX values that no other thread
  ever read. That capped occupancy at 1 blk/SM.
- v25 fixed shmem cost but spilled tk to local memory because dynamic
  `fi` forced indirect addressing.
- v25b's unrolled fi made tk indices compile-time constants. nvcc
  promoted what fits to registers and spilled only the leftover —
  spill dropped from 576 B to 400 B at B=7, and the register-resident
  portion eliminated dependent loads from the FMA path.

## Forward implications

The new bottleneck regime is fundamentally different from exp19/20:

- **Was:** compute pipe saturation, fixed-latency `wait` stalls on
  Phase-3 chain, capped at 21 M scat/s on idealized.
- **Now:** L2 bandwidth saturation from local-memory spill at
  achieved occupancy 16 %.

Levers for the next attack:

1. **Reduce regs/thread** to enable 3 blk/SM (need < 170 regs/thread;
   shmem already allows 4 blocks). This would halve per-block L2
   traffic share. Candidates: smaller `cv[B*ET]` Phase-3 buffer (B=5
   ET=4 already keeps 128 regs but spills more relatively), or
   compress the `cv` chain (FastSIMUS-9bj half2 idea).
2. **Reduce L2 traffic from tk spill** by chunking freqs in Phase 2
   + Phase 3 so only `chunk_size_per_thread` tk values are live at
   once. Trades extra cv re-init __sincosf calls for less spill.
3. **Combine v25b + v15 fp16 storage** to halve tk-spill bytes
   (would drop precision, defer until forced).

For now, FastSIMUS-99r is closed: **the lever did work**, +18% over
the prior fp16-precision champion and +35-66% over v11 fp32.

## Files

- `src/fast_simus/kernels/simus_fused_v25_regtx.cu`
- `src/fast_simus/kernels/simus_fused_v25b_regtx_unroll.cu`
- `tools/bench_v23.py`, `tools/validate_v23.py`, `tools/ncu_profile_v25.py`
- `4090_v25b_b7et4.ncu-rep`
