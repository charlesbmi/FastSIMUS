# Exp 9: v11 fp16 TX Buffer + Parameter Optimization

Epic: FastSIMUS-d7q

## Hypothesis

The TX buffer (2 * B_SCAT * N_FREQ floats) consumes 78% of shared memory at B=5.
Compressing it to fp16 halves its footprint, potentially allowing higher B_SCAT
(fewer atomics) while maintaining >= 2 blocks/SM.

Secondary: staggering element group processing order across blocks might reduce
L2 atomic cache line contention.

## Configuration

Base: v11 B=5 ET=8 (champion at 32.5ms, 3.08M scat/s)
Hardware: RTX A4000, 48 SMs, clocks locked 1560 MHz
N_FREQ=854, N_ELEM=64, N_ES=64 (N_SUB=1), TG_SIZE=128

## v15 fp16 TX Sweep Results

| B  | ET | shmem KB | regs | local | blk/SM | ms   | scat/s  |
|----|-----|---------|------|-------|--------|------|---------|
| 5  | 4   | 26.2    | 128  | 64B   | **3**  | 36.1 | 2.77M   |
| 5  | 6   | 26.2    | 168  | 64B   | **3**  | 35.9 | 2.78M   |
| **5** | **8** | **26.2** | **224** | **64B** | **2** | **31.6** | **3.17M** |
| 6  | 4   | 31.3    | 144  | 64B   | **3**  | 34.2 | 2.92M   |
| 6  | 8   | 31.3    | 254  | 64B   | 2      | 32.6 | 3.07M   |
| 7  | 4   | 36.4    | 167  | 64B   | 2      | 34.6 | 2.89M   |
| 7  | 8   | 36.4    | 255  | 96B   | 2      | 34.0 | 2.94M   |
| 8  | 8   | 41.4    | 255  | 240B  | 2      | 41.3 | 2.42M   |
| 9  | 4   | 46.5    | 192  | 72B   | 2      | 33.4 | 3.00M   |
| 10 | 4   | 51.6    | 208  | 72B   | **1**  | 48.0 | 2.08M   |

## v11-stagger Results (Element Group Reordering)

v11 B=5 ET=8 with `eg = (ego + blockIdx.x) % N_ELEM_GROUPS`:
- Time: 32.8ms (3.05M scat/s) -- no improvement (within noise of 32.5ms)

## Accuracy Analysis (v15 fp16 TX vs v11 fp32)

N_SCAT=10K, comparing identical inputs:

| Metric         | re      | im      | magnitude |
|----------------|---------|---------|-----------|
| Max rel error  | 4.68    | 22.2    | **2.74%** |
| Mean rel error | 0.14%   | 0.24%   | 0.020%    |
| Median         | 0.020%  | 0.021%  | 0.012%    |
| P99            | 1.31%   | 1.37%   | 0.14%     |

The max errors on re/im are at near-zero crossings (division by ~0). Magnitude
error is more meaningful: P99 of 0.14% is acceptable, but max of 2.7% may not
meet the project's rtol=1e-4 validation tolerance.

## Key Findings

### 1. fp16 TX gives only ~2.5% improvement

v15 B=5 ET=8 at 31.6ms vs v11's 32.5ms. The improvement comes from reduced
shared memory (26.2 KB vs 42.9 KB) and slightly lower register count (224 vs 228),
not from enabling higher B_SCAT.

### 2. Higher B_SCAT hits the register wall

B=7+ with ET=8 pushes registers to the 255 limit, causing spills that negate the
atomic reduction benefit. B=9 ET=4 achieves 33.4ms at 2 blocks/SM (promising
but ET=4 lacks ILP). The B_SCAT * ELEM_TILE product drives cv/sv register arrays
(4 floats per pair), creating a hard trade-off between batching and ILP.

### 3. Three blocks/SM does NOT help

B=5 ET=4 achieves 3 blocks/SM (vs 2 for ET=8) but is 14% slower (36.1ms vs 31.6ms).
This further confirms that occupancy is NOT the performance lever -- ILP (from
higher ELEM_TILE) matters more than thread-level parallelism.

### 4. Atomic staggering has no measurable effect

Rotating element group start offsets per block does not reduce L2 atomic pressure.
The contention appears to come from aggregate atomic volume, not same-cycle
cache line collisions between specific blocks.

### 5. Register pressure is the binding constraint for higher B_SCAT

| B  | ET | cv/sv regs | Total | Max(255) headroom |
|----|-----|-----------|-------|-------------------|
| 5  | 8   | 160       | 224   | 31                |
| 6  | 8   | 192       | 254   | 1                 |
| 7  | 8   | 224       | 255   | 0 (spilling)      |
| 8  | 8   | 256       | 255   | -1 (heavy spill)  |

## Conclusion

The v11 architecture is operating near its theoretical optimum for the A4000.
The remaining 3.16x gap to 10ms cannot be closed by parameter tuning or
incremental shmem optimization. It requires either:

1. **Different hardware** (more SMs, higher atomic throughput)
2. **Algorithmic change** (reduce total work, not just reorganize it)
3. **Mixed-precision compute** (fp16 FMA at 2x throughput -- but not practical
   due to SM86's scalar fp16 having no throughput advantage over fp32; only
   packed __half2 helps, which doesn't suit complex multiplication)
