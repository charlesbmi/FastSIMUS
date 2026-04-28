# Exp 16: Two-Pass TX/RX on RTX 4090

## Hypothesis

The two-pass architecture with Grid-Y element partitioning should reduce L2
atomic contention by 8x. On the 4090, the 72MB L2 keeps TX buffer resident
via chunked execution (10K scatterer chunks = 68MB TX buffer per chunk).

## Architecture

1. TX kernel (simus_tx_v14.cu): one block per scatterer, writes TX[scat][freq]
2. RX kernel (simus_rx_gridy_v18.cu): Grid-Y `(blocks_x, N_ELEM_GROUPS, 1)`,
   each y-block handles one element group of ELEM_TILE elements

New file: `src/fast_simus/kernels/simus_rx_gridy_v18.cu`

## Results

| Config | Time | vs v11 |
|--------|------|--------|
| chunk=10K ET=16 rxblk=64 | 34.4ms | 0.25x |
| chunk=10K ET=8 rxblk=64 | 35.0ms | 0.25x |
| chunk=100K ET=8 rxblk=64 | 35.4ms | 0.24x |
| chunk=10K ET=8 rxblk=256 | 42.8ms | 0.20x |

**All configs 4x slower than v11.** Correctness PASS (2.2e-5 relative error).

## Root Cause

The two-pass architecture trades shared memory TX access (~100 TB/s bandwidth)
for L2 TX reads (~5 TB/s). The geometry recomputation per scatterer per element
is identical total work in both approaches. The reduced atomic contention
(~8x fewer atomics) does not compensate for:

1. TX global memory write + L2 read overhead (680MB per pass)
2. Per-scatterer geometry recompute (no batching like v11's B_SCAT=5)
3. 10 kernel launch pairs per 100K scatterers (chunking overhead)

## Conclusion

Two-pass is NOT viable for the SIMUS algorithm on Ada Lovelace, consistent with
the A4000 result (Exp 8b: -43%). The fused kernel's shared memory TX access and
B_SCAT batching are irreplaceable advantages.
