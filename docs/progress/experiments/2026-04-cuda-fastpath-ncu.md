# CUDA fast-path NCU baseline

## Setup

- GPU: RTX 4090
- Kernel: `simus_fused_kernel` v25c (`B_SCAT=9`, `ELEM_TILE=2`, `TG_SIZE=128`, `TILE_SE=16`)
- Benchmark: `test_bench_simus_scaling[cupy-100000]`
- NCU report: `/tmp/4090_fastpath_wrapper_100k.ncu-rep`

## Fast-path benchmark

Unprofiled pytest-benchmark fast-path result from
`.benchmarks/Linux-CPython-3.12-64bit/0005_74696bdbaff2cfaefc4c857a1f5c951b57ce18ed_20260428_212045_uncommited-changes.json`:

- 100K scatterers: 8.4212 ms mean, 11.87 M scat/s
- 1M scatterers: 60.0641 ms mean, 16.65 M scat/s

The NCU wrapper run itself reported 23.5783 ms mean for the 100K benchmark because profiling overhead is included in the
pytest-benchmark timing. Use the unprofiled JSON above for throughput decisions.

## NCU metrics

Metrics imported from `/tmp/4090_fastpath_wrapper_100k.ncu-rep`:

- Kernel duration: 6.55 ms
- Compute (SM) Throughput: 63.32%
- L2 Cache Throughput: 67.09%
- Achieved Occupancy: 16.47%
- Block Limit Registers: 2 blocks
- Registers Per Thread: 255 registers/thread
- Eligible Warps / Scheduler: 1.03 warp
- Local Memory / Thread: not present in this report's imported details output

Local-memory-related counters present in the raw report:

- `sass__inst_executed_local_loads`: 3,200,256
- `sass__inst_executed_local_stores`: 11,600,928
- `memory_l2_theoretical_sectors_local`: 70,405,632
- Local load L1/TEX hit rate: 30.73%
- Local store L1/TEX hit rate: 9.24%

NCU reported local-memory access warnings:

- Local loads from L2: estimated speedup 45.09%; average 0.9 of 32 bytes utilized per sector.
- Local stores to L2: estimated speedup 59.03%; average 1.0 of 32 bytes utilized per sector.

## Interpretation

The kernel remains register-bound and spill/local-memory-heavy. Occupancy is still capped at 2 blocks/SM by 255
registers/thread, and the local load/store counters plus NCU warnings show inefficient local-memory traffic. Compute and
L2 throughput are close to the exp22 v25c profile, so the fast-path dispatch refactor did not change the underlying
kernel limiter.

The next experiment should be **Experiment C: Kernel tuning from exp22 follow-up**, focused on reducing register
pressure and local-memory traffic. Wrapper prep is a visible 100K fixed cost, but the user's priority is the 1M scale,
where the kernel dominates the time sink.
