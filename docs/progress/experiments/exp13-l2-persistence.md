# Exp 13: L2 Cache Persistence Hints

**Hypothesis**: Pinning the output array (438 KB) in L2 cache via
`CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW` with `hitProp=PERSISTING`
ensures atomicAdd operations always hit L2 (never DRAM), reducing latency.

**Result**: **No measurable effect.** 32.7ms with persistence vs 32.6ms
without (within measurement noise).

## Setup

- GPU: RTX A4000 (48 SMs, 4 MB L2 cache), clocks locked at 1560 MHz
- Kernel: v11 B=5 ET=8, 192 blocks
- Output array: 64 x 854 x 2 x 4 = 438 KB (fits in L2)
- Used CUDA driver API to set persisting L2 cache window:
  1. `cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, 438KB)`
  2. `cuStreamSetAttribute(stream, ACCESS_POLICY_WINDOW, {persisting})`

## Results

| Config | Run 1 | Run 2 | Run 3 | Median | vs control |
|--------|-------|-------|-------|--------|-----------|
| L2 persist ON | 32.8ms | 32.7ms | 32.6ms | **32.7ms** | +0.3% |
| L2 persist OFF | 32.9ms | 32.6ms | 32.5ms | **32.6ms** | baseline |

## Analysis

The output array is already L2-resident during kernel execution without hints.

At 438 KB, the output array occupies ~11% of the A4000's 4 MB L2 cache.
The kernel's atomicAdd operations create L2 cache lines on first access,
and subsequent atomics to the same cache line hit L2 naturally. With only
438 KB of output data and 100s of atomics per cache line per kernel
invocation, the output working set has very high temporal locality.

The persistence hint is most useful when:
- Multiple kernels share L2 and evict each other's data
- The working set is large relative to L2 capacity
- Data is accessed infrequently (low temporal locality)

None of these apply here: v11 is a single persistent kernel with a
small, heavily-accessed output array.

## Relevance to RTX 4090

Even less relevant on the 4090 (72 MB L2). The output array (438 KB)
is 0.6% of L2 capacity. L2 persistence hints would have zero effect.

The technique could matter if the two-pass architecture (Exp 8) is used:
the TX intermediate buffer is 682 MB (far exceeds L2). Pinning the
output array while streaming the TX buffer could improve L2 utilization.
But even then, the 4090's 72 MB L2 provides enough capacity for natural
caching of the hot output region.
