"""CUDA resource requirements for the custom SIMUS kernel.

This module stays importable without CuPy so dispatch can remain lazy. Device
inspection happens only when CUDA eligibility is evaluated.
"""

from __future__ import annotations

# Pinned tuning from ``cuda_simus``. Keep the resource calculation beside the
# capability policy so dispatch and launch cannot disagree.
CUDA_SCATTERER_TILE = 10

DEFAULT_DYNAMIC_SHARED_MEMORY_BYTES = 48 * 1024


def required_cuda_shared_memory(n_elements: int, n_sub: int) -> int:
    """Return dynamic shared memory required by the fused SIMUS kernel."""
    n_element_segments = n_elements * n_sub
    return (7 * CUDA_SCATTERER_TILE * n_element_segments + 3 * n_elements) * 4


def cuda_dynamic_shared_memory_limit() -> int:
    """Return the active CUDA device's opt-in per-block shared-memory limit."""
    import cupy as cp  # noqa: PLC0415  # Keep optional CUDA dependency lazy.

    return int(cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"])


def cuda_shared_memory_unsupported_reason(n_elements: int, n_sub: int) -> str | None:
    """Explain why a SIMUS shape exceeds the active CUDA device limit."""
    required = required_cuda_shared_memory(n_elements, n_sub)
    device_limit = cuda_dynamic_shared_memory_limit()
    if required <= device_limit:
        return None
    return f"required CUDA shared memory ({required} bytes) exceeds the device limit ({device_limit} bytes)"
