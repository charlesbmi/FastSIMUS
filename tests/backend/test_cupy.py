"""CuPy-specific tests for FastSIMUS.

Tests in this module verify features that require CuPy and go beyond the
Array API abstraction: ``cupy.RawModule`` NVRTC compile path, kernel
cache behavior, and CUDA device placement.
"""

import numpy as np
import pymust
import pytest

from tests.conftest import _cupy_has_cuda_device

cp = pytest.importorskip("cupy")
if not _cupy_has_cuda_device(cp):
    pytest.skip("CuPy CUDA device not available", allow_module_level=True)

from fast_simus import BackendKind
from fast_simus.kernels._cuda_capabilities import (
    DEFAULT_DYNAMIC_SHARED_MEMORY_BYTES,
    cuda_shared_memory_unsupported_reason,
    required_cuda_shared_memory,
)
from fast_simus.kernels.cuda_simus import _get_kernel, _kernel_cache
from fast_simus.simus import simus
from fast_simus.transducer_presets import C5_2v, L11_5v, P4_2v
from fast_simus.utils.geometry import element_positions


def test_kernel_cache_hits_on_repeat_call():
    """Same shape -> same RawKernel object (no recompile)."""
    _kernel_cache.clear()
    k1 = _get_kernel(64, 1, 854)
    k2 = _get_kernel(64, 1, 854)
    assert k1 is k2


def test_kernel_cache_miss_on_different_shapes():
    """Different shape -> different compile."""
    _kernel_cache.clear()
    k1 = _get_kernel(64, 1, 854)
    k2 = _get_kernel(64, 1, 600)
    assert k1 is not k2


def test_shmem_under_default_cap():
    """Pinned config must fit under the 48 KB default dynamic-shmem cap."""
    assert required_cuda_shared_memory(64, 1) < DEFAULT_DYNAMIC_SHARED_MEMORY_BYTES
    assert required_cuda_shared_memory(128, 1) < DEFAULT_DYNAMIC_SHARED_MEMORY_BYTES


def test_device_shared_memory_limit_matches_cuda_eligibility():
    """CUDA eligibility reflects the active device rather than a fixed ceiling."""
    params = L11_5v()
    reason = cuda_shared_memory_unsupported_reason(params.n_elements, 2)

    if reason is None:
        assert (
            required_cuda_shared_memory(params.n_elements, 2)
            <= cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
        )
    else:
        assert "device limit" in reason


def test_simus_cuda_on_probe_face_is_finite():
    """A scatterer at an element center on z=0 must not create NaNs."""
    params = P4_2v()
    elements, _, _ = element_positions(params.n_elements, params.pitch, params.radius, cp)
    scatterers = cp.reshape(elements[params.n_elements // 2], (1, 2))
    coefficients = cp.ones(1, dtype=cp.float32)
    delays = cp.zeros(params.n_elements, dtype=cp.float32)

    result = simus(
        scatterers,
        coefficients,
        delays,
        params,
        backend=BackendKind.CUDA,
        element_splitting=1,
    )

    assert bool(cp.all(cp.isfinite(result.rf)))
    assert bool(cp.all(cp.isfinite(result.spectrum)))


def test_simus_cuda_l11_5v_recompile():
    """L11-5v verifies a distinct 128-element NVRTC compile path."""
    params = L11_5v()
    n_scat = 4
    scat = cp.asarray(np.stack([np.zeros(n_scat), np.linspace(1e-2, 4e-2, n_scat)], axis=-1).astype(np.float32))
    rc = cp.ones(n_scat, dtype=cp.float32)
    delays = cp.zeros(params.n_elements, dtype=cp.float32)

    result = simus(scat, rc, delays, params, backend=BackendKind.CUDA, element_splitting=1)
    assert result.rf.shape[1] == params.n_elements
    assert bool(cp.all(cp.isfinite(result.rf)))


@pytest.mark.parametrize(
    ("probe_name", "preset"),
    [("P4-2v", P4_2v), ("L11-5v", L11_5v), ("C5-2v", C5_2v)],
)
def test_simus_cuda_matches_portable_and_pymust(probe_name, preset):
    """CUDA agrees with portable CuPy and PyMUST for supported probes."""
    params = preset()
    n_scat = 6
    x = np.linspace(-1e-2, 1e-2, n_scat).astype(np.float32)
    z = np.linspace(1.5e-2, 6e-2, n_scat).astype(np.float32)
    scatterers = cp.asarray(np.stack([x, z], axis=-1))
    rc = cp.ones(n_scat, dtype=cp.float32)
    delays_np = np.zeros(params.n_elements, dtype=np.float32)
    delays = cp.asarray(delays_np)
    fs = 4.0 * params.freq_center

    cuda = simus(
        scatterers,
        rc,
        delays,
        params,
        fs=fs,
        element_splitting=1,
        backend=BackendKind.CUDA,
    )
    portable = simus(
        scatterers,
        rc,
        delays,
        params,
        fs=fs,
        element_splitting=1,
        backend=BackendKind.CUPY,
    )

    pymust_params = pymust.getparam(probe_name)
    pymust_params.fs = fs
    options = pymust.utils.Options()
    options.dBThresh = -60.0
    options.ElementSplitting = 1
    reference, _ = pymust.simus(x, z, np.ones(n_scat), delays_np[None, :], pymust_params, options)

    cuda_rf = cp.asnumpy(cuda.rf)
    portable_rf = cp.asnumpy(portable.rf)
    min_len = min(cuda_rf.shape[0], portable_rf.shape[0], reference.shape[0])
    cuda_rf = cuda_rf[:min_len]
    portable_rf = portable_rf[:min_len]
    reference = reference[:min_len]
    portable_peak = max(float(np.max(np.abs(cuda_rf))), float(np.max(np.abs(portable_rf))))
    reference_peak = max(float(np.max(np.abs(cuda_rf))), float(np.max(np.abs(reference))))

    np.testing.assert_allclose(cuda_rf, portable_rf, rtol=0.0, atol=5e-3 * portable_peak)
    np.testing.assert_allclose(cuda_rf, reference, rtol=0.0, atol=2e-2 * reference_peak)
