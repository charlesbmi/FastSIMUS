"""CuPy-specific tests for FastSIMUS.

Tests in this module verify features that require CuPy and go beyond the
Array API abstraction: ``cupy.RawModule`` NVRTC compile path, kernel
cache behavior, and CUDA device placement.
"""

from importlib import import_module
from typing import cast

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from fast_simus.kernels.cuda_simus import _B_SCAT, _DEFAULT_SHMEM_CAP_BYTES, _get_kernel, _kernel_cache, _shmem_bytes
from fast_simus.simus import SimusStrategy, simus
from fast_simus.transducer_presets import L11_5v, P4_2v
from fast_simus.utils._array_api import Array

simus_mod = import_module("fast_simus.simus")


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
    assert _shmem_bytes(64, 1) < _DEFAULT_SHMEM_CAP_BYTES
    assert _shmem_bytes(128, 1) < _DEFAULT_SHMEM_CAP_BYTES


def test_simus_cuda_output_is_cupy():
    """End-to-end smoke: result stays on the CuPy device."""
    params = P4_2v()
    n = 3
    scat = cp.asarray(np.stack([np.zeros(n), np.linspace(1e-2, 5e-2, n)], axis=-1).astype(np.float32))
    rc = cp.ones(n, dtype=cp.float32)
    delays = cp.zeros(params.n_elements, dtype=cp.float32)

    result = simus(scat, rc, delays, params, strategy=SimusStrategy.CUDA)

    assert isinstance(result.rf, cp.ndarray)
    assert isinstance(result.spectrum, cp.ndarray)
    assert result.rf.shape[1] == params.n_elements
    assert bool(cp.all(cp.isfinite(result.rf)))


def test_simus_cuda_does_not_prepare_python_sweep(monkeypatch):
    """CUDA dispatch must skip _prepare_simus_sweep; v25c prepares flat inputs itself."""
    params = P4_2v()
    n_scat = _B_SCAT
    scat_np = np.stack([np.zeros(n_scat), np.linspace(1e-2, 5e-2, n_scat)], axis=-1).astype(np.float32)
    rc_np = np.ones(n_scat, dtype=np.float32)
    delays_np = np.zeros(params.n_elements, dtype=np.float32)

    scatterers = cp.asarray(scat_np)
    rc = cp.asarray(rc_np)
    delays = cp.asarray(delays_np)
    plan = simus_mod.simus_precompute(scatterers, rc, delays, params)

    def fail_prepare_sweep(*args, **kwargs):
        raise AssertionError("CUDA dispatch should not build _prepare_simus_sweep")

    monkeypatch.setattr(simus_mod, "_prepare_simus_sweep", fail_prepare_sweep)

    result = simus_mod.simus_compute(
        scatterers,
        rc,
        delays,
        plan,
        params,
        strategy=SimusStrategy.CUDA,
    )

    assert isinstance(result.rf, cp.ndarray)
    assert result.rf.shape[1] == params.n_elements
    assert bool(cp.all(cp.isfinite(result.rf)))


def test_simus_cuda_matches_python_strategy():
    """CUDA result must match Python strategy within ATOL_PEAK = 5e-3."""
    params = P4_2v()
    n_scat = 6
    scat_np = np.stack([np.zeros(n_scat), np.linspace(1e-2, 5e-2, n_scat)], axis=-1).astype(np.float32)
    rc_np = np.ones(n_scat, dtype=np.float32)
    delays_np = np.zeros(params.n_elements, dtype=np.float32)

    rf_py = np.asarray(
        simus(
            cast(Array, scat_np),
            cast(Array, rc_np),
            cast(Array, delays_np),
            params,
            strategy=SimusStrategy.PYTHON,
        ).rf,
    )
    rf_cu_cp = simus(
        cast(Array, cp.asarray(scat_np)),
        cast(Array, cp.asarray(rc_np)),
        cast(Array, cp.asarray(delays_np)),
        params,
        strategy=SimusStrategy.CUDA,
    ).rf
    rf_cu = cp.asnumpy(rf_cu_cp)

    peak = float(np.max(np.abs(rf_py)))
    assert np.allclose(rf_py, rf_cu, atol=5e-3 * peak, rtol=0)


def test_simus_cuda_l11_5v_recompile():
    """L11-5v has 128 elements; verifies a second NVRTC compile path works."""
    params = L11_5v()
    n_scat = 4
    scat = cp.asarray(np.stack([np.zeros(n_scat), np.linspace(1e-2, 4e-2, n_scat)], axis=-1).astype(np.float32))
    rc = cp.ones(n_scat, dtype=cp.float32)
    delays = cp.zeros(params.n_elements, dtype=cp.float32)

    result = simus(scat, rc, delays, params, strategy=SimusStrategy.CUDA)
    assert result.rf.shape[1] == params.n_elements
    assert bool(cp.all(cp.isfinite(result.rf)))
