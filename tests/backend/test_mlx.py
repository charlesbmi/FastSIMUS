"""MLX-specific tests for FastSIMUS.

Tests in this module verify features that require MLX and go beyond the
Array API abstraction: mx.compile, Apple Silicon GPU acceleration, etc.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from fast_simus import jit
from fast_simus.pfield import pfield_compute, pfield_precompute, pfield_spectrum_compute
from fast_simus.scattering import scattering_pfield_spectrum
from fast_simus.simus import simus_compute, simus_precompute
from fast_simus.transducer_presets import P4_2v


def _make_positions(x_range: tuple[float, float], z_range: tuple[float, float], n: int) -> np.ndarray:
    x = np.linspace(x_range[0], x_range[1], n)
    z = np.linspace(z_range[0], z_range[1], n)
    x_grid, z_grid = np.meshgrid(x, z)
    return np.stack([x_grid, z_grid], axis=-1)


@pytest.mark.slow
def test_mlx_compile_pfield_compute():
    """pfield_compute compiles and produces valid output through the public API."""
    params = P4_2v()
    positions_np = _make_positions((-2e-2, 2e-2), (params.pitch, 5e-2), n=50)
    delays_np = np.zeros(params.n_elements)

    positions_mx = mx.array(positions_np.astype(np.float32))
    delays_mx = mx.array(delays_np.astype(np.float32))

    plan = pfield_precompute(positions_mx, delays_mx, params)

    compiled = jit(lambda pos, dl: pfield_compute(pos, dl, plan, params), xp=mx)
    result = compiled(positions_mx, delays_mx)

    assert result.shape == (50, 50)
    assert bool(mx.all(result >= 0))


@pytest.mark.slow
def test_mlx_compile_pfield_spectrum_compute():
    """The split spectrum path is compilable through the public API."""
    params = P4_2v()
    positions = mx.array(_make_positions((-1e-2, 1e-2), (params.pitch, 3e-2), n=6).astype(np.float32))
    delays = mx.zeros(params.n_elements)
    plan = pfield_precompute(positions, delays, params)

    compute = jit(lambda pos, dl: pfield_spectrum_compute(pos, dl, plan, params), xp=mx)
    result = compute(positions, delays)

    assert result.shape[:2] == (6, 6)
    assert bool(mx.max(mx.abs(result)) > 0)


@pytest.mark.slow
def test_mlx_simus_compute():
    """simus_compute produces valid output with MLX arrays (Metal backend)."""
    from typing import cast

    from fast_simus.backends.mlx import ensure_compat
    from fast_simus.utils._array_api import Array

    ensure_compat(mx)

    params = P4_2v()
    n_scat = 6
    scatterers_np = np.stack([np.zeros(n_scat), np.linspace(1e-2, 5e-2, n_scat)], axis=-1).astype(np.float32)
    rc_np = np.ones(n_scat, dtype=np.float32)
    delays_np = np.zeros(params.n_elements, dtype=np.float32)

    scatterers = cast(Array, mx.array(scatterers_np))
    rc = cast(Array, mx.array(rc_np))
    delays = cast(Array, mx.array(delays_np))

    plan = simus_precompute(scatterers, rc, delays, params)
    result = simus_compute(scatterers, rc, delays, plan, params)

    rf = result.rf
    assert rf.ndim == 2
    assert rf.shape[1] == params.n_elements
    assert bool(mx.max(mx.abs(rf)) > 0)


@pytest.mark.slow
def test_mlx_scattering_materializes_each_observer_chunk(monkeypatch: pytest.MonkeyPatch):
    """Chunk barriers bound the MLX graph while retaining numerical agreement."""
    from typing import cast

    import fast_simus.backends.mlx as mlx_backend
    import fast_simus.scattering as scattering_module
    from fast_simus.backends.mlx import ensure_compat
    from fast_simus.utils._array_api import Array

    ensure_compat(mx)
    monkeypatch.setattr(scattering_module, "_MAX_PAIR_ELEMENTS", 4)
    materializations = 0
    original_eval_eager = mlx_backend.eval_eager

    def recording_eval_eager(*arrays):
        nonlocal materializations
        materializations += 1
        original_eval_eager(*arrays)

    monkeypatch.setattr(mlx_backend, "eval_eager", recording_eval_eager)

    params = P4_2v()
    positions_np = np.column_stack([np.linspace(-2e-3, 2e-3, 9), np.linspace(20e-3, 28e-3, 9)]).astype(np.float32)
    scatterers_np = np.asarray([[-1e-3, 12e-3], [0.0, 14e-3], [1e-3, 16e-3]], dtype=np.float32)
    rc_np = np.asarray([0.002, 0.003, 0.004], dtype=np.float32)
    delays_np = np.zeros(params.n_elements, dtype=np.float32)

    mlx_result = scattering_pfield_spectrum(
        cast(Array, mx.array(positions_np)),
        cast(Array, mx.array(scatterers_np)),
        cast(Array, mx.array(rc_np)),
        cast(Array, mx.array(delays_np)),
        params,
        element_splitting=1,
        frequency_step=10.0,
    )
    numpy_result = scattering_pfield_spectrum(
        cast(Array, positions_np),
        cast(Array, scatterers_np),
        cast(Array, rc_np),
        cast(Array, delays_np),
        params,
        element_splitting=1,
        frequency_step=10.0,
    )
    mx.eval(mlx_result.scattered)

    assert type(mlx_result.scattered).__module__.startswith("mlx")
    assert materializations == 5
    assert bool(mx.all(mx.isfinite(mlx_result.scattered)))
    np.testing.assert_allclose(np.asarray(mlx_result.scattered), numpy_result.scattered, rtol=2e-4, atol=2e-7)
