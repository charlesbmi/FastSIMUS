"""MLX-specific tests for FastSIMUS.

Tests in this module verify features that require MLX and go beyond the
Array API abstraction: mx.compile, Apple Silicon GPU acceleration, etc.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from fast_simus.pfield import pfield_compute, pfield_precompute
from fast_simus.simus import simus_compute, simus_precompute
from fast_simus.transducer_presets import P4_2v


def _make_positions(x_range: tuple[float, float], z_range: tuple[float, float], n: int) -> np.ndarray:
    x = np.linspace(x_range[0], x_range[1], n)
    z = np.linspace(z_range[0], z_range[1], n)
    x_grid, z_grid = np.meshgrid(x, z)
    return np.stack([x_grid, z_grid], axis=-1)


@pytest.mark.slow
def test_mlx_compile_pfield_compute():
    """pfield_compute compiles and produces valid output under mx.compile.

    mx.compile fuses operations in the computation graph, similar to JAX's jit.
    Non-array arguments (plan scalars, params, medium) are captured as constants
    in the compiled graph via a closure.
    """
    params = P4_2v()
    positions_np = _make_positions((-2e-2, 2e-2), (params.pitch, 5e-2), n=50)
    delays_np = np.zeros(params.n_elements)

    positions_mx = mx.array(positions_np.astype(np.float32))
    delays_mx = mx.array(delays_np.astype(np.float32))

    plan = pfield_precompute(positions_mx, delays_mx, params)

    compiled = mx.compile(lambda pos, dl: pfield_compute(pos, dl, plan, params))
    result = compiled(positions_mx, delays_mx)

    assert result.shape == (50, 50)
    assert bool(mx.all(result >= 0))


@pytest.mark.slow
def test_mlx_simus_compute():
    """simus_compute produces valid output with MLX arrays (Metal strategy).

    Unlike pfield, simus cannot use mx.compile because the Metal kernel path
    does eager float() on plan arrays. The Metal kernels are already
    GPU-optimized, so mx.compile is not needed.
    """
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
