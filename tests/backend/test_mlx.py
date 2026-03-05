"""MLX-specific tests for FastSIMUS.

Tests in this module verify features that require MLX and go beyond the
Array API abstraction: mx.compile, Apple Silicon GPU acceleration, etc.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from fast_simus.pfield import pfield_compute, pfield_precompute  # noqa: E402
from fast_simus.transducer_presets import P4_2v  # noqa: E402


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
