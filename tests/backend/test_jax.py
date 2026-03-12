"""JAX-specific tests for FastSIMUS.

Tests in this module verify features that require JAX and go beyond the
Array API abstraction: JIT compilation, gradient tracing, vmap, etc.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
eqx = pytest.importorskip("equinox")

from fast_simus.pfield import pfield_compute, pfield_precompute  # noqa: E402
from fast_simus.transducer_presets import P4_2v  # noqa: E402


def _make_positions(x_range: tuple[float, float], z_range: tuple[float, float], n: int) -> np.ndarray:
    x = np.linspace(x_range[0], x_range[1], n)
    z = np.linspace(z_range[0], z_range[1], n)
    x_grid, z_grid = np.meshgrid(x, z)
    return np.stack([x_grid, z_grid], axis=-1)


@pytest.mark.slow
def test_jax_jit_pfield_compute():
    """pfield_compute compiles and produces valid output under eqx.filter_jit.

    eqx.filter_jit automatically splits arguments into:
    - JAX arrays (traced): positions, delays, plan.selected_freqs, etc.
    - Everything else (static): plan.n_sub, params fields, ...

    freq_start/freq_step are derived from plan.selected_freqs at compute time,
    so they become traced scalars (fewer recompilations than static floats).
    """
    params = P4_2v()
    positions_np = _make_positions((-2e-2, 2e-2), (params.pitch, 5e-2), n=50)
    delays_np = np.zeros(params.n_elements)

    plan = pfield_precompute(jnp.asarray(positions_np), jnp.asarray(delays_np), params)

    jitted = eqx.filter_jit(pfield_compute)
    result = jitted(jnp.asarray(positions_np), jnp.asarray(delays_np), plan, params)

    assert result.shape == (50, 50)
    assert bool(jnp.all(result >= 0))
