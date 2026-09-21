"""JAX-specific tests for FastSIMUS.

Tests in this module verify features that require JAX and go beyond the
Array API abstraction: JIT compilation, gradient tracing, vmap, etc.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from fast_simus import jit
from fast_simus.pfield import pfield_compute, pfield_precompute, pfield_spectrum_compute
from fast_simus.simus import simus_compute, simus_precompute
from fast_simus.transducer_presets import P4_2v


def _make_positions(x_range: tuple[float, float], z_range: tuple[float, float], n: int) -> np.ndarray:
    x = np.linspace(x_range[0], x_range[1], n)
    z = np.linspace(z_range[0], z_range[1], n)
    x_grid, z_grid = np.meshgrid(x, z)
    return np.stack([x_grid, z_grid], axis=-1)


@pytest.mark.slow
def test_jax_jit_pfield_compute():
    """pfield_compute compiles and produces valid output through the public API."""
    params = P4_2v()
    positions_np = _make_positions((-2e-2, 2e-2), (params.pitch, 5e-2), n=50)
    delays_np = np.zeros(params.n_elements)

    plan = pfield_precompute(jnp.asarray(positions_np), jnp.asarray(delays_np), params)

    compute = jit(lambda pos, dl: pfield_compute(pos, dl, plan, params), xp=jnp)
    result = compute(jnp.asarray(positions_np), jnp.asarray(delays_np))

    assert result.shape == (50, 50)
    assert bool(jnp.all(result >= 0))


@pytest.mark.slow
def test_jax_jit_pfield_spectrum_compute():
    """The split spectrum path is compilable through the public API."""
    params = P4_2v()
    positions = jnp.asarray(_make_positions((-1e-2, 1e-2), (params.pitch, 3e-2), n=6))
    delays = jnp.zeros(params.n_elements)
    plan = pfield_precompute(positions, delays, params)

    compute = jit(lambda pos, dl: pfield_spectrum_compute(pos, dl, plan, params), xp=jnp)
    result = compute(positions, delays)

    assert result.shape[:2] == (6, 6)
    assert bool(jnp.max(jnp.abs(result)) > 0)


@pytest.mark.slow
def test_jax_jit_simus_compute():
    """simus_compute compiles and produces valid output through the public API."""
    params = P4_2v()
    n_scat = 6
    scatterers_np = np.stack([np.zeros(n_scat), np.linspace(1e-2, 5e-2, n_scat)], axis=-1)
    rc_np = np.ones(n_scat)
    delays_np = np.zeros(params.n_elements)

    scatterers = jnp.asarray(scatterers_np)
    rc = jnp.asarray(rc_np)
    delays = jnp.asarray(delays_np)

    plan = simus_precompute(scatterers, rc, delays, params)

    compute = jit(lambda scat, coeff, dl: simus_compute(scat, coeff, dl, plan, params), xp=jnp)
    result = compute(scatterers, rc, delays)

    rf = result.rf
    assert rf.ndim == 2
    assert rf.shape[1] == params.n_elements
    assert bool(jnp.max(jnp.abs(rf)) > 0)
