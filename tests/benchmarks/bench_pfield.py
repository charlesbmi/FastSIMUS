"""Benchmarks for pfield_compute across array backends (NumPy, JAX, MLX).

Synchronization barriers (``jax.block_until_ready``, ``mx.eval``) ensure
wall-clock time includes actual computation, not just async dispatch.
JIT warmup is handled by pytest-benchmark's ``warmup=True`` marker.

Run with:  poe benchmark
Correctness tests live in: tests/backend/test_jax.py, tests/backend/test_mlx.py
"""

from __future__ import annotations

import contextlib
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest
from array_api_compat import is_jax_namespace

from fast_simus.pfield import pfield_compute, pfield_precompute
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils._array_api import is_mlx_namespace

from ._bench_sync import sync_benchmark_array

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_simus.utils._array_api import Array, _ArrayNamespace

_eqx = None
with contextlib.suppress(ImportError):
    import equinox as _eqx

_mx = None
with contextlib.suppress(ImportError):
    import mlx.core as _mx


def _make_positions(grid_length: int, xp: _ArrayNamespace) -> Array:
    """Create (grid_length, grid_length) square grid for P4-2v transducer."""
    params = P4_2v()
    x = xp.linspace(-2e-2, 2e-2, grid_length)
    z = xp.linspace(float(params.pitch), 5e-2, grid_length)
    x_grid, z_grid = xp.meshgrid(x, z)
    return xp.stack([x_grid, z_grid], axis=-1)


def _make_compute(plan, params, xp: _ArrayNamespace) -> Callable:
    """Return a (positions, delays) -> result callable with backend-specific JIT."""
    if is_jax_namespace(cast(ModuleType, xp)):
        assert _eqx is not None
        compute_kernel = _eqx.filter_jit(pfield_compute)
    elif _mx is not None and is_mlx_namespace(xp):
        return _mx.compile(lambda pos, dl: pfield_compute(pos, dl, plan, params))
    else:
        compute_kernel = pfield_compute

    return lambda pos, dl: compute_kernel(pos, dl, plan, params)


@pytest.mark.benchmark(
    group="pfield_compute",
    min_time=0.1,
    max_time=5.0,
    min_rounds=3,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.parametrize("grid_n", [100])
def test_bench_pfield_compute(benchmark, xp, grid_n):
    """Benchmark pfield_compute (P4-2v) across array backends."""
    params = P4_2v()
    positions = _make_positions(grid_n, xp)
    delays = xp.zeros(params.n_elements)
    plan = pfield_precompute(positions, delays, params)
    compute = _make_compute(plan, params, xp)

    def run():
        result = compute(positions, delays)
        sync_benchmark_array(result, xp)
        return result

    benchmark(run)
