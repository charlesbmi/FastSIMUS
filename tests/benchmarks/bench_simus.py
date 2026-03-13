"""Benchmarks for simus_compute across array backends (NumPy, JAX).

Run with:  poe benchmark
"""

from __future__ import annotations

import contextlib
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest
from array_api_compat import is_jax_namespace

from fast_simus.simus import simus_compute, simus_precompute
from fast_simus.transducer_presets import P4_2v

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_simus.utils._array_api import Array, _ArrayNamespace

_jax = None
_eqx = None
with contextlib.suppress(ImportError):
    import equinox as _eqx
    import jax as _jax


def _make_scatterers(n_scat: int, xp: _ArrayNamespace) -> tuple[Array, Array]:
    """Create n_scat scatterers on the z-axis."""
    x = xp.zeros(n_scat)
    z = xp.linspace(1e-2, 8e-2, n_scat)
    scatterers = xp.stack([x, z], axis=-1)
    rc = xp.ones(n_scat)
    return scatterers, rc


def _make_compute(plan, params, xp: _ArrayNamespace) -> Callable:
    """Return a (scatterers, rc, delays) -> result callable with backend-specific JIT."""
    if is_jax_namespace(cast(ModuleType, xp)):
        assert _eqx is not None
        jitted = _eqx.filter_jit(simus_compute)
        return lambda scat, rc, dl: jitted(scat, rc, dl, plan, params)

    return lambda scat, rc, dl: simus_compute(scat, rc, dl, plan, params)


def _sync(result, xp: _ArrayNamespace) -> None:
    """Block until result is ready for async backends."""
    if is_jax_namespace(cast(ModuleType, xp)):
        assert _jax is not None
        _jax.block_until_ready(result)


@pytest.mark.benchmark(
    group="simus_compute",
    min_time=0.1,
    max_time=10.0,
    min_rounds=3,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.parametrize("n_scat", [100])
def test_bench_simus_compute(benchmark, xp, n_scat):
    """Benchmark simus_compute (P4-2v) across array backends."""
    params = P4_2v()
    scatterers, rc = _make_scatterers(n_scat, xp)
    delays = xp.zeros(params.n_elements)
    plan = simus_precompute(scatterers, rc, delays, params)
    compute = _make_compute(plan, params, xp)

    def run():
        result = compute(scatterers, rc, delays)
        _sync(result, xp)
        return result

    benchmark(run)
