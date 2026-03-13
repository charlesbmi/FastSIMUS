"""Scaling benchmarks for simus_compute with large scatterer counts.

These are skip-by-default. Run with:
    poe benchmark-scaling
    pytest tests/benchmarks/ --benchmark-only -m scaling -p no:xdist

Designed to track throughput regressions as kernel optimizations evolve.
"""

from __future__ import annotations

import contextlib
from types import ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from array_api_compat import is_jax_namespace

from fast_simus.simus import simus_compute, simus_precompute
from fast_simus.transducer_presets import P4_2v

if TYPE_CHECKING:
    from fast_simus.utils._array_api import Array, _ArrayNamespace

_jax = None
_eqx = None
with contextlib.suppress(ImportError):
    import equinox as _eqx
    import jax as _jax

_mx = None
with contextlib.suppress(ImportError):
    import mlx.core as _mx


def _make_random_scatterers(n_scat: int, xp: _ArrayNamespace, seed: int = 0) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2e-2, 2e-2, n_scat).astype(np.float32)
    z = rng.uniform(1e-3, 8e-2, n_scat).astype(np.float32)
    scatterers = xp.asarray(np.stack([x, z], axis=-1))
    rc = xp.asarray(rng.uniform(0.5, 1.5, n_scat).astype(np.float32))
    return scatterers, rc


def _make_compute(plan, params, xp: _ArrayNamespace):
    if is_jax_namespace(cast(ModuleType, xp)):
        assert _eqx is not None
        jitted = _eqx.filter_jit(simus_compute)
        return lambda scat, rc, dl: jitted(scat, rc, dl, plan, params)
    return lambda scat, rc, dl: simus_compute(scat, rc, dl, plan, params)


def _sync(result, xp: _ArrayNamespace) -> None:
    if is_jax_namespace(cast(ModuleType, xp)):
        assert _jax is not None
        _jax.block_until_ready(result)
        return
    if _mx is not None and xp is _mx:
        _mx.eval(result.rf)


@pytest.mark.scaling
@pytest.mark.benchmark(
    group="simus_scaling",
    min_time=0.5,
    max_time=30.0,
    min_rounds=3,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.parametrize("n_scat", [1_000, 10_000, 100_000, 1_000_000])
def test_bench_simus_scaling(benchmark, xp, n_scat):
    """Scaling benchmark: simus_compute at 1K-1M scatterers."""
    params = P4_2v()
    scatterers, rc = _make_random_scatterers(n_scat, xp)
    delays = xp.zeros(params.n_elements)
    plan = simus_precompute(scatterers, rc, delays, params)
    compute = _make_compute(plan, params, xp)

    def run():
        result = compute(scatterers, rc, delays)
        _sync(result, xp)
        return result

    result = benchmark(run)
    assert result is not None
