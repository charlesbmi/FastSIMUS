"""Scaling benchmarks for simus_compute with large scatterer counts.

These are skip-by-default. Run with:
    poe benchmark-scaling
    pytest tests/benchmarks/ --benchmark-only -m scaling -p no:xdist

Designed to track throughput regressions as kernel optimizations evolve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_simus.simus import simus_precompute
from fast_simus.transducer_presets import P4_2v
from tests.benchmarks._simus_bench_util import make_simus_compute, sync_simus_result

if TYPE_CHECKING:
    from fast_simus.utils._array_api import Array, _ArrayNamespace


def _make_random_scatterers(n_scat: int, xp: _ArrayNamespace, seed: int = 0) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2e-2, 2e-2, n_scat).astype(np.float32)
    z = rng.uniform(1e-3, 8e-2, n_scat).astype(np.float32)
    scatterers = xp.asarray(np.stack([x, z], axis=-1))
    rc = xp.asarray(rng.uniform(0.5, 1.5, n_scat).astype(np.float32))
    return scatterers, rc


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
    compute = make_simus_compute(plan, params, xp)

    def run():
        result = compute(scatterers, rc, delays)
        sync_simus_result(result, xp)
        return result

    result = benchmark(run)
    assert result is not None
