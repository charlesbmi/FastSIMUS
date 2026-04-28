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

from ._bench_sync import sync_benchmark_array
from ._simus_bench_util import make_simus_compute

if TYPE_CHECKING:
    from fast_simus.utils._array_api import Array, _ArrayNamespace


_N_SCAT_SWEEP = (1_000, 10_000, 100_000, 1_000_000)


def _make_random_scatterers(n_scat: int, xp: _ArrayNamespace, seed: int = 0) -> tuple[Array, Array]:
    """1D float32 scatterer arrays matching the PyMUST benchmark distribution."""
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
    min_rounds=1,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.parametrize("n_scat", _N_SCAT_SWEEP)
def test_bench_simus_scaling(benchmark, xp, n_scat):
    """Scaling benchmark: simus_compute at 1K-1M scatterers across backends."""
    params = P4_2v()
    scatterers, rc = _make_random_scatterers(n_scat, xp)
    delays = xp.zeros(params.n_elements)
    # Pin element_splitting=1 to match the public PyMUST SIMUS benchmark
    # config. For P4-2v the auto formula already lands on 1, but pinning it
    # keeps this config reproducible across probes and backends.
    plan = simus_precompute(scatterers, rc, delays, params, element_splitting=1)
    compute = make_simus_compute(plan, params, xp)

    benchmark.extra_info.update(
        {
            "backend": xp.__name__.split(".")[0],
            "probe": "P4-2v",
            "n_scat": n_scat,
            "element_splitting": 1,
        }
    )

    def run():
        result = compute(scatterers, rc, delays)
        sync_benchmark_array(result.rf, xp)
        return result

    benchmark(run)
