"""Benchmarks for simus_compute across array backends (NumPy, JAX).

Run with:  poe benchmark
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_simus.simus import simus_precompute
from fast_simus.transducer_presets import P4_2v

from ._bench_sync import sync_benchmark_array
from ._simus_bench_util import make_simus_compute

if TYPE_CHECKING:
    from fast_simus.utils._array_api import Array, _ArrayNamespace


def _make_scatterers(n_scat: int, xp: _ArrayNamespace) -> tuple[Array, Array]:
    """Create n_scat scatterers on the z-axis."""
    x = xp.zeros(n_scat)
    z = xp.linspace(1e-2, 8e-2, n_scat)
    scatterers = xp.stack([x, z], axis=-1)
    rc = xp.ones(n_scat)
    return scatterers, rc


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
    compute = make_simus_compute(plan, params, xp)

    def run():
        result = compute(scatterers, rc, delays)
        sync_benchmark_array(result.rf, xp)
        return result

    benchmark(run)
