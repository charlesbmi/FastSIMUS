"""PyMUST scaling benchmark — CPU reference for `simus_scaling` plot.

Runs `pymust.simus` at the same `n_scat` values as ``bench_simus_scaling.py``
so PyMUST lands in the same ``simus_scaling`` benchmark group in the
``pytest-benchmark`` JSON output.

Skipped entirely when PyMUST isn't installed (e.g. on a CUDA-only host).

Run with:
    poe benchmark-scaling
    pytest tests/benchmarks/ --benchmark-only -m scaling -p no:xdist
"""

from __future__ import annotations

import numpy as np
import pytest

pymust = pytest.importorskip("pymust")


def _make_random_scatterers(n_scat: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D float32 scatterer arrays matching ``bench_simus_scaling`` distribution."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2e-2, 2e-2, n_scat).astype(np.float32)
    z = rng.uniform(1e-3, 8e-2, n_scat).astype(np.float32)
    rc = rng.uniform(0.5, 1.5, n_scat).astype(np.float32)
    return x, z, rc


@pytest.mark.scaling
@pytest.mark.benchmark(
    group="simus_scaling",
    min_time=0.5,
    max_time=30.0,
    min_rounds=1,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.parametrize("n_scat", [1_000, 10_000, 100_000, 1_000_000])
def test_bench_pymust_scaling(benchmark, n_scat):
    """Scaling benchmark: pymust.simus (P4-2v) at 1K-1M scatterers."""
    param = pymust.getparam("P4-2v")
    options = pymust.utils.Options()
    options.dBThresh = -60.0
    delays = np.zeros((1, param.Nelements))
    x, z, rc = _make_random_scatterers(n_scat)

    def run():
        return pymust.simus(x, z, rc, delays, param, options)

    benchmark(run)
