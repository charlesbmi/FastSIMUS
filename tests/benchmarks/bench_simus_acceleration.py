"""Warmed custom-kernel versus portable Array API SIMUS benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_simus import BackendKind, get_backend
from fast_simus.simus import simus_precompute
from fast_simus.transducer_presets import P4_2v

from ._bench_sync import sync_benchmark_array
from ._simus_bench_util import make_simus_compute


@pytest.fixture(params=[BackendKind.METAL, BackendKind.MLX, BackendKind.CUDA, BackendKind.CUPY])
def acceleration_backend(request: pytest.FixtureRequest):
    """Return each accelerated backend and its same-library baseline."""
    try:
        return get_backend(request.param)
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.mark.benchmark(
    group="simus_acceleration",
    min_time=0.1,
    max_time=30.0,
    min_rounds=3,
    warmup=True,
    warmup_iterations=1,
)
@pytest.mark.parametrize("n_scat", [100, 1_000, 10_000])
def test_bench_simus_acceleration(benchmark: Any, acceleration_backend, n_scat: int) -> None:
    """Measure custom-kernel and portable execution with identical inputs."""
    backend = acceleration_backend
    xp = backend.xp
    params = P4_2v()
    rng = np.random.default_rng(0)
    scatterers = xp.asarray(
        np.stack(
            [rng.uniform(-2e-2, 2e-2, n_scat), rng.uniform(1e-3, 8e-2, n_scat)],
            axis=-1,
        ).astype(np.float32)
    )
    rc = xp.asarray(rng.uniform(0.5, 1.5, n_scat).astype(np.float32))
    delays = xp.zeros(params.n_elements)
    plan = simus_precompute(scatterers, rc, delays, params, element_splitting=1)
    compute = make_simus_compute(plan, params, xp, backend=backend)

    warmed = compute(scatterers, rc, delays)
    sync_benchmark_array(warmed.rf, xp)
    benchmark.extra_info.update(
        {
            "backend": backend.kind.value,
            "n_scat": n_scat,
            "probe": "P4-2v",
            "accelerated": backend.kind in (BackendKind.METAL, BackendKind.CUDA),
        }
    )

    def run():
        result = compute(scatterers, rc, delays)
        sync_benchmark_array(result.rf, xp)
        return result

    benchmark(run)
