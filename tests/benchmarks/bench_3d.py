"""Opt-in finite 3D RF benchmarks with reproducible scenes and explicit workspace."""

import numpy as np
import pytest

from fast_simus import ExecutionOptions, Transducer, matrix_aperture, simus_compute, simus_precompute
from tests.conftest import to_numpy


@pytest.mark.scaling
@pytest.mark.parametrize("side,count", [(16, 10_000), (32, 100_000)])
def test_3d_rf_scaling(benchmark, xp, side, count):
    """Record synchronized portable RF throughput with input/output accounting."""
    rng = np.random.default_rng(2026)
    points = xp.asarray(rng.uniform([-0.005, -0.005, 0.01], [0.005, 0.005, 0.03], (count, 3)), dtype=xp.float32)
    rc = xp.asarray(rng.normal(size=count), dtype=xp.float32)
    aperture = matrix_aperture(shape=(side, side), pitch=(0.0003, 0.0003), size=(0.0002, 0.0002), xp=xp)
    transducer = Transducer(aperture, "3d", 2e6)
    delays = xp.zeros(side * side, dtype=xp.float32)
    plan = simus_precompute(points, rc, delays, transducer, execution=ExecutionOptions(16 * 1024 * 1024))
    benchmark.extra_info.update(
        points=count,
        elements=side * side,
        frequency_bins=plan.selected_freqs.shape[0],
        subdivisions=plan._counts,
        estimated_workspace_bytes=plan.estimated_workspace_bytes,
        input_bytes=(count * 4 + side * side) * 4,
        output_bytes=(plan.n_freq_full * 8 + (plan.n_fft + 1) // 2 * 4) * side * side,
    )

    def run():
        result = simus_compute(points, rc, delays, plan, transducer)
        to_numpy(result.rf)  # Synchronize asynchronous backends.
        return result

    benchmark(run)
