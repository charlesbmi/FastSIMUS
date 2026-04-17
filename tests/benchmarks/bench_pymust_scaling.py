"""PyMUST scaling benchmark — CPU reference for `simus_scaling` plot.

Runs `pymust.simus` at the same ``n_scat`` sweep as ``bench_simus_scaling.py``
so PyMUST lands in the same ``simus_scaling`` benchmark group in the
``pytest-benchmark`` JSON output. The ``n_scat`` sweep is supplied by
``conftest.pytest_generate_tests`` (override with ``--n-scat=...``).

Skipped entirely when PyMUST isn't installed (e.g. on a CUDA-only host).

Run with:
    poe benchmark-scaling
    pytest tests/benchmarks/ --benchmark-only -m scaling -p no:xdist
"""

from __future__ import annotations

from typing import Any

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


def _dump_pymust_param(param: object) -> dict[str, Any]:
    """Pull JSON-safe scalar attrs off a pymust param for benchmark.extra_info.

    Whitelist matches the TransducerParams fields surfaced for FastSIMUS so
    PyMUST and FastSIMUS rows carry comparable metadata. Attributes that are
    absent or None are dropped (rather than surfacing as null) to keep
    downstream consumers simple.
    """
    field_map = {
        "fc_hz": "fc",
        "fs_hz": "fs",
        "c_mps": "c",
        "n_elements": "Nelements",
        "pitch_m": "pitch",
        "kerf_m": "kerf",
        "width_m": "width",
        "height_m": "height",
        "bandwidth": "bandwidth",
        "elev_focus_m": "focus",
    }
    out: dict[str, Any] = {}
    for out_key, attr in field_map.items():
        value = getattr(param, attr, None)
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, float, bool, str)):
            out[out_key] = value
    return out


@pytest.mark.scaling
@pytest.mark.benchmark(
    group="simus_scaling",
    min_time=0.5,
    max_time=30.0,
    min_rounds=1,
    warmup=True,
    warmup_iterations=1,
)
def test_bench_pymust_scaling(benchmark, n_scat):
    """Scaling benchmark: pymust.simus (P4-2v) at configurable scatterer counts."""
    param = pymust.getparam("P4-2v")
    # Pin fs and c to match FastSIMUS's simus_precompute defaults so the
    # scaling plot compares implementations at identical acquisition params.
    param.fs = 4.0 * param.fc
    param.c = 1540.0
    options = pymust.utils.Options()
    options.dBThresh = -60.0
    delays = np.zeros((1, param.Nelements))
    x, z, rc = _make_random_scatterers(n_scat)

    benchmark.extra_info.update(
        {
            "backend": "pymust",
            "probe": "P4-2v",
            "n_scat": n_scat,
            "db_thresh": options.dBThresh,
            **_dump_pymust_param(param),
        }
    )

    def run():
        return pymust.simus(x, z, rc, delays, param, options)

    benchmark(run)
