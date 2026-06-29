"""pytest-benchmark configuration for FastSIMUS scaling benchmarks.

Exposes two concerns:

* **Machine labeling** — adds an optional ``fast_simus_device_label`` entry
  to ``machine_info`` when the ``FASTSIMUS_DEVICE_LABEL`` environment
  variable is set. Lets cross-machine benchmark runs carry a human-readable
  tag (e.g. "M2 Max MLX", "H100 JAX-CUDA") that the scaling-plot script
  can use to distinguish series.

* **Configurable scatterer sweep** — adds a ``--n-scat`` pytest CLI
  option. Benchmarks marked ``@pytest.mark.scaling`` that take an
  ``n_scat`` fixture are parametrized over either the default sweep or
  the caller-supplied comma-separated list. Keeps dense / custom sweeps
  out of each bench file and central to one place.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

_DEVICE_LABEL_ENV_VAR = "FASTSIMUS_DEVICE_LABEL"
_DEVICE_LABEL_MACHINE_INFO_KEY = "fast_simus_device_label"

_N_SCAT_OPTION = "--n-scat"
_DEFAULT_N_SCAT: tuple[int, ...] = (1_000, 10_000, 100_000, 1_000_000)


def pytest_benchmark_update_machine_info(config: object, machine_info: dict[str, Any]) -> None:
    """Inject ``FASTSIMUS_DEVICE_LABEL`` (if set) into pytest-benchmark machine_info."""
    label = os.environ.get(_DEVICE_LABEL_ENV_VAR, "")
    if label:
        machine_info[_DEVICE_LABEL_MACHINE_INFO_KEY] = label


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--n-scat`` CLI option."""
    parser.addoption(
        _N_SCAT_OPTION,
        action="store",
        default=None,
        dest="n_scat",
        help=(
            "Comma-separated list of scatterer counts for scaling benchmarks "
            f"(default: {','.join(str(n) for n in _DEFAULT_N_SCAT)}). "
            "Example: --n-scat=1000,3162,10000,31623,100000,316228,1000000"
        ),
    )


def parse_n_scat_option(raw: str | None) -> tuple[int, ...]:
    """Parse the ``--n-scat`` CLI value into a tuple of positive ints.

    ``None`` or empty string -> default sweep. Duplicates are removed while
    preserving first-occurrence order. Non-integer or non-positive entries
    raise ``ValueError`` so a typo surfaces at collection time.
    """
    if not raw:
        return _DEFAULT_N_SCAT
    values: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            n = int(token)
        except ValueError as exc:
            msg = f"--n-scat: expected comma-separated integers, got {token!r}"
            raise ValueError(msg) from exc
        if n <= 0:
            msg = f"--n-scat: values must be positive, got {n}"
            raise ValueError(msg)
        if n not in seen:
            seen.add(n)
            values.append(n)
    if not values:
        return _DEFAULT_N_SCAT
    return tuple(values)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize scaling benchmarks taking ``n_scat`` with the resolved sweep."""
    if "n_scat" not in metafunc.fixturenames:
        return
    if not metafunc.definition.get_closest_marker("scaling"):
        return
    raw = metafunc.config.getoption("n_scat")
    values = parse_n_scat_option(raw)
    metafunc.parametrize("n_scat", values)
