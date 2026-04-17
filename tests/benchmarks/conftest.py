"""pytest-benchmark machine-info augmentation for FastSIMUS scaling benchmarks.

Adds an optional ``fast_simus_device_label`` entry to ``machine_info`` when
the ``FASTSIMUS_DEVICE_LABEL`` environment variable is set. Lets cross-machine
benchmark runs carry a human-readable tag (e.g. "M2 Max MLX", "H100 JAX-CUDA")
that the scaling-plot script can use to distinguish series.
"""

from __future__ import annotations

import os
from typing import Any

_DEVICE_LABEL_ENV_VAR = "FASTSIMUS_DEVICE_LABEL"
_DEVICE_LABEL_MACHINE_INFO_KEY = "fast_simus_device_label"


def pytest_benchmark_update_machine_info(config: object, machine_info: dict[str, Any]) -> None:
    """Inject ``FASTSIMUS_DEVICE_LABEL`` (if set) into pytest-benchmark machine_info."""
    label = os.environ.get(_DEVICE_LABEL_ENV_VAR, "")
    if label:
        machine_info[_DEVICE_LABEL_MACHINE_INFO_KEY] = label
