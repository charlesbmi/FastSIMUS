"""Unit tests for the benchmark conftest machine-label hook.

Lives at `tests/` (not `tests/benchmarks/`) so `pytest -p no:benchmark` can
collect it without triggering pluggy validation of the pytest-benchmark hook.
"""

from __future__ import annotations

import pytest

from tests.benchmarks.conftest import (
    _DEVICE_LABEL_ENV_VAR,
    _DEVICE_LABEL_MACHINE_INFO_KEY,
    pytest_benchmark_update_machine_info,
)


def test_label_injected_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hook adds `fast_simus_device_label` when env var is set."""
    monkeypatch.setenv(_DEVICE_LABEL_ENV_VAR, "M2 Max MLX")
    machine_info: dict[str, object] = {}
    pytest_benchmark_update_machine_info(config=None, machine_info=machine_info)
    assert machine_info[_DEVICE_LABEL_MACHINE_INFO_KEY] == "M2 Max MLX"


def test_label_absent_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hook leaves machine_info untouched when env var is unset."""
    monkeypatch.delenv(_DEVICE_LABEL_ENV_VAR, raising=False)
    machine_info: dict[str, object] = {"node": "existing"}
    pytest_benchmark_update_machine_info(config=None, machine_info=machine_info)
    assert _DEVICE_LABEL_MACHINE_INFO_KEY not in machine_info
    assert machine_info == {"node": "existing"}


def test_label_absent_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty string is treated as unset (prevents empty-label noise)."""
    monkeypatch.setenv(_DEVICE_LABEL_ENV_VAR, "")
    machine_info: dict[str, object] = {}
    pytest_benchmark_update_machine_info(config=None, machine_info=machine_info)
    assert _DEVICE_LABEL_MACHINE_INFO_KEY not in machine_info
