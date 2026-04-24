"""Unit tests for the benchmark conftest hooks and option parsing.

Lives at `tests/` (not `tests/benchmarks/`) so `pytest -p no:benchmark` can
collect it without triggering pluggy validation of the pytest-benchmark hook.
"""

from __future__ import annotations

import pytest

from tests.benchmarks.conftest import (
    _DEFAULT_N_SCAT,
    _DEVICE_LABEL_ENV_VAR,
    _DEVICE_LABEL_MACHINE_INFO_KEY,
    parse_n_scat_option,
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


class TestParseNScatOption:
    """Unit tests for ``parse_n_scat_option``."""

    def test_none_returns_default(self) -> None:
        """``None`` yields the default sweep."""
        assert parse_n_scat_option(None) == _DEFAULT_N_SCAT

    def test_empty_string_returns_default(self) -> None:
        """Empty string yields the default sweep."""
        assert parse_n_scat_option("") == _DEFAULT_N_SCAT

    def test_whitespace_only_returns_default(self) -> None:
        """All-whitespace tokens yield the default sweep."""
        assert parse_n_scat_option(" , ,  ") == _DEFAULT_N_SCAT

    def test_custom_sweep_preserves_order(self) -> None:
        """A comma-separated list yields the exact sweep in order."""
        assert parse_n_scat_option("1000,3162,10000") == (1000, 3162, 10000)

    def test_dedups_preserving_first_occurrence(self) -> None:
        """Duplicate entries collapse to first-occurrence order."""
        assert parse_n_scat_option("1000,10000,1000,10000") == (1000, 10000)

    def test_rejects_non_integer(self) -> None:
        """Non-integer tokens raise ValueError with the bad token in the message."""
        with pytest.raises(ValueError, match="1k"):
            parse_n_scat_option("1000,1k,2000")

    def test_rejects_non_positive(self) -> None:
        """Non-positive values raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            parse_n_scat_option("1000,0,2000")
        with pytest.raises(ValueError, match="positive"):
            parse_n_scat_option("1000,-5")
