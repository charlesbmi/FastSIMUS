"""Unit tests for scripts/plot_benchmark_scaling.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "plot_benchmark_scaling.py"


def _load_plot_module():
    """Import scripts/plot_benchmark_scaling.py as a module (not a package)."""
    spec = importlib.util.spec_from_file_location("plot_benchmark_scaling", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["plot_benchmark_scaling"] = module
    spec.loader.exec_module(module)
    return module


plot_module = _load_plot_module()


def _make_benchmark_json(
    path: Path,
    *,
    device_label: str | None,
    commit_id: str,
    entries: list[tuple[str, dict[str, Any], float]],
) -> None:
    """Write a minimal pytest-benchmark-shaped JSON file to ``path``.

    Each entry is ``(name, params, mean_s)``. Group is fixed to "simus_scaling".
    """
    machine_info: dict[str, Any] = {"node": "testnode"}
    if device_label is not None:
        machine_info["fast_simus_device_label"] = device_label
    data: dict[str, Any] = {
        "machine_info": machine_info,
        "commit_info": {"id": commit_id},
        "benchmarks": [
            {
                "name": name,
                "group": "simus_scaling",
                "params": params,
                "stats": {"mean": mean_s, "stddev": mean_s * 0.01},
            }
            for name, params, mean_s in entries
        ],
    }
    path.write_text(json.dumps(data))


class TestExtractBackend:
    """Backend label parsing from pytest-benchmark names."""

    def test_pymust_name(self) -> None:
        assert plot_module._extract_backend("test_bench_pymust_scaling[1000]") == "pymust"

    def test_xp_parametrized_name(self) -> None:
        assert plot_module._extract_backend("test_bench_simus_scaling[numpy-1000]") == "numpy"
        assert plot_module._extract_backend("test_bench_simus_scaling[jax-1000]") == "jax"
        assert plot_module._extract_backend("test_bench_simus_scaling[mlx-1000]") == "mlx"

    def test_unknown_shape(self) -> None:
        assert plot_module._extract_backend("weird_name_no_brackets") == "unknown"


class TestBuildDataFrame:
    """End-to-end DataFrame construction from JSON fixtures."""

    def test_single_file_multi_backend(self, tmp_path: Path) -> None:
        fixture = tmp_path / "run1.json"
        _make_benchmark_json(
            fixture,
            device_label="M2 Max",
            commit_id="abc1234",
            entries=[
                ("test_bench_simus_scaling[numpy-1000]", {"n_scat": 1000}, 0.050),
                ("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.010),
                ("test_bench_pymust_scaling[1000]", {"n_scat": 1000}, 0.052),
            ],
        )
        df = plot_module.build_dataframe([fixture], group="simus_scaling")
        assert len(df) == 3
        assert set(df["backend"]) == {"numpy", "jax", "pymust"}
        assert (df["machine"] == "M2 Max").all()
        assert (df["throughput"] > 0).all()

    def test_multi_file_merges_machines(self, tmp_path: Path) -> None:
        mac = tmp_path / "mac.json"
        _make_benchmark_json(
            mac,
            device_label="M2 Max",
            commit_id="abc1234",
            entries=[("test_bench_pymust_scaling[1000]", {"n_scat": 1000}, 0.050)],
        )
        cuda = tmp_path / "cuda.json"
        _make_benchmark_json(
            cuda,
            device_label="H100 JAX-CUDA",
            commit_id="abc1234",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.001)],
        )
        df = plot_module.build_dataframe([mac, cuda], group="simus_scaling")
        assert set(df["machine"]) == {"M2 Max", "H100 JAX-CUDA"}
        assert len(df) == 2

    def test_fallback_machine_label_when_device_unset(self, tmp_path: Path) -> None:
        fixture = tmp_path / "nolabel.json"
        _make_benchmark_json(
            fixture,
            device_label=None,
            commit_id="abc1234",
            entries=[("test_bench_simus_scaling[numpy-1000]", {"n_scat": 1000}, 0.050)],
        )
        df = plot_module.build_dataframe([fixture], group="simus_scaling")
        assert df["machine"].iloc[0] == "testnode (numpy)"

    def test_group_filter(self, tmp_path: Path) -> None:
        fixture = tmp_path / "run.json"
        data = {
            "machine_info": {"node": "x"},
            "commit_info": {"id": "abc"},
            "benchmarks": [
                {
                    "name": "test_bench_other[1000]",
                    "group": "other_group",
                    "params": {"n_scat": 1000},
                    "stats": {"mean": 0.1, "stddev": 0.0},
                }
            ],
        }
        fixture.write_text(json.dumps(data))
        df = plot_module.build_dataframe([fixture], group="simus_scaling")
        assert df.empty


class TestCli:
    """CLI entry-point behavior."""

    def test_main_produces_png(self, tmp_path: Path) -> None:
        fixture = tmp_path / "run.json"
        _make_benchmark_json(
            fixture,
            device_label="M2 Max",
            commit_id="abc1234",
            entries=[
                ("test_bench_pymust_scaling[1000]", {"n_scat": 1000}, 0.050),
                ("test_bench_pymust_scaling[10000]", {"n_scat": 10000}, 0.150),
                ("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.005),
                ("test_bench_simus_scaling[jax-10000]", {"n_scat": 10000}, 0.010),
            ],
        )
        out = tmp_path / "plot.png"
        rc = plot_module.main([str(fixture), "-o", str(out)])
        assert rc == 0
        assert out.exists()
        assert out.stat().st_size > 0

    def test_main_exits_2_when_no_benchmarks(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        empty = tmp_path / "empty.json"
        _make_benchmark_json(empty, device_label=None, commit_id="abc", entries=[])
        rc = plot_module.main([str(empty), "-o", str(tmp_path / "out.png")])
        assert rc == 2
        assert "no benchmarks in group" in capsys.readouterr().err

    def test_main_exits_2_when_missing_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = plot_module.main([str(tmp_path / "nope.json"), "-o", str(tmp_path / "out.png")])
        assert rc == 2
        assert "no such file" in capsys.readouterr().err

    def test_main_malformed_json_surfaces_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(SystemExit) as excinfo:
            plot_module.main([str(bad), "-o", str(tmp_path / "out.png")])
        assert str(bad) in str(excinfo.value)
