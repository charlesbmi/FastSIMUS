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
    datetime: str = "2026-01-01T00:00:00",
) -> None:
    """Write a minimal pytest-benchmark-shaped JSON file to ``path``.

    Each entry is ``(name, params, mean_s)``. Group is fixed to "simus_scaling".
    ``datetime`` controls the JSON top-level timestamp used by dedupe ordering.
    """
    machine_info: dict[str, Any] = {"node": "testnode"}
    if device_label is not None:
        machine_info["fast_simus_device_label"] = device_label
    data: dict[str, Any] = {
        "machine_info": machine_info,
        "commit_info": {"id": commit_id},
        "datetime": datetime,
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
        """`test_bench_pymust_*` names resolve to the "pymust" backend."""
        assert plot_module._extract_backend("test_bench_pymust_scaling[1000]") == "pymust"

    def test_xp_parametrized_name(self) -> None:
        """Names with `[backend-n_scat]` params resolve to the first non-numeric token."""
        assert plot_module._extract_backend("test_bench_simus_scaling[numpy-1000]") == "numpy"
        assert plot_module._extract_backend("test_bench_simus_scaling[jax-1000]") == "jax"
        assert plot_module._extract_backend("test_bench_simus_scaling[mlx-1000]") == "mlx"

    def test_unknown_shape(self) -> None:
        """Names without brackets fall back to "unknown"."""
        assert plot_module._extract_backend("weird_name_no_brackets") == "unknown"


class TestBuildDataFrame:
    """End-to-end DataFrame construction from JSON fixtures."""

    def test_single_file_multi_backend(self, tmp_path: Path) -> None:
        """One JSON with multiple backends yields one row per backend."""
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
        assert (df["commit"] == "abc1234").all()
        assert (df["datetime"] == "2026-01-01T00:00:00").all()

    def test_multi_file_merges_machines(self, tmp_path: Path) -> None:
        """Rows from multiple JSONs merge into one DataFrame keyed by machine label."""
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
        assert set(df["commit"]) == {"abc1234"}

    def test_fallback_machine_label_when_device_unset(self, tmp_path: Path) -> None:
        """Missing device label falls back to `node (backend)` in machine column."""
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
        """Benchmarks outside the requested group are excluded from the DataFrame."""
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
        """Happy-path CLI run writes a non-empty PNG and returns 0."""
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
        """JSON with zero matching benchmarks -> exit 2 + stderr message."""
        empty = tmp_path / "empty.json"
        _make_benchmark_json(empty, device_label=None, commit_id="abc", entries=[])
        rc = plot_module.main([str(empty), "-o", str(tmp_path / "out.png")])
        assert rc == 2
        assert "no benchmarks in group" in capsys.readouterr().err

    def test_main_exits_2_when_missing_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Non-existent input path -> exit 2 + stderr message naming the path."""
        rc = plot_module.main([str(tmp_path / "nope.json"), "-o", str(tmp_path / "out.png")])
        assert rc == 2
        assert "no such file" in capsys.readouterr().err

    def test_main_malformed_json_surfaces_path(self, tmp_path: Path) -> None:
        """Malformed JSON raises SystemExit whose message includes the offending path."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(SystemExit) as excinfo:
            plot_module.main([str(bad), "-o", str(tmp_path / "out.png")])
        assert str(bad) in str(excinfo.value)


class TestDedupe:
    """Default dedupe: keep newest row per (machine, backend, n_scat)."""

    def test_dedupe_keeps_latest_per_tuple(self, tmp_path: Path) -> None:
        """Two JSONs with the same (machine, backend, n_scat) collapse to the newer row."""
        old = tmp_path / "old.json"
        _make_benchmark_json(
            old,
            device_label="M2 Max",
            commit_id="aaaaaaa",
            datetime="2026-01-01T00:00:00",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.050)],
        )
        new = tmp_path / "new.json"
        _make_benchmark_json(
            new,
            device_label="M2 Max",
            commit_id="bbbbbbb",
            datetime="2026-02-01T00:00:00",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.005)],
        )
        df = plot_module.build_dataframe([old, new], group="simus_scaling")
        assert len(df) == 1
        assert df.iloc[0]["mean_s"] == 0.005
        assert df.iloc[0]["commit"] == "bbbbbbb"

    def test_dedupe_preserves_distinct_tuples(self, tmp_path: Path) -> None:
        """Distinct (machine, backend, n_scat) tuples all survive dedupe."""
        a = tmp_path / "a.json"
        _make_benchmark_json(
            a,
            device_label="M2 Max",
            commit_id="aaaaaaa",
            datetime="2026-01-01T00:00:00",
            entries=[
                ("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.010),
                ("test_bench_simus_scaling[jax-10000]", {"n_scat": 10000}, 0.050),
            ],
        )
        df = plot_module.build_dataframe([a], group="simus_scaling")
        assert len(df) == 2

    def test_all_commits_preserves_every_row(self, tmp_path: Path) -> None:
        """dedupe=False (i.e. --all-commits) keeps duplicate (machine, backend, n_scat) rows."""
        old = tmp_path / "old.json"
        _make_benchmark_json(
            old,
            device_label="M2 Max",
            commit_id="aaaaaaa",
            datetime="2026-01-01T00:00:00",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.050)],
        )
        new = tmp_path / "new.json"
        _make_benchmark_json(
            new,
            device_label="M2 Max",
            commit_id="bbbbbbb",
            datetime="2026-02-01T00:00:00",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.005)],
        )
        df = plot_module.build_dataframe([old, new], group="simus_scaling", dedupe=False)
        assert len(df) == 2
        assert set(df["commit"]) == {"aaaaaaa", "bbbbbbb"}


class TestCommitFilter:
    """Commit-prefix filter on build_dataframe / --commit flag."""

    def test_commit_filter_selects_single_commit(self, tmp_path: Path) -> None:
        """Only rows whose commit starts with the given prefix survive."""
        a = tmp_path / "a.json"
        _make_benchmark_json(
            a,
            device_label="M2 Max",
            commit_id="abcdef1234",
            datetime="2026-01-01T00:00:00",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.010)],
        )
        b = tmp_path / "b.json"
        _make_benchmark_json(
            b,
            device_label="H100 JAX-CUDA",
            commit_id="9876abcd",
            datetime="2026-02-01T00:00:00",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.001)],
        )
        df = plot_module.build_dataframe([a, b], group="simus_scaling", commit_filter="abcdef")
        assert len(df) == 1
        assert df.iloc[0]["commit"] == "abcdef1234"

    def test_commit_filter_no_match_is_empty(self, tmp_path: Path) -> None:
        """Filter that matches nothing returns an empty frame (main() then exits 2)."""
        fixture = tmp_path / "run.json"
        _make_benchmark_json(
            fixture,
            device_label="M2 Max",
            commit_id="abcdef1234",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.010)],
        )
        df = plot_module.build_dataframe([fixture], group="simus_scaling", commit_filter="zzzzz")
        assert df.empty


class TestCliFilters:
    """CLI wiring of --commit and --all-commits."""

    def _two_commit_fixtures(self, tmp_path: Path) -> tuple[Path, Path]:
        old = tmp_path / "old.json"
        _make_benchmark_json(
            old,
            device_label="M2 Max",
            commit_id="aaaaaaa",
            datetime="2026-01-01T00:00:00",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.050)],
        )
        new = tmp_path / "new.json"
        _make_benchmark_json(
            new,
            device_label="M2 Max",
            commit_id="bbbbbbb",
            datetime="2026-02-01T00:00:00",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.005)],
        )
        return old, new

    def test_commit_flag_filters(self, tmp_path: Path) -> None:
        """`--commit aaa` keeps the matching commit and drops others."""
        old, new = self._two_commit_fixtures(tmp_path)
        out = tmp_path / "plot.png"
        rc = plot_module.main([str(old), str(new), "-o", str(out), "--commit", "aaa"])
        assert rc == 0
        assert out.exists()

    def test_commit_flag_no_match_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Unknown commit prefix -> exit 2 with explanatory stderr."""
        old, new = self._two_commit_fixtures(tmp_path)
        rc = plot_module.main([str(old), str(new), "-o", str(tmp_path / "out.png"), "--commit", "zzzzz"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "--commit" in err
        assert "zzzzz" in err

    def test_all_commits_flag_disables_dedupe(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """`--all-commits` plots every row and the stderr summary reports both commits."""
        old, new = self._two_commit_fixtures(tmp_path)
        out = tmp_path / "plot.png"
        rc = plot_module.main([str(old), str(new), "-o", str(out), "--all-commits"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "Plotting 2 row(s)" in err
        assert "aaaaaaa" in err
        assert "bbbbbbb" in err

    def test_commit_and_all_commits_mutually_exclusive(self, tmp_path: Path) -> None:
        """Argparse refuses `--commit` + `--all-commits` with SystemExit 2."""
        fixture = tmp_path / "run.json"
        _make_benchmark_json(
            fixture,
            device_label="M2 Max",
            commit_id="abc",
            entries=[("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.010)],
        )
        with pytest.raises(SystemExit) as excinfo:
            plot_module.main([str(fixture), "-o", str(tmp_path / "out.png"), "--commit", "abc", "--all-commits"])
        assert excinfo.value.code == 2


class TestSpeedups:
    """`_compute_speedups` normalizes non-PyMUST rows against a PyMUST reference."""

    def test_speedup_single_machine(self, tmp_path: Path) -> None:
        """JAX 10x faster than PyMUST at the same n_scat -> speedup == 10."""
        fixture = tmp_path / "run.json"
        _make_benchmark_json(
            fixture,
            device_label="M2 Max",
            commit_id="abc1234",
            entries=[
                ("test_bench_pymust_scaling[1000]", {"n_scat": 1000}, 0.100),
                ("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.010),
            ],
        )
        df = plot_module.build_dataframe([fixture], group="simus_scaling")
        speedup = plot_module._compute_speedups(df)
        assert len(speedup) == 1
        row = speedup.iloc[0]
        assert row["backend"] == "jax"
        assert row["speedup"] == pytest.approx(10.0)

    def test_speedup_drops_n_scat_without_reference(self, tmp_path: Path) -> None:
        """n_scat values with no PyMUST row are excluded from the speedup frame."""
        fixture = tmp_path / "run.json"
        _make_benchmark_json(
            fixture,
            device_label="M2 Max",
            commit_id="abc1234",
            entries=[
                ("test_bench_pymust_scaling[1000]", {"n_scat": 1000}, 0.100),
                ("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.020),
                ("test_bench_simus_scaling[jax-10000]", {"n_scat": 10000}, 0.050),
            ],
        )
        df = plot_module.build_dataframe([fixture], group="simus_scaling")
        speedup = plot_module._compute_speedups(df)
        assert list(speedup["n_scat"]) == [1000]

    def test_speedup_empty_without_reference(self, tmp_path: Path) -> None:
        """No PyMUST rows -> empty speedup frame (so the 3rd panel is skipped)."""
        fixture = tmp_path / "run.json"
        _make_benchmark_json(
            fixture,
            device_label="M2 Max",
            commit_id="abc1234",
            entries=[
                ("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.020),
            ],
        )
        df = plot_module.build_dataframe([fixture], group="simus_scaling")
        speedup = plot_module._compute_speedups(df)
        assert speedup.empty

    def test_speedup_median_across_multiple_pymust_samples(self, tmp_path: Path) -> None:
        """Multiple PyMUST rows collapse to the median before normalizing."""
        fixture = tmp_path / "run.json"
        _make_benchmark_json(
            fixture,
            device_label="A",
            commit_id="abc1234",
            entries=[
                ("test_bench_pymust_scaling[1000]", {"n_scat": 1000}, 0.090),
                ("test_bench_simus_scaling[jax-1000]", {"n_scat": 1000}, 0.010),
            ],
        )
        fixture2 = tmp_path / "run2.json"
        _make_benchmark_json(
            fixture2,
            device_label="B",
            commit_id="abc1234",
            entries=[
                ("test_bench_pymust_scaling[1000]", {"n_scat": 1000}, 0.110),
            ],
        )
        df = plot_module.build_dataframe([fixture, fixture2], group="simus_scaling")
        speedup = plot_module._compute_speedups(df)
        assert len(speedup) == 1
        assert speedup.iloc[0]["speedup"] == pytest.approx(10.0)


class TestCommitSummary:
    """_commit_summary() title-suffix formatting."""

    def test_single_commit(self) -> None:
        """Single commit is rendered as `@ <7-char>`."""
        import pandas as pd

        df = pd.DataFrame({"commit": ["abcdef1234567"]})
        assert plot_module._commit_summary(df) == "@ abcdef1"

    def test_multi_commit_lists_first_three(self) -> None:
        """Multiple commits list up to three short hashes."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "commit": [
                    "aaaaaaa0000",
                    "bbbbbbb0000",
                    "ccccccc0000",
                    "ddddddd0000",
                ]
            }
        )
        summary = plot_module._commit_summary(df)
        assert summary.startswith("@ 4 commits:")
        assert "aaaaaaa" in summary
        assert "bbbbbbb" in summary
        assert "ccccccc" in summary
        assert "ddddddd" not in summary
        assert summary.endswith("...")

    def test_empty_df(self) -> None:
        """Empty frame yields no suffix."""
        import pandas as pd

        assert plot_module._commit_summary(pd.DataFrame(columns=["commit"])) == ""
