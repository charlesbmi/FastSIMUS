"""Unit tests for the Nsight Compute pytest wrapper."""

from __future__ import annotations

import argparse

import pytest

from scripts import ncu_pytest


def test_build_ncu_command_uses_ncu_compatible_options() -> None:
    """NCU command avoids options that fail on the local Nsight Compute CLI."""
    args = argparse.Namespace(
        k="test_bench_simus_scaling and 100000 and cupy",
        output="reports/4090_fastpath_100k.ncu-rep",
        launch_skip=5,
        launch_count=1,
        kernel_regex="simus_fused_kernel",
        ncu="/usr/local/cuda/bin/ncu",
        bench_path="tests/benchmarks/bench_simus_scaling.py",
        set="full",
        section_folder=None,
    )

    cmd = ncu_pytest._build_ncu_command(args, python_executable="python")

    assert "--" not in cmd
    assert "--kernel-id" not in cmd
    assert "--kernel-name" in cmd
    assert "regex:simus_fused_kernel" in cmd
    assert "--export" in cmd
    assert "reports/4090_fastpath_100k.ncu-rep" in cmd
    assert cmd[cmd.index("-k") + 1] == "test_bench_simus_scaling and 100000 and cupy"


def test_build_ncu_command_includes_section_folder_when_provided() -> None:
    """Flox-packaged NCU can be pointed at its packaged section files."""
    args = argparse.Namespace(
        k="cupy",
        output="reports/smoke.ncu-rep",
        launch_skip=0,
        launch_count=1,
        kernel_regex="simus_fused_kernel",
        ncu="ncu",
        bench_path="tests/benchmarks/bench_simus_scaling.py",
        set="speedOfLight",
        section_folder="/nix/store/example-nsight-compute/sections",
    )

    cmd = ncu_pytest._build_ncu_command(args, python_executable="python")

    assert cmd[1:3] == ["--section-folder", "/nix/store/example-nsight-compute/sections"]


def test_default_section_folder_resolves_bare_binary_name(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Section-folder autodetection should resolve bare command names via PATH lookup."""
    ncu_bin = tmp_path / "bin" / "ncu"
    ncu_bin.parent.mkdir(parents=True)
    ncu_bin.write_text("", encoding="utf-8")
    sections = tmp_path / "sections"
    sections.mkdir()

    monkeypatch.setattr(
        ncu_pytest.shutil,
        "which",
        lambda name: str(ncu_bin) if name == "ncu" else None,
    )
    assert ncu_pytest._default_section_folder("ncu") == str(sections)
