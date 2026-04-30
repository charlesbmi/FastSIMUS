"""Unit tests for the Nsight Compute pytest wrapper."""

from __future__ import annotations

import argparse

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
    )

    cmd = ncu_pytest._build_ncu_command(args, python_executable="python")

    assert "--" not in cmd
    assert "--kernel-id" not in cmd
    assert "--kernel-name" in cmd
    assert "regex:simus_fused_kernel" in cmd
    assert "--export" in cmd
    assert "reports/4090_fastpath_100k.ncu-rep" in cmd
    assert cmd[cmd.index("-k") + 1] == "test_bench_simus_scaling and 100000 and cupy"
