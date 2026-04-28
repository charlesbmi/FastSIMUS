r"""Wrap any pytest-benchmark invocation under NVIDIA Nsight Compute (NCU).

Skips warmup launches via ``--launch-skip``, captures one steady-state launch
of ``simus_fused_kernel``, and writes an ``.ncu-rep`` report.

Replaces the per-kernel ``tools/ncu_profile_v*.py`` proliferation that lived
on the experimentation branch -- any benchmark in ``tests/benchmarks/`` can
now be deeply profiled with one command.

Example:
    sudo $(which python) scripts/ncu_pytest.py \
        -k "test_bench_simus_scaling and 100000 and cupy" \
        -o /tmp/4090_simus_100k.ncu-rep
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

NCU_DEFAULT = "/usr/local/cuda/bin/ncu"


def main() -> int:
    """Argparse + subprocess driver. Returns NCU's exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-k",
        required=True,
        help="pytest -k expression (passed through verbatim)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help=".ncu-rep output path",
    )
    parser.add_argument(
        "--launch-skip",
        type=int,
        default=5,
        help="Skip the first N kernel launches (warmup). Default: 5.",
    )
    parser.add_argument(
        "--launch-count",
        type=int,
        default=1,
        help="Number of launches to capture after the skip. Default: 1.",
    )
    parser.add_argument(
        "--kernel-regex",
        default="simus_fused_kernel",
        help="ncu --kernel-id ::regex:<this>. Default: simus_fused_kernel.",
    )
    parser.add_argument(
        "--ncu",
        default=os.environ.get("NCU", NCU_DEFAULT),
        help=f"Path to ncu binary. Default: {NCU_DEFAULT} (or $NCU).",
    )
    parser.add_argument(
        "--bench-path",
        default="tests/benchmarks/",
        help="pytest target path. Default: tests/benchmarks/.",
    )
    parser.add_argument(
        "--set",
        default="full",
        help="ncu --set value. Default: full.",
    )
    args = parser.parse_args()

    cmd = [
        args.ncu,
        "--target-processes",
        "all",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        "--set",
        args.set,
        "--kernel-id",
        f"::regex:{args.kernel_regex}",
        "-f",
        "-o",
        args.output,
        "--",
        sys.executable,
        "-m",
        "pytest",
        args.bench_path,
        "--benchmark-only",
        "-p",
        "no:xdist",
        "-k",
        args.k,
        # Keep pytest-benchmark from rerunning the timing loop -- NCU
        # profiling already takes 30+ seconds per launch.
        "--benchmark-min-rounds=1",
        "--benchmark-min-time=0",
    ]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)  # noqa: S603 - args are constructed from argparse, not user shell input


if __name__ == "__main__":
    sys.exit(main())
