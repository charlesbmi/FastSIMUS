r"""Wrap any pytest-benchmark invocation under NVIDIA Nsight Compute (NCU).

Skips warmup launches via ``--launch-skip``, captures one steady-state launch
of ``simus_fused_kernel``, and writes an ``.ncu-rep`` report.

Replaces the per-kernel ``tools/ncu_profile_v*.py`` proliferation that lived
on the experimentation branch -- any benchmark in ``tests/benchmarks/`` can
now be deeply profiled with one command.

Example:
    env NCU=$(command -v ncu) $(which python) scripts/ncu_pytest.py \
        -k "test_bench_simus_scaling and 100000 and cupy" \
        -o /tmp/4090_simus_100k.ncu-rep
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

NCU_DEFAULT = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"


def _default_section_folder(ncu_path: str) -> str | None:
    """Return Flox/Nix-packaged Nsight Compute sections directory when present."""
    resolved = Path(ncu_path).resolve()
    candidates = [
        resolved.parent.parent / "sections",
        resolved.parent / "sections",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return None


def _build_ncu_command(args: argparse.Namespace, *, python_executable: str) -> list[str]:
    """Build an Nsight Compute command for profiling a pytest benchmark."""
    command = [
        args.ncu,
        "--target-processes",
        "all",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        "--set",
        args.set,
        "--kernel-name",
        f"regex:{args.kernel_regex}",
        "--force-overwrite",
        "--export",
        args.output,
        python_executable,
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
    if section_folder := getattr(args, "section_folder", None):
        command[1:1] = ["--section-folder", section_folder]
    return command


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
        help="ncu --kernel-name regex:<this>. Default: simus_fused_kernel.",
    )
    parser.add_argument(
        "--ncu",
        default=os.environ.get("NCU", NCU_DEFAULT),
        help=f"Path to ncu binary. Default: {NCU_DEFAULT} (or $NCU).",
    )
    parser.add_argument(
        "--bench-path",
        default="tests/benchmarks/bench_simus_scaling.py",
        help="pytest target path. Default: tests/benchmarks/bench_simus_scaling.py.",
    )
    parser.add_argument(
        "--section-folder",
        default=os.environ.get("NCU_SECTION_FOLDER"),
        help="Nsight Compute section folder. Defaults to $NCU_SECTION_FOLDER or Flox package sections when found.",
    )
    parser.add_argument(
        "--set",
        default="full",
        help="ncu --set value. Default: full.",
    )
    args = parser.parse_args()
    if args.section_folder is None:
        args.section_folder = _default_section_folder(args.ncu)

    cmd = _build_ncu_command(args, python_executable=sys.executable)
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)  # noqa: S603 - args are constructed from argparse, not user shell input


if __name__ == "__main__":
    sys.exit(main())
