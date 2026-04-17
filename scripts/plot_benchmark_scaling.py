"""Plot pytest-benchmark scaling results across machines and backends.

Loads one or more ``pytest-benchmark --benchmark-autosave`` JSON files and
produces a 1x2 log-log PNG: runtime (s) vs n_scat on the left, throughput
(scatterers/s) vs n_scat on the right. One line per (machine, backend).

Typical usage:

    # Single machine
    python scripts/plot_benchmark_scaling.py .benchmarks/**/*.json

    # Merge cross-machine results
    python scripts/plot_benchmark_scaling.py mac/*.json cuda/*.json

    # Custom output
    python scripts/plot_benchmark_scaling.py -o /tmp/scaling.png .benchmarks/**/*.json

Cross-machine workflow:

    1. Mac:   FASTSIMUS_DEVICE_LABEL="M-series MLX+CPU" poe benchmark-scaling
    2. CUDA:  FASTSIMUS_DEVICE_LABEL="H100 JAX-CUDA"     poe benchmark-scaling
    3. Copy the CUDA JSON(s) back to the Mac.
    4. poe benchmark-plot mac/*.json cuda/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_DEFAULT_GROUP = "simus_scaling"
_DEFAULT_OUTPUT = Path(".benchmarks") / "scaling_plot.png"
_DEVICE_LABEL_KEY = "fast_simus_device_label"

# test_bench_<something>[backend-1000] or test_bench_<something>[1000]
# Captures the content between the first "[" and last "]".
_PARAM_BRACKET_RE = re.compile(r"\[(?P<params>.+)\]$")


@dataclass(frozen=True)
class BenchmarkRow:
    """One (machine, backend, n_scat) result extracted from a JSON file."""

    source_file: str
    machine: str
    backend: str
    n_scat: int
    mean_s: float
    stddev_s: float


def _expand_paths(args: list[str]) -> list[Path]:
    """Expand a list of paths-or-globs to a de-duplicated sorted list of files."""
    seen: set[Path] = set()
    results: list[Path] = []
    for arg in args:
        matches = glob.glob(arg, recursive=True) if any(c in arg for c in "*?[") else [arg]
        if not matches:
            # Treat as literal path; downstream check will surface a useful error.
            matches = [arg]
        for match in matches:
            p = Path(match).resolve()
            if p not in seen:
                seen.add(p)
                results.append(p)
    return results


def _extract_backend(benchmark_name: str) -> str:
    """Infer backend label from the pytest-benchmark name.

    Rules:
      * If the test function name starts with ``test_bench_pymust`` -> "pymust".
      * Otherwise, look at the first non-numeric token inside the ``[...]`` suffix
        (e.g. ``test_bench_simus_scaling[jax-1000]`` -> "jax").
      * Fall back to "unknown" if nothing matches.
    """
    # Split off any [param] suffix before checking the function name.
    func_name = benchmark_name.split("[", 1)[0]
    if func_name.startswith("test_bench_pymust"):
        return "pymust"

    match = _PARAM_BRACKET_RE.search(benchmark_name)
    if not match:
        return "unknown"
    for token in match.group("params").split("-"):
        if not token.isdigit():
            return token
    return "unknown"


def _machine_label(machine_info: dict[str, object], backend: str) -> str:
    """Build a machine label from pytest-benchmark machine_info."""
    label = machine_info.get(_DEVICE_LABEL_KEY)
    if isinstance(label, str) and label:
        return label
    node = machine_info.get("node", "unknown")
    return f"{node} ({backend})"


def _load_rows(path: Path, group: str) -> list[BenchmarkRow]:
    """Load rows from one pytest-benchmark JSON, filtered to ``group``."""
    try:
        with path.open("r") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        msg = f"Failed to parse JSON at {path}: {exc}"
        raise SystemExit(msg) from exc

    machine_info = data.get("machine_info", {})
    rows: list[BenchmarkRow] = []
    for bench in data.get("benchmarks", []):
        if bench.get("group") != group:
            continue
        params = bench.get("params") or {}
        stats = bench.get("stats") or {}
        n_scat = params.get("n_scat")
        mean_s = stats.get("mean")
        stddev_s = stats.get("stddev", 0.0)
        if n_scat is None or mean_s is None:
            continue
        backend = _extract_backend(bench.get("name", ""))
        rows.append(
            BenchmarkRow(
                source_file=str(path),
                machine=_machine_label(machine_info, backend),
                backend=backend,
                n_scat=int(n_scat),
                mean_s=float(mean_s),
                stddev_s=float(stddev_s),
            )
        )
    return rows


def build_dataframe(paths: list[Path], group: str) -> pd.DataFrame:
    """Load all JSONs, filter to ``group``, return a tidy DataFrame."""
    rows: list[BenchmarkRow] = []
    for path in paths:
        rows.extend(_load_rows(path, group))
    if not rows:
        return pd.DataFrame(columns=["source_file", "machine", "backend", "n_scat", "mean_s", "stddev_s", "throughput"])
    df = pd.DataFrame([r.__dict__ for r in rows])
    df["throughput"] = df["n_scat"] / df["mean_s"]
    return df.sort_values(["backend", "machine", "n_scat"]).reset_index(drop=True)


def _commit_summary(paths: list[Path]) -> str:
    """Return a short title suffix describing the git commit(s) across inputs."""
    commits: set[str] = set()
    for path in paths:
        try:
            with path.open("r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        commit_id = (data.get("commit_info") or {}).get("id")
        if isinstance(commit_id, str) and commit_id:
            commits.add(commit_id[:7])
    if not commits:
        return ""
    if len(commits) == 1:
        return f"@ {next(iter(commits))}"
    return f"@ mixed commits ({len(commits)})"


def render_plot(df: pd.DataFrame, output: Path, commit_summary: str = "") -> None:
    """Render the 1x2 log-log figure and save to ``output`` as PNG."""
    long_df = pd.concat(
        [
            df.assign(metric="Runtime (s)", value=df["mean_s"]),
            df.assign(metric="Throughput (scatterers/s)", value=df["throughput"]),
        ],
        ignore_index=True,
    )

    sns.set_theme(context="notebook", style="whitegrid")
    g = sns.relplot(
        data=long_df,
        x="n_scat",
        y="value",
        hue="backend",
        style="machine",
        col="metric",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=4.2,
        aspect=1.25,
    )
    for ax in g.axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of scatterers")
        ax.grid(True, which="both", alpha=0.3)
    axes = list(g.axes.flat)
    axes[0].set_ylabel("Runtime (s)")
    if len(axes) > 1:
        axes[1].set_ylabel("Throughput (scatterers/s)")
    title = "SIMUS scaling: PyMUST vs FastSIMUS backends"
    if commit_summary:
        title += f" {commit_summary}"
    g.figure.suptitle(title, y=1.02)

    output.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(g.figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more pytest-benchmark JSON files or shell globs (e.g. '.benchmarks/**/*.json').",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "-g",
        "--group",
        default=_DEFAULT_GROUP,
        help=f"pytest-benchmark group to plot (default: {_DEFAULT_GROUP!r}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    paths = _expand_paths(args.paths)
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"error: no such file: {p}", file=sys.stderr)
        return 2

    df = build_dataframe(paths, group=args.group)
    if df.empty:
        print(
            f"error: no benchmarks in group {args.group!r} found across {len(paths)} JSON file(s).",
            file=sys.stderr,
        )
        return 2

    render_plot(df, args.output, commit_summary=_commit_summary(paths))
    print(
        f"Wrote {args.output} ({len(df)} rows across {df['backend'].nunique()} backend(s), "
        f"{df['machine'].nunique()} machine(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
