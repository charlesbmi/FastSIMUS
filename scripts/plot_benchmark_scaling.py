"""Plot pytest-benchmark scaling results across machines and backends.

Loads one or more ``pytest-benchmark --benchmark-autosave`` JSON files and
produces a log-log PNG with runtime (s) vs n_scat and throughput
(scatterers/s) vs n_scat. When PyMUST reference rows are present, the plot
also includes a third panel showing speedup vs PyMUST. One line per
(machine, backend).

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
    datetime: str  # Top-level ISO timestamp from the pytest-benchmark JSON.
    commit: str  # Full commit_info.id (short-hashed only for display).


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
    file_datetime = str(data.get("datetime") or "")
    commit_id = str((data.get("commit_info") or {}).get("id") or "")
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
                datetime=file_datetime,
                commit=commit_id,
            )
        )
    return rows


_DATAFRAME_COLUMNS = [
    "source_file",
    "machine",
    "backend",
    "n_scat",
    "mean_s",
    "stddev_s",
    "datetime",
    "commit",
    "throughput",
]


def build_dataframe(
    paths: list[Path],
    group: str,
    *,
    dedupe: bool = True,
    commit_filter: str | None = None,
) -> pd.DataFrame:
    """Load all JSONs, filter to ``group``, return a tidy DataFrame.

    Args:
        paths: Expanded list of JSON files to parse.
        group: pytest-benchmark group name to keep.
        dedupe: If True (default), collapse duplicate (machine, backend, n_scat)
            rows by keeping the latest ``datetime``. Set False to preserve every
            row (e.g. for `--all-commits`).
        commit_filter: If set, drop rows whose ``commit`` does not start with
            this prefix.
    """
    rows: list[BenchmarkRow] = []
    for path in paths:
        rows.extend(_load_rows(path, group))
    if not rows:
        return pd.DataFrame(columns=_DATAFRAME_COLUMNS)
    df = pd.DataFrame([r.__dict__ for r in rows])
    if commit_filter:
        df = df[df["commit"].str.startswith(commit_filter)]
    if dedupe and not df.empty:
        df = df.sort_values("datetime", kind="stable").drop_duplicates(
            subset=["machine", "backend", "n_scat"], keep="last"
        )
    if df.empty:
        return pd.DataFrame(columns=_DATAFRAME_COLUMNS)
    df = df.copy()
    df["throughput"] = df["n_scat"] / df["mean_s"]
    return df.sort_values(["backend", "machine", "n_scat"]).reset_index(drop=True)


_REFERENCE_BACKEND = "pymust"
_RUNTIME_METRIC = "Runtime (s)"
_THROUGHPUT_METRIC = "Throughput (scatterers/s)"
_SPEEDUP_METRIC = "Speedup vs PyMUST"


def _compute_speedups(df: pd.DataFrame, reference_backend: str = _REFERENCE_BACKEND) -> pd.DataFrame:
    """Compute per-row speedup relative to the reference backend at the same ``n_scat``.

    The reference mean is the median of all reference-backend runtimes at each
    ``n_scat`` (robust to multiple PyMUST samples across machines/commits).
    Reference rows themselves are excluded from the output. Returns an empty
    DataFrame if there are no reference-backend rows (e.g. a run that skipped
    PyMUST), so callers can drop the speedup panel gracefully.
    """
    if df.empty or "backend" not in df.columns:
        return df.iloc[0:0].assign(speedup=pd.Series(dtype="float64"))
    ref = df[df["backend"] == reference_backend]
    others = df[df["backend"] != reference_backend]
    if ref.empty or others.empty:
        return df.iloc[0:0].assign(speedup=pd.Series(dtype="float64"))
    ref_mean = ref.groupby("n_scat")["mean_s"].median().rename("_ref_mean_s")
    merged = others.merge(ref_mean, on="n_scat", how="inner")
    if merged.empty:
        return df.iloc[0:0].assign(speedup=pd.Series(dtype="float64"))
    merged = merged.copy()
    merged["speedup"] = merged["_ref_mean_s"] / merged["mean_s"]
    return merged.drop(columns=["_ref_mean_s"]).reset_index(drop=True)


def _unique_short_commits(df: pd.DataFrame) -> list[str]:
    """Return the ordered set of short (7-char) commits actually present in ``df``."""
    if df.empty or "commit" not in df.columns:
        return []
    seen: list[str] = []
    for commit in df["commit"]:
        if not isinstance(commit, str) or not commit:
            continue
        short = commit[:7]
        if short not in seen:
            seen.append(short)
    return seen


def _commit_summary(df: pd.DataFrame) -> str:
    """Return a short title suffix describing the git commit(s) actually plotted."""
    commits = _unique_short_commits(df)
    if not commits:
        return ""
    if len(commits) == 1:
        return f"@ {commits[0]}"
    preview = ", ".join(commits[:3])
    if len(commits) > 3:
        preview += ", ..."
    return f"@ {len(commits)} commits: {preview}"


def render_plot(df: pd.DataFrame, output: Path, commit_summary: str = "") -> None:
    """Render the log-log figure and save to ``output`` as PNG.

    Two panels (Runtime, Throughput) are always shown; a third "Speedup vs
    PyMUST" panel is appended when at least one reference PyMUST row is
    available to normalize against.
    """
    speedup_df = _compute_speedups(df)
    frames = [
        df.assign(metric=_RUNTIME_METRIC, value=df["mean_s"]),
        df.assign(metric=_THROUGHPUT_METRIC, value=df["throughput"]),
    ]
    col_order = [_RUNTIME_METRIC, _THROUGHPUT_METRIC]
    if not speedup_df.empty:
        frames.append(speedup_df.assign(metric=_SPEEDUP_METRIC, value=speedup_df["speedup"]))
        col_order.append(_SPEEDUP_METRIC)
    long_df = pd.concat(frames, ignore_index=True)

    sns.set_theme(context="notebook", style="whitegrid")
    g = sns.relplot(
        data=long_df,
        x="n_scat",
        y="value",
        hue="backend",
        style="machine",
        col="metric",
        col_order=col_order,
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=4.2,
        aspect=1.25,
    )
    axes = list(g.axes.flat)
    for ax, col_name in zip(axes, col_order, strict=True):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of scatterers")
        ax.grid(True, which="both", alpha=0.3)
        if col_name == _RUNTIME_METRIC:
            ax.set_ylabel("Runtime (s)")
        elif col_name == _THROUGHPUT_METRIC:
            ax.set_ylabel("Throughput (scatterers/s)")
        elif col_name == _SPEEDUP_METRIC:
            ax.set_ylabel("Speedup (x, vs PyMUST)")
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    title = "SIMUS scaling: PyMUST vs FastSIMUS backends"
    if commit_summary:
        title += f" {commit_summary}"
    g.figure.suptitle(title, y=1.02)

    output.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output, dpi=150, bbox_inches="tight")
    g.figure.close()


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
    commit_group = parser.add_mutually_exclusive_group()
    commit_group.add_argument(
        "--commit",
        default=None,
        help="Only plot rows whose commit_info.id starts with this prefix (e.g. 'a3a98ed').",
    )
    commit_group.add_argument(
        "--all-commits",
        "--no-dedupe",
        dest="all_commits",
        action="store_true",
        help=(
            "Disable the default 'keep newest per (machine, backend, n_scat)' dedupe "
            "and plot every row from every input JSON. Useful for surveying historical runs."
        ),
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

    raw_df = build_dataframe(paths, group=args.group, dedupe=False)
    if raw_df.empty:
        print(
            f"error: no benchmarks in group {args.group!r} found across {len(paths)} JSON file(s).",
            file=sys.stderr,
        )
        return 2

    df = build_dataframe(
        paths,
        group=args.group,
        dedupe=not args.all_commits,
        commit_filter=args.commit,
    )
    if df.empty:
        if args.commit:
            print(
                f"error: no benchmarks in group {args.group!r} match --commit {args.commit!r} "
                f"across {len(paths)} JSON file(s).",
                file=sys.stderr,
            )
        else:
            print(
                f"error: no benchmarks in group {args.group!r} found across {len(paths)} JSON file(s).",
                file=sys.stderr,
            )
        return 2

    dropped = len(raw_df) - len(df)
    commits = _unique_short_commits(df)
    commits_str = ", ".join(commits) if commits else "none"
    print(
        f"Plotting {len(df)} row(s) from {len(paths)} JSON file(s)"
        f" (dropped {dropped} stale/filtered row(s); commits: {commits_str}).",
        file=sys.stderr,
    )

    render_plot(df, args.output, commit_summary=_commit_summary(df))
    print(
        f"Wrote {args.output} ({len(df)} rows across {df['backend'].nunique()} backend(s), "
        f"{df['machine'].nunique()} machine(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
