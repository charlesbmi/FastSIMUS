"""Unit tests for scaling benchmark plot helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import plot_benchmark_scaling


def test_build_long_dataframe_uses_custom_reference_label() -> None:
    """Speedup metric names can describe a NumPy SIMUS reference."""
    df = pd.DataFrame(
        [
            {
                "source_file": "cpu.json",
                "machine": "SIMUS (NumPy)",
                "backend": "numpy",
                "n_scat": 1000,
                "mean_s": 2.0,
                "stddev_s": 0.0,
                "datetime": "2026-04-30T00:00:00",
                "commit": "abcdef1",
                "throughput": 500.0,
            },
            {
                "source_file": "cuda.json",
                "machine": "Proposed (CUDA)",
                "backend": "cupy",
                "n_scat": 1000,
                "mean_s": 0.5,
                "stddev_s": 0.0,
                "datetime": "2026-04-30T00:00:00",
                "commit": "abcdef1",
                "throughput": 2000.0,
            },
        ]
    )

    long_df, col_order = plot_benchmark_scaling._build_long_dataframe(
        df,
        reference_backend="numpy",
        reference_label="SIMUS (NumPy)",
        series_column="machine",
    )

    assert "Speedup vs SIMUS (NumPy)" in col_order
    speedup = long_df[long_df["metric"] == "Speedup vs SIMUS (NumPy)"].iloc[0]
    assert speedup["series"] == "Proposed (CUDA)"
    assert speedup["value"] == 4.0


def test_parse_args_accepts_custom_plot_labels() -> None:
    """CLI exposes the label controls needed for publication figures."""
    args = plot_benchmark_scaling.parse_args(
        [
            "cpu.json",
            "cuda.json",
            "--reference-backend",
            "numpy",
            "--reference-label",
            "SIMUS (NumPy)",
            "--series-column",
            "machine",
            "--title",
            "SIMUS scaling: SIMUS (NumPy) vs Proposed",
            "-o",
            "figure.png",
        ]
    )

    assert args.paths == ["cpu.json", "cuda.json"]
    assert args.reference_backend == "numpy"
    assert args.reference_label == "SIMUS (NumPy)"
    assert args.series_column == "machine"
    assert args.title == "SIMUS scaling: SIMUS (NumPy) vs Proposed"
    assert args.output == Path("figure.png")
