"""Behavioral contracts for cached HTTPS downloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from fast_simus.io.utils import cached_download


def test_cached_download_rejects_non_https(tmp_path: Path) -> None:
    """Downloads are limited to HTTPS URLs."""
    with pytest.raises(ValueError, match="non-HTTPS"):
        cached_download("http://example.invalid/file.bin", cache_dir=tmp_path)


def test_cached_download_reuses_matching_cache(tmp_path: Path) -> None:
    """A cached file of the expected size is returned without a network fetch."""
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"abc")
    mtime_ns = dest.stat().st_mtime_ns

    result = cached_download(
        "https://example.invalid/file.bin",
        cache_dir=tmp_path,
        filename="file.bin",
        expected_size=3,
    )

    assert result == dest
    assert result.read_bytes() == b"abc"
    assert dest.stat().st_mtime_ns == mtime_ns
