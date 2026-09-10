"""Behavioral contracts for PICMUS contrast artifact downloads."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from fast_simus.io.picmus import (
    PICMUS_CONTRAST_PHANTOM_EXPECTED_SIZE,
    PICMUS_CONTRAST_PHANTOM_NAME,
    PICMUS_CONTRAST_SCAN_EXPECTED_SIZE,
    PICMUS_CONTRAST_SCAN_NAME,
    cached_picmus_contrast_phantom,
    cached_picmus_contrast_scan,
)


@pytest.mark.parametrize(
    ("loader", "filename", "expected_size"),
    [
        (cached_picmus_contrast_phantom, PICMUS_CONTRAST_PHANTOM_NAME, PICMUS_CONTRAST_PHANTOM_EXPECTED_SIZE),
        (cached_picmus_contrast_scan, PICMUS_CONTRAST_SCAN_NAME, PICMUS_CONTRAST_SCAN_EXPECTED_SIZE),
    ],
)
def test_picmus_contrast_helpers_reuse_valid_cache(
    tmp_path: Path,
    loader: Callable[..., Path],
    filename: str,
    expected_size: int,
) -> None:
    """A correctly sized cache file is returned as the named PICMUS artifact."""
    dest = tmp_path / filename
    dest.write_bytes(b"x" * expected_size)
    mtime_ns = dest.stat().st_mtime_ns

    result = loader(cache_dir=tmp_path)

    assert result == dest
    assert dest.stat().st_mtime_ns == mtime_ns


def test_picmus_contrast_scan_downloads_public_release(tmp_path: Path) -> None:
    """The public GitHub Release scan file is downloaded, verified, and then cached."""
    stale = tmp_path / PICMUS_CONTRAST_SCAN_NAME
    stale.write_bytes(b"stale")

    path = cached_picmus_contrast_scan(cache_dir=tmp_path)
    assert path == stale
    assert path.stat().st_size == PICMUS_CONTRAST_SCAN_EXPECTED_SIZE

    mtime_ns = path.stat().st_mtime_ns
    cached = cached_picmus_contrast_scan(cache_dir=tmp_path)
    assert cached == path
    assert cached.stat().st_mtime_ns == mtime_ns
