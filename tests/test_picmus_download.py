"""Tests for cached PICMUS archive downloads."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tests._picmus_download import _extract_zip_member, cached_download, cached_picmus_contrast_phantom


def test_cached_download_skips_existing_file_with_matching_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An already-cached file of the expected size is not downloaded again."""
    cached = tmp_path / "file.bin"
    cached.write_bytes(b"abc")
    calls = 0

    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("urlopen should not run for a valid cache")

    monkeypatch.setattr("tests._picmus_download.urlopen", fail_urlopen)
    result = cached_download("https://example.invalid/file.bin", cached, expected_size=3)
    assert result == cached
    assert calls == 0


def test_cached_download_rejects_non_https(tmp_path: Path) -> None:
    """Downloads are limited to HTTPS URLs."""
    with pytest.raises(ValueError, match="non-HTTPS"):
        cached_download("http://example.invalid/file.bin", tmp_path / "file.bin")


def test_cached_picmus_contrast_phantom_uses_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-cached phantom is returned without contacting the archive."""
    dest = tmp_path / "contrast_speckle_simu_phantom.hdf5"
    dest.write_bytes(b"cached-phantom")

    def fail_download(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("cached phantom should not download the archive")

    monkeypatch.setattr("tests._picmus_download.cached_download", fail_download)

    result = cached_picmus_contrast_phantom(cache_dir=tmp_path)
    assert result.read_bytes() == b"cached-phantom"


def test_extract_zip_member_writes_named_file(tmp_path: Path) -> None:
    """Zip member extraction writes the requested file to the destination."""
    archive = tmp_path / "archive.zip"
    member = "database/simulation/contrast_speckle/contrast_speckle_simu_phantom.hdf5"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, b"phantom-bytes")
    dest = tmp_path / "contrast_speckle_simu_phantom.hdf5"
    written = _extract_zip_member(archive, member, dest)
    assert written.read_bytes() == b"phantom-bytes"
