"""Tests for cached PICMUS GitHub Release downloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._picmus_download import (
    PICMUS_CONTRAST_PHANTOM_EXPECTED_SIZE,
    PICMUS_CONTRAST_PHANTOM_URL,
    PICMUS_CONTRAST_SCAN_EXPECTED_SIZE,
    PICMUS_CONTRAST_SCAN_URL,
    cached_download,
    cached_picmus_contrast_phantom,
    cached_picmus_contrast_scan,
)


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
    """An already-cached phantom is returned without contacting GitHub."""
    dest = tmp_path / "contrast_speckle_simu_phantom.hdf5"
    dest.write_bytes(b"x" * PICMUS_CONTRAST_PHANTOM_EXPECTED_SIZE)

    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("urlopen should not run for a valid phantom cache")

    monkeypatch.setattr("tests._picmus_download.urlopen", fail_urlopen)

    result = cached_picmus_contrast_phantom(cache_dir=tmp_path)
    assert result == dest


def test_cached_picmus_contrast_phantom_downloads_github_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing phantom is fetched from the public GitHub Release asset URL."""
    calls: list[tuple[str, Path, int | None]] = []

    def fake_download(url: str, output_path: Path, *, expected_size: int | None = None) -> Path:
        calls.append((url, output_path, expected_size))
        output_path.write_bytes(b"phantom")
        return output_path

    monkeypatch.setattr("tests._picmus_download.cached_download", fake_download)

    result = cached_picmus_contrast_phantom(cache_dir=tmp_path)
    assert result.name == "contrast_speckle_simu_phantom.hdf5"
    assert calls == [
        (
            PICMUS_CONTRAST_PHANTOM_URL,
            tmp_path / "contrast_speckle_simu_phantom.hdf5",
            PICMUS_CONTRAST_PHANTOM_EXPECTED_SIZE,
        )
    ]
    assert PICMUS_CONTRAST_PHANTOM_URL.startswith("https://github.com/charlesbmi/FastSIMUS/releases/download/")


def test_cached_picmus_contrast_scan_downloads_github_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing scan file is fetched from the public GitHub Release asset URL."""
    calls: list[tuple[str, Path, int | None]] = []

    def fake_download(url: str, output_path: Path, *, expected_size: int | None = None) -> Path:
        calls.append((url, output_path, expected_size))
        output_path.write_bytes(b"scan")
        return output_path

    monkeypatch.setattr("tests._picmus_download.cached_download", fake_download)

    result = cached_picmus_contrast_scan(cache_dir=tmp_path)
    assert result.name == "contrast_speckle_simu_scan.hdf5"
    assert calls == [
        (
            PICMUS_CONTRAST_SCAN_URL,
            tmp_path / "contrast_speckle_simu_scan.hdf5",
            PICMUS_CONTRAST_SCAN_EXPECTED_SIZE,
        )
    ]
    assert PICMUS_CONTRAST_SCAN_URL.startswith("https://github.com/charlesbmi/FastSIMUS/releases/download/")
