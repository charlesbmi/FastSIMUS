"""Download and cache files over HTTPS."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

CACHE_DIR = Path.home() / ".cache" / "fast_simus"
_USER_AGENT = "fast-simus"


def cached_download(
    url: str,
    cache_dir: str | Path = CACHE_DIR,
    filename: str | Path | None = None,
    timeout: int = 60,
    *,
    expected_size: int | None = None,
) -> Path:
    """Download `url` into `cache_dir` unless a matching file is already cached.

    Args:
        url: HTTPS URL to fetch.
        cache_dir: Directory for the cached file. Ignored when `filename` is absolute.
        filename: Destination name. Defaults to the URL path basename.
        timeout: Socket timeout in seconds.
        expected_size: If set, reuse or accept the file only when its size matches.

    Returns:
        Path to the cached file.

    Raises:
        ValueError: If `url` is not HTTPS.
        RuntimeError: If the download fails the size check.
    """
    if not url.startswith("https://"):
        raise ValueError(f"Refusing to download non-HTTPS URL: {url}")

    output_path = _cached_path(url, cache_dir, filename)
    if _cached_file_matches(output_path, expected_size=expected_size):
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    with urlopen(request, timeout=timeout) as response, output_path.open("wb") as handle:  # noqa: S310
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    if not _cached_file_matches(output_path, expected_size=expected_size):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file failed size check: {output_path}")
    return output_path


def _cached_path(url: str, cache_dir: str | Path, filename: str | Path | None) -> Path:
    if filename is None:
        return Path(cache_dir) / Path(url).name
    path = Path(filename)
    if path.is_absolute():
        return path
    return Path(cache_dir) / path


def _cached_file_matches(path: Path, *, expected_size: int | None) -> bool:
    if not path.is_file():
        return False
    if expected_size is None:
        return path.stat().st_size > 0
    return path.stat().st_size == expected_size
