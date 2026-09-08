"""Download and cache public PICMUS HDF5 files for contrast figure generation."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlopen

CACHE_DIR = Path.home() / ".cache" / "fast_simus" / "picmus"
PICMUS_ARCHIVE_URL = (
    "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/sites/"
    "www.creatis.insa-lyon.fr.Challenge.IEEE_IUS_2016/files/archive_to_download.zip"
)
PICMUS_ARCHIVE_EXPECTED_SIZE = 564_074_891
PICMUS_CONTRAST_PHANTOM_MEMBER = (
    "archive_to_download/database/simulation/contrast_speckle/contrast_speckle_simu_phantom.hdf5"
)
PICMUS_CONTRAST_SCAN_MEMBER = "archive_to_download/database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5"
PICMUS_CONTRAST_PHANTOM_NAME = "contrast_speckle_simu_phantom.hdf5"
PICMUS_CONTRAST_SCAN_NAME = "contrast_speckle_simu_scan.hdf5"


def cached_download(url: str, output_path: Path, *, expected_size: int | None = None) -> Path:
    """Download `url` to `output_path` unless a matching file is already cached."""
    if not url.startswith("https://"):
        raise ValueError(f"Refusing to download non-HTTPS URL: {url}")
    output_path = Path(output_path)
    if _cached_file_matches(output_path, expected_size=expected_size):
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response, output_path.open("wb") as handle:  # noqa: S310
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if not _cached_file_matches(output_path, expected_size=expected_size):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file failed size check: {output_path}")
    return output_path


def cached_picmus_contrast_phantom(*, cache_dir: Path | None = None) -> Path:
    """Return the public PICMUS contrast scatterer phantom, downloading if needed."""
    return _cached_picmus_member(
        PICMUS_CONTRAST_PHANTOM_MEMBER,
        PICMUS_CONTRAST_PHANTOM_NAME,
        cache_dir=cache_dir,
    )


def cached_picmus_contrast_scan(*, cache_dir: Path | None = None) -> Path:
    """Return the public PICMUS contrast scan axes file, downloading if needed."""
    return _cached_picmus_member(
        PICMUS_CONTRAST_SCAN_MEMBER,
        PICMUS_CONTRAST_SCAN_NAME,
        cache_dir=cache_dir,
    )


def _cached_picmus_member(member: str, filename: str, *, cache_dir: Path | None) -> Path:
    dest_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    dest = dest_dir / filename
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    archive = cached_download(
        PICMUS_ARCHIVE_URL,
        dest_dir / "archive_to_download.zip",
        expected_size=PICMUS_ARCHIVE_EXPECTED_SIZE,
    )
    _extract_zip_member(archive, member, dest)
    return dest


def _extract_zip_member(archive: Path, member: str, dest: Path) -> Path:
    """Extract one archive member to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle, handle.open(member) as source, dest.open("wb") as output:
        output.write(source.read())
    return dest


def _cached_file_matches(path: Path, *, expected_size: int | None) -> bool:
    if not path.is_file():
        return False
    if expected_size is None:
        return path.stat().st_size > 0
    return path.stat().st_size == expected_size
