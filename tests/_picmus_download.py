"""Download and cache public PICMUS HDF5 files for contrast figure generation.

The scatterer phantom and scan axes are hosted as a GitHub Release artifact so
callers do not download the 564 MB CREATIS challenge zip. The files remain
PICMUS challenge data: free of use with attribution. See
https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/node/48 and
Liebgott et al., IEEE IUS 2016, https://doi.org/10.1109/ULTSYM.2016.7728908.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

CACHE_DIR = Path.home() / ".cache" / "fast_simus" / "picmus"
PICMUS_RELEASE_TAG = "picmus-contrast-artifacts-v1"
PICMUS_RELEASE_DOWNLOAD_BASE = f"https://github.com/charlesbmi/FastSIMUS/releases/download/{PICMUS_RELEASE_TAG}"
PICMUS_CONTRAST_PHANTOM_NAME = "contrast_speckle_simu_phantom.hdf5"
PICMUS_CONTRAST_SCAN_NAME = "contrast_speckle_simu_scan.hdf5"
PICMUS_CONTRAST_PHANTOM_URL = f"{PICMUS_RELEASE_DOWNLOAD_BASE}/{PICMUS_CONTRAST_PHANTOM_NAME}"
PICMUS_CONTRAST_SCAN_URL = f"{PICMUS_RELEASE_DOWNLOAD_BASE}/{PICMUS_CONTRAST_SCAN_NAME}"
PICMUS_CONTRAST_PHANTOM_EXPECTED_SIZE = 2_304_364
PICMUS_CONTRAST_SCAN_EXPECTED_SIZE = 16_360


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
    dest_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    return cached_download(
        PICMUS_CONTRAST_PHANTOM_URL,
        dest_dir / PICMUS_CONTRAST_PHANTOM_NAME,
        expected_size=PICMUS_CONTRAST_PHANTOM_EXPECTED_SIZE,
    )


def cached_picmus_contrast_scan(*, cache_dir: Path | None = None) -> Path:
    """Return the public PICMUS contrast scan axes file, downloading if needed."""
    dest_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    return cached_download(
        PICMUS_CONTRAST_SCAN_URL,
        dest_dir / PICMUS_CONTRAST_SCAN_NAME,
        expected_size=PICMUS_CONTRAST_SCAN_EXPECTED_SIZE,
    )


def _cached_file_matches(path: Path, *, expected_size: int | None) -> bool:
    if not path.is_file():
        return False
    if expected_size is None:
        return path.stat().st_size > 0
    return path.stat().st_size == expected_size
