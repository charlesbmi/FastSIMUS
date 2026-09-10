"""Download and cache public PICMUS HDF5 files for contrast figure generation."""

from __future__ import annotations

from pathlib import Path

from fast_simus.io.utils import cached_download

CACHE_DIR = Path.home() / ".cache" / "fast_simus" / "picmus"
PICMUS_RELEASE_TAG = "picmus-contrast-artifacts-v1"
PICMUS_RELEASE_DOWNLOAD_BASE = f"https://github.com/charlesbmi/FastSIMUS/releases/download/{PICMUS_RELEASE_TAG}"
PICMUS_CONTRAST_PHANTOM_NAME = "contrast_speckle_simu_phantom.hdf5"
PICMUS_CONTRAST_SCAN_NAME = "contrast_speckle_simu_scan.hdf5"
PICMUS_CONTRAST_PHANTOM_URL = f"{PICMUS_RELEASE_DOWNLOAD_BASE}/{PICMUS_CONTRAST_PHANTOM_NAME}"
PICMUS_CONTRAST_SCAN_URL = f"{PICMUS_RELEASE_DOWNLOAD_BASE}/{PICMUS_CONTRAST_SCAN_NAME}"
PICMUS_CONTRAST_PHANTOM_EXPECTED_SIZE = 2_304_364
PICMUS_CONTRAST_SCAN_EXPECTED_SIZE = 16_360


def cached_picmus_contrast_phantom(*, cache_dir: Path | None = None) -> Path:
    """Return the public PICMUS contrast scatterer phantom, downloading if needed."""
    dest_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    return cached_download(
        PICMUS_CONTRAST_PHANTOM_URL,
        cache_dir=dest_dir,
        filename=PICMUS_CONTRAST_PHANTOM_NAME,
        expected_size=PICMUS_CONTRAST_PHANTOM_EXPECTED_SIZE,
    )


def cached_picmus_contrast_scan(*, cache_dir: Path | None = None) -> Path:
    """Return the public PICMUS contrast scan axes file, downloading if needed."""
    dest_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    return cached_download(
        PICMUS_CONTRAST_SCAN_URL,
        cache_dir=dest_dir,
        filename=PICMUS_CONTRAST_SCAN_NAME,
        expected_size=PICMUS_CONTRAST_SCAN_EXPECTED_SIZE,
    )
