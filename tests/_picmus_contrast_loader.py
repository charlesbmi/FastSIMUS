"""HDF5 loader for the public PICMUS contrast-speckle phantom."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np

PICMUS_CONTRAST_GROUP = "/US/US_DATASET0000"
MIN_PUBLIC_CONTRAST_SCATTERERS = 100_000
PLANAR_Y_ATOL_M = 1e-12
N_PICMUS_CONTRAST_CYSTS = 9


class ContrastPhantomData(NamedTuple):
    """Exact public PICMUS contrast scatterers and cyst metadata."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    rc: np.ndarray
    cyst_center_x_m: np.ndarray
    cyst_center_z_m: np.ndarray
    cyst_diameter_m: np.ndarray
    roi_metadata: dict[str, float | tuple[float, ...]]


def load_picmus_contrast_phantom(path: Path) -> ContrastPhantomData:
    """Load exact PICMUS contrast scatterers from a public HDF5 phantom file.

    Args:
        path: HDF5 file containing `/US/US_DATASET0000`.

    Returns:
        Contrast phantom data with positions in meters and scatterer amplitudes.

    Raises:
        ValueError: If the file does not match the expected public PICMUS schema.
    """
    with h5py.File(path, "r") as handle:
        if PICMUS_CONTRAST_GROUP not in handle:
            raise ValueError(f"PICMUS contrast group missing: {PICMUS_CONTRAST_GROUP}")
        group = handle[PICMUS_CONTRAST_GROUP]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"PICMUS contrast path is not a group: {PICMUS_CONTRAST_GROUP}")

        positions_m = _read_dataset(group, "scatterers_positions")
        amplitudes = _read_dataset(group, "scatterers_amplitude")
        cyst_center_x_m = _read_cyst_array(group, "phantom_occlusionCenterX")
        cyst_center_z_m = _read_cyst_array(group, "phantom_occlusionCenterZ")
        cyst_diameter_m = _read_cyst_array(group, "phantom_occlusionDiameter")
        roi_metadata = _read_roi_metadata(group)

    _validate_positions(positions_m)
    n_scatterers = positions_m.shape[1]
    _validate_amplitudes(amplitudes, n_scatterers)
    _validate_planar_y(positions_m[1])

    return ContrastPhantomData(
        x_m=_readonly_array(positions_m[0]),
        y_m=_readonly_array(positions_m[1]),
        z_m=_readonly_array(positions_m[2]),
        rc=_readonly_array(amplitudes),
        cyst_center_x_m=_readonly_array(cyst_center_x_m),
        cyst_center_z_m=_readonly_array(cyst_center_z_m),
        cyst_diameter_m=_readonly_array(cyst_diameter_m),
        roi_metadata=roi_metadata,
    )


def _read_dataset(group: h5py.Group, name: str) -> np.ndarray:
    """Read a required HDF5 dataset as a NumPy array."""
    if name not in group:
        raise ValueError(f"PICMUS contrast dataset missing: {PICMUS_CONTRAST_GROUP}/{name}")
    dataset = group[name]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"PICMUS contrast path is not a dataset: {PICMUS_CONTRAST_GROUP}/{name}")
    return np.asarray(dataset[()], dtype=np.float64)


def _read_cyst_array(group: h5py.Group, name: str) -> np.ndarray:
    """Read and validate a required 9-cyst metadata array."""
    cyst_array = _read_dataset(group, name)
    if cyst_array.shape != (N_PICMUS_CONTRAST_CYSTS,):
        raise ValueError(
            f"{name} must have shape ({N_PICMUS_CONTRAST_CYSTS},), got {cyst_array.shape}",
        )
    if not np.all(np.isfinite(cyst_array)):
        raise ValueError(f"{name} must contain only finite values")
    return cyst_array


def _read_roi_metadata(group: h5py.Group) -> dict[str, float | tuple[float, ...]]:
    """Read optional ROI metadata from scalar/vector attributes and datasets."""
    metadata: dict[str, float | tuple[float, ...]] = {}
    for name, value in group.attrs.items():
        if "ROI" in name:
            metadata[name] = _metadata_value(value)
    for name, value in group.items():
        if "ROI" in name and isinstance(value, h5py.Dataset):
            metadata[name] = _metadata_value(value[()])
    return metadata


def _metadata_value(value: object) -> float | tuple[float, ...]:
    """Convert numeric HDF5 metadata into immutable Python values."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        return float(array)
    return tuple(float(entry) for entry in array.ravel())


def _validate_positions(positions_m: np.ndarray) -> None:
    """Validate public PICMUS contrast scatterer positions."""
    if positions_m.ndim != 2 or positions_m.shape[0] != 3:
        raise ValueError(f"scatterers_positions must have shape (3, n_scatterers), got {positions_m.shape}")
    if positions_m.shape[1] <= MIN_PUBLIC_CONTRAST_SCATTERERS:
        raise ValueError(
            "scatterers_positions must contain more than "
            f"{MIN_PUBLIC_CONTRAST_SCATTERERS} scatterers, got {positions_m.shape[1]}",
        )
    if not np.all(np.isfinite(positions_m)):
        raise ValueError("scatterers_positions must contain only finite values")


def _validate_amplitudes(amplitudes: np.ndarray, n_scatterers: int) -> None:
    """Validate public PICMUS contrast scatterer amplitudes."""
    if amplitudes.shape != (n_scatterers,):
        raise ValueError(f"scatterers_amplitude must have shape ({n_scatterers},), got {amplitudes.shape}")
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("scatterers_amplitude must contain only finite values")
    if float(np.var(amplitudes)) <= 0.0:
        raise ValueError("scatterers_amplitude must have nonzero variance")


def _validate_planar_y(y_m: np.ndarray) -> None:
    """Validate the public contrast phantom is planar before 2D simulation."""
    if not np.allclose(y_m, 0.0, atol=PLANAR_Y_ATOL_M, rtol=0.0):
        max_abs_y_m = float(np.max(np.abs(y_m)))
        raise ValueError(f"PICMUS contrast y coordinates must be zero before 2D projection; max |y|={max_abs_y_m:g} m")


def _readonly_array(array: np.ndarray) -> np.ndarray:
    """Return a read-only copy of an array loaded from HDF5."""
    readonly = np.array(array, dtype=np.float64, copy=True)
    readonly.setflags(write=False)
    return readonly
