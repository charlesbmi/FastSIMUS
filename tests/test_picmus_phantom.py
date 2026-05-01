"""Integration test scaffold for the PICMUS resolution phantom."""

from __future__ import annotations

import contextlib
import time
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, cast

import h5py
import numpy as np
import pytest
from beartype import beartype
from jaxtyping import Complex, Float, jaxtyped

from tests._picmus_contrast_loader import (
    MIN_PUBLIC_CONTRAST_SCATTERERS,
    PICMUS_CONTRAST_GROUP,
    ContrastPhantomData,
    load_picmus_contrast_phantom,
)
from tests._picmus_phantom_shared import (
    DYNAMIC_RANGE_DB,
    L11_PICMUS_ATTENUATION_DB_CM_MHZ,
    L11_PICMUS_FREQ_CENTER_HZ,
    L11_PICMUS_PITCH_M,
    L11_PICMUS_SAMPLING_FREQUENCY_HZ,
    L11_PICMUS_SIMUS_DB_THRESH,
    L11_PICMUS_SPEED_OF_SOUND_M_S,
    L11_PICMUS_TX_N_WAVELENGTHS,
    PICMUS_GRID_WARN_PIXELS,
    PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS,
    Phantom,
    PhantomCase,
    ReconstructedIq,
    RfStacks,
    _angle_cache_key,
    _assert_iq_close,
    _fastsimus_array_namespace,
    _iq_residual_display_db,
    _iq_to_display_db,
    _reconstruct_iq,
    _simulate_rf_stacks_for_case,
    make_l11_picmus_matched_params,
)
from tests._picmus_phantom_shared import (
    _expects_large_reconstruction_warning as _shared_expects_large_reconstruction_warning,
)
from tests._picmus_phantom_shared import (
    _image_axes_for_grid_wavelengths as _shared_image_axes_for_grid_wavelengths,
)

HAS_MATPLOTLIB = False
plt: ModuleType | None = None
with contextlib.suppress(ImportError):
    import matplotlib.pyplot as _matplotlib_pyplot

    plt = _matplotlib_pyplot
    HAS_MATPLOTLIB = True

HAS_SEABORN = False
sns: ModuleType | None = None
with contextlib.suppress(ImportError):
    import seaborn as _seaborn

    sns = _seaborn
    HAS_SEABORN = True


class ContrastBeamformedData(NamedTuple):
    """Beamformed full-angle contrast data for cache and figure generation."""

    fastsimus_iq: np.ndarray
    pymust_iq: np.ndarray
    x_axis_m: np.ndarray
    z_axis_m: np.ndarray
    selected_angles_rad: np.ndarray
    phantom_cache_key: tuple[str, ...]


class PicmusContrastPanelSpec(NamedTuple):
    """Display data and color scaling for one contrast comparison panel."""

    title: str
    image_db: np.ndarray
    cmap: str
    vmin_db: float
    vmax_db: float
    colorbar_label: str


class PicmusContrastScanAxes(NamedTuple):
    """Public PICMUS scan axes loaded next to the contrast phantom file."""

    x_axis_m: np.ndarray
    z_axis_m: np.ndarray


FREQ_CENTER_HZ = L11_PICMUS_FREQ_CENTER_HZ
PITCH_M = L11_PICMUS_PITCH_M
TX_N_WAVELENGTHS = L11_PICMUS_TX_N_WAVELENGTHS
SPEED_OF_SOUND_M_S = L11_PICMUS_SPEED_OF_SOUND_M_S
PICMUS_WAVELENGTH_M = SPEED_OF_SOUND_M_S / FREQ_CENTER_HZ
# Attenuation is not stored in the PICMUS HDF5 files; the prototype uses 0.5 dB/cm/MHz.
ATTENUATION_DB_CM_MHZ = L11_PICMUS_ATTENUATION_DB_CM_MHZ
SAMPLING_FREQUENCY_HZ = L11_PICMUS_SAMPLING_FREQUENCY_HZ
SIMUS_DB_THRESH = L11_PICMUS_SIMUS_DB_THRESH

PICMUS_PHANTOM_X_M = 1e-3 * np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -15.0,
        -10.0,
        -5.0,
        5.0,
        10.0,
        15.0,
        -15.0,
        -10.0,
        -5.0,
        5.0,
        10.0,
        15.0,
    ]
)
PICMUS_PHANTOM_Z_M = 1e-3 * np.array(
    [
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
    ]
)
PICMUS_FULL_ANGLES_RAD = np.deg2rad(np.linspace(-16.0, 16.0, 75))
PICMUS_BROADSIDE_ANGLES_RAD = PICMUS_FULL_ANGLES_RAD[36:39].copy()
PICMUS_APERTURE_X_M = 19.05000114440918e-3
IMAGE_X_MIN_M = -PICMUS_APERTURE_X_M
IMAGE_X_MAX_M = PICMUS_APERTURE_X_M
IMAGE_Z_MIN_M = 5.0e-3
IMAGE_Z_MAX_M = 50.0e-3

IMAGE_X_M = np.linspace(IMAGE_X_MIN_M, IMAGE_X_MAX_M, 96)
IMAGE_Z_M = np.linspace(IMAGE_Z_MIN_M, IMAGE_Z_MAX_M, 128)
IQ_ATOL_PEAK = 5e-2
PROPOSED_LABEL = "Proposed"
REFERENCE_LABEL = "PyMUST"
RESIDUAL_LABEL = f"|{PROPOSED_LABEL} - {REFERENCE_LABEL}|"
# 12 in x 360 dpi gives a 4320 px-wide three-panel PNG, close to a 4k export target.
PICMUS_PLOT_FIGSIZE_IN = (12.0, 4.0)
PICMUS_PLOT_MIN_WIDTH_PX = 4096
PICMUS_PLOT_DPI = 360
PICMUS_PLOT_INTERPOLATION = "nearest"
PICMUS_CONTRAST_GRID_WAVELENGTHS = 0.25
PICMUS_CONTRAST_RESIDUAL_DYNAMIC_RANGE_DB = 100.0
PICMUS_CONTRAST_FIGURE_HEIGHT_IN = 2.4
PICMUS_CONTRAST_COMPARISON_FIGSIZE_IN = (7.2, PICMUS_CONTRAST_FIGURE_HEIGHT_IN)
PICMUS_CONTRAST_TWO_PANEL_FIGSIZE_IN = (4.6, PICMUS_CONTRAST_FIGURE_HEIGHT_IN)
PICMUS_CONTRAST_TWO_PANEL_WSPACE = 0.02
PICMUS_CONTRAST_TWO_PANEL_YLABEL_PAD = 2.0
PICMUS_CONTRAST_FASTSIMUS_FIGSIZE_IN = (2.4, PICMUS_CONTRAST_FIGURE_HEIGHT_IN)
PICMUS_CONTRAST_IMAGE_ASPECT = "equal"
PICMUS_CONTRAST_SCAN_FILENAME = "contrast_speckle_simu_scan.hdf5"
PICMUS_CONTRAST_BEAMFORMED_CACHE_VERSION = "picmus-contrast-beamformed-v1"
PICMUS_CONTRAST_REFERENCE_TITLE = "PyMUST SIMUS baseline"
PICMUS_CONTRAST_PROPOSED_TITLE = "Proposed"
PICMUS_CONTRAST_RESIDUAL_TITLE = "Residual"
PICMUS_CONTRAST_DEBUG_CYST_SUFFIX = "_debug_cysts"
VALID_TEST_CONTRAST_SCATTERERS = MIN_PUBLIC_CONTRAST_SCATTERERS + 1
# The public contrast dataset exercises different simulators, so this is a loose
# finite/nonzero smoke gate rather than a numerical equivalence claim.
RF_PUBLIC_CONTRAST_ATOL_PEAK = 1.5

for constant_array in (
    PICMUS_PHANTOM_X_M,
    PICMUS_PHANTOM_Z_M,
    PICMUS_FULL_ANGLES_RAD,
    PICMUS_BROADSIDE_ANGLES_RAD,
    IMAGE_X_M,
    IMAGE_Z_M,
):
    constant_array.setflags(write=False)


@jaxtyped(typechecker=beartype)
def _assert_rf_stack_layout(
    fastsimus: Float[np.ndarray, "samples channels firings"],
    pymust: Float[np.ndarray, "samples channels firings"],
    angles_rad: Float[np.ndarray, "firings"],  # noqa: F821, UP037 - jaxtyping axis name.
    *,
    n_elements: int,
    n_firings: int,
) -> None:
    """Assert RF stacks use the expected PICMUS samples/channels/firings layout."""
    assert pymust.shape == fastsimus.shape
    assert fastsimus.shape[1:] == (n_elements, n_firings)
    assert angles_rad.shape == (n_firings,)


@jaxtyped(typechecker=beartype)
def _assert_reconstructed_iq_layout(
    fastsimus: Complex[np.ndarray, "z x"],
    pymust: Complex[np.ndarray, "z x"],
    *,
    n_z: int,
    n_x: int,
) -> None:
    """Assert reconstructed IQ images use the expected PICMUS z/x grid layout."""
    assert fastsimus.shape == (n_z, n_x)
    assert pymust.shape == (n_z, n_x)


def _angles_for_plot_mode(case: PhantomCase, angle_mode: str) -> np.ndarray:
    """Return the PICMUS angle sequence requested for diagnostic plotting."""
    if angle_mode == "broadside":
        return np.array(case.default_angles_rad, dtype=np.float64)
    if angle_mode == "full":
        if case.diagnostic_angles_rad is None:
            raise ValueError(f"Phantom case {case.id!r} has no full diagnostic angle sequence")
        return np.array(case.diagnostic_angles_rad, dtype=np.float64)
    raise ValueError(f"Unknown PICMUS phantom angle mode: {angle_mode}")


def _skip_unless_picmus_contrast_requested(
    *,
    run_picmus_contrast: bool,
    picmus_contrast_phantom_path: Path | None,
) -> Path:
    """Return the supplied contrast phantom path only when explicit gates are present."""
    if not run_picmus_contrast:
        pytest.skip("--run-picmus-contrast not provided")
    if picmus_contrast_phantom_path is None:
        pytest.skip("--picmus-contrast-phantom not provided")
    if not picmus_contrast_phantom_path.is_file():
        pytest.skip(f"--picmus-contrast-phantom does not exist: {picmus_contrast_phantom_path}")
    return picmus_contrast_phantom_path


def _write_picmus_contrast_hdf5(
    path: Path,
    *,
    positions_m: np.ndarray,
    amplitudes: np.ndarray,
) -> None:
    """Write the PICMUS contrast schema subset used by loader tests."""
    with h5py.File(path, "w") as handle:
        group = handle.create_group("/US/US_DATASET0000")
        group.create_dataset("scatterers_positions", data=positions_m)
        group.create_dataset("scatterers_amplitude", data=amplitudes)
        group.create_dataset("phantom_occlusionCenterX", data=np.linspace(-12.0e-3, 12.0e-3, 9))
        group.create_dataset("phantom_occlusionCenterZ", data=np.linspace(12.0e-3, 44.0e-3, 9))
        group.create_dataset("phantom_occlusionDiameter", data=np.full(9, 4.0e-3))
        group.attrs["phantom_ROIWidth"] = 38.0e-3
        group.attrs["phantom_ROIHeight"] = 45.0e-3


def _write_picmus_contrast_scan_hdf5(
    path: Path,
    *,
    x_axis_m: np.ndarray,
    z_axis_m: np.ndarray,
) -> None:
    """Write the public PICMUS scan-axis datasets used by contrast artifact tests."""
    with h5py.File(path, "w") as handle:
        group = handle.create_group("/US/US_DATASET0000")
        group.create_dataset("x_axis", data=x_axis_m)
        group.create_dataset("z_axis", data=z_axis_m)


def _valid_picmus_contrast_positions(n_scatterers: int = VALID_TEST_CONTRAST_SCATTERERS) -> np.ndarray:
    """Return plausible public contrast scatterer positions in meters."""
    positions_m = np.zeros((3, n_scatterers), dtype=np.float64)
    positions_m[0] = np.linspace(-18.0e-3, 18.0e-3, n_scatterers)
    positions_m[2] = np.linspace(5.0e-3, 50.0e-3, n_scatterers)
    return positions_m


def _valid_picmus_contrast_amplitudes(n_scatterers: int = VALID_TEST_CONTRAST_SCATTERERS) -> np.ndarray:
    """Return finite amplitudes with nonzero variance."""
    return np.linspace(-1.0, 1.0, n_scatterers, dtype=np.float64)


def _angles_for_picmus_contrast_mode(angle_mode: str) -> np.ndarray:
    """Return the contrast angle sequence for an explicit opt-in mode."""
    if angle_mode == "center":
        return np.array([PICMUS_FULL_ANGLES_RAD[PICMUS_FULL_ANGLES_RAD.size // 2]], dtype=np.float64)
    if angle_mode == "full":
        return PICMUS_FULL_ANGLES_RAD.copy()
    raise ValueError(f"Unknown PICMUS contrast angle mode: {angle_mode}")


def _picmus_contrast_scan_path_for_phantom(path: Path) -> Path:
    """Return the public contrast scan file expected next to a phantom file."""
    return path.with_name(PICMUS_CONTRAST_SCAN_FILENAME)


def _validate_picmus_contrast_scan_axis(axis_m: np.ndarray, *, name: str) -> np.ndarray:
    """Return a validated, read-only public scan axis."""
    axis_m = np.asarray(axis_m, dtype=np.float64)
    if axis_m.ndim != 1 or axis_m.size < 2:
        raise ValueError(f"{name} must be a one-dimensional scan axis with at least two samples")
    if not np.all(np.isfinite(axis_m)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all(np.diff(axis_m) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    axis_m.setflags(write=False)
    return axis_m


def _load_picmus_contrast_scan_axes(path: Path) -> PicmusContrastScanAxes | None:
    """Load public PICMUS scan axes from the sibling scan HDF5 file when present."""
    scan_path = _picmus_contrast_scan_path_for_phantom(path)
    if not scan_path.is_file():
        return None
    with h5py.File(scan_path, "r") as handle:
        group = handle[PICMUS_CONTRAST_GROUP]
        x_axis_m = _validate_picmus_contrast_scan_axis(np.asarray(group["x_axis"]), name="x_axis")
        z_axis_m = _validate_picmus_contrast_scan_axis(np.asarray(group["z_axis"]), name="z_axis")
    return PicmusContrastScanAxes(x_axis_m=x_axis_m, z_axis_m=z_axis_m)


def _picmus_contrast_scan_identity(path: Path) -> tuple[str, ...]:
    """Return cache-key fields for the optional public scan HDF5 file."""
    scan_path = _picmus_contrast_scan_path_for_phantom(path)
    if not scan_path.is_file():
        return ("scan=missing",)
    stat = scan_path.stat()
    return (
        f"scan_path={scan_path.resolve()}",
        f"scan_size={stat.st_size}",
        f"scan_mtime_ns={stat.st_mtime_ns}",
    )


def _picmus_contrast_artifact_axes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return lambda/4 beamforming axes covering the public contrast scan scope."""
    scan_axes = _load_picmus_contrast_scan_axes(path)
    if scan_axes is None:
        x_bounds_m = (IMAGE_X_MIN_M, IMAGE_X_MAX_M)
        z_bounds_m = (IMAGE_Z_MIN_M, IMAGE_Z_MAX_M)
    else:
        x_bounds_m = (float(scan_axes.x_axis_m[0]), float(scan_axes.x_axis_m[-1]))
        z_bounds_m = (float(scan_axes.z_axis_m[0]), float(scan_axes.z_axis_m[-1]))
    return _shared_image_axes_for_grid_wavelengths(
        PICMUS_CONTRAST_GRID_WAVELENGTHS,
        default_x_axis_m=np.array(x_bounds_m, dtype=np.float64),
        default_z_axis_m=np.array(z_bounds_m, dtype=np.float64),
        x_bounds_m=x_bounds_m,
        z_bounds_m=z_bounds_m,
        wavelength_m=PICMUS_WAVELENGTH_M,
        warn_pixels=PICMUS_GRID_WARN_PIXELS,
        stacklevel=3,
    )


def _make_picmus_contrast_phantom(data: ContrastPhantomData) -> Phantom:
    """Convert loaded public contrast data into the 2D shared Phantom shape."""
    if not np.allclose(data.y_m, 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("PICMUS contrast y coordinates must be zero before 2D Phantom projection")
    return Phantom(
        x=np.array(data.x_m, dtype=np.float64, copy=True),
        z=np.array(data.z_m, dtype=np.float64, copy=True),
        rc=np.array(data.rc, dtype=np.float64, copy=True),
    )


def _picmus_contrast_file_identity(path: Path, data: ContrastPhantomData) -> tuple[str, ...]:
    """Return public-data identity fields for contrast cache keys."""
    stat = path.stat()
    return (
        f"path={path.resolve()}",
        f"size={stat.st_size}",
        f"mtime_ns={stat.st_mtime_ns}",
        f"n_scatterers={data.x_m.size}",
    )


def _make_picmus_contrast_case(path: Path, data: ContrastPhantomData, *, angle_mode: str) -> PhantomCase:
    """Build a gated contrast case from exact public HDF5 scatterers."""
    angles_rad = _angles_for_picmus_contrast_mode(angle_mode)
    angles_key = _angle_cache_key(angles_rad)
    x_axis_m, z_axis_m = _picmus_contrast_artifact_axes(path)
    cache_key = (
        "contrast_speckle",
        "picmus-contrast-speckle-public-v1",
        *_picmus_contrast_file_identity(path, data),
        *_picmus_contrast_scan_identity(path),
        "l11-picmus-matched-v1",
        f"fs={SAMPLING_FREQUENCY_HZ:.6g}",
        f"tx={TX_N_WAVELENGTHS:g}",
        f"db={SIMUS_DB_THRESH:g}",
        f"grid_wavelengths={PICMUS_CONTRAST_GRID_WAVELENGTHS:g}",
        f"angle_mode={angle_mode}",
        f"angles={angles_key!r}",
    )
    return PhantomCase(
        id="contrast_speckle",
        cache_key=cache_key,
        phantom_factory=lambda: _make_picmus_contrast_phantom(data),
        matched_params_factory=make_l11_picmus_matched_params,
        default_angles_rad=angles_key,
        diagnostic_angles_rad=None,
        x_bounds_m=(float(x_axis_m[0]), float(x_axis_m[-1])),
        z_bounds_m=(float(z_axis_m[0]), float(z_axis_m[-1])),
        default_x_axis_m=tuple(float(x) for x in x_axis_m),
        default_z_axis_m=tuple(float(z) for z in z_axis_m),
        wavelength_m=PICMUS_WAVELENGTH_M,
        validation_mode="rf_public_contrast",
        iq_atol_peak=None,
    )


def _assert_rf_public_contrast_stacks_close(
    stacks: RfStacks,
    *,
    n_elements: int,
    n_firings: int,
    atol_peak: float = RF_PUBLIC_CONTRAST_ATOL_PEAK,
) -> None:
    """Assert public contrast RF stacks are finite, nonzero, and loosely matched."""
    _assert_rf_stack_layout(
        stacks.fastsimus,
        stacks.pymust,
        stacks.angles_rad,
        n_elements=n_elements,
        n_firings=n_firings,
    )
    if not np.all(np.isfinite(stacks.fastsimus)):
        raise AssertionError("FastSIMUS public contrast RF contains non-finite values")
    if not np.all(np.isfinite(stacks.pymust)):
        raise AssertionError("PyMUST public contrast RF contains non-finite values")

    fastsimus_peak = float(np.max(np.abs(stacks.fastsimus)))
    pymust_peak = float(np.max(np.abs(stacks.pymust)))
    reference_peak = max(fastsimus_peak, pymust_peak)
    if reference_peak <= 0.0:
        raise AssertionError("public contrast RF reference peak must be nonzero")

    residual = float(np.max(np.abs(stacks.fastsimus - stacks.pymust)) / reference_peak)
    residual_db = 20.0 * np.log10(residual) if residual > 0.0 else -np.inf
    if residual > atol_peak:
        raise AssertionError(
            f"public contrast RF residual {residual:.3e} ({residual_db:.1f} dB) "
            f"exceeds peak-normalized tolerance {atol_peak:.3e}; "
            f"FastSIMUS peak={fastsimus_peak:.3e}, PyMUST peak={pymust_peak:.3e}, "
            f"shape={stacks.fastsimus.shape}, firings={n_firings}",
        )


def _beamformed_data_from_reconstruction(
    reconstructed: ReconstructedIq,
    *,
    selected_angles_rad: np.ndarray,
    phantom_cache_key: tuple[str, ...],
) -> ContrastBeamformedData:
    """Package reconstructed full-angle contrast IQ for cache and figures."""
    return ContrastBeamformedData(
        fastsimus_iq=np.array(reconstructed.fastsimus, dtype=np.complex128, copy=True),
        pymust_iq=np.array(reconstructed.pymust, dtype=np.complex128, copy=True),
        x_axis_m=np.array(reconstructed.x_grid_m[0], dtype=np.float64, copy=True),
        z_axis_m=np.array(reconstructed.z_grid_m[:, 0], dtype=np.float64, copy=True),
        selected_angles_rad=np.array(selected_angles_rad, dtype=np.float64, copy=True),
        phantom_cache_key=phantom_cache_key,
    )


def _save_picmus_contrast_beamformed(path: Path, data: ContrastBeamformedData) -> None:
    """Save beamformed contrast data to a reusable HDF5 cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["cache_format_version"] = PICMUS_CONTRAST_BEAMFORMED_CACHE_VERSION
        handle.attrs["grid_wavelengths"] = PICMUS_CONTRAST_GRID_WAVELENGTHS
        handle.attrs["residual_dynamic_range_db"] = PICMUS_CONTRAST_RESIDUAL_DYNAMIC_RANGE_DB
        handle.attrs["reference_title"] = PICMUS_CONTRAST_REFERENCE_TITLE
        handle.attrs["proposed_title"] = PICMUS_CONTRAST_PROPOSED_TITLE
        handle.attrs["residual_title"] = PICMUS_CONTRAST_RESIDUAL_TITLE
        handle.create_dataset("x_axis_m", data=data.x_axis_m)
        handle.create_dataset("z_axis_m", data=data.z_axis_m)
        handle.create_dataset("fastsimus_iq_real", data=np.real(data.fastsimus_iq))
        handle.create_dataset("fastsimus_iq_imag", data=np.imag(data.fastsimus_iq))
        handle.create_dataset("pymust_iq_real", data=np.real(data.pymust_iq))
        handle.create_dataset("pymust_iq_imag", data=np.imag(data.pymust_iq))
        handle.create_dataset("selected_angles_rad", data=data.selected_angles_rad)
        handle.create_dataset(
            "phantom_cache_key",
            data=np.array([entry.encode("utf-8") for entry in data.phantom_cache_key]),
        )
    print(f"Saved PICMUS contrast beamformed cache to {path}", flush=True)


def _load_picmus_contrast_beamformed(path: Path) -> ContrastBeamformedData:
    """Load beamformed contrast data saved by `_save_picmus_contrast_beamformed`."""
    with h5py.File(path, "r") as handle:
        cache_version = handle.attrs.get("cache_format_version")
        if cache_version != PICMUS_CONTRAST_BEAMFORMED_CACHE_VERSION:
            raise ValueError(
                f"Unsupported PICMUS contrast beamformed cache format {cache_version!r}; "
                f"expected {PICMUS_CONTRAST_BEAMFORMED_CACHE_VERSION!r}",
            )
        fastsimus_iq = np.asarray(handle["fastsimus_iq_real"]) + 1j * np.asarray(handle["fastsimus_iq_imag"])
        pymust_iq = np.asarray(handle["pymust_iq_real"]) + 1j * np.asarray(handle["pymust_iq_imag"])
        raw_cache_key = np.asarray(handle["phantom_cache_key"])
        phantom_cache_key = tuple(
            entry.decode("utf-8") if isinstance(entry, bytes) else str(entry) for entry in raw_cache_key
        )
        return ContrastBeamformedData(
            fastsimus_iq=np.asarray(fastsimus_iq, dtype=np.complex128),
            pymust_iq=np.asarray(pymust_iq, dtype=np.complex128),
            x_axis_m=np.asarray(handle["x_axis_m"], dtype=np.float64),
            z_axis_m=np.asarray(handle["z_axis_m"], dtype=np.float64),
            selected_angles_rad=np.asarray(handle["selected_angles_rad"], dtype=np.float64),
            phantom_cache_key=phantom_cache_key,
        )


def _picmus_contrast_extent_mm(data: ContrastBeamformedData) -> tuple[float, float, float, float]:
    """Return an imshow extent for beamformed contrast images."""
    return (
        float(data.x_axis_m[0] * 1e3),
        float(data.x_axis_m[-1] * 1e3),
        float(data.z_axis_m[-1] * 1e3),
        float(data.z_axis_m[0] * 1e3),
    )


def _picmus_contrast_residual_display_db(
    fastsimus_iq: np.ndarray,
    pymust_iq: np.ndarray,
    *,
    reference_peak: float,
) -> np.ndarray:
    """Return contrast residual display values clipped to the submission residual range."""
    return _iq_to_display_db(
        fastsimus_iq - pymust_iq,
        reference_peak=reference_peak,
        dynamic_range_db=PICMUS_CONTRAST_RESIDUAL_DYNAMIC_RANGE_DB,
    )


def _picmus_contrast_comparison_panel_specs(
    data: ContrastBeamformedData,
) -> tuple[PicmusContrastPanelSpec, PicmusContrastPanelSpec, PicmusContrastPanelSpec]:
    """Return comparison panel images, labels, and color scales for paper figures."""
    reference_peak = max(float(np.max(np.abs(data.fastsimus_iq))), float(np.max(np.abs(data.pymust_iq))))
    return (
        PicmusContrastPanelSpec(
            title=PICMUS_CONTRAST_REFERENCE_TITLE,
            image_db=_iq_to_display_db(data.pymust_iq, reference_peak=reference_peak),
            cmap="gray",
            vmin_db=-DYNAMIC_RANGE_DB,
            vmax_db=0.0,
            colorbar_label="Amplitude [dB]",
        ),
        PicmusContrastPanelSpec(
            title=PICMUS_CONTRAST_PROPOSED_TITLE,
            image_db=_iq_to_display_db(data.fastsimus_iq, reference_peak=reference_peak),
            cmap="gray",
            vmin_db=-DYNAMIC_RANGE_DB,
            vmax_db=0.0,
            colorbar_label="Amplitude [dB]",
        ),
        PicmusContrastPanelSpec(
            title=PICMUS_CONTRAST_RESIDUAL_TITLE,
            image_db=_picmus_contrast_residual_display_db(
                data.fastsimus_iq,
                data.pymust_iq,
                reference_peak=reference_peak,
            ),
            cmap="magma",
            vmin_db=-PICMUS_CONTRAST_RESIDUAL_DYNAMIC_RANGE_DB,
            vmax_db=0.0,
            colorbar_label="Residual [dB, -100 dB floor]",
        ),
    )


def _overlay_contrast_cysts(ax: Any, phantom_data: ContrastPhantomData) -> None:
    """Overlay public contrast cyst positions on an image axis."""
    if plt is None:
        pytest.skip("PICMUS contrast figure generation requires matplotlib")
    for x_center_m, z_center_m, diameter_m in zip(
        phantom_data.cyst_center_x_m,
        phantom_data.cyst_center_z_m,
        phantom_data.cyst_diameter_m,
        strict=True,
    ):
        ax.add_patch(
            cast("Any", plt).Circle(
                (float(x_center_m * 1e3), float(z_center_m * 1e3)),
                float(0.5 * diameter_m * 1e3),
                color="yellow",
                fill=False,
                linewidth=0.8,
            ),
        )


def _render_picmus_contrast_comparison(
    data: ContrastBeamformedData,
    phantom_data: ContrastPhantomData,
    output_path: Path,
    *,
    overlay_cysts: bool,
) -> None:
    """Render one PyMUST/FastSIMUS/residual comparison figure."""
    pyplot = cast("Any", plt)
    extent_mm = _picmus_contrast_extent_mm(data)
    panel_specs = _picmus_contrast_comparison_panel_specs(data)
    fig, axes = pyplot.subplots(1, 3, figsize=PICMUS_CONTRAST_COMPARISON_FIGSIZE_IN, constrained_layout=True)
    for ax, spec in zip(axes, panel_specs, strict=True):
        im = ax.imshow(
            spec.image_db,
            extent=extent_mm,
            cmap=spec.cmap,
            vmin=spec.vmin_db,
            vmax=spec.vmax_db,
            aspect=PICMUS_CONTRAST_IMAGE_ASPECT,
            interpolation=PICMUS_PLOT_INTERPOLATION,
        )
        if overlay_cysts:
            _overlay_contrast_cysts(ax, phantom_data)
        ax.set_title(spec.title)
        ax.set_xlabel("Lateral [mm]")
        ax.set_ylabel("Depth [mm]")
        fig.colorbar(im, ax=ax, label=spec.colorbar_label)
    fig.savefig(output_path, dpi=PICMUS_PLOT_DPI, bbox_inches="tight")
    pyplot.close(fig)


def _render_picmus_contrast_two_panel(data: ContrastBeamformedData, output_path: Path) -> None:
    """Render PyMUST and proposed images with shared axes labels and colorbar."""
    pyplot = cast("Any", plt)
    extent_mm = _picmus_contrast_extent_mm(data)
    panel_specs = _picmus_contrast_comparison_panel_specs(data)[:2]
    fig, axes = pyplot.subplots(
        1,
        2,
        figsize=PICMUS_CONTRAST_TWO_PANEL_FIGSIZE_IN,
        constrained_layout=True,
        gridspec_kw={"wspace": PICMUS_CONTRAST_TWO_PANEL_WSPACE},
        sharex=True,
        sharey=True,
    )
    last_image = None
    for ax, spec in zip(axes, panel_specs, strict=True):
        last_image = ax.imshow(
            spec.image_db,
            extent=extent_mm,
            cmap=spec.cmap,
            vmin=spec.vmin_db,
            vmax=spec.vmax_db,
            aspect=PICMUS_CONTRAST_IMAGE_ASPECT,
            interpolation=PICMUS_PLOT_INTERPOLATION,
        )
        ax.set_title(spec.title)
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0].set_ylabel("Depth [mm]", labelpad=PICMUS_CONTRAST_TWO_PANEL_YLABEL_PAD)
    fig.supxlabel("Lateral [mm]")
    fig.colorbar(last_image, ax=axes, label="Amplitude [dB]", shrink=0.84, pad=0.015)
    fig.savefig(output_path, dpi=PICMUS_PLOT_DPI, bbox_inches="tight")
    pyplot.close(fig)


def _render_picmus_contrast_figures(
    data: ContrastBeamformedData,
    phantom_data: ContrastPhantomData,
    figure_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Render FastSIMUS-only and PyMUST-vs-FastSIMUS contrast figures."""
    if plt is None:
        pytest.skip("PICMUS contrast figure generation requires matplotlib")

    pyplot = cast("Any", plt)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fastsimus_path = figure_dir / "picmus_contrast_fastsimus_mlx_75_angle.png"
    comparison_path = figure_dir / "picmus_contrast_pymust_vs_fastsimus_mlx_75_angle.png"
    two_panel_path = figure_dir / "picmus_contrast_pymust_vs_fastsimus_mlx_75_angle_two_panel.png"
    debug_comparison_path = (
        figure_dir / f"picmus_contrast_pymust_vs_fastsimus_mlx_75_angle{PICMUS_CONTRAST_DEBUG_CYST_SUFFIX}.png"
    )
    extent_mm = _picmus_contrast_extent_mm(data)

    style_context = cast("Any", sns).axes_style("ticks") if sns is not None else contextlib.nullcontext()
    paper_context = cast("Any", sns).plotting_context("paper") if sns is not None else contextlib.nullcontext()
    with style_context, paper_context:
        fastsimus_peak = float(np.max(np.abs(data.fastsimus_iq)))
        # The standalone Proposed image is self-normalized for visual inspection;
        # the comparison figure below uses a shared PyMUST/FastSIMUS reference peak.
        fastsimus_db = _iq_to_display_db(data.fastsimus_iq, reference_peak=fastsimus_peak)
        fig, ax = pyplot.subplots(
            1,
            1,
            figsize=PICMUS_CONTRAST_FASTSIMUS_FIGSIZE_IN,
            constrained_layout=True,
        )
        im = ax.imshow(
            fastsimus_db,
            extent=extent_mm,
            cmap="gray",
            vmin=-DYNAMIC_RANGE_DB,
            vmax=0.0,
            aspect=PICMUS_CONTRAST_IMAGE_ASPECT,
            interpolation=PICMUS_PLOT_INTERPOLATION,
        )
        ax.set_title(PICMUS_CONTRAST_PROPOSED_TITLE)
        ax.set_xlabel("Lateral [mm]")
        ax.set_ylabel("Depth [mm]")
        fig.colorbar(im, ax=ax, label="Amplitude [dB]")
        fig.savefig(fastsimus_path, dpi=PICMUS_PLOT_DPI, bbox_inches="tight")
        pyplot.close(fig)

        _render_picmus_contrast_comparison(data, phantom_data, comparison_path, overlay_cysts=False)
        _render_picmus_contrast_two_panel(data, two_panel_path)
        _render_picmus_contrast_comparison(data, phantom_data, debug_comparison_path, overlay_cysts=True)
    return fastsimus_path, comparison_path, two_panel_path, debug_comparison_path


def _expects_large_reconstruction_warning(
    case: PhantomCase,
    *,
    grid_wavelengths: float | None,
    n_firings: int,
) -> bool:
    """Return whether a requested diagnostic reconstruction is expected to warn."""
    return _shared_expects_large_reconstruction_warning(
        grid_wavelengths=grid_wavelengths,
        n_firings=n_firings,
        default_x_axis_m=_case_default_x_axis(case),
        default_z_axis_m=_case_default_z_axis(case),
        x_bounds_m=case.x_bounds_m,
        z_bounds_m=case.z_bounds_m,
        wavelength_m=case.wavelength_m,
        warn_pixels=PICMUS_GRID_WARN_PIXELS,
        warn_pixel_firings=PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS,
    )


def _make_picmus_resolution_phantom() -> Phantom:
    """Create a compact in-code version of the PICMUS resolution phantom."""
    return Phantom(
        PICMUS_PHANTOM_X_M.copy(),
        PICMUS_PHANTOM_Z_M.copy(),
        np.ones(PICMUS_PHANTOM_X_M.shape, dtype=np.float64),
    )


RESOLUTION_CASE = PhantomCase(
    id="resolution",
    cache_key=(
        "resolution",
        "picmus-resolution-distorsion",
        "phantom-20-point-v1",
        "l11-picmus-matched-v1",
        f"fs={SAMPLING_FREQUENCY_HZ:.6g}",
        f"tx={TX_N_WAVELENGTHS:g}",
        f"db={SIMUS_DB_THRESH:g}",
    ),
    phantom_factory=_make_picmus_resolution_phantom,
    matched_params_factory=make_l11_picmus_matched_params,
    default_angles_rad=_angle_cache_key(PICMUS_BROADSIDE_ANGLES_RAD),
    diagnostic_angles_rad=_angle_cache_key(PICMUS_FULL_ANGLES_RAD),
    x_bounds_m=(IMAGE_X_MIN_M, IMAGE_X_MAX_M),
    z_bounds_m=(IMAGE_Z_MIN_M, IMAGE_Z_MAX_M),
    default_x_axis_m=tuple(float(x) for x in IMAGE_X_M),
    default_z_axis_m=tuple(float(z) for z in IMAGE_Z_M),
    wavelength_m=PICMUS_WAVELENGTH_M,
    validation_mode="iq_residual",
    iq_atol_peak=IQ_ATOL_PEAK,
)
ACTIVE_PHANTOM_CASES = (RESOLUTION_CASE,)
# Future contrast/speckle phantoms should use metric-specific case collections,
# not the IQ residual comparison used by point-scatterer resolution data.
IQ_RESIDUAL_CASES = tuple(case for case in ACTIVE_PHANTOM_CASES if case.validation_mode == "iq_residual")
_CASES_BY_CACHE_KEY = {case.cache_key: case for case in ACTIVE_PHANTOM_CASES}


def _case_for_cache_key(case_cache_key: tuple[str, ...]) -> PhantomCase:
    """Return a registered phantom case by its explicit cache identity."""
    return _CASES_BY_CACHE_KEY[case_cache_key]


def _case_default_x_axis(case: PhantomCase) -> np.ndarray:
    """Return a writable copy of a case's default lateral image axis."""
    return np.array(case.default_x_axis_m, dtype=np.float64)


def _case_default_z_axis(case: PhantomCase) -> np.ndarray:
    """Return a writable copy of a case's default axial image axis."""
    return np.array(case.default_z_axis_m, dtype=np.float64)


def _image_axes_for_case_grid(case: PhantomCase, grid_wavelengths: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Return image axes for a case's default or wavelength-spaced grid."""
    return _shared_image_axes_for_grid_wavelengths(
        grid_wavelengths,
        default_x_axis_m=_case_default_x_axis(case),
        default_z_axis_m=_case_default_z_axis(case),
        x_bounds_m=case.x_bounds_m,
        z_bounds_m=case.z_bounds_m,
        wavelength_m=case.wavelength_m,
        warn_pixels=PICMUS_GRID_WARN_PIXELS,
        stacklevel=3,
    )


def _rf_stacks_cache_key(
    case: PhantomCase,
    angles_key: tuple[float, ...],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Return the RF cache key, including the case's scientific identity."""
    return case.cache_key, angles_key


def _reconstructed_iq_cache_key(
    case: PhantomCase,
    angles_key: tuple[float, ...],
    grid_wavelengths: float | None,
) -> tuple[tuple[str, ...], tuple[float, ...], float | None]:
    """Return the IQ cache key, including case, angles, and grid identity."""
    return case.cache_key, angles_key, grid_wavelengths


@lru_cache(maxsize=1)
def _simulate_rf_stacks(case: PhantomCase = RESOLUTION_CASE) -> RfStacks:
    """Simulate matched PyMUST and FastSIMUS RF stacks for a phantom case."""
    return _simulate_rf_stacks_for_case_key(*_rf_stacks_cache_key(case, case.default_angles_rad))


@lru_cache(maxsize=2)
def _simulate_rf_stacks_for_case_key(
    case_cache_key: tuple[str, ...],
    angles_key: tuple[float, ...],
) -> RfStacks:
    """Simulate matched RF stacks for the requested case and angle sequence."""
    case = _case_for_cache_key(case_cache_key)
    angles_rad = np.array(angles_key, dtype=np.float64)
    phantom = case.phantom_factory()
    params = case.matched_params_factory()
    return _simulate_rf_stacks_for_case(phantom, params, angles_rad)


@lru_cache(maxsize=1)
def _reconstruct_picmus_iq(case: PhantomCase = RESOLUTION_CASE) -> ReconstructedIq:
    """Reconstruct the cached PICMUS phantom RF stacks once per test session."""
    return _reconstruct_picmus_iq_for_grid(case, case.default_angles_rad, None)


@lru_cache(maxsize=4)
def _reconstruct_picmus_iq_for_grid(
    case: PhantomCase,
    angles_key: tuple[float, ...],
    grid_wavelengths: float | None,
) -> ReconstructedIq:
    """Reconstruct the PICMUS phantom for requested angles and image-grid spacing."""
    return _reconstruct_picmus_iq_for_cache_key(*_reconstructed_iq_cache_key(case, angles_key, grid_wavelengths))


@lru_cache(maxsize=4)
def _reconstruct_picmus_iq_for_cache_key(
    case_cache_key: tuple[str, ...],
    angles_key: tuple[float, ...],
    grid_wavelengths: float | None,
) -> ReconstructedIq:
    """Reconstruct the PICMUS phantom for requested case, angles, and image-grid spacing."""
    case = _case_for_cache_key(case_cache_key)
    x_axis_m, z_axis_m = _image_axes_for_case_grid(case, grid_wavelengths)
    return _reconstruct_iq(
        _simulate_rf_stacks_for_case_key(case_cache_key, angles_key),
        case.matched_params_factory(),
        x_axis_m=x_axis_m,
        z_axis_m=z_axis_m,
        warn_pixel_firings=PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS,
    )


def _render_picmus_phantom_plot(reconstructed: ReconstructedIq, output_path: Path) -> None:
    """Write the optional three-panel PICMUS phantom diagnostic figure."""
    if plt is None:
        pytest.skip("--plot-picmus-phantom requires matplotlib")

    pyplot = cast("Any", plt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_peak = max(
        float(np.max(np.abs(reconstructed.fastsimus))),
        float(np.max(np.abs(reconstructed.pymust))),
    )
    fastsimus_db = _iq_to_display_db(reconstructed.fastsimus, reference_peak=reference_peak)
    pymust_db = _iq_to_display_db(reconstructed.pymust, reference_peak=reference_peak)
    residual_db = _iq_residual_display_db(
        reconstructed.fastsimus,
        reconstructed.pymust,
        reference_peak=reference_peak,
    )
    extent_mm = (
        float(reconstructed.x_grid_m.min() * 1e3),
        float(reconstructed.x_grid_m.max() * 1e3),
        float(reconstructed.z_grid_m.max() * 1e3),
        float(reconstructed.z_grid_m.min() * 1e3),
    )

    fig, axes = pyplot.subplots(1, 3, figsize=PICMUS_PLOT_FIGSIZE_IN, constrained_layout=True)
    panel_specs = (
        (PROPOSED_LABEL, fastsimus_db, "gray", -DYNAMIC_RANGE_DB, 0.0, "Amplitude [dB]"),
        (REFERENCE_LABEL, pymust_db, "gray", -DYNAMIC_RANGE_DB, 0.0, "Amplitude [dB]"),
        (RESIDUAL_LABEL, residual_db, "magma", -DYNAMIC_RANGE_DB, 0.0, "Residual amplitude [dB]"),
    )
    for ax, (title, image, cmap, vmin, vmax, colorbar_label) in zip(axes, panel_specs, strict=True):
        im = ax.imshow(
            image,
            extent=extent_mm,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation=PICMUS_PLOT_INTERPOLATION,
        )
        ax.set_title(title)
        ax.set_xlabel("Lateral [mm]")
        ax.set_ylabel("Depth [mm]")
        fig.colorbar(im, ax=ax, label=colorbar_label)

    fig.savefig(output_path, dpi=PICMUS_PLOT_DPI)
    pyplot.close(fig)


def test_picmus_phantom_constants_match_hdf5_values() -> None:
    """Phantom constants are copied from the PICMUS resolution_distorsion HDF5 phantom."""
    phantom = RESOLUTION_CASE.phantom_factory()

    np.testing.assert_allclose(phantom.x, PICMUS_PHANTOM_X_M)
    np.testing.assert_allclose(phantom.z, PICMUS_PHANTOM_Z_M)
    np.testing.assert_array_equal(phantom.rc, np.ones(20, dtype=np.float64))
    np.testing.assert_allclose(np.array(RESOLUTION_CASE.default_angles_rad), PICMUS_BROADSIDE_ANGLES_RAD)
    assert PICMUS_FULL_ANGLES_RAD.shape == (75,)
    assert PICMUS_FULL_ANGLES_RAD[37] == pytest.approx(0.0)
    np.testing.assert_allclose(PICMUS_FULL_ANGLES_RAD[36:39], PICMUS_BROADSIDE_ANGLES_RAD)
    assert pytest.approx(0.30e-3, rel=1e-5) == PITCH_M
    assert pytest.approx(1540.0) == SPEED_OF_SOUND_M_S
    assert pytest.approx(max(20.832e6, 4.001 * FREQ_CENTER_HZ)) == SAMPLING_FREQUENCY_HZ
    assert pytest.approx(0.5) == ATTENUATION_DB_CM_MHZ
    assert IMAGE_X_M[0] <= -PICMUS_APERTURE_X_M
    assert IMAGE_X_M[-1] >= PICMUS_APERTURE_X_M
    assert pytest.approx(5.0e-3) == IMAGE_Z_MIN_M
    assert pytest.approx(50.0e-3) == IMAGE_Z_MAX_M
    assert IMAGE_Z_M[0] <= IMAGE_Z_MIN_M
    assert IMAGE_Z_M[-1] >= IMAGE_Z_MAX_M


def test_picmus_phantom_angle_modes_select_expected_sequences() -> None:
    """Plot angle modes select either the fast subset or full PICMUS sequence."""
    np.testing.assert_allclose(_angles_for_plot_mode(RESOLUTION_CASE, "broadside"), PICMUS_BROADSIDE_ANGLES_RAD)
    np.testing.assert_allclose(_angles_for_plot_mode(RESOLUTION_CASE, "full"), PICMUS_FULL_ANGLES_RAD)

    with pytest.raises(ValueError, match="Unknown PICMUS phantom angle mode"):
        _angles_for_plot_mode(RESOLUTION_CASE, "unsupported")


def test_fastsimus_rf_backend_prefers_mlx_for_metal_when_available() -> None:
    """FastSIMUS RF helpers use MLX so contrast artifacts match the Metal label."""
    mlx_core = pytest.importorskip("mlx.core")

    assert _fastsimus_array_namespace() is mlx_core


def test_picmus_contrast_loader_reads_public_schema(tmp_path: Path) -> None:
    """The contrast loader reads exact scatterers and cyst metadata from HDF5."""
    phantom_path = tmp_path / "contrast_speckle_simu_phantom.hdf5"
    positions_m = _valid_picmus_contrast_positions()
    amplitudes = _valid_picmus_contrast_amplitudes()
    _write_picmus_contrast_hdf5(phantom_path, positions_m=positions_m, amplitudes=amplitudes)

    data = load_picmus_contrast_phantom(phantom_path)

    assert data.x_m.shape == (VALID_TEST_CONTRAST_SCATTERERS,)
    np.testing.assert_allclose(data.x_m, positions_m[0])
    np.testing.assert_allclose(data.y_m, positions_m[1])
    np.testing.assert_allclose(data.z_m, positions_m[2])
    np.testing.assert_allclose(data.rc, amplitudes)
    assert data.cyst_center_x_m.shape == (9,)
    assert data.cyst_center_z_m.shape == (9,)
    assert data.cyst_diameter_m.shape == (9,)
    assert data.roi_metadata["phantom_ROIWidth"] == pytest.approx(38.0e-3)


def test_picmus_contrast_loader_rejects_bad_positions_shape(tmp_path: Path) -> None:
    """The contrast loader validates the published positions shape."""
    phantom_path = tmp_path / "bad_positions.hdf5"
    positions_m = _valid_picmus_contrast_positions()[:2]
    amplitudes = _valid_picmus_contrast_amplitudes()
    _write_picmus_contrast_hdf5(phantom_path, positions_m=positions_m, amplitudes=amplitudes)

    with pytest.raises(ValueError, match=r"scatterers_positions.*shape"):
        load_picmus_contrast_phantom(phantom_path)


def test_picmus_contrast_loader_rejects_nonzero_y_before_2d_projection(tmp_path: Path) -> None:
    """The contrast loader requires 3D public data to be planar before dropping y."""
    phantom_path = tmp_path / "nonzero_y.hdf5"
    positions_m = _valid_picmus_contrast_positions()
    positions_m[1, 0] = 1.0e-6
    _write_picmus_contrast_hdf5(
        phantom_path,
        positions_m=positions_m,
        amplitudes=_valid_picmus_contrast_amplitudes(),
    )

    with pytest.raises(ValueError, match="y coordinates"):
        load_picmus_contrast_phantom(phantom_path)


def test_picmus_contrast_public_hdf5_schema_loads(
    picmus_contrast_phantom_path: Path | None,
) -> None:
    """A caller-supplied public contrast HDF5 file passes loader/schema checks."""
    if picmus_contrast_phantom_path is None:
        pytest.skip("--picmus-contrast-phantom not provided")
    if not picmus_contrast_phantom_path.is_file():
        pytest.skip(f"--picmus-contrast-phantom does not exist: {picmus_contrast_phantom_path}")

    data = load_picmus_contrast_phantom(picmus_contrast_phantom_path)

    assert data.x_m.shape == data.z_m.shape == data.rc.shape
    assert data.x_m.size > 100_000
    assert np.all(np.isfinite(data.x_m))
    assert np.all(np.isfinite(data.z_m))
    assert np.all(np.isfinite(data.rc))
    np.testing.assert_allclose(data.y_m, 0.0, atol=1e-12)


def test_picmus_contrast_angle_modes_select_center_or_full_sequence() -> None:
    """Contrast angle modes select the center plane wave or full public sequence."""
    center_angles = _angles_for_picmus_contrast_mode("center")
    full_angles = _angles_for_picmus_contrast_mode("full")

    assert center_angles.shape == (1,)
    assert center_angles[0] == pytest.approx(0.0)
    np.testing.assert_allclose(full_angles, PICMUS_FULL_ANGLES_RAD)

    with pytest.raises(ValueError, match="Unknown PICMUS contrast angle mode"):
        _angles_for_picmus_contrast_mode("unsupported")


def test_picmus_contrast_case_uses_public_identity_and_rf_validation(tmp_path: Path) -> None:
    """The contrast case is a gated RF case keyed by public data identity."""
    phantom_path = tmp_path / "contrast_speckle_simu_phantom.hdf5"
    _write_picmus_contrast_hdf5(
        phantom_path,
        positions_m=_valid_picmus_contrast_positions(),
        amplitudes=_valid_picmus_contrast_amplitudes(),
    )
    data = load_picmus_contrast_phantom(phantom_path)

    center_case = _make_picmus_contrast_case(phantom_path, data, angle_mode="center")
    full_case = _make_picmus_contrast_case(phantom_path, data, angle_mode="full")

    assert center_case.id == "contrast_speckle"
    assert center_case.validation_mode == "rf_public_contrast"
    assert center_case.iq_atol_peak is None
    assert len(center_case.default_angles_rad) == 1
    assert len(full_case.default_angles_rad) == 75
    assert center_case.cache_key != full_case.cache_key
    assert any(str(phantom_path.resolve()) in entry for entry in center_case.cache_key)
    assert any(entry == f"n_scatterers={VALID_TEST_CONTRAST_SCATTERERS}" for entry in center_case.cache_key)
    assert any(entry == "angle_mode=center" for entry in center_case.cache_key)
    assert any(entry == "angle_mode=full" for entry in full_case.cache_key)
    assert center_case.validation_mode != "iq_residual"
    assert center_case not in IQ_RESIDUAL_CASES
    assert center_case not in ACTIVE_PHANTOM_CASES


def test_picmus_contrast_case_uses_public_scan_scope_and_lambda_over_four_grid(tmp_path: Path) -> None:
    """Contrast artifact beamforming uses public scan bounds at lambda/4 spacing."""
    phantom_path = tmp_path / "contrast_speckle_simu_phantom.hdf5"
    scan_path = tmp_path / "contrast_speckle_simu_scan.hdf5"
    x_scan_m = np.array([-19.05e-3, 0.0, 18.996594982078844e-3])
    z_scan_m = np.array([5.0e-3, 20.0e-3, 49.94623655913978e-3])
    _write_picmus_contrast_hdf5(
        phantom_path,
        positions_m=_valid_picmus_contrast_positions(),
        amplitudes=_valid_picmus_contrast_amplitudes(),
    )
    _write_picmus_contrast_scan_hdf5(scan_path, x_axis_m=x_scan_m, z_axis_m=z_scan_m)
    data = load_picmus_contrast_phantom(phantom_path)

    case = _make_picmus_contrast_case(phantom_path, data, angle_mode="full")
    x_axis_m = np.array(case.default_x_axis_m)
    z_axis_m = np.array(case.default_z_axis_m)

    assert any(str(scan_path.resolve()) in entry for entry in case.cache_key)
    assert any(entry.startswith("scan_size=") for entry in case.cache_key)
    assert x_axis_m[0] == pytest.approx(float(x_scan_m[0]))
    assert x_axis_m[-1] == pytest.approx(float(x_scan_m[-1]))
    assert z_axis_m[0] == pytest.approx(float(z_scan_m[0]))
    assert z_axis_m[-1] == pytest.approx(float(z_scan_m[-1]))
    assert float(np.max(np.diff(x_axis_m))) <= PICMUS_WAVELENGTH_M / 4.0 + 1e-15
    assert float(np.max(np.diff(z_axis_m))) <= PICMUS_WAVELENGTH_M / 4.0 + 1e-15
    assert x_axis_m.size > IMAGE_X_M.size
    assert z_axis_m.size > IMAGE_Z_M.size


def test_picmus_contrast_phantom_factory_copies_planar_loaded_data(tmp_path: Path) -> None:
    """The contrast Phantom factory preserves loaded arrays and projects x/z explicitly."""
    phantom_path = tmp_path / "contrast_speckle_simu_phantom.hdf5"
    _write_picmus_contrast_hdf5(
        phantom_path,
        positions_m=_valid_picmus_contrast_positions(),
        amplitudes=_valid_picmus_contrast_amplitudes(),
    )
    data = load_picmus_contrast_phantom(phantom_path)
    case = _make_picmus_contrast_case(phantom_path, data, angle_mode="center")

    phantom = case.phantom_factory()

    np.testing.assert_allclose(phantom.x, data.x_m)
    np.testing.assert_allclose(phantom.z, data.z_m)
    np.testing.assert_allclose(phantom.rc, data.rc)
    assert not np.shares_memory(phantom.x, data.x_m)
    assert not np.shares_memory(phantom.z, data.z_m)
    assert not np.shares_memory(phantom.rc, data.rc)


def test_picmus_contrast_rf_assertion_reports_peak_normalized_residual() -> None:
    """Public contrast RF comparison failures include residual diagnostics."""
    stacks = RfStacks(
        fastsimus=np.ones((2, 2, 1), dtype=np.float64),
        pymust=np.zeros((2, 2, 1), dtype=np.float64),
        angles_rad=np.array([0.0], dtype=np.float64),
    )

    with pytest.raises(AssertionError, match="public contrast RF residual"):
        _assert_rf_public_contrast_stacks_close(stacks, n_elements=2, n_firings=1, atol_peak=0.1)


def test_picmus_contrast_beamformed_hdf5_round_trips(tmp_path: Path) -> None:
    """Beamformed contrast cache stores reusable IQ arrays and metadata."""
    beamformed = ContrastBeamformedData(
        fastsimus_iq=np.array([[1.0 + 0.5j, 0.5 + 0.25j]]),
        pymust_iq=np.array([[0.9 + 0.4j, 0.4 + 0.2j]]),
        x_axis_m=np.array([-1.0e-3, 1.0e-3]),
        z_axis_m=np.array([10.0e-3]),
        selected_angles_rad=np.array([0.0]),
        phantom_cache_key=("contrast_speckle", "n_scatterers=100001"),
    )
    cache_path = tmp_path / "beamformed.hdf5"

    _save_picmus_contrast_beamformed(cache_path, beamformed)
    loaded = _load_picmus_contrast_beamformed(cache_path)

    with h5py.File(cache_path, "r") as handle:
        assert handle.attrs["cache_format_version"] == PICMUS_CONTRAST_BEAMFORMED_CACHE_VERSION
        assert handle.attrs["grid_wavelengths"] == PICMUS_CONTRAST_GRID_WAVELENGTHS
    np.testing.assert_allclose(loaded.fastsimus_iq, beamformed.fastsimus_iq)
    np.testing.assert_allclose(loaded.pymust_iq, beamformed.pymust_iq)
    np.testing.assert_allclose(loaded.x_axis_m, beamformed.x_axis_m)
    np.testing.assert_allclose(loaded.z_axis_m, beamformed.z_axis_m)
    np.testing.assert_allclose(loaded.selected_angles_rad, beamformed.selected_angles_rad)
    assert loaded.phantom_cache_key == beamformed.phantom_cache_key


def test_picmus_contrast_beamformed_hdf5_rejects_unknown_format(tmp_path: Path) -> None:
    """Beamformed contrast caches fail fast when their schema version is unsupported."""
    cache_path = tmp_path / "beamformed.hdf5"
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["cache_format_version"] = "old-format"

    with pytest.raises(ValueError, match="Unsupported PICMUS contrast beamformed cache format"):
        _load_picmus_contrast_beamformed(cache_path)


def test_picmus_contrast_comparison_panel_specs_match_submission_labels_and_residual_floor() -> None:
    """The comparison figure uses submission labels and a -100 dB residual floor."""
    beamformed = ContrastBeamformedData(
        fastsimus_iq=np.array([[1.0 + 0.0j, 0.1 + 0.0j]]),
        pymust_iq=np.array([[0.5 + 0.0j, 0.1 + 0.0j]]),
        x_axis_m=np.array([-1.0e-3, 1.0e-3]),
        z_axis_m=np.array([10.0e-3]),
        selected_angles_rad=PICMUS_FULL_ANGLES_RAD,
        phantom_cache_key=("contrast_speckle",),
    )

    panel_specs = _picmus_contrast_comparison_panel_specs(beamformed)

    assert [spec.title for spec in panel_specs] == [
        "PyMUST SIMUS baseline",
        "Proposed (Metal)",
        "Residual",
    ]
    assert panel_specs[0].vmin_db == -DYNAMIC_RANGE_DB
    assert panel_specs[1].vmin_db == -DYNAMIC_RANGE_DB
    assert panel_specs[2].vmin_db == -PICMUS_CONTRAST_RESIDUAL_DYNAMIC_RANGE_DB
    assert float(np.min(panel_specs[2].image_db)) == -PICMUS_CONTRAST_RESIDUAL_DYNAMIC_RANGE_DB
    assert panel_specs[2].image_db[0, 0] == pytest.approx(20.0 * np.log10(0.5))


def test_picmus_contrast_figures_are_written_without_default_overlays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final contrast figures use equal aspect and keep lesion circles in debug output."""
    if not HAS_MATPLOTLIB:
        pytest.skip("matplotlib not available")

    phantom_path = tmp_path / "contrast_speckle_simu_phantom.hdf5"
    _write_picmus_contrast_hdf5(
        phantom_path,
        positions_m=_valid_picmus_contrast_positions(),
        amplitudes=_valid_picmus_contrast_amplitudes(),
    )
    phantom_data = load_picmus_contrast_phantom(phantom_path)
    beamformed = ContrastBeamformedData(
        fastsimus_iq=np.array([[1.0 + 0.0j, 0.5 + 0.0j], [0.25 + 0.0j, 0.1 + 0.0j]]),
        pymust_iq=np.array([[0.9 + 0.0j, 0.45 + 0.0j], [0.2 + 0.0j, 0.08 + 0.0j]]),
        x_axis_m=np.array([-2.0e-3, 2.0e-3]),
        z_axis_m=np.array([10.0e-3, 20.0e-3]),
        selected_angles_rad=PICMUS_FULL_ANGLES_RAD,
        phantom_cache_key=("contrast_speckle",),
    )
    overlay_calls = 0
    original_overlay = _overlay_contrast_cysts

    def count_overlay(ax: Any, phantom_data: ContrastPhantomData) -> None:
        nonlocal overlay_calls
        overlay_calls += 1
        original_overlay(ax, phantom_data)

    monkeypatch.setattr("tests.test_picmus_phantom._overlay_contrast_cysts", count_overlay)

    fastsimus_path, comparison_path, two_panel_path, debug_path = _render_picmus_contrast_figures(
        beamformed,
        phantom_data,
        tmp_path,
    )

    assert fastsimus_path.name == "picmus_contrast_fastsimus_mlx_75_angle.png"
    assert comparison_path.name == "picmus_contrast_pymust_vs_fastsimus_mlx_75_angle.png"
    assert two_panel_path.name == "picmus_contrast_pymust_vs_fastsimus_mlx_75_angle_two_panel.png"
    assert debug_path.name == "picmus_contrast_pymust_vs_fastsimus_mlx_75_angle_debug_cysts.png"
    assert fastsimus_path.is_file()
    assert comparison_path.is_file()
    assert two_panel_path.is_file()
    assert debug_path.is_file()
    assert fastsimus_path.stat().st_mtime_ns <= comparison_path.stat().st_mtime_ns
    assert comparison_path.stat().st_mtime_ns <= two_panel_path.stat().st_mtime_ns
    assert two_panel_path.stat().st_mtime_ns <= debug_path.stat().st_mtime_ns
    assert overlay_calls == 3
    assert PICMUS_CONTRAST_IMAGE_ASPECT == "equal"
    assert PICMUS_CONTRAST_TWO_PANEL_FIGSIZE_IN[1] == PICMUS_CONTRAST_FIGURE_HEIGHT_IN
    assert PICMUS_CONTRAST_TWO_PANEL_WSPACE <= 0.02
    assert PICMUS_CONTRAST_TWO_PANEL_YLABEL_PAD <= 2.0


@pytest.mark.slow
def test_picmus_contrast_public_rf_simulation_is_gated(
    run_picmus_contrast: bool,
    picmus_contrast_phantom_path: Path | None,
    picmus_contrast_angle_mode: str,
    picmus_contrast_figure_dir: Path | None,
    picmus_contrast_save_beamformed_path: Path | None,
    picmus_contrast_load_beamformed_path: Path | None,
) -> None:
    """Explicitly gated public contrast simulation returns finite RF stacks."""
    phantom_path = _skip_unless_picmus_contrast_requested(
        run_picmus_contrast=run_picmus_contrast,
        picmus_contrast_phantom_path=picmus_contrast_phantom_path,
    )
    data = load_picmus_contrast_phantom(phantom_path)

    if picmus_contrast_load_beamformed_path is not None:
        if not picmus_contrast_load_beamformed_path.is_file():
            pytest.fail(f"--picmus-contrast-load-beamformed does not exist: {picmus_contrast_load_beamformed_path}")
        if picmus_contrast_figure_dir is None:
            pytest.skip("--picmus-contrast-load-beamformed requires --picmus-contrast-figure-dir")
        beamformed = _load_picmus_contrast_beamformed(picmus_contrast_load_beamformed_path)
        _render_picmus_contrast_figures(beamformed, data, picmus_contrast_figure_dir)
        return

    artifact_requested = picmus_contrast_save_beamformed_path is not None or picmus_contrast_figure_dir is not None
    if artifact_requested and picmus_contrast_angle_mode != "full":
        pytest.skip("PICMUS contrast beamformed artifacts require --picmus-contrast-angle-mode full")

    case = _make_picmus_contrast_case(phantom_path, data, angle_mode=picmus_contrast_angle_mode)
    params = case.matched_params_factory()
    start_time_s = time.perf_counter()
    stacks = _simulate_rf_stacks_for_case(
        case.phantom_factory(),
        params,
        np.array(case.default_angles_rad, dtype=np.float64),
        progress_label=f"PICMUS contrast {picmus_contrast_angle_mode}",
    )
    elapsed_s = time.perf_counter() - start_time_s
    print(
        f"PICMUS contrast {picmus_contrast_angle_mode} RF simulation produced "
        f"{stacks.fastsimus.shape} in {elapsed_s:.1f}s",
    )

    _assert_rf_public_contrast_stacks_close(
        stacks,
        n_elements=params.fastsimus_transducer.n_elements,
        n_firings=len(case.default_angles_rad),
    )

    if artifact_requested:
        reconstruction_context = (
            pytest.warns(RuntimeWarning, match="DAS reconstruction will be slow")
            if _expects_large_reconstruction_warning(
                case,
                grid_wavelengths=None,
                n_firings=len(case.default_angles_rad),
            )
            else contextlib.nullcontext()
        )
        with reconstruction_context:
            reconstructed = _reconstruct_iq(
                stacks,
                params,
                x_axis_m=_case_default_x_axis(case),
                z_axis_m=_case_default_z_axis(case),
                warn_pixel_firings=PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS,
                progress_label="PICMUS contrast",
            )
        beamformed = _beamformed_data_from_reconstruction(
            reconstructed,
            selected_angles_rad=stacks.angles_rad,
            phantom_cache_key=case.cache_key,
        )
        if picmus_contrast_save_beamformed_path is not None:
            _save_picmus_contrast_beamformed(picmus_contrast_save_beamformed_path, beamformed)
        if picmus_contrast_figure_dir is not None:
            _render_picmus_contrast_figures(beamformed, data, picmus_contrast_figure_dir)


def test_resolution_case_uses_explicit_scientific_cache_key() -> None:
    """The first phantom case carries a stable cache key beyond its display ID."""
    assert RESOLUTION_CASE.id == "resolution"
    assert RESOLUTION_CASE.cache_key != (RESOLUTION_CASE.id,)
    assert RESOLUTION_CASE.cache_key[0] == RESOLUTION_CASE.id
    assert "picmus-resolution-distorsion" in RESOLUTION_CASE.cache_key
    assert _rf_stacks_cache_key(RESOLUTION_CASE, RESOLUTION_CASE.default_angles_rad)[0] == RESOLUTION_CASE.cache_key


def test_iq_residual_cases_select_supported_validation_modes() -> None:
    """IQ residual tests only collect cases configured for complex-IQ comparison."""
    assert IQ_RESIDUAL_CASES == (RESOLUTION_CASE,)
    assert all(case.validation_mode == "iq_residual" for case in IQ_RESIDUAL_CASES)


def test_resolution_case_is_the_only_active_phantom_case() -> None:
    """Contrast remains a future, explicitly gated extension point."""
    assert ACTIVE_PHANTOM_CASES == (RESOLUTION_CASE,)


def test_diagnostic_warning_uses_case_grid_identity() -> None:
    """Diagnostic warning prediction uses the case bounds, wavelength, and grid."""
    assert _expects_large_reconstruction_warning(
        RESOLUTION_CASE,
        grid_wavelengths=0.1,
        n_firings=len(RESOLUTION_CASE.diagnostic_angles_rad or ()),
    )


@pytest.mark.parametrize("case", IQ_RESIDUAL_CASES, ids=lambda case: case.id)
def test_simulated_rf_stacks_are_finite_nonzero_and_matched(case) -> None:
    """PyMUST and FastSIMUS produce finite, nonzero RF stacks with matched layout."""
    stacks = _simulate_rf_stacks(case)
    params = case.matched_params_factory()

    _assert_rf_stack_layout(
        stacks.fastsimus,
        stacks.pymust,
        stacks.angles_rad,
        n_elements=params.fastsimus_transducer.n_elements,
        n_firings=len(case.default_angles_rad),
    )
    assert np.all(np.isfinite(stacks.pymust))
    assert np.all(np.isfinite(stacks.fastsimus))
    assert np.max(np.abs(stacks.pymust)) > 0.0
    assert np.max(np.abs(stacks.fastsimus)) > 0.0


@pytest.mark.parametrize("case", IQ_RESIDUAL_CASES, ids=lambda case: case.id)
def test_reconstructed_complex_iq_matches_pymust(case) -> None:
    """Shared DAS reconstruction returns finite complex IQ matching PyMUST."""
    reconstructed = _reconstruct_picmus_iq(case)

    _assert_reconstructed_iq_layout(
        reconstructed.fastsimus,
        reconstructed.pymust,
        n_z=len(case.default_z_axis_m),
        n_x=len(case.default_x_axis_m),
    )
    assert np.all(np.isfinite(reconstructed.fastsimus))
    assert np.all(np.isfinite(reconstructed.pymust))
    assert np.max(np.abs(reconstructed.fastsimus)) > 0.0
    assert np.max(np.abs(reconstructed.pymust)) > 0.0
    assert case.iq_atol_peak is not None
    _assert_iq_close(reconstructed.fastsimus, reconstructed.pymust, atol_peak=case.iq_atol_peak)


def test_optional_picmus_phantom_plot_is_written(
    picmus_phantom_plot: Path | None,
    picmus_phantom_angles: str,
    picmus_phantom_grid_wavelengths: float | None,
) -> None:
    """The optional CLI path writes a diagnostic PNG and stays inactive by default."""
    if picmus_phantom_plot is None:
        pytest.skip("--plot-picmus-phantom not provided")
    if not HAS_MATPLOTLIB:
        pytest.skip("--plot-picmus-phantom requires matplotlib")

    angles_rad = _angles_for_plot_mode(RESOLUTION_CASE, picmus_phantom_angles)
    warning_context = (
        pytest.warns(RuntimeWarning, match="reconstruction will be slow")
        if _expects_large_reconstruction_warning(
            RESOLUTION_CASE,
            grid_wavelengths=picmus_phantom_grid_wavelengths,
            n_firings=angles_rad.size,
        )
        else contextlib.nullcontext()
    )
    with warning_context:
        _render_picmus_phantom_plot(
            _reconstruct_picmus_iq_for_grid(
                RESOLUTION_CASE,
                _angle_cache_key(angles_rad),
                picmus_phantom_grid_wavelengths,
            ),
            picmus_phantom_plot,
        )

    assert picmus_phantom_plot.is_file()
    assert picmus_phantom_plot.stat().st_size > 0
    pyplot = cast("Any", plt)
    assert pyplot.imread(picmus_phantom_plot).shape[1] >= PICMUS_PLOT_MIN_WIDTH_PX
