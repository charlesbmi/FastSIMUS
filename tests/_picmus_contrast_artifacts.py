"""Artifact helpers for the public PICMUS contrast phantom tests."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, cast

import h5py
import numpy as np
import pytest

from tests._picmus_contrast_loader import PICMUS_CONTRAST_GROUP, ContrastPhantomData
from tests._picmus_phantom_shared import (
    DYNAMIC_RANGE_DB,
    L11_PICMUS_FREQ_CENTER_HZ,
    L11_PICMUS_SPEED_OF_SOUND_M_S,
    PICMUS_GRID_WARN_PIXELS,
    ReconstructedIq,
    _image_axes_for_grid_wavelengths,
    _iq_to_display_db,
)

logger = logging.getLogger(__name__)

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
PICMUS_CONTRAST_PLOT_DPI = 360
PICMUS_CONTRAST_PLOT_INTERPOLATION = "nearest"
PICMUS_CONTRAST_WAVELENGTH_M = L11_PICMUS_SPEED_OF_SOUND_M_S / L11_PICMUS_FREQ_CENTER_HZ


def picmus_contrast_artifact_axes(
    path: Path,
    *,
    fallback_x_bounds_m: tuple[float, float],
    fallback_z_bounds_m: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return lambda/4 beamforming axes covering the public contrast scan scope."""
    scan_axes = _load_picmus_contrast_scan_axes(path)
    if scan_axes is None:
        x_bounds_m = fallback_x_bounds_m
        z_bounds_m = fallback_z_bounds_m
    else:
        x_bounds_m = (float(scan_axes.x_axis_m[0]), float(scan_axes.x_axis_m[-1]))
        z_bounds_m = (float(scan_axes.z_axis_m[0]), float(scan_axes.z_axis_m[-1]))
    return _image_axes_for_grid_wavelengths(
        PICMUS_CONTRAST_GRID_WAVELENGTHS,
        default_x_axis_m=np.array(x_bounds_m, dtype=np.float64),
        default_z_axis_m=np.array(z_bounds_m, dtype=np.float64),
        x_bounds_m=x_bounds_m,
        z_bounds_m=z_bounds_m,
        wavelength_m=PICMUS_CONTRAST_WAVELENGTH_M,
        warn_pixels=PICMUS_GRID_WARN_PIXELS,
        stacklevel=3,
    )


def picmus_contrast_scan_identity(path: Path) -> tuple[str, ...]:
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


def beamformed_data_from_reconstruction(
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


def save_picmus_contrast_beamformed(path: Path, data: ContrastBeamformedData) -> None:
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
    logger.info("Saved PICMUS contrast beamformed cache to %s", path)


def load_picmus_contrast_beamformed(path: Path) -> ContrastBeamformedData:
    """Load beamformed contrast data saved by `save_picmus_contrast_beamformed`."""
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


def picmus_contrast_comparison_panel_specs(
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


def render_picmus_contrast_figures(
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
            interpolation=PICMUS_CONTRAST_PLOT_INTERPOLATION,
        )
        ax.set_title(PICMUS_CONTRAST_PROPOSED_TITLE)
        ax.set_xlabel("Lateral [mm]")
        ax.set_ylabel("Depth [mm]")
        fig.colorbar(im, ax=ax, label="Amplitude [dB]")
        fig.savefig(fastsimus_path, dpi=PICMUS_CONTRAST_PLOT_DPI, bbox_inches="tight")
        pyplot.close(fig)

        _render_picmus_contrast_comparison(data, phantom_data, comparison_path, overlay_cysts=False)
        _render_picmus_contrast_two_panel(data, two_panel_path)
        _render_picmus_contrast_comparison(data, phantom_data, debug_comparison_path, overlay_cysts=True)
    return fastsimus_path, comparison_path, two_panel_path, debug_comparison_path


def overlay_contrast_cysts(ax: Any, phantom_data: ContrastPhantomData) -> None:
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
    panel_specs = picmus_contrast_comparison_panel_specs(data)
    fig, axes = pyplot.subplots(1, 3, figsize=PICMUS_CONTRAST_COMPARISON_FIGSIZE_IN, constrained_layout=True)
    for ax, spec in zip(axes, panel_specs, strict=True):
        im = ax.imshow(
            spec.image_db,
            extent=extent_mm,
            cmap=spec.cmap,
            vmin=spec.vmin_db,
            vmax=spec.vmax_db,
            aspect=PICMUS_CONTRAST_IMAGE_ASPECT,
            interpolation=PICMUS_CONTRAST_PLOT_INTERPOLATION,
        )
        if overlay_cysts:
            overlay_contrast_cysts(ax, phantom_data)
        ax.set_title(spec.title)
        ax.set_xlabel("Lateral [mm]")
        ax.set_ylabel("Depth [mm]")
        fig.colorbar(im, ax=ax, label=spec.colorbar_label)
    fig.savefig(output_path, dpi=PICMUS_CONTRAST_PLOT_DPI, bbox_inches="tight")
    pyplot.close(fig)


def _render_picmus_contrast_two_panel(data: ContrastBeamformedData, output_path: Path) -> None:
    """Render PyMUST and proposed images with shared axes labels and colorbar."""
    pyplot = cast("Any", plt)
    extent_mm = _picmus_contrast_extent_mm(data)
    panel_specs = picmus_contrast_comparison_panel_specs(data)[:2]
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
            interpolation=PICMUS_CONTRAST_PLOT_INTERPOLATION,
        )
        ax.set_title(spec.title)
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0].set_ylabel("Depth [mm]", labelpad=PICMUS_CONTRAST_TWO_PANEL_YLABEL_PAD)
    fig.supxlabel("Lateral [mm]")
    fig.colorbar(last_image, ax=axes, label="Amplitude [dB]", shrink=0.84, pad=0.015)
    fig.savefig(output_path, dpi=PICMUS_CONTRAST_PLOT_DPI, bbox_inches="tight")
    pyplot.close(fig)
