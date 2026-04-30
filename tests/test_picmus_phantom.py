"""Integration test scaffold for the PICMUS resolution phantom."""

from __future__ import annotations

import contextlib
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from beartype import beartype
from jaxtyping import Complex, Float, jaxtyped

from tests._picmus_phantom_shared import (
    DYNAMIC_RANGE_DB,
    L11_PICMUS_ATTENUATION_DB_CM_MHZ,
    L11_PICMUS_BANDWIDTH_PERCENT,
    L11_PICMUS_ELEVATION_FOCUS_M,
    L11_PICMUS_FREQ_CENTER_HZ,
    L11_PICMUS_HEIGHT_M,
    L11_PICMUS_N_ELEMENTS,
    L11_PICMUS_PITCH_M,
    L11_PICMUS_SAMPLING_FREQUENCY_HZ,
    L11_PICMUS_SIMUS_DB_THRESH,
    L11_PICMUS_SPEED_OF_SOUND_M_S,
    L11_PICMUS_TX_N_WAVELENGTHS,
    L11_PICMUS_WIDTH_M,
    PICMUS_GRID_WARN_PIXELS,
    PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS,
    Phantom,
    ReconstructedIq,
    RfStacks,
    _angle_cache_key,
    _assert_iq_close,
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

N_ELEMENTS = L11_PICMUS_N_ELEMENTS
FREQ_CENTER_HZ = L11_PICMUS_FREQ_CENTER_HZ
PITCH_M = L11_PICMUS_PITCH_M
WIDTH_M = L11_PICMUS_WIDTH_M
KERF_M = PITCH_M - WIDTH_M
HEIGHT_M = L11_PICMUS_HEIGHT_M
ELEVATION_FOCUS_M = L11_PICMUS_ELEVATION_FOCUS_M
BANDWIDTH_PERCENT = L11_PICMUS_BANDWIDTH_PERCENT
BANDWIDTH_FRACTION = BANDWIDTH_PERCENT / 100.0
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

CPWC_ANGLES_RAD = PICMUS_BROADSIDE_ANGLES_RAD.copy()
PHANTOM_X_M = PICMUS_PHANTOM_X_M.copy()
PHANTOM_Z_M = PICMUS_PHANTOM_Z_M.copy()
PHANTOM_RC = np.ones(PHANTOM_X_M.shape, dtype=np.float64)

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

for constant_array in (
    PICMUS_PHANTOM_X_M,
    PICMUS_PHANTOM_Z_M,
    PICMUS_FULL_ANGLES_RAD,
    PICMUS_BROADSIDE_ANGLES_RAD,
    CPWC_ANGLES_RAD,
    PHANTOM_X_M,
    PHANTOM_Z_M,
    PHANTOM_RC,
    IMAGE_X_M,
    IMAGE_Z_M,
):
    constant_array.setflags(write=False)


@jaxtyped(typechecker=beartype)
def _assert_rf_stack_layout(
    fastsimus: Float[np.ndarray, "samples channels firings"],
    pymust: Float[np.ndarray, "samples channels firings"],
    angles_rad: Float[np.ndarray, "firings"],  # noqa: F821, UP037 - jaxtyping axis name.
) -> None:
    """Assert RF stacks use the expected PICMUS samples/channels/firings layout."""
    assert fastsimus.shape[1:] == (N_ELEMENTS, CPWC_ANGLES_RAD.size)
    assert angles_rad.shape == (CPWC_ANGLES_RAD.size,)


@jaxtyped(typechecker=beartype)
def _assert_reconstructed_iq_layout(
    fastsimus: Complex[np.ndarray, "z x"],
    pymust: Complex[np.ndarray, "z x"],
) -> None:
    """Assert reconstructed IQ images use the expected PICMUS z/x grid layout."""
    assert fastsimus.shape == (IMAGE_Z_M.size, IMAGE_X_M.size)


def _angles_for_plot_mode(angle_mode: str) -> np.ndarray:
    """Return the PICMUS angle sequence requested for diagnostic plotting."""
    if angle_mode == "broadside":
        return PICMUS_BROADSIDE_ANGLES_RAD.copy()
    if angle_mode == "full":
        return PICMUS_FULL_ANGLES_RAD.copy()
    raise ValueError(f"Unknown PICMUS phantom angle mode: {angle_mode}")


def _image_axes_for_grid_wavelengths(grid_wavelengths: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Return image axes for either the fast test grid or a wavelength-spaced plot grid."""
    return _shared_image_axes_for_grid_wavelengths(
        grid_wavelengths,
        default_x_axis_m=IMAGE_X_M,
        default_z_axis_m=IMAGE_Z_M,
        x_bounds_m=(IMAGE_X_MIN_M, IMAGE_X_MAX_M),
        z_bounds_m=(IMAGE_Z_MIN_M, IMAGE_Z_MAX_M),
        wavelength_m=PICMUS_WAVELENGTH_M,
        warn_pixels=PICMUS_GRID_WARN_PIXELS,
        stacklevel=3,
    )


def _expects_large_reconstruction_warning(
    *,
    grid_wavelengths: float | None,
    n_firings: int,
) -> bool:
    """Return whether a requested diagnostic reconstruction is expected to warn."""
    return _shared_expects_large_reconstruction_warning(
        grid_wavelengths=grid_wavelengths,
        n_firings=n_firings,
        default_x_axis_m=IMAGE_X_M,
        default_z_axis_m=IMAGE_Z_M,
        x_bounds_m=(IMAGE_X_MIN_M, IMAGE_X_MAX_M),
        z_bounds_m=(IMAGE_Z_MIN_M, IMAGE_Z_MAX_M),
        wavelength_m=PICMUS_WAVELENGTH_M,
        warn_pixels=PICMUS_GRID_WARN_PIXELS,
        warn_pixel_firings=PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS,
    )


def _make_picmus_resolution_phantom() -> Phantom:
    """Create a compact in-code version of the PICMUS resolution phantom."""
    return Phantom(PHANTOM_X_M.copy(), PHANTOM_Z_M.copy(), PHANTOM_RC.copy())


@lru_cache(maxsize=1)
def _simulate_rf_stacks() -> RfStacks:
    """Simulate matched PyMUST and FastSIMUS RF stacks for the compact phantom."""
    return _simulate_rf_stacks_for_angles(_angle_cache_key(CPWC_ANGLES_RAD))


@lru_cache(maxsize=2)
def _simulate_rf_stacks_for_angles(angles_key: tuple[float, ...]) -> RfStacks:
    """Simulate matched RF stacks for the requested PICMUS angle sequence."""
    angles_rad = np.array(angles_key, dtype=np.float64)
    phantom = _make_picmus_resolution_phantom()
    params = make_l11_picmus_matched_params()
    return _simulate_rf_stacks_for_case(phantom, params, angles_rad)


@lru_cache(maxsize=1)
def _reconstruct_picmus_iq() -> ReconstructedIq:
    """Reconstruct the cached PICMUS phantom RF stacks once per test session."""
    return _reconstruct_picmus_iq_for_grid(_angle_cache_key(CPWC_ANGLES_RAD), None)


@lru_cache(maxsize=4)
def _reconstruct_picmus_iq_for_grid(
    angles_key: tuple[float, ...],
    grid_wavelengths: float | None,
) -> ReconstructedIq:
    """Reconstruct the PICMUS phantom for requested angles and image-grid spacing."""
    x_axis_m, z_axis_m = _image_axes_for_grid_wavelengths(grid_wavelengths)
    return _reconstruct_iq(
        _simulate_rf_stacks_for_angles(angles_key),
        make_l11_picmus_matched_params(),
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
    np.testing.assert_allclose(PHANTOM_X_M, PICMUS_PHANTOM_X_M)
    np.testing.assert_allclose(PHANTOM_Z_M, PICMUS_PHANTOM_Z_M)
    np.testing.assert_array_equal(PHANTOM_RC, np.ones(20, dtype=np.float64))
    np.testing.assert_allclose(CPWC_ANGLES_RAD, PICMUS_BROADSIDE_ANGLES_RAD)
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
    np.testing.assert_allclose(_angles_for_plot_mode("broadside"), PICMUS_BROADSIDE_ANGLES_RAD)
    np.testing.assert_allclose(_angles_for_plot_mode("full"), PICMUS_FULL_ANGLES_RAD)

    with pytest.raises(ValueError, match="Unknown PICMUS phantom angle mode"):
        _angles_for_plot_mode("unsupported")


def test_simulated_rf_stacks_are_finite_nonzero_and_matched() -> None:
    """PyMUST and FastSIMUS produce finite, nonzero RF stacks with matched layout."""
    stacks = _simulate_rf_stacks()

    _assert_rf_stack_layout(stacks.fastsimus, stacks.pymust, stacks.angles_rad)
    assert np.all(np.isfinite(stacks.pymust))
    assert np.all(np.isfinite(stacks.fastsimus))
    assert np.max(np.abs(stacks.pymust)) > 0.0
    assert np.max(np.abs(stacks.fastsimus)) > 0.0


def test_reconstructed_complex_iq_matches_pymust() -> None:
    """Shared DAS reconstruction returns finite complex IQ matching PyMUST."""
    reconstructed = _reconstruct_picmus_iq()

    _assert_reconstructed_iq_layout(reconstructed.fastsimus, reconstructed.pymust)
    assert np.all(np.isfinite(reconstructed.fastsimus))
    assert np.all(np.isfinite(reconstructed.pymust))
    assert np.max(np.abs(reconstructed.fastsimus)) > 0.0
    assert np.max(np.abs(reconstructed.pymust)) > 0.0
    _assert_iq_close(reconstructed.fastsimus, reconstructed.pymust, atol_peak=IQ_ATOL_PEAK)


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

    angles_rad = _angles_for_plot_mode(picmus_phantom_angles)
    warning_context = (
        pytest.warns(RuntimeWarning, match="reconstruction will be slow")
        if _expects_large_reconstruction_warning(
            grid_wavelengths=picmus_phantom_grid_wavelengths,
            n_firings=angles_rad.size,
        )
        else contextlib.nullcontext()
    )
    with warning_context:
        _render_picmus_phantom_plot(
            _reconstruct_picmus_iq_for_grid(_angle_cache_key(angles_rad), picmus_phantom_grid_wavelengths),
            picmus_phantom_plot,
        )

    assert picmus_phantom_plot.is_file()
    assert picmus_phantom_plot.stat().st_size > 0
    pyplot = cast("Any", plt)
    assert pyplot.imread(picmus_phantom_plot).shape[1] >= PICMUS_PLOT_MIN_WIDTH_PX
