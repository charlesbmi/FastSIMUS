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
