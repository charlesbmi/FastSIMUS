"""Integration test scaffold for the PICMUS resolution phantom."""

from __future__ import annotations

import contextlib
import copy
import warnings
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, cast

import array_api_strict
import numpy as np
import pymust
import pytest

from fast_simus import MediumParams, TransducerParams, element_positions, plane_wave, simus_compute, simus_precompute
from fast_simus.utils._array_api import _ArrayNamespace

HAS_MATPLOTLIB = False
plt: ModuleType | None = None
with contextlib.suppress(ImportError):
    import matplotlib.pyplot as _matplotlib_pyplot

    plt = _matplotlib_pyplot
    HAS_MATPLOTLIB = True

N_ELEMENTS = 128
FREQ_CENTER_HZ = 5.208e6
PITCH_M = 0.30e-3
WIDTH_M = 0.27e-3
KERF_M = PITCH_M - WIDTH_M
HEIGHT_M = 5.0e-3
ELEVATION_FOCUS_M = 20.0e-3
BANDWIDTH_PERCENT = 67.0
BANDWIDTH_FRACTION = BANDWIDTH_PERCENT / 100.0
TX_N_WAVELENGTHS = 2.5
SPEED_OF_SOUND_M_S = 1540.0
PICMUS_WAVELENGTH_M = SPEED_OF_SOUND_M_S / FREQ_CENTER_HZ
# Attenuation is not stored in the PICMUS HDF5 files; the prototype uses 0.5 dB/cm/MHz.
ATTENUATION_DB_CM_MHZ = 0.5
SAMPLING_FREQUENCY_HZ = 4.001 * FREQ_CENTER_HZ
SIMUS_DB_THRESH = -60.0

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
DYNAMIC_RANGE_DB = 60.0
PICMUS_GRID_WARN_PIXELS = 1_000_000
PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS = 5_000_000
PROPOSED_LABEL = "Proposed"
REFERENCE_LABEL = "PyMUST"
RESIDUAL_LABEL = f"|{PROPOSED_LABEL} - {REFERENCE_LABEL}|"
# 12 in x 360 dpi gives a 4320 px-wide three-panel PNG, close to a 4k export target.
PICMUS_PLOT_FIGSIZE_IN = (12.0, 4.0)
PICMUS_PLOT_MIN_WIDTH_PX = 4096
PICMUS_PLOT_DPI = 360
PICMUS_PLOT_INTERPOLATION = "nearest"
# array_api_strict exercises the FastSIMUS simulation path; PyMUST reconstruction is NumPy-only.
XP_STRICT = cast("_ArrayNamespace", array_api_strict)

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


class Phantom(NamedTuple):
    """Point-scatterer phantom used by both simulators."""

    x: np.ndarray
    z: np.ndarray
    rc: np.ndarray


class MatchedSimulatorParams(NamedTuple):
    """Matched PyMUST and FastSIMUS physical parameters."""

    pymust_param: pymust.utils.Param
    fastsimus_transducer: TransducerParams
    fastsimus_medium: MediumParams


class RfStacks(NamedTuple):
    """RF simulation outputs stacked as samples, channels, firings."""

    fastsimus: np.ndarray
    pymust: np.ndarray
    angles_rad: np.ndarray


class ReconstructedIq(NamedTuple):
    """Complex IQ images reconstructed through a shared DAS path."""

    fastsimus: np.ndarray
    pymust: np.ndarray
    x_grid_m: np.ndarray
    z_grid_m: np.ndarray


def _angle_cache_key(angles_rad: np.ndarray) -> tuple[float, ...]:
    """Convert angle arrays into a stable cache key."""
    return tuple(float(angle) for angle in angles_rad)


def _angles_for_plot_mode(angle_mode: str) -> np.ndarray:
    """Return the PICMUS angle sequence requested for diagnostic plotting."""
    if angle_mode == "broadside":
        return PICMUS_BROADSIDE_ANGLES_RAD.copy()
    if angle_mode == "full":
        return PICMUS_FULL_ANGLES_RAD.copy()
    raise ValueError(f"Unknown PICMUS phantom angle mode: {angle_mode}")


def _image_axes_for_grid_wavelengths(grid_wavelengths: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Return image axes for either the fast test grid or a wavelength-spaced plot grid."""
    if grid_wavelengths is None:
        return IMAGE_X_M.copy(), IMAGE_Z_M.copy()

    n_x, n_z = _image_axis_sizes_for_grid_wavelengths(grid_wavelengths)
    n_pixels = n_x * n_z
    if n_pixels > PICMUS_GRID_WARN_PIXELS:
        warnings.warn(
            f"PICMUS phantom grid spacing {grid_wavelengths:g} wavelengths produces "
            f"{n_x}x{n_z} = {n_pixels} pixels; reconstruction will be slow",
            RuntimeWarning,
            stacklevel=2,
        )
    return (
        np.linspace(IMAGE_X_MIN_M, IMAGE_X_MAX_M, n_x),
        np.linspace(IMAGE_Z_MIN_M, IMAGE_Z_MAX_M, n_z),
    )


def _image_axis_sizes_for_grid_wavelengths(grid_wavelengths: float | None) -> tuple[int, int]:
    """Return image-axis lengths for a plot grid spacing."""
    if grid_wavelengths is None:
        return IMAGE_X_M.size, IMAGE_Z_M.size
    if not np.isfinite(grid_wavelengths) or grid_wavelengths <= 0.0:
        raise ValueError("grid spacing must be finite and positive")

    spacing_m = grid_wavelengths * PICMUS_WAVELENGTH_M
    n_x = int(np.ceil((IMAGE_X_MAX_M - IMAGE_X_MIN_M) / spacing_m)) + 1
    n_z = int(np.ceil((IMAGE_Z_MAX_M - IMAGE_Z_MIN_M) / spacing_m)) + 1
    return n_x, n_z


def _expects_large_reconstruction_warning(
    *,
    grid_wavelengths: float | None,
    n_firings: int,
) -> bool:
    """Return whether a requested diagnostic reconstruction is expected to warn."""
    n_x, n_z = _image_axis_sizes_for_grid_wavelengths(grid_wavelengths)
    n_pixels = n_x * n_z
    return n_pixels > PICMUS_GRID_WARN_PIXELS or n_pixels * n_firings > PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS


def _warn_if_reconstruction_is_large(*, n_pixels: int, n_firings: int) -> None:
    """Warn when a diagnostic DAS reconstruction is expected to be slow."""
    pixel_firings = n_pixels * n_firings
    if pixel_firings > PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS:
        warnings.warn(
            f"PICMUS phantom DAS reconstruction will be slow: {n_pixels} pixels x "
            f"{n_firings} firings = {pixel_firings} pixel-firings",
            RuntimeWarning,
            stacklevel=2,
        )


def _make_picmus_resolution_phantom() -> Phantom:
    """Create a compact in-code version of the PICMUS resolution phantom."""
    return Phantom(PHANTOM_X_M.copy(), PHANTOM_Z_M.copy(), PHANTOM_RC.copy())


def _make_matched_simulator_params() -> MatchedSimulatorParams:
    """Build matched PyMUST and FastSIMUS simulation parameters."""
    pymust_param = pymust.utils.Param()
    pymust_param.fc = FREQ_CENTER_HZ
    pymust_param.pitch = PITCH_M
    pymust_param.width = WIDTH_M
    pymust_param.kerf = KERF_M
    pymust_param.Nelements = N_ELEMENTS
    pymust_param.height = HEIGHT_M
    pymust_param.focus = ELEVATION_FOCUS_M
    pymust_param.radius = np.inf
    pymust_param.bandwidth = BANDWIDTH_PERCENT
    pymust_param.TXnow = cast("Any", TX_N_WAVELENGTHS)
    pymust_param.c = SPEED_OF_SOUND_M_S
    pymust_param.fs = SAMPLING_FREQUENCY_HZ
    pymust_param.attenuation = ATTENUATION_DB_CM_MHZ

    fastsimus_transducer = TransducerParams(
        freq_center=FREQ_CENTER_HZ,
        pitch=PITCH_M,
        width=WIDTH_M,
        n_elements=N_ELEMENTS,
        height=HEIGHT_M,
        elev_focus=ELEVATION_FOCUS_M,
        radius=float("inf"),
        bandwidth=BANDWIDTH_FRACTION,
    )
    fastsimus_medium = MediumParams(
        speed_of_sound=SPEED_OF_SOUND_M_S,
        attenuation=ATTENUATION_DB_CM_MHZ,
    )
    return MatchedSimulatorParams(pymust_param, fastsimus_transducer, fastsimus_medium)


def _pymust_plane_wave_delays(param: pymust.utils.Param, angles_rad: np.ndarray) -> np.ndarray:
    """Compute PyMUST plane-wave delays for each CPWC firing."""
    return np.stack([np.asarray(pymust.txdelay(param, float(angle))) for angle in angles_rad], axis=0)


def _fastsimus_plane_wave_delays(
    transducer: TransducerParams,
    medium: MediumParams,
    angles_rad: np.ndarray,
) -> np.ndarray:
    """Compute FastSIMUS plane-wave delays for each CPWC firing."""
    positions, _, apex_offset = element_positions(transducer.n_elements, transducer.pitch, transducer.radius, XP_STRICT)
    delays = [
        plane_wave(
            positions,
            float(angle),
            speed_of_sound=medium.speed_of_sound,
            radius=transducer.radius,
            apex_offset=apex_offset,
        )
        for angle in angles_rad
    ]
    return np.stack([np.asarray(delay) for delay in delays], axis=0)


def _stack_rf_firings(firings: list[np.ndarray], n_elements: int) -> np.ndarray:
    """Pad RF firings to a common sample count and stack as samples, channels, firings."""
    n_samples = max(rf.shape[0] for rf in firings)
    dtype = np.result_type(*(rf.dtype for rf in firings), np.float32)
    stack = np.zeros((n_samples, n_elements, len(firings)), dtype=dtype)
    for firing_idx, rf in enumerate(firings):
        stack[: rf.shape[0], :, firing_idx] = rf
    return stack


def _pad_rf_stack_to_samples(stack: np.ndarray, n_samples: int) -> np.ndarray:
    """Pad an RF stack along samples without changing channels or firings."""
    if stack.shape[0] >= n_samples:
        return stack
    pad_samples = n_samples - stack.shape[0]
    return np.pad(stack, ((0, pad_samples), (0, 0), (0, 0)))


def _simulate_pymust_rf(
    phantom: Phantom,
    param: pymust.utils.Param,
    delays: np.ndarray,
) -> np.ndarray:
    """Simulate one PyMUST RF firing per configured plane-wave delay."""
    options = pymust.utils.Options()
    options.dBThresh = SIMUS_DB_THRESH
    firings = []
    for firing_delays in delays:
        rf, _ = pymust.simus(phantom.x, phantom.z, phantom.rc, firing_delays, param, options)
        firings.append(np.asarray(rf))
    return _stack_rf_firings(firings, N_ELEMENTS)


def _simulate_fastsimus_rf(
    phantom: Phantom,
    params: MatchedSimulatorParams,
    delays: np.ndarray,
) -> np.ndarray:
    """Simulate one FastSIMUS RF firing per configured plane-wave delay."""
    scatterers = XP_STRICT.asarray(np.stack([phantom.x, phantom.z], axis=-1))
    rc = XP_STRICT.asarray(phantom.rc)
    delays_xp = [XP_STRICT.asarray(firing_delays) for firing_delays in delays]
    delay_spans = np.ptp(delays, axis=1)
    # Use the largest delay spread so the precomputed frequency grid covers every firing.
    precompute_idx = int(np.argmax(delay_spans))
    assert np.all(delay_spans <= delay_spans[precompute_idx])

    plan = simus_precompute(
        scatterers,
        rc,
        delays_xp[precompute_idx],
        params.fastsimus_transducer,
        params.fastsimus_medium,
        fs=SAMPLING_FREQUENCY_HZ,
        tx_n_wavelengths=TX_N_WAVELENGTHS,
        db_thresh=SIMUS_DB_THRESH,
    )
    firings = [
        np.asarray(
            simus_compute(
                scatterers,
                rc,
                firing_delays,
                plan,
                params.fastsimus_transducer,
                params.fastsimus_medium,
            ).rf,
        )
        for firing_delays in delays_xp
    ]
    return _stack_rf_firings(firings, N_ELEMENTS)


@lru_cache(maxsize=1)
def _simulate_rf_stacks() -> RfStacks:
    """Simulate matched PyMUST and FastSIMUS RF stacks for the compact phantom."""
    return _simulate_rf_stacks_for_angles(_angle_cache_key(CPWC_ANGLES_RAD))


@lru_cache(maxsize=2)
def _simulate_rf_stacks_for_angles(angles_key: tuple[float, ...]) -> RfStacks:
    """Simulate matched RF stacks for the requested PICMUS angle sequence."""
    angles_rad = np.array(angles_key, dtype=np.float64)
    phantom = _make_picmus_resolution_phantom()
    params = _make_matched_simulator_params()
    pymust_delays = _pymust_plane_wave_delays(params.pymust_param, angles_rad)
    fastsimus_delays = _fastsimus_plane_wave_delays(
        params.fastsimus_transducer,
        params.fastsimus_medium,
        angles_rad,
    )

    pymust_rf = _simulate_pymust_rf(phantom, params.pymust_param, pymust_delays)
    fastsimus_rf = _simulate_fastsimus_rf(phantom, params, fastsimus_delays)
    n_samples = max(pymust_rf.shape[0], fastsimus_rf.shape[0])

    result = RfStacks(
        fastsimus=_pad_rf_stack_to_samples(fastsimus_rf, n_samples),
        pymust=_pad_rf_stack_to_samples(pymust_rf, n_samples),
        angles_rad=angles_rad.copy(),
    )
    result.fastsimus.setflags(write=False)
    result.pymust.setflags(write=False)
    result.angles_rad.setflags(write=False)
    return result


def _reconstruct_iq(
    stacks: RfStacks,
    *,
    x_axis_m: np.ndarray | None = None,
    z_axis_m: np.ndarray | None = None,
) -> ReconstructedIq:
    """Reconstruct FastSIMUS and PyMUST RF stacks through the same PyMUST DAS matrices."""
    params = _make_matched_simulator_params()
    param_for_das = copy.copy(params.pymust_param)
    pymust_delays = _pymust_plane_wave_delays(params.pymust_param, stacks.angles_rad)
    x_axis = IMAGE_X_M if x_axis_m is None else x_axis_m
    z_axis = IMAGE_Z_M if z_axis_m is None else z_axis_m
    x_grid, z_grid = np.meshgrid(x_axis, z_axis)
    _warn_if_reconstruction_is_large(n_pixels=x_grid.size, n_firings=stacks.angles_rad.size)
    fastsimus_image = np.zeros(x_grid.shape, dtype=np.complex128)
    pymust_image = np.zeros(x_grid.shape, dtype=np.complex128)

    for firing_idx, firing_delays in enumerate(pymust_delays):
        fastsimus_iq = pymust.rf2iq(
            stacks.fastsimus[..., firing_idx],
            SAMPLING_FREQUENCY_HZ,
            FREQ_CENTER_HZ,
        )
        pymust_iq = pymust.rf2iq(
            stacks.pymust[..., firing_idx],
            SAMPLING_FREQUENCY_HZ,
            FREQ_CENTER_HZ,
        )
        if fastsimus_iq.shape != pymust_iq.shape:
            raise AssertionError(f"IQ shape mismatch: {fastsimus_iq.shape} != {pymust_iq.shape}")

        param_for_das.TXdelay = firing_delays
        das_matrix = pymust.dasmtx(1j * np.array(pymust_iq.shape), x_grid, z_grid, param_for_das)
        fastsimus_image += (das_matrix @ fastsimus_iq.flatten(order="F")).reshape(x_grid.shape, order="F")
        pymust_image += (das_matrix @ pymust_iq.flatten(order="F")).reshape(x_grid.shape, order="F")

    result = ReconstructedIq(
        fastsimus=fastsimus_image,
        pymust=pymust_image,
        x_grid_m=x_grid,
        z_grid_m=z_grid,
    )
    result.fastsimus.setflags(write=False)
    result.pymust.setflags(write=False)
    result.x_grid_m.setflags(write=False)
    result.z_grid_m.setflags(write=False)
    return result


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
    return _reconstruct_iq(_simulate_rf_stacks_for_angles(angles_key), x_axis_m=x_axis_m, z_axis_m=z_axis_m)


def _assert_iq_close(actual: np.ndarray, expected: np.ndarray, *, atol_peak: float = IQ_ATOL_PEAK) -> None:
    """Assert complex IQ images match within a peak-normalized absolute tolerance."""
    if actual.shape != expected.shape:
        raise AssertionError(f"complex IQ shape mismatch: {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)):
        raise AssertionError("FastSIMUS complex IQ contains non-finite values")
    if not np.all(np.isfinite(expected)):
        raise AssertionError("PyMUST complex IQ contains non-finite values")

    # Match the plan's shared-peak metric so both images use one amplitude reference.
    reference_peak = max(float(np.max(np.abs(actual))), float(np.max(np.abs(expected))))
    if reference_peak <= 0.0:
        raise AssertionError("complex IQ reference peak must be nonzero")

    residual = float(np.max(np.abs(actual - expected)) / reference_peak)
    residual_db = 20.0 * np.log10(residual) if residual > 0.0 else -np.inf
    if residual > atol_peak:
        raise AssertionError(
            f"complex IQ residual {residual:.3e} ({residual_db:.1f} dB) "
            f"exceeds peak-normalized tolerance {atol_peak:.3e}",
        )


def _iq_to_display_db(
    iq: np.ndarray,
    *,
    reference_peak: float,
    dynamic_range_db: float = DYNAMIC_RANGE_DB,
) -> np.ndarray:
    """Convert complex IQ to clipped dB display values using an explicit reference peak."""
    if not np.isfinite(reference_peak) or reference_peak <= 0.0:
        raise ValueError("reference_peak must be finite and positive")

    display_db = _iq_to_reference_db(iq, reference_peak=reference_peak)
    return np.clip(display_db, -dynamic_range_db, 0.0)


def _iq_to_reference_db(iq: np.ndarray, *, reference_peak: float) -> np.ndarray:
    """Convert complex IQ magnitude to unclipped dB values using a shared reference peak."""
    if not np.isfinite(reference_peak) or reference_peak <= 0.0:
        raise ValueError("reference_peak must be finite and positive")

    normalized = np.maximum(np.abs(iq) / reference_peak, np.finfo(np.float64).tiny)
    return 20.0 * np.log10(normalized)


def _iq_residual_display_db(fastsimus_iq: np.ndarray, pymust_iq: np.ndarray, *, reference_peak: float) -> np.ndarray:
    """Display complex-domain residual magnitude relative to the shared image peak."""
    return _iq_to_display_db(fastsimus_iq - pymust_iq, reference_peak=reference_peak)


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


def test_picmus_phantom_scaffold(picmus_phantom_plot: Path | None) -> None:
    """Scaffold constants describe a compact PICMUS resolution phantom test."""
    if picmus_phantom_plot is not None and not HAS_MATPLOTLIB:
        pytest.skip("--plot-picmus-phantom requires matplotlib")

    phantom = Phantom(PHANTOM_X_M, PHANTOM_Z_M, PHANTOM_RC)

    assert picmus_phantom_plot is None or isinstance(picmus_phantom_plot, Path)
    assert phantom.x.shape == phantom.z.shape == phantom.rc.shape
    assert CPWC_ANGLES_RAD.shape == (3,)
    assert IMAGE_Z_M.size > IMAGE_X_M.size
    assert isinstance(HAS_MATPLOTLIB, bool)
    assert plt is None or isinstance(plt, ModuleType)
    assert pytest.approx(5e-2) == IQ_ATOL_PEAK


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


def test_picmus_phantom_plot_labels_are_anonymous() -> None:
    """Diagnostic figures label the proposed simulator anonymously."""
    assert PROPOSED_LABEL == "Proposed"
    assert "FastSIMUS" not in PROPOSED_LABEL
    assert "FastSIMUS" not in RESIDUAL_LABEL


def test_picmus_phantom_plot_export_targets_4k_width() -> None:
    """Diagnostic figures are exported near 4k width without display smoothing."""
    assert PICMUS_PLOT_DPI > 150
    assert PICMUS_PLOT_FIGSIZE_IN[0] * PICMUS_PLOT_DPI >= PICMUS_PLOT_MIN_WIDTH_PX
    assert PICMUS_PLOT_MIN_WIDTH_PX >= 4096
    assert PICMUS_PLOT_INTERPOLATION == "nearest"


def test_picmus_phantom_grid_wavelength_spacing_defaults_to_test_grid() -> None:
    """The optional plot grid defaults to the compact test grid."""
    x_axis_m, z_axis_m = _image_axes_for_grid_wavelengths(None)

    np.testing.assert_allclose(x_axis_m, IMAGE_X_M)
    np.testing.assert_allclose(z_axis_m, IMAGE_Z_M)


def test_picmus_phantom_grid_wavelength_spacing_supports_lambda_quarter() -> None:
    """A 0.25 wavelength plot grid beamforms close to lambda/4 sample spacing."""
    spacing_m = 0.25 * PICMUS_WAVELENGTH_M
    x_axis_m, z_axis_m = _image_axes_for_grid_wavelengths(0.25)

    assert x_axis_m.size == int(np.ceil((IMAGE_X_MAX_M - IMAGE_X_MIN_M) / spacing_m)) + 1
    assert z_axis_m.size == int(np.ceil((IMAGE_Z_MAX_M - IMAGE_Z_MIN_M) / spacing_m)) + 1
    assert np.max(np.diff(x_axis_m)) <= spacing_m
    assert np.max(np.diff(z_axis_m)) <= spacing_m
    assert x_axis_m[0] == pytest.approx(IMAGE_X_MIN_M)
    assert x_axis_m[-1] == pytest.approx(IMAGE_X_MAX_M)
    assert z_axis_m[0] == pytest.approx(IMAGE_Z_MIN_M)
    assert z_axis_m[-1] == pytest.approx(IMAGE_Z_MAX_M)

    with pytest.raises(ValueError, match="grid spacing must be finite and positive"):
        _image_axes_for_grid_wavelengths(0.0)

    with pytest.raises(ValueError, match="grid spacing must be finite and positive"):
        _image_axes_for_grid_wavelengths(np.inf)

    with pytest.raises(ValueError, match="grid spacing must be finite and positive"):
        _image_axes_for_grid_wavelengths(np.nan)

    with pytest.raises(ValueError, match="grid spacing must be finite and positive"):
        _image_axes_for_grid_wavelengths(-1.0)


def test_picmus_phantom_grid_wavelength_spacing_warns_for_large_grids() -> None:
    """Very fine plot grids warn before expensive DAS reconstruction."""
    with pytest.warns(RuntimeWarning, match="reconstruction will be slow"):
        _image_axes_for_grid_wavelengths(0.1)


def test_picmus_phantom_reconstruction_warns_for_large_grid_angle_products() -> None:
    """Fine grids with many firings warn before expensive DAS reconstruction."""
    with pytest.warns(RuntimeWarning, match="DAS reconstruction will be slow"):
        _warn_if_reconstruction_is_large(n_pixels=315_000, n_firings=75)


def test_simulated_rf_stacks_are_finite_and_nonzero() -> None:
    """PyMUST and FastSIMUS produce finite, nonzero RF stacks."""
    stacks = _simulate_rf_stacks()

    assert np.all(np.isfinite(stacks.pymust))
    assert np.all(np.isfinite(stacks.fastsimus))
    assert np.max(np.abs(stacks.pymust)) > 0.0
    assert np.max(np.abs(stacks.fastsimus)) > 0.0


def test_simulated_rf_stacks_have_matched_shape() -> None:
    """RF stacks use samples, channels, firings layout after padding."""
    stacks = _simulate_rf_stacks()

    assert stacks.pymust.ndim == 3
    assert stacks.fastsimus.ndim == 3
    assert stacks.pymust.shape == stacks.fastsimus.shape
    assert stacks.pymust.shape[1] == N_ELEMENTS
    assert stacks.pymust.shape[2] == CPWC_ANGLES_RAD.size
    assert stacks.angles_rad.shape == CPWC_ANGLES_RAD.shape


def test_reconstructed_iq_arrays_are_finite_and_nonzero() -> None:
    """Shared DAS reconstruction returns finite, nonzero complex IQ images."""
    reconstructed = _reconstruct_picmus_iq()

    assert reconstructed.fastsimus.shape == reconstructed.pymust.shape
    assert reconstructed.fastsimus.shape == (IMAGE_Z_M.size, IMAGE_X_M.size)
    assert np.all(np.isfinite(reconstructed.fastsimus))
    assert np.all(np.isfinite(reconstructed.pymust))
    assert np.max(np.abs(reconstructed.fastsimus)) > 0.0
    assert np.max(np.abs(reconstructed.pymust)) > 0.0


def test_reconstructed_complex_iq_matches_pymust() -> None:
    """FastSIMUS and PyMUST reconstructed complex IQ agree before display transforms."""
    reconstructed = _reconstruct_picmus_iq()

    _assert_iq_close(reconstructed.fastsimus, reconstructed.pymust)


def test_iq_assertion_reports_residual_db() -> None:
    """IQ assertion failure includes the peak-normalized residual in dB."""
    actual = np.ones((2, 2), dtype=np.complex64)
    expected = np.zeros((2, 2), dtype=np.complex64)

    with pytest.raises(AssertionError, match=r"complex IQ residual .* dB"):
        _assert_iq_close(actual, expected, atol_peak=1e-6)


def test_display_db_uses_shared_reference_and_clips() -> None:
    """Display conversion uses the supplied peak and clips to the configured dynamic range."""
    iq = np.array([[0.5 + 0.0j, 0.05 + 0.0j, 0.0 + 0.0j]])

    display_db = _iq_to_display_db(iq, reference_peak=1.0)

    np.testing.assert_allclose(display_db, np.array([[-6.02059991, -26.02059991, -DYNAMIC_RANGE_DB]]))

    with pytest.raises(ValueError, match="reference_peak must be finite and positive"):
        _iq_to_display_db(iq, reference_peak=np.inf)


def test_residual_display_uses_complex_domain_difference() -> None:
    """Difference panel displays complex residual magnitude, not dB-image subtraction."""
    fastsimus_iq = np.array([[1.0 + 0.0j, 0.9 + 0.0j]])
    pymust_iq = np.array([[1.0 + 0.0j, 1.0 + 0.0j]])

    residual_db = _iq_residual_display_db(fastsimus_iq, pymust_iq, reference_peak=1.0)

    np.testing.assert_allclose(residual_db, np.array([[-DYNAMIC_RANGE_DB, -20.0]]))


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
