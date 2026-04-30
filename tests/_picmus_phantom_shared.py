"""Shared helpers for PICMUS-style phantom integration tests."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import array_api_strict
import numpy as np
import pymust

from fast_simus import MediumParams, TransducerParams, element_positions, plane_wave, simus_compute, simus_precompute
from fast_simus.utils._array_api import _ArrayNamespace

L11_PICMUS_N_ELEMENTS = 128
L11_PICMUS_FREQ_CENTER_HZ = 5.208e6
L11_PICMUS_PITCH_M = 0.30e-3
L11_PICMUS_WIDTH_M = 0.27e-3
L11_PICMUS_HEIGHT_M = 5.0e-3
L11_PICMUS_ELEVATION_FOCUS_M = 20.0e-3
L11_PICMUS_BANDWIDTH_PERCENT = 67.0
L11_PICMUS_TX_N_WAVELENGTHS = 2.5
L11_PICMUS_SPEED_OF_SOUND_M_S = 1540.0
L11_PICMUS_ATTENUATION_DB_CM_MHZ = 0.5
L11_PICMUS_SAMPLING_FREQUENCY_HZ = 4.001 * L11_PICMUS_FREQ_CENTER_HZ
L11_PICMUS_SIMUS_DB_THRESH = -60.0

DYNAMIC_RANGE_DB = 60.0
PICMUS_GRID_WARN_PIXELS = 1_000_000
PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS = 5_000_000
XP_STRICT = cast("_ArrayNamespace", array_api_strict)


class Phantom(NamedTuple):
    """Point-scatterer phantom used by PyMUST and FastSIMUS simulators."""

    x: np.ndarray
    z: np.ndarray
    rc: np.ndarray


class MatchedSimulatorParams(NamedTuple):
    """Matched PyMUST and FastSIMUS physical and simulation parameters."""

    pymust_param: pymust.utils.Param
    fastsimus_transducer: TransducerParams
    fastsimus_medium: MediumParams
    sampling_frequency_hz: float
    tx_n_wavelengths: float
    simus_db_thresh: float


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


class PhantomCase(NamedTuple):
    """Configuration for a reusable phantom test case."""

    id: str
    cache_key: tuple[str, ...]
    phantom_factory: Callable[[], Phantom]
    matched_params_factory: Callable[[], MatchedSimulatorParams]
    default_angles_rad: tuple[float, ...]
    diagnostic_angles_rad: tuple[float, ...] | None
    x_bounds_m: tuple[float, float]
    z_bounds_m: tuple[float, float]
    default_x_axis_m: tuple[float, ...]
    default_z_axis_m: tuple[float, ...]
    wavelength_m: float
    validation_mode: str
    iq_atol_peak: float | None = None


def _angle_cache_key(angles_rad: np.ndarray) -> tuple[float, ...]:
    """Convert angle arrays into a stable cache key."""
    return tuple(float(angle) for angle in angles_rad)


def _image_axes_for_grid_wavelengths(
    grid_wavelengths: float | None,
    *,
    default_x_axis_m: np.ndarray,
    default_z_axis_m: np.ndarray,
    x_bounds_m: tuple[float, float],
    z_bounds_m: tuple[float, float],
    wavelength_m: float,
    warn_pixels: int = PICMUS_GRID_WARN_PIXELS,
    stacklevel: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return image axes for either a default grid or a wavelength-spaced plot grid."""
    if grid_wavelengths is None:
        return default_x_axis_m.copy(), default_z_axis_m.copy()

    n_x, n_z = _image_axis_sizes_for_grid_wavelengths(
        grid_wavelengths,
        default_x_axis_m=default_x_axis_m,
        default_z_axis_m=default_z_axis_m,
        x_bounds_m=x_bounds_m,
        z_bounds_m=z_bounds_m,
        wavelength_m=wavelength_m,
    )
    n_pixels = n_x * n_z
    if n_pixels > warn_pixels:
        warnings.warn(
            f"PICMUS phantom grid spacing {grid_wavelengths:g} wavelengths produces "
            f"{n_x}x{n_z} = {n_pixels} pixels; reconstruction will be slow",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    return (
        np.linspace(x_bounds_m[0], x_bounds_m[1], n_x),
        np.linspace(z_bounds_m[0], z_bounds_m[1], n_z),
    )


def _image_axis_sizes_for_grid_wavelengths(
    grid_wavelengths: float | None,
    *,
    default_x_axis_m: np.ndarray,
    default_z_axis_m: np.ndarray,
    x_bounds_m: tuple[float, float],
    z_bounds_m: tuple[float, float],
    wavelength_m: float,
) -> tuple[int, int]:
    """Return image-axis lengths for a plot grid spacing."""
    if grid_wavelengths is None:
        return default_x_axis_m.size, default_z_axis_m.size
    if not np.isfinite(grid_wavelengths) or grid_wavelengths <= 0.0:
        raise ValueError("grid spacing must be finite and positive")

    spacing_m = grid_wavelengths * wavelength_m
    n_x = int(np.ceil((x_bounds_m[1] - x_bounds_m[0]) / spacing_m)) + 1
    n_z = int(np.ceil((z_bounds_m[1] - z_bounds_m[0]) / spacing_m)) + 1
    return n_x, n_z


def _expects_large_reconstruction_warning(
    *,
    grid_wavelengths: float | None,
    n_firings: int,
    default_x_axis_m: np.ndarray,
    default_z_axis_m: np.ndarray,
    x_bounds_m: tuple[float, float],
    z_bounds_m: tuple[float, float],
    wavelength_m: float,
    warn_pixels: int = PICMUS_GRID_WARN_PIXELS,
    warn_pixel_firings: int = PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS,
) -> bool:
    """Return whether a requested diagnostic reconstruction is expected to warn."""
    n_x, n_z = _image_axis_sizes_for_grid_wavelengths(
        grid_wavelengths,
        default_x_axis_m=default_x_axis_m,
        default_z_axis_m=default_z_axis_m,
        x_bounds_m=x_bounds_m,
        z_bounds_m=z_bounds_m,
        wavelength_m=wavelength_m,
    )
    n_pixels = n_x * n_z
    return n_pixels > warn_pixels or n_pixels * n_firings > warn_pixel_firings


def _warn_if_reconstruction_is_large(
    *,
    n_pixels: int,
    n_firings: int,
    warn_pixel_firings: int = PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS,
    stacklevel: int = 2,
) -> None:
    """Warn when a diagnostic DAS reconstruction is expected to be slow."""
    pixel_firings = n_pixels * n_firings
    if pixel_firings > warn_pixel_firings:
        warnings.warn(
            f"PICMUS phantom DAS reconstruction will be slow: {n_pixels} pixels x "
            f"{n_firings} firings = {pixel_firings} pixel-firings",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def make_l11_picmus_matched_params() -> MatchedSimulatorParams:
    """Build matched PyMUST and FastSIMUS parameters for the PICMUS L11-class probe."""
    pymust_param = pymust.utils.Param()
    pymust_param.fc = L11_PICMUS_FREQ_CENTER_HZ
    pymust_param.pitch = L11_PICMUS_PITCH_M
    pymust_param.width = L11_PICMUS_WIDTH_M
    pymust_param.kerf = L11_PICMUS_PITCH_M - L11_PICMUS_WIDTH_M
    pymust_param.Nelements = L11_PICMUS_N_ELEMENTS
    pymust_param.height = L11_PICMUS_HEIGHT_M
    pymust_param.focus = L11_PICMUS_ELEVATION_FOCUS_M
    pymust_param.radius = np.inf
    pymust_param.bandwidth = L11_PICMUS_BANDWIDTH_PERCENT
    pymust_param.TXnow = cast("Any", L11_PICMUS_TX_N_WAVELENGTHS)
    pymust_param.c = L11_PICMUS_SPEED_OF_SOUND_M_S
    pymust_param.fs = L11_PICMUS_SAMPLING_FREQUENCY_HZ
    pymust_param.attenuation = L11_PICMUS_ATTENUATION_DB_CM_MHZ

    fastsimus_transducer = TransducerParams(
        freq_center=L11_PICMUS_FREQ_CENTER_HZ,
        pitch=L11_PICMUS_PITCH_M,
        width=L11_PICMUS_WIDTH_M,
        n_elements=L11_PICMUS_N_ELEMENTS,
        height=L11_PICMUS_HEIGHT_M,
        elev_focus=L11_PICMUS_ELEVATION_FOCUS_M,
        radius=float("inf"),
        bandwidth=L11_PICMUS_BANDWIDTH_PERCENT / 100.0,
    )
    fastsimus_medium = MediumParams(
        speed_of_sound=L11_PICMUS_SPEED_OF_SOUND_M_S,
        attenuation=L11_PICMUS_ATTENUATION_DB_CM_MHZ,
    )
    return MatchedSimulatorParams(
        pymust_param=pymust_param,
        fastsimus_transducer=fastsimus_transducer,
        fastsimus_medium=fastsimus_medium,
        sampling_frequency_hz=L11_PICMUS_SAMPLING_FREQUENCY_HZ,
        tx_n_wavelengths=L11_PICMUS_TX_N_WAVELENGTHS,
        simus_db_thresh=L11_PICMUS_SIMUS_DB_THRESH,
    )


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
    *,
    n_elements: int,
    db_thresh: float,
) -> np.ndarray:
    """Simulate one PyMUST RF firing per configured plane-wave delay."""
    options = pymust.utils.Options()
    options.dBThresh = db_thresh
    firings = []
    for firing_delays in delays:
        rf, _ = pymust.simus(phantom.x, phantom.z, phantom.rc, firing_delays, param, options)
        firings.append(np.asarray(rf))
    return _stack_rf_firings(firings, n_elements)


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
    precompute_idx = int(np.argmax(delay_spans))
    assert np.all(delay_spans <= delay_spans[precompute_idx])

    plan = simus_precompute(
        scatterers,
        rc,
        delays_xp[precompute_idx],
        params.fastsimus_transducer,
        params.fastsimus_medium,
        fs=params.sampling_frequency_hz,
        tx_n_wavelengths=params.tx_n_wavelengths,
        db_thresh=params.simus_db_thresh,
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
    return _stack_rf_firings(firings, params.fastsimus_transducer.n_elements)


def _simulate_rf_stacks_for_case(
    phantom: Phantom,
    params: MatchedSimulatorParams,
    angles_rad: np.ndarray,
) -> RfStacks:
    """Simulate matched RF stacks for a concrete phantom and angle sequence."""
    pymust_delays = _pymust_plane_wave_delays(params.pymust_param, angles_rad)
    fastsimus_delays = _fastsimus_plane_wave_delays(
        params.fastsimus_transducer,
        params.fastsimus_medium,
        angles_rad,
    )

    pymust_rf = _simulate_pymust_rf(
        phantom,
        params.pymust_param,
        pymust_delays,
        n_elements=params.fastsimus_transducer.n_elements,
        db_thresh=params.simus_db_thresh,
    )
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
    params: MatchedSimulatorParams,
    *,
    x_axis_m: np.ndarray,
    z_axis_m: np.ndarray,
    warn_pixel_firings: int = PICMUS_RECONSTRUCTION_WARN_PIXEL_FIRINGS,
) -> ReconstructedIq:
    """Reconstruct FastSIMUS and PyMUST RF stacks through the same PyMUST DAS matrices."""
    param_for_das = copy.copy(params.pymust_param)
    pymust_delays = _pymust_plane_wave_delays(params.pymust_param, stacks.angles_rad)
    x_grid, z_grid = np.meshgrid(x_axis_m, z_axis_m)
    _warn_if_reconstruction_is_large(
        n_pixels=x_grid.size,
        n_firings=stacks.angles_rad.size,
        warn_pixel_firings=warn_pixel_firings,
        stacklevel=3,
    )
    fastsimus_image = np.zeros(x_grid.shape, dtype=np.complex128)
    pymust_image = np.zeros(x_grid.shape, dtype=np.complex128)

    for firing_idx, firing_delays in enumerate(pymust_delays):
        fastsimus_iq = pymust.rf2iq(
            stacks.fastsimus[..., firing_idx],
            params.sampling_frequency_hz,
            params.fastsimus_transducer.freq_center,
        )
        pymust_iq = pymust.rf2iq(
            stacks.pymust[..., firing_idx],
            params.sampling_frequency_hz,
            params.fastsimus_transducer.freq_center,
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


def _assert_iq_close(actual: np.ndarray, expected: np.ndarray, *, atol_peak: float) -> None:
    """Assert complex IQ images match within a peak-normalized absolute tolerance."""
    if actual.shape != expected.shape:
        raise AssertionError(f"complex IQ shape mismatch: {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)):
        raise AssertionError("FastSIMUS complex IQ contains non-finite values")
    if not np.all(np.isfinite(expected)):
        raise AssertionError("PyMUST complex IQ contains non-finite values")

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
