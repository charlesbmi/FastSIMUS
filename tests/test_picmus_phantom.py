"""Integration test scaffold for the PICMUS resolution phantom."""

from __future__ import annotations

import contextlib
import copy
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

CPWC_ANGLES_RAD = PICMUS_BROADSIDE_ANGLES_RAD.copy()
PHANTOM_X_M = PICMUS_PHANTOM_X_M.copy()
PHANTOM_Z_M = PICMUS_PHANTOM_Z_M.copy()
PHANTOM_RC = np.ones(PHANTOM_X_M.shape, dtype=np.float64)

IMAGE_X_M = np.linspace(-PICMUS_APERTURE_X_M, PICMUS_APERTURE_X_M, 96)
IMAGE_Z_M = np.linspace(5.0e-3, 50.0e-3, 128)
IQ_ATOL_PEAK = 5e-2
DYNAMIC_RANGE_DB = 60.0
PROPOSED_LABEL = "Proposed"
REFERENCE_LABEL = "PyMUST"
RESIDUAL_LABEL = f"|{PROPOSED_LABEL} - {REFERENCE_LABEL}|"
PICMUS_PLOT_FIGSIZE_IN = (12.0, 4.0)
PICMUS_PLOT_MIN_WIDTH_PX = 4096
PICMUS_PLOT_DPI = 360
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


def _reconstruct_iq(stacks: RfStacks) -> ReconstructedIq:
    """Reconstruct FastSIMUS and PyMUST RF stacks through the same PyMUST DAS matrices."""
    params = _make_matched_simulator_params()
    param_for_das = copy.copy(params.pymust_param)
    pymust_delays = _pymust_plane_wave_delays(params.pymust_param, stacks.angles_rad)
    x_grid, z_grid = np.meshgrid(IMAGE_X_M, IMAGE_Z_M)
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
    return _reconstruct_picmus_iq_for_angles(_angle_cache_key(CPWC_ANGLES_RAD))


@lru_cache(maxsize=2)
def _reconstruct_picmus_iq_for_angles(angles_key: tuple[float, ...]) -> ReconstructedIq:
    """Reconstruct the PICMUS phantom for a requested angle sequence."""
    return _reconstruct_iq(_simulate_rf_stacks_for_angles(angles_key))


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
        im = ax.imshow(image, extent=extent_mm, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
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
    assert IMAGE_Z_M[0] <= 5.0e-3
    assert IMAGE_Z_M[-1] >= 50.0e-3


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
) -> None:
    """The optional CLI path writes a diagnostic PNG and stays inactive by default."""
    if picmus_phantom_plot is None:
        pytest.skip("--plot-picmus-phantom not provided")
    if not HAS_MATPLOTLIB:
        pytest.skip("--plot-picmus-phantom requires matplotlib")

    angles_rad = _angles_for_plot_mode(picmus_phantom_angles)
    _render_picmus_phantom_plot(_reconstruct_picmus_iq_for_angles(_angle_cache_key(angles_rad)), picmus_phantom_plot)

    assert picmus_phantom_plot.is_file()
    assert picmus_phantom_plot.stat().st_size > 0
    pyplot = cast("Any", plt)
    assert pyplot.imread(picmus_phantom_plot).shape[1] >= PICMUS_PLOT_MIN_WIDTH_PX
