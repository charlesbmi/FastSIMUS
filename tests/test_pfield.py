"""Tests for pfield (pressure field) computation.

Reference tests compare FastSIMUS pfield against PyMUST's pfield output.
Tests are structured as invariants that must hold at every refactoring step.
"""

from typing import NamedTuple, cast

import array_api_strict
import numpy as np
import pymust
import pytest

from fast_simus.pfield import pfield
from fast_simus.transducer_params import TransducerParams
from fast_simus.transducer_presets import C5_2v, L11_5v, P4_2v
from fast_simus.utils._array_api import _ArrayNamespace

# Array API backend for FastSIMUS calls
xp = cast(_ArrayNamespace, array_api_strict)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID_SIZE = 50


class ReferenceData(NamedTuple):
    """Typed reference data from PyMUST pfield."""

    rp: np.ndarray
    positions: np.ndarray
    delays: np.ndarray
    probe: str
    x_grid: np.ndarray
    z_grid: np.ndarray
    focus: tuple[float, float] | None = None
    tilt_rad: float | None = None


def _make_grid(
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    n: int = GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 2D meshgrid for pressure field evaluation.

    Returns:
        Tuple of (x_grid, z_grid) each with shape (n, n).
    """
    x_lin = np.linspace(x_range[0], x_range[1], n)
    z_lin = np.linspace(z_range[0], z_range[1], n)
    return np.meshgrid(x_lin, z_lin)


def _make_positions(
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    n: int = GRID_SIZE,
) -> np.ndarray:
    """Create positions array for pfield evaluation.

    Returns:
        Positions array with shape (n, n, 2) where [..., 0] is x and [..., 1] is z.
    """
    x_grid, z_grid = _make_grid(x_range, z_range, n)
    return np.stack([x_grid, z_grid], axis=-1)


def _pymust_reference(
    probe_name: str,
    delays: np.ndarray,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    n: int = GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference pressure field via PyMUST.

    Returns:
        Tuple of (rp, x_grid, z_grid, positions).
    """
    param = pymust.getparam(probe_name)
    x_grid, z_grid = _make_grid(x_range, z_range, n)
    positions = np.stack([x_grid, z_grid], axis=-1)
    rp, _spect, _idx = pymust.pfield(x_grid, [], z_grid, delays, param)  # type: ignore[arg-type]
    return rp, x_grid, z_grid, positions


# ---------------------------------------------------------------------------
# Reference data fixtures (computed on-the-fly via PyMUST)
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("P4-2v-focused", id="P4-2v-focused"),
        pytest.param("L11-5v-plane", id="L11-5v-plane"),
        pytest.param("C5-2v-focused", id="C5-2v-focused"),
    ]
)
def reference(request: pytest.FixtureRequest) -> ReferenceData:
    """Parametrized reference data from PyMUST for P4-2v, L11-5v, C5-2v."""
    case = request.param
    if case == "P4-2v-focused":
        param = pymust.getparam("P4-2v")
        x0, z0 = 0.02, 0.05
        delays = pymust.txdelayFocused(param, x0, z0)
        rp, x_grid, z_grid, positions = _pymust_reference(
            "P4-2v",
            delays,
            (-4e-2, 4e-2),
            (param.pitch, 10e-2),  # type: ignore[arg-type]
        )
        return ReferenceData(rp, positions, delays, "P4-2v", x_grid, z_grid, focus=(x0, z0))
    if case == "L11-5v-plane":
        param = pymust.getparam("L11-5v")
        tilt_rad = np.deg2rad(10.0)
        delays = pymust.txdelayPlane(param, tilt_rad)
        rp, x_grid, z_grid, positions = _pymust_reference(
            "L11-5v",
            delays,
            (-2e-2, 2e-2),
            (param.pitch, 4e-2),  # type: ignore[arg-type]
        )
        return ReferenceData(rp, positions, delays, "L11-5v", x_grid, z_grid, tilt_rad=tilt_rad)
    if case == "C5-2v-focused":
        param = pymust.getparam("C5-2v")
        x0, z0 = 0.0, 0.06
        delays = pymust.txdelayFocused(param, x0, z0)
        rp, x_grid, z_grid, positions = _pymust_reference(
            "C5-2v",
            delays,
            (-4e-2, 4e-2),
            (param.pitch, 10e-2),  # type: ignore[arg-type]
        )
        return ReferenceData(rp, positions, delays, "C5-2v", x_grid, z_grid, focus=(x0, z0))
    raise ValueError(f"Unknown reference case: {case}")


def _preset_for_probe(probe: str):
    """Return preset callable for probe name."""
    return {"P4-2v": P4_2v, "L11-5v": L11_5v, "C5-2v": C5_2v}[probe]


def _assert_valid_pfield_output(rp, expected_shape: tuple[int, ...], *, expect_zero: bool = False) -> None:
    """Assert pfield output has correct shape and valid values."""
    assert rp.shape == expected_shape
    rp_np = np.asarray(rp)
    assert np.all(rp_np >= 0)
    if expect_zero:
        assert np.max(rp_np) < 1e-10
    else:
        assert np.max(rp_np) > 0


# ---------------------------------------------------------------------------
# Phase 1: Reference sanity tests (PyMUST produces valid output)
# ---------------------------------------------------------------------------


class TestPyMUSTReference:
    """Validate that PyMUST reference data is sane."""

    def test_shape_and_positivity(self, reference: ReferenceData):
        """Reference output has expected shape and all-positive values."""
        assert reference.rp.shape == (GRID_SIZE, GRID_SIZE)
        assert np.all(reference.rp >= 0), "RMS pressure must be non-negative"
        assert np.max(reference.rp) > 0, "Pressure field should not be all zeros"

    @pytest.mark.parametrize("reference", ["P4-2v-focused", "C5-2v-focused"], indirect=True)
    def test_peak_near_focus(self, reference: ReferenceData):
        """Peak pressure should be near the focal point (focused beams only)."""
        assert reference.focus is not None
        x0, z0 = reference.focus
        peak_idx = np.unravel_index(np.argmax(reference.rp), reference.rp.shape)
        x_peak = reference.x_grid[peak_idx]
        z_peak = reference.z_grid[peak_idx]
        dist = np.sqrt((x_peak - x0) ** 2 + (z_peak - z0) ** 2)
        assert dist < 0.01, f"Peak at ({x_peak:.4f}, {z_peak:.4f}), focus at ({x0}, {z0}), dist={dist:.4f}"


# ---------------------------------------------------------------------------
# Comparison tests (FastSIMUS pfield vs. PyMUST reference)
# ---------------------------------------------------------------------------


def _fastsimus_pfield(
    preset_fn,
    delays: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Call FastSIMUS pfield with a preset and return RP.

    Args:
        preset_fn: Transducer preset callable (e.g., P4_2v, L11_5v).
        delays: Transmit delays. Shape (1, n_elements) or (n_elements,).
        positions: Grid positions. Shape (*grid_shape, 2).

    Returns:
        RMS pressure field with shape (*grid_shape,).
    """
    params = preset_fn()
    # Convert to array-api-strict for FastSIMUS call
    delays_strict = xp.asarray(np.asarray(delays))
    delays_1d = xp.reshape(delays_strict, (-1,))
    positions_strict = xp.asarray(np.asarray(positions))
    # Call pfield with array-api-strict arrays
    result = pfield(positions_strict, delays_1d, params)
    # Convert back to numpy for comparison
    return np.asarray(result)


# ---------------------------------------------------------------------------
# Tolerance for PyMUST comparison
# ---------------------------------------------------------------------------
#
# PyMUST uses float32/complex64 precision throughout its pfield computation
# (coordinates, distances, propagation exponentials are all explicitly cast).
# FastSIMUS uses backend default float/complex (float64 NumPy, float32 JAX).
#
# When PyMUST and FastSIMUS both use float64, the two implementations agree.
# to -234 dB (machine epsilon) -- confirming zero algorithmic differences.
# The entire error budget is therefore set by PyMUST's float32 quantization:
#
#   Probe       Error (dB)   Error (frac of peak)
#   C5-2v       -53.9        2.0e-3       (convex geometry, largest kw*r)
#   L11-5v      -68.9        3.6e-4
#   P4-2v       -75.6        1.7e-4
#
# We use a peak-normalized absolute tolerance (atol_peak) instead of rtol
# because rtol penalises sidelobe regions where pressure is << peak but
# float32 noise is a fixed absolute floor. The tolerance of 2.5e-3 (-52 dB)
# is the tightest achievable against a float32 reference.
_PYMUST_ATOL_PEAK = 2.5e-3  # -52 dB re peak; limited by PyMUST's float32


def _compute_peak_normalized_error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """Compute peak-normalized max error and report in dB.

    Args:
        actual: Computed pressure field.
        expected: Reference pressure field.

    Returns:
        Tuple of (max_error_fraction, max_error_db) where:
        - max_error_fraction is max(|actual - expected|) / peak
        - max_error_db is 20*log10(max_error_fraction)
    """
    peak = max(np.max(np.abs(actual)), np.max(np.abs(expected)))
    if peak == 0:
        return 0.0, -np.inf
    max_error = np.max(np.abs(actual - expected))
    max_error_fraction = max_error / peak
    max_error_db = 20 * np.log10(max_error_fraction) if max_error_fraction > 0 else -np.inf
    return max_error_fraction, max_error_db


def _assert_pfield_close(actual: np.ndarray, expected: np.ndarray, atol_peak: float = 1e-3, desc: str = "") -> None:
    """Assert pressure fields match using peak-normalized metric.

    Compares fields as fraction of peak value, so atol_peak=1e-3 means
    all differences are within -60dB of the peak pressure. This is more
    appropriate than rtol for pressure fields where sidelobe levels can
    be 1/1000 of the peak.

    Args:
        actual: Computed pressure field.
        expected: Reference pressure field.
        atol_peak: Absolute tolerance as fraction of peak (default: 1e-3 = -60dB).
        desc: Description for error message.
    """
    peak = max(np.max(np.abs(actual)), np.max(np.abs(expected)))
    if peak == 0:
        return

    max_error_fraction, max_error_db = _compute_peak_normalized_error(actual, expected)

    # Print diagnostic info
    if desc:
        print(f"\n{desc}: max error = {max_error_db:.1f} dB ({max_error_fraction:.2e} of peak)")

    # Normalize and compare
    np.testing.assert_allclose(
        actual / peak,
        expected / peak,
        atol=atol_peak,
        rtol=0,
        err_msg=f"{desc}: max error {max_error_db:.1f} dB exceeds tolerance {20 * np.log10(atol_peak):.1f} dB",
    )


class TestPfieldMatchesPyMUST:
    """Compare FastSIMUS pfield output against PyMUST reference."""

    def test_matches_pymust(self, reference: ReferenceData):
        """FastSIMUS pfield must match PyMUST reference."""
        preset_fn = _preset_for_probe(reference.probe)
        fastsimus_rp = _fastsimus_pfield(preset_fn, reference.delays, reference.positions)
        _assert_pfield_close(fastsimus_rp, reference.rp, atol_peak=_PYMUST_ATOL_PEAK, desc=reference.probe)

    def test_full_frequency_directivity(self, reference: ReferenceData):
        """Full frequency directivity path should give similar results to center-frequency only."""
        params = _preset_for_probe(reference.probe)()
        delays_strict = xp.asarray(np.asarray(reference.delays))
        delays_1d = xp.reshape(delays_strict, (-1,))
        positions_strict = xp.asarray(np.asarray(reference.positions))

        rp_default = pfield(positions_strict, delays_1d, params, full_frequency_directivity=False)
        rp_full_freq = pfield(positions_strict, delays_1d, params, full_frequency_directivity=True)

        rp_default_np = np.asarray(rp_default)
        rp_full_freq_np = np.asarray(rp_full_freq)

        _assert_pfield_close(
            rp_full_freq_np,
            rp_default_np,
            atol_peak=0.05,
            desc=f"{reference.probe} full_frequency_directivity",
        )


class TestPfieldEdgeCases:
    """Edge-case tests for pfield."""

    def test_single_element_array(self):
        """Single-element array produces valid output."""
        params = TransducerParams(freq_center=3e6, pitch=1e-3, n_elements=1, width=1e-3)
        positions = _make_positions((-2e-2, 2e-2), (1e-2, 5e-2), n=20)
        rp = pfield(xp.asarray(positions), xp.asarray(np.zeros(1)), params)
        _assert_valid_pfield_output(rp, positions.shape[:-1])

    def test_all_nan_delays_zero_apodization(self):
        """All-NaN delays yield zero apodization and valid (near-zero) field."""
        params = L11_5v()
        positions = _make_positions((-1e-2, 1e-2), (params.pitch, 2e-2), n=10)
        delays = np.full(params.n_elements, np.nan)
        rp = pfield(xp.asarray(positions), xp.asarray(delays), params)
        _assert_valid_pfield_output(rp, positions.shape[:-1], expect_zero=True)

    def test_custom_baffle_float(self):
        """Custom baffle (float impedance ratio) produces valid output."""
        params = P4_2v().model_copy(update={"baffle": 1.75})
        positions = _make_positions((-2e-2, 2e-2), (params.pitch, 3e-2), n=15)
        rp = pfield(xp.asarray(positions), xp.asarray(np.zeros(params.n_elements)), params)
        _assert_valid_pfield_output(rp, positions.shape[:-1])

    @pytest.mark.parametrize(
        "pfield_kwargs",
        [
            {"element_splitting": 1},
            {"frequency_step": 2.0},
            {"tx_n_wavelengths": 10.0},
        ],
        ids=["element_splitting=1", "frequency_step=2", "tx_n_wavelengths=10"],
    )
    def test_pfield_options_produce_valid_output(self, pfield_kwargs):
        """Pfield produces valid output with various option overrides."""
        params = P4_2v()
        positions = _make_positions((-2e-2, 2e-2), (params.pitch, 3e-2), n=15)
        rp = pfield(
            xp.asarray(positions),
            xp.asarray(np.zeros(params.n_elements)),
            params,
            **pfield_kwargs,
        )
        _assert_valid_pfield_output(rp, positions.shape[:-1])

    def test_1d_positions_input(self):
        """1D positions (single line) produces valid output."""
        params = P4_2v()
        positions = np.stack(
            [np.zeros(30), np.linspace(params.pitch, 5e-2, 30)],
            axis=-1,
        )
        rp = pfield(xp.asarray(positions), xp.asarray(np.zeros(params.n_elements)), params)
        _assert_valid_pfield_output(rp, (30,))
