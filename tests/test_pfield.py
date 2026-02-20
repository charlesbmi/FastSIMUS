"""Tests for pfield (pressure field) computation.

Reference tests compare FastSIMUS pfield against PyMUST's pfield output.
Tests are structured as invariants that must hold at every refactoring step.
"""

from typing import Any, cast

import array_api_strict
import numpy as np
import pymust
import pytest

from fast_simus.pfield import pfield
from fast_simus.transducer_presets import C5_2v, L11_5v, P4_2v
from fast_simus.utils._array_api import _ArrayNamespace

# Array API backend for FastSIMUS calls
xp = cast(_ArrayNamespace, array_api_strict)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID_SIZE = 50


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
) -> dict[str, Any]:
    """Compute reference pressure field via PyMUST.

    Returns:
        Dictionary with keys:
        - 'rp': RMS pressure field, shape (n, n)
        - 'x_grid': x coordinates, shape (n, n)
        - 'z_grid': z coordinates, shape (n, n)
        - 'positions': positions array, shape (n, n, 2)
    """
    param = pymust.getparam(probe_name)
    x_grid, z_grid = _make_grid(x_range, z_range, n)
    positions = np.stack([x_grid, z_grid], axis=-1)
    rp, _spect, _idx = pymust.pfield(x_grid, [], z_grid, delays, param)  # type: ignore[arg-type]
    return {"rp": rp, "x_grid": x_grid, "z_grid": z_grid, "positions": positions}


# ---------------------------------------------------------------------------
# Reference data fixtures (computed on-the-fly via PyMUST)
# ---------------------------------------------------------------------------


@pytest.fixture()
def p4_2v_focused_reference() -> dict[str, Any]:
    """P4-2v phased array, focused beam at (2cm, 5cm)."""
    param = pymust.getparam("P4-2v")
    x0, z0 = 0.02, 0.05
    delays = pymust.txdelayFocused(param, x0, z0)
    ref = _pymust_reference("P4-2v", delays, (-4e-2, 4e-2), (param.pitch, 10e-2))  # type: ignore[arg-type]
    ref["delays"] = delays
    ref["focus"] = (x0, z0)
    ref["probe"] = "P4-2v"
    return ref


@pytest.fixture()
def l11_5v_plane_reference() -> dict[str, Any]:
    """L11-5v linear array, plane wave at 10 degrees."""
    param = pymust.getparam("L11-5v")
    tilt_rad = np.deg2rad(10.0)
    delays = pymust.txdelayPlane(param, tilt_rad)
    ref = _pymust_reference("L11-5v", delays, (-2e-2, 2e-2), (param.pitch, 4e-2))  # type: ignore[arg-type]
    ref["delays"] = delays
    ref["tilt_rad"] = tilt_rad
    ref["probe"] = "L11-5v"
    return ref


@pytest.fixture()
def c5_2v_focused_reference() -> dict[str, Any]:
    """C5-2v convex array, focused beam at (0, 6cm)."""
    param = pymust.getparam("C5-2v")
    x0, z0 = 0.0, 0.06
    delays = pymust.txdelayFocused(param, x0, z0)
    ref = _pymust_reference("C5-2v", delays, (-4e-2, 4e-2), (param.pitch, 10e-2))  # type: ignore[arg-type]
    ref["delays"] = delays
    ref["focus"] = (x0, z0)
    ref["probe"] = "C5-2v"
    return ref


# ---------------------------------------------------------------------------
# Phase 1: Reference sanity tests (PyMUST produces valid output)
# ---------------------------------------------------------------------------


class TestPyMUSTReference:
    """Validate that PyMUST reference data is sane."""

    @pytest.mark.parametrize(
        "ref_fixture",
        ["p4_2v_focused_reference", "l11_5v_plane_reference", "c5_2v_focused_reference"],
    )
    def test_shape_and_positivity(self, ref_fixture, request):
        """Reference output has expected shape and all-positive values."""
        ref = request.getfixturevalue(ref_fixture)
        rp = ref["rp"]
        assert rp.shape == (GRID_SIZE, GRID_SIZE)
        assert np.all(rp >= 0), "RMS pressure must be non-negative"
        assert np.max(rp) > 0, "Pressure field should not be all zeros"

    @pytest.mark.parametrize(
        "ref_fixture",
        ["p4_2v_focused_reference", "c5_2v_focused_reference"],
    )
    def test_peak_near_focus(self, ref_fixture, request):
        """Peak pressure should be near the focal point (focused beams only)."""
        ref = request.getfixturevalue(ref_fixture)
        rp = ref["rp"]
        x_grid, z_grid = ref["x_grid"], ref["z_grid"]
        x0, z0 = ref["focus"]

        # Find peak location
        peak_idx = np.unravel_index(np.argmax(rp), rp.shape)
        x_peak = x_grid[peak_idx]
        z_peak = z_grid[peak_idx]

        # Peak should be within 1cm of the intended focus
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
# FastSIMUS uses float64/complex128, which is strictly more accurate.
#
# When PyMUST is monkey-patched to use float64, the two implementations agree
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

    @pytest.mark.parametrize(
        ("preset_fn", "ref_fixture", "desc"),
        [
            (P4_2v, "p4_2v_focused_reference", "P4-2v focused"),
            (L11_5v, "l11_5v_plane_reference", "L11-5v plane"),
            (C5_2v, "c5_2v_focused_reference", "C5-2v focused"),
        ],
    )
    def test_matches_pymust(self, preset_fn, ref_fixture, desc, request):
        """FastSIMUS pfield must match PyMUST reference."""
        ref = request.getfixturevalue(ref_fixture)
        fastsimus_rp = _fastsimus_pfield(preset_fn, ref["delays"], ref["positions"])
        _assert_pfield_close(fastsimus_rp, ref["rp"], atol_peak=_PYMUST_ATOL_PEAK, desc=desc)

    @pytest.mark.parametrize(
        ("preset_fn", "ref_fixture", "desc"),
        [
            (P4_2v, "p4_2v_focused_reference", "P4-2v focused"),
            (L11_5v, "l11_5v_plane_reference", "L11-5v plane"),
            (C5_2v, "c5_2v_focused_reference", "C5-2v focused"),
        ],
    )
    def test_full_frequency_directivity(self, preset_fn, ref_fixture, desc, request):
        """Full frequency directivity path should give similar results to center-frequency only."""
        ref = request.getfixturevalue(ref_fixture)
        params = preset_fn()
        # Convert to array-api-strict for FastSIMUS calls
        delays_strict = xp.asarray(np.asarray(ref["delays"]))
        delays_1d = xp.reshape(delays_strict, (-1,))
        positions_strict = xp.asarray(np.asarray(ref["positions"]))

        # Default: center-frequency directivity
        rp_default = pfield(positions_strict, delays_1d, params, full_frequency_directivity=False)

        # Full frequency-dependent directivity
        rp_full_freq = pfield(positions_strict, delays_1d, params, full_frequency_directivity=True)

        # Convert results back to numpy for comparison
        rp_default_np = np.asarray(rp_default)
        rp_full_freq_np = np.asarray(rp_full_freq)

        # Results should be very close (within a few percent)
        # Full frequency directivity is more accurate but the difference is small for typical bandwidths
        _assert_pfield_close(rp_full_freq_np, rp_default_np, atol_peak=0.05, desc=f"{desc} full_frequency_directivity")
