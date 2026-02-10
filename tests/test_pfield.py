"""Tests for pfield (pressure field) computation.

Reference tests compare FastSIMUS pfield against PyMUST's pfield output.
Tests are structured as invariants that must hold at every refactoring step.
"""

import numpy as np
import pytest

from fast_simus.pfield import pfield
from fast_simus.transducer_presets import C5_2v, L11_5v, P4_2v

# PyMUST may not be available (Python 3.14 syntax errors)
try:
    from pymust import getparam, txdelayFocused, txdelayPlane
    from pymust import pfield as pymust_pfield

    PYMUST_AVAILABLE = True
except (SyntaxError, ImportError):
    PYMUST_AVAILABLE = False

requires_pymust = pytest.mark.skipif(
    not PYMUST_AVAILABLE, reason="PyMUST not available"
)

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


def _pymust_reference(
    probe_name: str,
    delays: np.ndarray,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    n: int = GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference pressure field via PyMUST.

    Returns:
        Tuple of (RP, x_grid, z_grid).
        RP has shape (n, n), x_grid and z_grid have shape (n, n).
    """
    param = getparam(probe_name)
    x_grid, z_grid = _make_grid(x_range, z_range, n)
    rp, _spect, _idx = pymust_pfield(x_grid, [], z_grid, delays, param)
    return rp, x_grid, z_grid


# ---------------------------------------------------------------------------
# Reference data fixtures (computed on-the-fly via PyMUST)
# ---------------------------------------------------------------------------


@pytest.fixture()
def p4_2v_focused_reference():
    """P4-2v phased array, focused beam at (2cm, 5cm)."""
    param = getparam("P4-2v")
    x0, z0 = 0.02, 0.05
    delays = txdelayFocused(param, x0, z0)
    rp, x_grid, z_grid = _pymust_reference(
        "P4-2v", delays, (-4e-2, 4e-2), (param.pitch, 10e-2)
    )
    return {
        "rp": rp,
        "x_grid": x_grid,
        "z_grid": z_grid,
        "delays": delays,
        "focus": (x0, z0),
        "probe": "P4-2v",
    }


@pytest.fixture()
def l11_5v_plane_reference():
    """L11-5v linear array, plane wave at 10 degrees."""
    param = getparam("L11-5v")
    tilt_rad = np.deg2rad(10.0)
    delays = txdelayPlane(param, tilt_rad)
    rp, x_grid, z_grid = _pymust_reference(
        "L11-5v", delays, (-2e-2, 2e-2), (param.pitch, 4e-2)
    )
    return {
        "rp": rp,
        "x_grid": x_grid,
        "z_grid": z_grid,
        "delays": delays,
        "tilt_rad": tilt_rad,
        "probe": "L11-5v",
    }


@pytest.fixture()
def c5_2v_focused_reference():
    """C5-2v convex array, focused beam at (0, 6cm)."""
    param = getparam("C5-2v")
    x0, z0 = 0.0, 0.06
    delays = txdelayFocused(param, x0, z0)
    rp, x_grid, z_grid = _pymust_reference(
        "C5-2v", delays, (-4e-2, 4e-2), (param.pitch, 10e-2)
    )
    return {
        "rp": rp,
        "x_grid": x_grid,
        "z_grid": z_grid,
        "delays": delays,
        "focus": (x0, z0),
        "probe": "C5-2v",
    }


# ---------------------------------------------------------------------------
# Phase 1: Reference sanity tests (PyMUST produces valid output)
# ---------------------------------------------------------------------------


@requires_pymust
class TestPyMUSTReference:
    """Validate that PyMUST reference data is sane."""

    def test_p4_2v_focused_shape_and_positivity(self, p4_2v_focused_reference):
        """P4-2v focused: output has expected shape and all-positive values."""
        rp = p4_2v_focused_reference["rp"]
        assert rp.shape == (GRID_SIZE, GRID_SIZE)
        assert np.all(rp >= 0), "RMS pressure must be non-negative"
        assert np.max(rp) > 0, "Pressure field should not be all zeros"

    def test_p4_2v_focused_peak_near_focus(self, p4_2v_focused_reference):
        """P4-2v focused: peak pressure should be near the focal point."""
        ref = p4_2v_focused_reference
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

    def test_l11_5v_plane_shape_and_positivity(self, l11_5v_plane_reference):
        """L11-5v plane wave: output has expected shape."""
        rp = l11_5v_plane_reference["rp"]
        assert rp.shape == (GRID_SIZE, GRID_SIZE)
        assert np.all(rp >= 0)
        assert np.max(rp) > 0

    def test_c5_2v_focused_shape_and_positivity(self, c5_2v_focused_reference):
        """C5-2v convex focused: output has expected shape."""
        rp = c5_2v_focused_reference["rp"]
        assert rp.shape == (GRID_SIZE, GRID_SIZE)
        assert np.all(rp >= 0)
        assert np.max(rp) > 0

    def test_c5_2v_focused_peak_near_focus(self, c5_2v_focused_reference):
        """C5-2v focused: peak should be near the focal point."""
        ref = c5_2v_focused_reference
        rp = ref["rp"]
        x_grid, z_grid = ref["x_grid"], ref["z_grid"]
        x0, z0 = ref["focus"]

        peak_idx = np.unravel_index(np.argmax(rp), rp.shape)
        x_peak = x_grid[peak_idx]
        z_peak = z_grid[peak_idx]

        dist = np.sqrt((x_peak - x0) ** 2 + (z_peak - z0) ** 2)
        assert dist < 0.01, f"Peak at ({x_peak:.4f}, {z_peak:.4f}), focus at ({x0}, {z0}), dist={dist:.4f}"


# ---------------------------------------------------------------------------
# Comparison tests (FastSIMUS pfield vs. PyMUST reference)
# ---------------------------------------------------------------------------


def _fastsimus_pfield(
    preset_fn,
    delays: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    """Call FastSIMUS pfield with a preset and return RP.

    Handles the translation from PyMUST delay format (1, n_elements)
    to FastSIMUS delay format (n_elements,).
    """
    params = preset_fn()
    delays_1d = np.squeeze(delays)
    return pfield(x_grid, z_grid, delays_1d, params)


# Tolerance for PyMUST comparison.
# PyMUST uses complex64 for the recursive exponential multiplication in its
# frequency loop, so small float32 rounding differences accumulate over
# hundreds of iterations. Our implementation uses the same precision chain but
# minor code-path differences (Array API vs raw NumPy) introduce ~5e-4 max
# relative error at edge points near the transducer. The 99th-percentile
# relative error is <1e-4 and mean is ~1e-5.
_PYMUST_RTOL = 1e-3


@requires_pymust
class TestPfieldMatchesPyMUST:
    """Compare FastSIMUS pfield output against PyMUST reference."""

    def test_p4_2v_focused_matches(self, p4_2v_focused_reference):
        """P4-2v focused beam must match PyMUST."""
        ref = p4_2v_focused_reference
        our_rp = _fastsimus_pfield(P4_2v, ref["delays"], ref["x_grid"], ref["z_grid"])
        np.testing.assert_allclose(our_rp, ref["rp"], rtol=_PYMUST_RTOL, atol=1e-10)

    def test_l11_5v_plane_matches(self, l11_5v_plane_reference):
        """L11-5v plane wave must match PyMUST."""
        ref = l11_5v_plane_reference
        our_rp = _fastsimus_pfield(L11_5v, ref["delays"], ref["x_grid"], ref["z_grid"])
        np.testing.assert_allclose(our_rp, ref["rp"], rtol=_PYMUST_RTOL, atol=1e-10)

    def test_c5_2v_focused_matches(self, c5_2v_focused_reference):
        """C5-2v convex focused beam must match PyMUST."""
        ref = c5_2v_focused_reference
        our_rp = _fastsimus_pfield(C5_2v, ref["delays"], ref["x_grid"], ref["z_grid"])
        np.testing.assert_allclose(our_rp, ref["rp"], rtol=_PYMUST_RTOL, atol=1e-10)
