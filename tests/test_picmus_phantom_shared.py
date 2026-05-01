"""Characterization tests for shared PICMUS phantom helper behavior."""

from __future__ import annotations

import numpy as np
import pytest

from tests._picmus_phantom_shared import (
    DYNAMIC_RANGE_DB,
    _assert_iq_close,
    _image_axes_for_grid_wavelengths,
    _iq_residual_display_db,
    _iq_to_display_db,
    _warn_if_reconstruction_is_large,
)
from tests.test_picmus_phantom import (
    IMAGE_X_M,
    IMAGE_X_MAX_M,
    IMAGE_X_MIN_M,
    IMAGE_Z_M,
    IMAGE_Z_MAX_M,
    IMAGE_Z_MIN_M,
    PICMUS_WAVELENGTH_M,
)

IMAGE_X_BOUNDS_M = (IMAGE_X_MIN_M, IMAGE_X_MAX_M)
IMAGE_Z_BOUNDS_M = (IMAGE_Z_MIN_M, IMAGE_Z_MAX_M)


def _picmus_image_axes_for_grid_wavelengths(grid_wavelengths: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Return resolution phantom image axes through the shared helper."""
    return _image_axes_for_grid_wavelengths(
        grid_wavelengths,
        default_x_axis_m=IMAGE_X_M,
        default_z_axis_m=IMAGE_Z_M,
        x_bounds_m=IMAGE_X_BOUNDS_M,
        z_bounds_m=IMAGE_Z_BOUNDS_M,
        wavelength_m=PICMUS_WAVELENGTH_M,
    )


def test_grid_wavelength_spacing_defaults_to_test_grid() -> None:
    """The optional plot grid defaults to the compact test grid."""
    x_axis_m, z_axis_m = _picmus_image_axes_for_grid_wavelengths(None)

    np.testing.assert_allclose(x_axis_m, IMAGE_X_M)
    np.testing.assert_allclose(z_axis_m, IMAGE_Z_M)


def test_grid_wavelength_spacing_supports_lambda_quarter() -> None:
    """A 0.25 wavelength plot grid beamforms close to lambda/4 sample spacing."""
    spacing_m = 0.25 * PICMUS_WAVELENGTH_M
    x_axis_m, z_axis_m = _picmus_image_axes_for_grid_wavelengths(0.25)

    assert x_axis_m.size == int(np.ceil((IMAGE_X_MAX_M - IMAGE_X_MIN_M) / spacing_m)) + 1
    assert z_axis_m.size == int(np.ceil((IMAGE_Z_MAX_M - IMAGE_Z_MIN_M) / spacing_m)) + 1
    assert np.max(np.diff(x_axis_m)) <= spacing_m
    assert np.max(np.diff(z_axis_m)) <= spacing_m
    assert x_axis_m[0] == pytest.approx(IMAGE_X_MIN_M)
    assert x_axis_m[-1] == pytest.approx(IMAGE_X_MAX_M)
    assert z_axis_m[0] == pytest.approx(IMAGE_Z_MIN_M)
    assert z_axis_m[-1] == pytest.approx(IMAGE_Z_MAX_M)


@pytest.mark.parametrize("grid_wavelengths", [0.0, np.inf, np.nan, -1.0])
def test_grid_wavelength_spacing_rejects_invalid_values(grid_wavelengths: float) -> None:
    """Grid spacing must be finite and positive."""
    with pytest.raises(ValueError, match="grid spacing must be finite and positive"):
        _picmus_image_axes_for_grid_wavelengths(grid_wavelengths)


def test_grid_wavelength_spacing_warns_for_large_grids() -> None:
    """Very fine plot grids warn before expensive DAS reconstruction."""
    with pytest.warns(RuntimeWarning, match="reconstruction will be slow"):
        _picmus_image_axes_for_grid_wavelengths(0.1)


def test_reconstruction_warns_for_large_grid_angle_products() -> None:
    """Fine grids with many firings warn before expensive DAS reconstruction."""
    with pytest.warns(RuntimeWarning, match="DAS reconstruction will be slow"):
        _warn_if_reconstruction_is_large(n_pixels=315_000, n_firings=75)


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
