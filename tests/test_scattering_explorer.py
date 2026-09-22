"""Tests for deterministic scene and coordinate helpers used by the notebook."""

import numpy as np
import pytest

from examples._scattering_explorer import (
    append_drawn_points,
    canvas_to_physical,
    custom_rows_are_dirty,
    drawing_class_coefficients,
    estimate_field_movie_bytes,
    estimate_simulation_workload,
    estimate_spatial_pairs,
    grid_from_spacing,
    normalize_custom_rows,
    physical_to_canvas,
    picmus_point_targets,
    snapshot_custom_rows,
    spacing_in_mm,
    speckle_lesion,
    tukey_apodization,
    wavelength_mm,
)
from examples._scattering_simulation import probe_with_center_frequency
from fast_simus.transducer_presets import P4_2v


def test_picmus_point_targets_match_resolution_phantom() -> None:
    """The preset contains the exact 20 unique resolution targets."""
    positions, rc = picmus_point_targets()
    expected_axial = {(0.0, float(z)) for z in range(10, 50, 5)}
    expected_lateral = {(float(x), float(z)) for z in (20, 40) for x in range(-15, 20, 5)}

    assert positions.shape == (20, 2)
    assert {(float(x), float(z)) for x, z in positions} == expected_axial | expected_lateral
    np.testing.assert_array_equal(rc, np.full(20, 0.005))


def test_canvas_physical_round_trip() -> None:
    """Drawing coordinates round-trip through physical millimetres."""
    extent = (-20.0, 20.0, 5.0, 55.0)
    canvas = np.asarray([[0.0, 0.0], [320.0, 240.0], [640.0, 480.0]])

    physical = canvas_to_physical(canvas, extent, width=640, height=480)
    restored = physical_to_canvas(physical, extent, width=640, height=480)

    np.testing.assert_allclose(restored, canvas)
    np.testing.assert_allclose(physical[[0, -1]], [[-20.0, 5.0], [20.0, 55.0]])


def test_speckle_lesion_is_deterministic_and_hypoechoic() -> None:
    """The seeded scene uses nonnegative Rayleigh amplitudes and a lesion."""
    first_positions, first_rc = speckle_lesion(20_000, seed=7)
    second_positions, second_rc = speckle_lesion(20_000, seed=7)
    radius = np.sqrt(first_positions[:, 0] ** 2 + (first_positions[:, 1] - 35.0) ** 2)

    np.testing.assert_array_equal(first_positions, second_positions)
    np.testing.assert_array_equal(first_rc, second_rc)
    assert np.all(first_rc >= 0.0)
    assert np.any(radius < 7.0)
    assert np.any(radius >= 7.0)
    assert np.mean(first_rc[radius >= 7.0]) == pytest.approx(0.001, rel=0.03)
    assert np.mean(first_rc[radius < 7.0]) == pytest.approx(0.0002, rel=0.08)


def test_drawing_classes_map_to_nonnegative_reflectivity() -> None:
    """Drawdata class labels retain configurable target strengths."""
    values = (0.001, 0.002, 0.005, 0.02)

    coefficients = drawing_class_coefficients(["a", "b", "c", "d", "unknown"], values)

    np.testing.assert_array_equal(coefficients, [0.001, 0.002, 0.005, 0.02, 0.005])


def test_draw_table_draw_sequence_uses_one_canonical_table() -> None:
    """Draw commits append to the latest edited table without stale snapshots."""
    extent = (-20.0, 20.0, 5.0, 55.0)
    first_drawing = [
        {"x": 160.0, "y": 192.0, "label": "c"},
        {"x": 480.0, "y": 288.0, "label": "a"},
    ]
    rows = append_drawn_points([], first_drawing, extent, (0.001, 0.002, 0.005, 0.02), width=640, height=480)
    edited = normalize_custom_rows(
        [
            {"x_mm": rows[0]["x_mm"] + 1.0, "z_mm": rows[0]["z_mm"], "rc": 0.007},
            {"x_mm": 3.0, "z_mm": 42.0, "rc": 0.003},
        ]
    )
    final = append_drawn_points(
        edited,
        [{"x": 320.0, "y": 240.0, "label": "d"}],
        extent,
        (0.001, 0.002, 0.005, 0.02),
        width=640,
        height=480,
    )

    assert len(final) == 3
    assert final[:2] == edited
    assert final[-1] == {"x_mm": 0.0, "z_mm": 30.0, "rc": 0.02}


def test_custom_rows_are_normalized() -> None:
    """Editor values become plain physical-coordinate floats."""
    rows = normalize_custom_rows([{"x_mm": 1, "z_mm": 2, "rc": 0.5}])

    assert rows == [{"x_mm": 1.0, "z_mm": 2.0, "rc": 0.5}]


def test_custom_row_snapshot_is_an_independent_applied_copy() -> None:
    """Applying a draft creates a normalized scene that later edits cannot mutate."""
    draft = [{"x_mm": 1, "z_mm": 2, "rc": 0.5}]

    applied = snapshot_custom_rows(draft)
    draft[0]["x_mm"] = 9

    assert applied == [{"x_mm": 1.0, "z_mm": 2.0, "rc": 0.5}]


def test_custom_row_dirty_state_compares_normalized_scenes() -> None:
    """Dirty state reports only an observable difference from the applied scene."""
    applied = [{"x_mm": 1.0, "z_mm": 2.0, "rc": 0.5}]

    assert not custom_rows_are_dirty([{"x_mm": 1, "z_mm": 2, "rc": 0.5}], applied)
    assert custom_rows_are_dirty([{"x_mm": 2, "z_mm": 2, "rc": 0.5}], applied)
    assert custom_rows_are_dirty([], applied)
    assert not custom_rows_are_dirty([], [])


def test_empty_custom_scene_can_be_applied() -> None:
    """The Custom workflow supports deliberately simulating no scatterers."""
    assert snapshot_custom_rows([]) == []


def test_custom_rows_reject_negative_reflectivity() -> None:
    """The simplified notebook editor retains its last valid nonnegative scene."""
    last_valid = normalize_custom_rows([{"x_mm": 1.0, "z_mm": 2.0, "rc": 0.005}])
    with pytest.raises(ValueError, match="nonnegative"):
        normalize_custom_rows([{"x_mm": 1.0, "z_mm": 2.0, "rc": -0.1}])
    assert last_valid == [{"x_mm": 1.0, "z_mm": 2.0, "rc": 0.005}]
    assert normalize_custom_rows([{"x_mm": 3.0, "z_mm": 4.0, "rc": 0.002}]) == [{"x_mm": 3.0, "z_mm": 4.0, "rc": 0.002}]


def test_wavelength_grid_spacing_includes_roi_endpoints() -> None:
    """Wavelength-derived grids include endpoints and refine with frequency."""
    wavelength = wavelength_mm(1540.0, 2.72)
    spacing = wavelength / 3.0
    grid = grid_from_spacing((-20.0, 20.0), (0.0, 55.0), spacing)
    high_frequency_spacing = wavelength_mm(1540.0, 7.6) / 3.0
    high_frequency_grid = grid_from_spacing((-20.0, 20.0), (0.0, 55.0), high_frequency_spacing)

    assert wavelength == pytest.approx(1540.0 / 2.72e6 * 1e3)
    assert grid.nx == int(np.ceil(40.0 / spacing)) + 1
    assert grid.nz == int(np.ceil(55.0 / spacing)) + 1
    assert grid.dx_mm <= spacing
    assert grid.dz_mm <= spacing
    assert grid.z_min_mm == 0.0
    assert high_frequency_spacing < spacing
    assert high_frequency_grid.nx > grid.nx
    assert spacing_in_mm(1.0 / 3.0, wavelength, wavelength_units=True) == pytest.approx(spacing)
    assert spacing_in_mm(0.25, wavelength_mm(1540.0, 2.72), wavelength_units=False) == 0.25


def test_center_frequency_override_preserves_probe_geometry() -> None:
    """Frequency overrides retain preset geometry and fractional bandwidth."""
    base = P4_2v()
    overridden = probe_with_center_frequency(base, 7.5)

    assert overridden.freq_center == 7.5e6
    assert overridden.pitch == base.pitch
    assert overridden.n_elements == base.n_elements
    assert overridden.width == base.width
    assert overridden.radius == base.radius
    assert overridden.bandwidth == base.bandwidth


def test_tukey_apodization_endpoints_and_symmetry() -> None:
    """Tukey roll spans uniform through Hann without changing aperture size."""
    uniform = tukey_apodization(8, 0.0)
    tapered = tukey_apodization(8, 0.4)
    hann = tukey_apodization(8, 1.0)

    np.testing.assert_array_equal(uniform, np.ones(8))
    np.testing.assert_allclose(hann, np.hanning(8))
    np.testing.assert_allclose(tapered, tapered[::-1])
    assert tapered.shape == (8,)
    assert np.all((tapered >= 0.0) & (tapered <= 1.0))


def test_field_movie_memory_estimate_counts_both_components() -> None:
    """The warning estimate includes incident and scattered frame storage."""
    assert estimate_field_movie_bytes((10, 20), 30, bytes_per_value=8) == 10 * 20 * 30 * 8 * 2
    assert estimate_field_movie_bytes((10, 20), 30, temporal_oversampling=2) == 10 * 20 * 60 * 8 * 2
    assert estimate_spatial_pairs((10, 20), 7) == 1400


def test_simulation_workload_uses_roi_and_scatterer_extent() -> None:
    """Diagnostics share one tested record-length and memory estimate."""
    estimate = estimate_simulation_workload(
        grid_shape=(5, 6),
        x_limits_mm=(-2.0, 2.0),
        z_limits_mm=(0.0, 4.0),
        scatterers_mm=np.asarray([[6.0, 8.0]]),
        n_scatterers=3,
        propagation_speed=1_000.0,
        center_frequency_mhz=2.0,
        pulse_wavelengths=2.0,
        time_oversampling=2,
    )

    assert estimate.effective_dx_mm == 1.0
    assert estimate.effective_dz_mm == 0.8
    assert estimate.spatial_pairs == 5 * 6 * 3
    assert estimate.record_samples >= 168
    assert estimate.field_movie_bytes == 5 * 6 * estimate.record_samples * 2 * 8 * 2
