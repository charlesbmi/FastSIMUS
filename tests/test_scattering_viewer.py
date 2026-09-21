"""Data-contract tests for the interactive scattering viewer."""

import numpy as np

from examples._scattering_viewer import (
    ScatteringFigure,
    crop_receive_to_field_window,
    db_visibility,
    prepare_viewer_data,
    reflectivity_legend,
    reflectivity_strength,
    signed_db,
    time_index_for_time,
)


def test_receive_data_is_cropped_to_selectable_field_time() -> None:
    """The RF strip ends where the wavefield cursor can still move."""
    receive = np.arange(15).reshape(5, 3)
    receive_times = np.arange(5, dtype=float) * 1e-6
    field_times = np.asarray([0.0, 1e-6, 2.2e-6])

    cropped, cropped_times = crop_receive_to_field_window(receive, receive_times, field_times)

    np.testing.assert_array_equal(cropped, receive[:3])
    np.testing.assert_array_equal(cropped_times, receive_times[:3])


def test_receive_crop_preserves_one_sample_for_short_field_window() -> None:
    """A valid RF strip remains available when its first sample exceeds field time."""
    receive = np.arange(6).reshape(3, 2)
    receive_times = np.asarray([1e-6, 2e-6, 3e-6])

    cropped, cropped_times = crop_receive_to_field_window(receive, receive_times, np.asarray([0.0]))

    np.testing.assert_array_equal(cropped, receive[:1])
    np.testing.assert_array_equal(cropped_times, receive_times[:1])


def test_viewer_data_uses_separate_field_and_receive_references() -> None:
    """Field components share incident scaling while RF uses its own scale."""
    incident = np.asarray([[[0.0, 2.0], [-4.0, 0.0]]])
    scattered = np.asarray([[[0.0, 1.0], [2.0, 0.0]]])
    rms = np.asarray([[1.0, 2.0]])
    rf = np.asarray([[0.0, -20.0], [10.0, 0.0]])

    data = prepare_viewer_data(incident, scattered, rms, rf)

    assert data.field_shape == (1, 2, 2)
    assert data.rf_shape == (2, 2)
    assert data.field_reference == 4.0
    assert data.receive_reference == 20.0
    np.testing.assert_allclose(data.incident, incident / 4.0)
    np.testing.assert_allclose(data.scattered, scattered / 4.0)
    np.testing.assert_allclose(data.receive, rf / 20.0)
    np.testing.assert_allclose(data.incident_rms, rms / 2.0)


def test_viewer_data_handles_empty_and_nonfinite_receive_data() -> None:
    """An empty or invalid receive result still produces a finite strip."""
    incident = np.zeros((2, 3, 4))
    scattered = np.zeros_like(incident)
    rms = np.zeros((2, 3))
    rf = np.asarray([[np.nan, np.inf]])

    data = prepare_viewer_data(incident, scattered, rms, rf)

    assert data.field_reference == 1.0
    assert data.receive_reference == 1.0
    assert np.all(np.isfinite(data.receive))
    assert np.all(data.receive == 0.0)

    empty = prepare_viewer_data(incident, scattered, rms, np.empty((0, 0)))
    assert empty.receive.shape == (0, 0)
    assert empty.receive_reference == 1.0


def test_time_index_uses_absolute_time() -> None:
    """Receive-cursor time selects the nearest wavefield frame."""
    times = np.asarray([0.0, 2e-6, 4e-6, 6e-6])

    assert time_index_for_time(times, 3.1e-6) == 2
    assert time_index_for_time(times, -1.0) == 0
    assert time_index_for_time(times, 1.0) == 3


def test_anyplotlib_figure_preserves_shapes_and_physical_aspect() -> None:
    """The shared plotting framework receives the expected field and RF data."""
    data = prepare_viewer_data(
        np.zeros((2, 3, 4)),
        np.ones((2, 3, 4)),
        np.ones((2, 3)),
        np.ones((5, 2)),
    )
    viewer = ScatteringFigure.from_data(
        data,
        field_times=np.linspace(0.0, 3e-6, 4),
        receive_times=np.linspace(0.0, 4e-6, 5),
        extent=(-20.0, 20.0, 55.0, 5.0),
        elements=np.zeros((2, 2)),
        scatterers=np.zeros((0, 2)),
        coefficients=np.zeros(0),
        time_index=2,
    )

    assert viewer.data.field_shape == data.field_shape
    assert viewer.data.rf_shape == data.rf_shape
    assert viewer.time_index == 2
    assert viewer.field_aspect == 40.0 / 50.0
    assert viewer.receive_aspect > 0.0


def test_anyplotlib_cursor_updates_the_field_frame() -> None:
    """Dragging the RF cursor selects the nearest absolute field time."""
    data = prepare_viewer_data(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        np.zeros((2, 3, 4)),
        np.ones((2, 3)),
        np.ones((5, 2)),
    )
    viewer = ScatteringFigure.from_data(
        data,
        field_times=np.asarray([0.0, 2e-6, 4e-6, 6e-6]),
        receive_times=np.asarray([0.0, 1e-6, 3e-6, 5e-6, 7e-6]),
        extent=(-1.0, 1.0, 0.0, 2.0),
        elements=np.asarray([[-0.5, 0.0], [0.5, 0.0]]),
        scatterers=np.empty((0, 2)),
        coefficients=np.empty(0),
        time_index=2,
    )

    assert viewer.cursor.x == 2.0

    viewer.cursor.set(x=4.0)

    assert viewer.time_index == 3
    np.testing.assert_allclose(viewer.current_frame, signed_db(data.incident[..., 3], 60.0))


def test_viewer_display_defaults_use_range_without_gain() -> None:
    """The viewer exposes one waveform range and a tighter RMS backdrop."""
    data = prepare_viewer_data(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    viewer = ScatteringFigure.from_data(
        data,
        field_times=np.zeros(1),
        receive_times=np.zeros(1),
        extent=(0.0, 1.0, 0.0, 1.0),
        elements=np.zeros((1, 2)),
        scatterers=np.empty((0, 2)),
        coefficients=np.empty(0),
    )

    assert viewer.waveform_dynamic_range == 60.0
    assert viewer.rms_dynamic_range == 20.0
    assert viewer.rms_visible


def test_total_frame_is_derived_from_the_selected_time_slice() -> None:
    """Total display data is the selected incident-plus-scattered frame."""
    incident = np.arange(8, dtype=float).reshape(2, 2, 2)
    scattered = np.full_like(incident, 2.0)
    data = prepare_viewer_data(incident, scattered, np.ones((2, 2)), np.zeros((1, 1)))
    viewer = ScatteringFigure.from_data(
        data,
        field_times=np.asarray([0.0, 1e-6]),
        receive_times=np.zeros(1),
        extent=(0.0, 1.0, 0.0, 1.0),
        elements=np.zeros((1, 2)),
        scatterers=np.empty((0, 2)),
        coefficients=np.empty(0),
        time_index=1,
    )

    expected = signed_db(data.incident[..., 1] + data.scattered[..., 1], 60.0)
    np.testing.assert_allclose(viewer.current_frame, expected)


def test_db_visibility_clips_below_range() -> None:
    """Relative-dB rendering is neutral below range and reaches one at 0 dB."""
    values = np.asarray([0.0, 1e-4, 1e-3, 0.1, 1.0])

    visibility = db_visibility(values, dynamic_range_db=60.0)

    np.testing.assert_allclose(visibility, [0.0, 0.0, 0.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(signed_db(np.asarray([-1.0, -1e-4, 0.0, 0.1, 1.0]), 60.0), [-60, 0, 0, 40, 60])


def test_reflectivity_strength_is_grayscale_magnitude() -> None:
    """Scatterer darkness ignores sign and is monotonic in magnitude."""
    values = np.asarray([-0.02, -0.005, 0.0, 0.005, 0.02])

    strength = reflectivity_strength(values)

    np.testing.assert_allclose(strength, [1.0, 0.25, 0.0, 0.25, 1.0])
    assert "relative reflectivity 0-0.02" in reflectivity_legend(values)
