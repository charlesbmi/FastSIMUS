"""Data-contract tests for the interactive scattering viewer."""

import numpy as np

from examples._scattering_viewer import (
    ScatteringViewer,
    crop_receive_to_field_window,
    db_visibility,
    prepare_viewer_data,
    reflectivity_strength,
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


def test_widget_buffers_match_declared_shapes() -> None:
    """Binary widget payloads contain exactly the advertised float32 data."""
    data = prepare_viewer_data(
        np.zeros((2, 3, 4)),
        np.ones((2, 3, 4)),
        np.ones((2, 3)),
        np.ones((5, 2)),
    )
    viewer = ScatteringViewer.from_data(
        data,
        field_times=np.linspace(0.0, 3e-6, 4),
        receive_times=np.linspace(0.0, 4e-6, 5),
        extent=(-20.0, 20.0, 55.0, 5.0),
        elements=np.zeros((2, 2)),
        scatterers=np.zeros((0, 2)),
        coefficients=np.zeros(0),
        time_index=2,
    )

    assert tuple(viewer.field_shape) == data.field_shape
    assert tuple(viewer.receive_shape) == data.rf_shape
    assert len(viewer.incident) == np.prod(data.field_shape) * 4
    assert len(viewer.scattered) == np.prod(data.field_shape) * 4
    assert len(viewer.incident_rms) == 2 * 3 * 4
    assert len(viewer.receive) == 5 * 2 * 4
    assert viewer.time_index == 2


def test_viewer_display_defaults_use_range_without_gain() -> None:
    """The viewer exposes one waveform range and a tighter RMS backdrop."""
    viewer = ScatteringViewer()

    assert viewer.phase_dynamic_range == 60.0
    assert viewer.rms_dynamic_range == 20.0
    assert not viewer.has_trait("field_gain_db")
    assert not viewer.has_trait("receive_gain_db")


def test_db_visibility_clips_below_range() -> None:
    """Relative-dB rendering is neutral below range and reaches one at 0 dB."""
    values = np.asarray([0.0, 1e-4, 1e-3, 0.1, 1.0])

    visibility = db_visibility(values, dynamic_range_db=60.0)

    np.testing.assert_allclose(visibility, [0.0, 0.0, 0.0, 2.0 / 3.0, 1.0])


def test_reflectivity_strength_is_grayscale_magnitude() -> None:
    """Scatterer darkness ignores sign and is monotonic in magnitude."""
    values = np.asarray([-0.02, -0.005, 0.0, 0.005, 0.02])

    strength = reflectivity_strength(values)

    np.testing.assert_allclose(strength, [1.0, 0.25, 0.0, 0.25, 1.0])
