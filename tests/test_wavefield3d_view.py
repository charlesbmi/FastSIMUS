"""Physical plot transforms and optional fastplotlib resource lifecycle."""

import numpy as np
import pytest


def test_physical_planes_and_canvas_close(monkeypatch):
    """Image insertion must not replace physical coordinates with drawing layers."""
    monkeypatch.setenv("RENDERCANVAS_FORCE_OFFSCREEN", "1")
    pytest.importorskip("fastplotlib")
    from rendercanvas.offscreen import RenderCanvas

    from examples._wavefield3d import simulate
    from examples._wavefield3d_view import NotebookCanvas, WavefieldViewer

    simulation = simulate(
        dict(scene="Point", count=2, backend="NumPy", transmit="Plane wave", steer=0, focus=18, side=2, smoke=True)
    )
    canvas = RenderCanvas(size=(400, 300))
    viewer = WavefieldViewer(simulation, canvas=canvas)
    for graphic, indices in zip(viewer.planes, simulation.slices.indices, strict=True):
        row, column = np.array(indices.shape) // 2
        position = graphic.world_object.world.matrix @ np.array([column, row, 0, 1])
        expected = simulation.slices.points[indices[row, column]] * 1000
        np.testing.assert_allclose(position[:3], expected, atol=1e-5)
    viewer.configure("scattered", False, 10, True)
    index = int(np.argmin(np.abs(simulation.times - 2 * simulation.scatterers[0, 2] / simulation.sound_speed)))
    viewer.update(index)
    time = simulation.times[index]
    for cursor in viewer.received.cursors:
        np.testing.assert_allclose(cursor.data[:, 1], time * 1e6)
    before = [np.asarray(image.data[:]).copy() for image in viewer.received.images]
    viewer.configure("total", True, 20, False)
    for image, expected in zip(viewer.received.images, before, strict=True):
        np.testing.assert_array_equal(image.data[:], expected)
    canvas.draw()
    assert viewer.figure.export_numpy().shape[-1] == 4
    viewer.close()
    notebook_canvas = NotebookCanvas()
    notebook_canvas.close()
    notebook_canvas.close()
    assert notebook_canvas.get_closed()


@pytest.mark.parametrize("sound_speed", [1480.0, 1540.0])
def test_matrix_rf_slices_and_time_scaling(monkeypatch, sound_speed):
    """Non-square channel ordering, physical tick labels and equal projected time scales."""
    from types import SimpleNamespace

    monkeypatch.setenv("RENDERCANVAS_FORCE_OFFSCREEN", "1")
    fpl = pytest.importorskip("fastplotlib")
    from rendercanvas.offscreen import RenderCanvas

    from examples._wavefield3d_rf import ReceivedRFView

    nx, ny = 3, 4
    x, y = np.arange(nx) * 0.0003 - 0.0003, np.arange(ny) * 0.0005 - 0.00075
    xx, yy = np.meshgrid(x, y)
    rf = np.arange(5 * nx * ny, dtype=np.float32).reshape(5, ny, nx) - 25
    original = rf.copy()
    sim = SimpleNamespace(
        matrix_shape=(nx, ny),
        sound_speed=sound_speed,
        rf=rf.reshape(5, -1),
        rf_times=np.arange(5) * 1e-6,
        elements=np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)]),
    )
    canvas = RenderCanvas(size=(800, 400))
    figure = fpl.Figure(shape=(1, 2), canvas=canvas, controller_ids="sync")
    view = ReceivedRFView(figure[0, 0], figure[0, 1], sim)
    figure.show()
    view.fit()
    assert figure[0, 0].controller is figure[0, 1].controller
    assert (view.row, view.column) == (1, 1)
    view.select(3, 0)
    canvas.draw()
    canvas.draw()  # Axes margins settle after the first text layout.
    peak = np.max(np.abs(rf))
    np.testing.assert_array_equal(view.images[0].data[:], rf[:, 3, :] / peak)
    np.testing.assert_array_equal(view.images[1].data[:], rf[:, :, 0] / peak)
    for axis, image, panel in zip((x, y), view.images, figure, strict=True):
        for i, position in enumerate(axis):
            world = image.world_object.world.matrix @ [i, 2, 0, 1]
            np.testing.assert_allclose(world[:2], [position / sound_speed * 1e6, 2], atol=1e-6)
        assert float(panel.axes.x.tick_format(1, -1, 1)) == sound_speed * 1e-3
        matrix = panel.camera.camera_matrix
        width, height = panel.viewport.logical_size
        np.testing.assert_allclose(abs(matrix[0, 0]) * width, abs(matrix[1, 1]) * height, rtol=1e-6)
    view.update(2.5e-6)
    for cursor in view.cursors:
        np.testing.assert_allclose(cursor.data[:, 1], 2.5)
    for row, column in [(-1, 0), (ny, 0), (0, nx)]:
        with pytest.raises(ValueError, match="index"):
            view.select(row, column)
    np.testing.assert_array_equal(rf, original)
    canvas.draw()
    canvas.close()
