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
    index = int(np.argmin(np.abs(simulation.times - 2 * 0.017 / 1540)))
    viewer.update(index)
    time = simulation.times[index]
    np.testing.assert_allclose(viewer.received.cursor.data[:, 1], time * 1e6)
    expected = np.array([np.interp(time, simulation.rf_times, channel) for channel in simulation.rf.T])
    expected /= np.max(np.abs(simulation.rf))
    np.testing.assert_allclose(viewer.received.signal.data[:, 1], expected, atol=1e-6)
    viewer.configure("total", True, 20, False)
    np.testing.assert_allclose(viewer.received.signal.data[:, 1], expected, atol=1e-6)
    canvas.draw()
    assert viewer.figure.export_numpy().shape[-1] == 4
    viewer.close()
    notebook_canvas = NotebookCanvas()
    notebook_canvas.close()
    notebook_canvas.close()
    assert notebook_canvas.get_closed()
