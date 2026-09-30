"""Fastplotlib presentation boundary for physical orthoslices and scene overlays."""

from time import perf_counter

import numpy as np
from anywidget import AnyWidget
from matplotlib import colormaps
from rendercanvas.anywidget import RenderCanvas


class NotebookCanvas(RenderCanvas):
    """Avoid rendercanvas 2.7.2's recursive close-message dispatch."""

    def _rc_close(self):
        self._is_closed = True
        AnyWidget.close(self)


class WavefieldViewer:
    """Persistent scene: display changes update textures without running physics."""

    def __init__(self, simulation, canvas=None):
        import fastplotlib as fpl
        import pygfx

        self.simulation = simulation
        self.index = 0
        self.component = "incident"
        self.magnitude = False
        self.gain = 1.0
        self.playing = False
        self.last_frame = 0.0
        camera = pygfx.PerspectiveCamera(50)
        camera.local.z = -100  # Keep initial rulers away from the camera projection plane.
        self.figure = fpl.Figure(
            rects=[(0, 0, 0.5, 0.4), (0.5, 0, 0.5, 0.4)] + [(i / 4, 0.4, 0.25, 0.6) for i in range(4)],
            cameras=[camera, "2d", "2d", "2d", "2d", "2d"],
            controller_types=["orbit", "panzoom", "panzoom", "panzoom", "panzoom", "panzoom"],
            controller_ids=[[0, 1], [2, 3], [4, 4]],
            canvas=canvas or NotebookCanvas(size=(1100, 900), max_fps=20),
        )
        self.images = []
        self.planes = []
        self.scene = self.figure[0]
        self.scene.title = "Pressure slices (mm)"
        axes = [a * 1000 for a in simulation.slices.axes]
        x, y, z = axes
        cx, cy, cz = np.asarray(simulation.slices.center) * 1000
        origins = [(x[0], y[0], cz), (x[0], cy, z[0]), (cx, y[0], z[0])]
        scales = [(x[1] - x[0], y[1] - y[0], 1), (x[1] - x[0], z[1] - z[0], 1), (y[1] - y[0], z[1] - z[0], 1)]
        # Image local (column,row,normal) axes map to physical XY, XZ and YZ.
        rotations = [(0, 0, 0, 1), (2**-0.5, 0, 0, 2**-0.5), (0.5, 0.5, 0.5, 0.5)]
        labels = ["XY", "XZ", "YZ"]
        for i, indices in enumerate(simulation.slices.indices):
            data = np.zeros(indices.shape, np.float32)
            plane = self.scene.add_image(np.zeros((*data.shape, 4), dtype=np.float32), vmin=0, vmax=1)
            plane.scale = scales[i]
            plane.rotation = rotations[i]
            plane.offset = origins[i]
            self.planes.append(plane)
            subplot = self.figure[i + 1]
            subplot.title = labels[i] + " pressure (mm)"
            image = subplot.add_image(data, cmap="bwr", vmin=-1, vmax=1)
            image.scale = scales[i]
            image.offset = (
                origins[i][0] if i < 2 else origins[i][1],
                origins[i][1] if i == 0 else origins[i][2],
                0,
            )
            self.images.append(image)
        self.probe = self.scene.add_scatter(simulation.elements * 1000, sizes=5, colors="#ffc857")
        points = simulation.scatterers
        self.displayed_scatterers = min(1500, len(points))
        self.cloud = None
        if len(points):
            indices = np.linspace(0, len(points) - 1, self.displayed_scatterers, dtype=int)
            self.cloud = self.scene.add_scatter(points[indices] * 1000, sizes=2, colors=(0.4, 0.75, 0.9, 0.35))
        # Fastplotlib assigns integer z layers when images are added. Apply the
        # physical offsets after all graphics exist so no later insertion resets them.
        for plane, origin in zip(self.planes, origins, strict=True):
            plane.offset = origin
        self.peaks = {
            "incident": max(float(np.max(np.abs(simulation.incident))), 1e-30),
            "scattered": max(float(np.max(np.abs(simulation.scattered))), 1e-30),
            "total": max(float(np.max(np.abs(simulation.incident + simulation.scattered))), 1e-30),
        }
        self.received = ReceivedRFView(self.figure[4], self.figure[5], simulation)
        self.figure.add_animations(self.animate)
        self.widget = self.figure.show()
        self.received.fit()
        self.scene.camera.local.scale_y = 1
        self.scene.camera.show_object(self.scene.scene, view_dir=(-1, -1, -0.7), up=(0, 0, -1))
        self.update(0)

    def configure(self, component, magnitude, gain, geometry):
        """Adjust display only, with one fixed scale per component for every time."""
        self.component, self.magnitude, self.gain = component, magnitude, gain
        self.probe.visible = geometry
        if self.cloud is not None:
            self.cloud.visible = geometry
        for image in self.images:
            image.cmap = "viridis" if magnitude else "bwr"
            image.vmin = 0 if magnitude else -1
            image.vmax = 1
        self.update(self.index)

    def update(self, index):
        """Update six textures using one time sample in seconds."""
        self.index = int(index) % len(self.simulation.times)
        sim = self.simulation
        values = sim.incident[:, self.index] if self.component == "incident" else sim.scattered[:, self.index]
        if self.component == "total":
            values = values + sim.incident[:, self.index]
        if self.magnitude:
            values = np.abs(values)
        values = values * (self.gain / self.peaks[self.component])
        for indices, plane, image in zip(sim.slices.indices, self.planes, self.images, strict=True):
            data = np.ascontiguousarray(values[indices], dtype=np.float32)
            colors = colormaps["viridis" if self.magnitude else "bwr"](
                np.clip(data if self.magnitude else (data + 1) / 2, 0, 1)
            ).astype(np.float32)
            colors[..., 3] = np.clip(np.abs(data) * 4, 0, 0.95)
            plane.data = colors
            image.data = data
        self.received.update(sim.times[self.index])
        self.scene.title = f"{self.component.capitalize()} | {sim.times[self.index] * 1e6:.2f} us"

    def animate(self, *args):
        """Playback skips samples for a short movie while preserving physical time labels."""
        now = perf_counter()
        if self.playing and now - self.last_frame >= 0.08:
            self.update(self.index + max(1, len(self.simulation.times) // 200))
            self.last_frame = now

    def close(self):
        """Release the old renderer when replacing the latest result."""
        self.playing = False
        self.figure.remove_animation(self.animate)
        self.figure.canvas.close()


class ViewerSlot:
    """Own one live figure and release it when a new simulation replaces it."""

    def __init__(self):
        self.viewer = None

    def replace(self, simulation):
        """Close the preceding figure before retaining the new result."""
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = WavefieldViewer(simulation)
        return self.viewer


class ReceivedRFView:
    """Orthogonal matrix RF histories with equal lateral and temporal travel-time scales."""

    def __init__(self, x_panel, y_panel, simulation):
        self.panels = (x_panel, y_panel)
        self.times = simulation.rf_times
        nx, ny = simulation.matrix_shape
        self.sound_speed = simulation.sound_speed
        peak = max(float(np.max(np.abs(simulation.rf))), 1e-30)
        self.rf = np.asarray(simulation.rf / peak, dtype=np.float32).reshape(len(self.times), ny, nx)
        centers = simulation.elements.reshape(ny, nx, 3)
        self.axes = (centers[0, :, 0], centers[:, 0, 1])
        self.images, self.cursors, self.bounds = [], [], []
        for axis, panel in zip(self.axes, self.panels, strict=True):
            positions = axis / self.sound_speed * 1e6
            step = positions[1] - positions[0]
            image = panel.add_image(np.zeros((len(self.times), len(axis)), np.float32), cmap="bwr", vmin=-1, vmax=1)
            image.scale = (step, (self.times[1] - self.times[0]) * 1e6, 1)
            image.offset = (positions[0], self.times[0] * 1e6, 0)
            bounds = (positions[0] - step / 2, positions[-1] + step / 2)
            cursor = panel.add_line(
                np.array([[bounds[0], 0, 1], [bounds[1], 0, 1]], dtype=np.float32),
                colors="#ffc857",
                thickness=2,
            )
            panel.axes.x.tick_format = self.format_mm
            self.images.append(image)
            self.cursors.append(cursor)
            self.bounds.append(bounds)
        self.select((ny - 1) // 2, (nx - 1) // 2)

    def format_mm(self, value, minimum, maximum):
        """Label travel-time coordinates with their physical position in millimeters."""
        return f"{value * self.sound_speed * 1e-3:.4g}"

    def select(self, row, column):
        """Select a Y row and X column without changing normalization or camera state."""
        if not 0 <= row < self.rf.shape[1] or not 0 <= column < self.rf.shape[2]:
            raise ValueError("RF row or column index out of range")
        self.row, self.column = row, column
        self.images[0].data = np.ascontiguousarray(self.rf[:, row, :])
        self.images[1].data = np.ascontiguousarray(self.rf[:, :, column])
        self.panels[0].title = f"RF X (mm), t (us)\nY row {row}: {self.axes[1][row] * 1000:.2f} mm"
        self.panels[1].title = f"RF Y (mm), t (us)\nX column {column}: {self.axes[0][column] * 1000:.2f} mm"

    def fit(self):
        """Show full histories with shared bounds and an undistorted travel-time aspect."""
        left = min(bound[0] for bound in self.bounds)
        right = max(bound[1] for bound in self.bounds)
        dt = (self.times[1] - self.times[0]) * 1e6
        for panel in self.panels:
            panel.camera.maintain_aspect = True
            panel.camera.show_rect(left, right, self.times[0] * 1e6 - dt / 2, self.times[-1] * 1e6 + dt / 2)

    def update(self, time):
        """Move both cursors to the physical playback time in seconds."""
        for cursor in self.cursors:
            cursor.data[:, 1] = time * 1e6
