"""Matrix RF cross-sections in physical travel-time coordinates."""

import numpy as np


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
