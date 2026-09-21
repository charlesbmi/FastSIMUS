"""Data preparation and anyplotlib viewer for the scattering explorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anyplotlib as apl
import numpy as np


@dataclass(frozen=True)
class ViewerData:
    """Finite float32 arrays and independent display references."""

    incident: np.ndarray
    scattered: np.ndarray
    incident_rms: np.ndarray
    receive: np.ndarray
    field_reference: float
    receive_reference: float

    @property
    def field_shape(self) -> tuple[int, int, int]:
        return self.incident.shape

    @property
    def rf_shape(self) -> tuple[int, int]:
        return self.receive.shape


def _finite_reference(values: np.ndarray) -> float:
    finite = _finite_array(values)
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    return peak if peak > 0.0 else 1.0


def _finite_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values), copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_viewer_data(
    incident: np.ndarray,
    scattered: np.ndarray,
    incident_rms: np.ndarray,
    receive: np.ndarray,
) -> ViewerData:
    """Normalize field components together and receive data separately."""
    incident = _finite_array(incident)
    scattered = _finite_array(scattered)
    incident_rms = _finite_array(incident_rms)
    receive = _finite_array(receive)
    if incident.shape != scattered.shape or incident.ndim != 3:
        raise ValueError("Incident and scattered frames must share shape (nz, nx, n_times)")
    if incident_rms.shape != incident.shape[:-1]:
        raise ValueError("Incident RMS must match the field spatial shape")
    if receive.ndim != 2:
        raise ValueError("Receive data must have shape (n_times, n_channels)")

    field_reference = _finite_reference(incident)
    receive_reference = _finite_reference(receive)
    rms_reference = _finite_reference(incident_rms)
    return ViewerData(
        incident=np.ascontiguousarray(incident / field_reference, dtype=np.float32),
        scattered=np.ascontiguousarray(scattered / field_reference, dtype=np.float32),
        incident_rms=np.ascontiguousarray(incident_rms / rms_reference, dtype=np.float32),
        receive=np.ascontiguousarray(receive / receive_reference, dtype=np.float32),
        field_reference=field_reference,
        receive_reference=receive_reference,
    )


def crop_receive_to_field_window(
    receive: np.ndarray,
    receive_times: np.ndarray,
    field_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop receive samples to the latest selectable wavefield time."""
    receive = np.asarray(receive)
    receive_times = np.asarray(receive_times)
    field_times = np.asarray(field_times)
    if receive.ndim != 2 or receive_times.ndim != 1 or receive.shape[0] != receive_times.size:
        raise ValueError("Receive data and times must share their first dimension")
    if receive_times.size == 0 or field_times.size == 0:
        return receive.copy(), receive_times.copy()
    stop = int(np.searchsorted(receive_times, field_times[-1], side="right"))
    stop = max(1, min(stop, receive_times.size))
    return receive[:stop].copy(), receive_times[:stop].copy()


def time_index_for_time(times: np.ndarray, time: float) -> int:
    """Return the nearest frame index for an absolute time in seconds."""
    times = np.asarray(times)
    if times.size == 0:
        return 0
    return int(np.argmin(np.abs(times - time)))


def physical_to_image_coordinates(
    points: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> np.ndarray:
    """Map physical overlay coordinates to Anyplotlib image indices."""
    points = np.asarray(points, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape (n, 2)")
    if x_axis.ndim != 1 or y_axis.ndim != 1 or x_axis.size == 0 or y_axis.size == 0:
        raise ValueError("Image axes must be nonempty one-dimensional arrays")

    def axis_indices(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
        indices = np.arange(axis.size, dtype=float)
        if axis.size > 1 and axis[-1] < axis[0]:
            return np.interp(values, axis[::-1], indices[::-1])
        return np.interp(values, axis, indices)

    x_indices = axis_indices(points[:, 0], x_axis)
    y_indices = axis_indices(points[:, 1], y_axis)
    return np.column_stack((x_indices, y_indices))


def rasterize_receive_for_aspect(receive: np.ndarray, *, aspect: float) -> np.ndarray:
    """Repeat whole RF channel rows to match a physical display aspect."""
    receive = np.asarray(receive)
    if receive.ndim != 2:
        raise ValueError("Receive display data must have shape (channels, times)")
    if aspect <= 0.0:
        raise ValueError("Receive display aspect must be positive")
    n_channels, n_times = receive.shape
    if n_channels == 0 or n_times == 0:
        return receive.copy()
    display_channels = max(1, int(np.ceil(n_times / aspect)))
    indices = np.rint(np.linspace(0, n_channels - 1, display_channels)).astype(int)
    return receive[indices].copy()


def db_visibility(values: np.ndarray, *, dynamic_range_db: float) -> np.ndarray:
    """Map normalized amplitudes to visibility over a relative-dB range."""
    if dynamic_range_db <= 0.0:
        raise ValueError("Dynamic range must be positive")
    magnitude = np.abs(np.asarray(values, dtype=float))
    level_db = 20.0 * np.log10(np.maximum(magnitude, np.finfo(float).tiny))
    visibility = np.clip((level_db + dynamic_range_db) / dynamic_range_db, 0.0, 1.0)
    return np.where(magnitude == 0.0, 0.0, visibility)


def signed_db(values: np.ndarray, dynamic_range_db: float) -> np.ndarray:
    """Map normalized bipolar values to a clipped signed-decibel display."""
    return np.sign(values) * db_visibility(values, dynamic_range_db=dynamic_range_db) * dynamic_range_db


def reflectivity_strength(coefficients: np.ndarray) -> np.ndarray:
    """Normalize scatterer magnitude for grayscale marker rendering."""
    magnitude = np.abs(np.asarray(coefficients, dtype=float))
    reference = float(np.max(magnitude)) if magnitude.size else 0.0
    return np.zeros_like(magnitude) if reference == 0.0 else magnitude / reference


def reflectivity_legend(coefficients: np.ndarray) -> str:
    """Return a compact grayscale key for relative scatterer amplitudes."""
    values = np.abs(np.asarray(coefficients, dtype=float))
    reference = float(np.max(values)) if values.size else 0.0
    return (
        '<div style="display:flex;justify-content:flex-end;align-items:center;gap:.45rem;'
        'color:#64748b;font:12px ui-sans-serif,system-ui">'
        '<span style="width:5rem;height:4px;border-radius:99px;'
        'background:linear-gradient(90deg,#f5f5f5,#191919)"></span>'
        f"<span>relative reflectivity 0-{reference:.2g}</span></div>"
    )


def _component_frame(data: ViewerData, component: str, time_index: int) -> np.ndarray:
    if component == "Incident":
        return data.incident[..., time_index]
    if component == "Scattered":
        return data.scattered[..., time_index]
    if component == "Total":
        return data.incident[..., time_index] + data.scattered[..., time_index]
    raise ValueError(f"Unknown field component: {component}")


def _aperture_span(elements: np.ndarray) -> float:
    if elements.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(elements, axis=0), axis=1)))


def _grayscale_colors(coefficients: np.ndarray) -> list[str]:
    darkness = np.sqrt(reflectivity_strength(coefficients))
    shades = np.rint(245.0 - 220.0 * darkness).astype(int)
    return [f"#{shade:02x}{shade:02x}{shade:02x}" for shade in shades]


@dataclass
class ScatteringFigure:
    """Linked anyplotlib field and receive panels with a draggable cursor."""

    figure: apl.Figure
    field_plot: apl.Plot2D
    receive_plot: apl.Plot2D
    rms_layer: Any
    cursor: apl.VLineWidget
    data: ViewerData
    field_times: np.ndarray
    receive_times: np.ndarray
    field_aspect: float
    receive_aspect: float
    time_index: int = 0
    component: str = "Total"
    waveform_dynamic_range: float = 60.0
    rms_dynamic_range: float = 20.0
    rms_visible: bool = True

    @classmethod
    def from_data(
        cls,
        data: ViewerData,
        *,
        field_times: np.ndarray,
        receive_times: np.ndarray,
        extent: tuple[float, float, float, float],
        elements: np.ndarray,
        scatterers: np.ndarray,
        coefficients: np.ndarray,
        propagation_speed: float = 1540.0,
        focus: np.ndarray | None = None,
        time_index: int = 0,
    ) -> ScatteringFigure:
        """Create linked anyplotlib panels from normalized physical data."""
        field_times = np.asarray(field_times, dtype=float)
        receive_times = np.asarray(receive_times, dtype=float)
        elements = np.asarray(elements, dtype=float).reshape((-1, 2))
        scatterers = np.asarray(scatterers, dtype=float).reshape((-1, 2))
        coefficients = np.asarray(coefficients, dtype=float)
        if field_times.size != data.field_shape[-1]:
            raise ValueError("Field times must match the movie frame count")
        if receive_times.size != data.rf_shape[0]:
            raise ValueError("Receive times must match the RF sample count")

        time_index = max(0, min(time_index, field_times.size - 1)) if field_times.size else 0
        x_min, x_max, z_min, z_max = map(float, extent)
        x_span = max(abs(x_max - x_min), np.finfo(float).eps)
        z_span = max(abs(z_max - z_min), np.finfo(float).eps)
        field_aspect = float(x_span / z_span)
        aperture_span = _aperture_span(elements)
        propagation_span = (
            abs(float(receive_times[-1] - receive_times[0])) * propagation_speed * 1e3
            if receive_times.size > 1
            else 0.0
        )
        receive_aspect = propagation_span / aperture_span if aperture_span > 0.0 and propagation_span > 0.0 else 4.0

        plot_width = 820.0
        field_height = plot_width / field_aspect + 80.0
        receive_height = min(300.0, max(120.0, plot_width / receive_aspect + 70.0))
        figure, axes = apl.subplots(
            2,
            1,
            figsize=(900, int(field_height + receive_height)),
            height_ratios=[field_height, receive_height],
        )
        x_axis = np.linspace(x_min, x_max, data.field_shape[1])
        z_axis = np.linspace(z_min, z_max, data.field_shape[0])
        element_pixels = physical_to_image_coordinates(elements, x_axis, z_axis)
        scatterer_pixels = physical_to_image_coordinates(scatterers, x_axis, z_axis)
        initial_frame = signed_db(_component_frame(data, "Total", time_index), 60.0)
        field_plot = axes[0].imshow(
            initial_frame,
            axes=[x_axis, z_axis],
            units="mm",
            cmap="bwr",
            vmin=-60.0,
            vmax=60.0,
            origin="upper",
            tile=False,
        )
        field_plot.set_aspect(field_aspect)
        field_plot.set_tick_label_size(12.0)
        field_plot.set_xlabel("lateral position (mm)", fontsize=12.0)
        field_plot.set_ylabel("depth (mm)", fontsize=12.0)
        field_plot.set_colorbar_visible(True)
        field_plot.set_colorbar_label("relative pressure (dB)", fontsize=11.0)
        rms_layer = field_plot.add_layer(
            db_visibility(data.incident_rms, dynamic_range_db=20.0),
            tint="#5edae8",
            alpha=0.5,
            clim=(0.0, 1.0),
        )
        field_plot.add_points(
            element_pixels,
            name="probe elements",
            sizes=3.0,
            color="#1d4ed8",
            facecolors="#1d4ed8",
            linewidths=0.5,
            alpha=1.0,
            size_units="px",
            clip_display=False,
        )
        if scatterers.shape[0]:
            field_plot.add_points(
                scatterer_pixels,
                name="scatterers",
                sizes=4.0,
                color="#ffffff",
                facecolors=_grayscale_colors(coefficients),
                linewidths=1.0,
                alpha=1.0,
                size_units="px",
            )
        if focus is not None:
            focus_point = np.asarray(focus, dtype=float)
            focus_pixels = physical_to_image_coordinates(
                np.asarray([[focus_point[0], min(0.0, z_min)], focus_point]),
                x_axis,
                z_axis,
            )
            field_plot.add_lines(
                focus_pixels[None, ...],
                name="focus",
                edgecolors="#db2777",
                linewidths=1.5,
                clip_display=False,
            )

        receive_db = rasterize_receive_for_aspect(signed_db(data.receive.T, 60.0), aspect=receive_aspect)
        receive_axis_us = receive_times * 1e6
        channel_axis = np.linspace(0.0, max(data.rf_shape[1] - 1, 0), receive_db.shape[0])
        receive_plot = axes[1].imshow(
            receive_db,
            axes=[receive_axis_us, channel_axis],
            units=" ",
            cmap="bwr",
            vmin=-60.0,
            vmax=60.0,
            origin="upper",
            tile=False,
        )
        receive_plot.set_aspect(receive_aspect)
        receive_plot.set_tick_label_size(12.0)
        receive_plot.set_xlabel("time (µs)", fontsize=12.0)
        receive_plot.set_ylabel("receive element", fontsize=12.0)
        receive_plot.set_colorbar_visible(True)
        receive_plot.set_colorbar_label("relative receive RF (dB)", fontsize=11.0)
        receive_index = int(np.argmin(np.abs(receive_times - field_times[time_index]))) if receive_times.size else 0
        cursor = receive_plot.add_vline_widget(x=float(receive_index), color="#db2777", linewidth=2.0)
        cursor.set(_notify=False, snap_values=list(range(receive_times.size)))

        viewer = cls(
            figure=figure,
            field_plot=field_plot,
            receive_plot=receive_plot,
            rms_layer=rms_layer,
            cursor=cursor,
            data=data,
            field_times=field_times,
            receive_times=receive_times,
            field_aspect=field_aspect,
            receive_aspect=receive_aspect,
            time_index=time_index,
        )
        cursor.add_event_handler(lambda event: viewer._cursor_moved(event), "pointer_move")
        receive_plot.add_event_handler(lambda event: viewer._receive_clicked(event), "pointer_down")
        viewer._update_field()
        return viewer

    @property
    def current_frame(self) -> np.ndarray:
        """Return the displayed signed-dB field frame."""
        return signed_db(
            _component_frame(self.data, self.component, self.time_index),
            self.waveform_dynamic_range,
        )

    def _update_field(self) -> None:
        with self.figure.batch():
            self.field_plot.set_data(
                self.current_frame,
                clim=(-self.waveform_dynamic_range, self.waveform_dynamic_range),
                tile=False,
            )
            time_us = self.field_times[self.time_index] * 1e6 if self.field_times.size else 0.0
            self.field_plot.set_title(f"{self.component} field · {time_us:.1f} µs", fontsize=13.0)

    def _cursor_moved(self, _event: apl.Event) -> None:
        if not self.receive_times.size:
            return
        receive_index = max(0, min(round(float(self.cursor.x)), self.receive_times.size - 1))
        self.time_index = time_index_for_time(self.field_times, self.receive_times[receive_index])
        self._update_field()

    def _receive_clicked(self, event: apl.Event) -> None:
        if event.xdata is None or not self.receive_times.size:
            return
        receive_index = int(np.argmin(np.abs(self.receive_times * 1e6 - float(event.xdata))))
        self.cursor.set(_notify=False, x=float(receive_index))
        self.time_index = time_index_for_time(self.field_times, self.receive_times[receive_index])
        self._update_field()

    def update_display(
        self,
        *,
        component: str,
        waveform_dynamic_range: float,
        rms_visible: bool,
        rms_dynamic_range: float,
    ) -> None:
        """Update display-only state without rerunning simulation physics."""
        _component_frame(self.data, component, self.time_index)
        if waveform_dynamic_range <= 0.0 or rms_dynamic_range <= 0.0:
            raise ValueError("Display dynamic ranges must be positive")
        self.component = component
        self.waveform_dynamic_range = float(waveform_dynamic_range)
        self.rms_visible = bool(rms_visible)
        self.rms_dynamic_range = float(rms_dynamic_range)
        with self.figure.batch():
            self.rms_layer.set_data(db_visibility(self.data.incident_rms, dynamic_range_db=self.rms_dynamic_range))
            self.rms_layer.set(visible=self.rms_visible)
            self.receive_plot.set_data(
                rasterize_receive_for_aspect(
                    signed_db(self.data.receive.T, self.waveform_dynamic_range),
                    aspect=self.receive_aspect,
                ),
                clim=(-self.waveform_dynamic_range, self.waveform_dynamic_range),
                tile=False,
            )
            self._update_field()
