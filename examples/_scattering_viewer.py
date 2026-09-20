"""Data preparation and anywidget viewer for the scattering explorer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anywidget
import numpy as np
import traitlets


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


def db_visibility(values: np.ndarray, *, dynamic_range_db: float) -> np.ndarray:
    """Map normalized amplitudes to visibility over a relative-dB range."""
    if dynamic_range_db <= 0.0:
        raise ValueError("Dynamic range must be positive")
    magnitude = np.abs(np.asarray(values, dtype=float))
    level_db = 20.0 * np.log10(np.maximum(magnitude, np.finfo(float).tiny))
    visibility = np.clip((level_db + dynamic_range_db) / dynamic_range_db, 0.0, 1.0)
    return np.where(magnitude == 0.0, 0.0, visibility)


def reflectivity_strength(coefficients: np.ndarray) -> np.ndarray:
    """Normalize scatterer magnitude for grayscale marker rendering."""
    magnitude = np.abs(np.asarray(coefficients, dtype=float))
    reference = float(np.max(magnitude)) if magnitude.size else 0.0
    return np.zeros_like(magnitude) if reference == 0.0 else magnitude / reference


class ScatteringViewer(anywidget.AnyWidget):
    """Responsive field and receive-time canvas with a draggable cursor."""

    _esm = Path(__file__).with_name("_scattering_viewer.js")
    _css = Path(__file__).with_name("_scattering_viewer.css")

    incident = traitlets.Bytes().tag(sync=True)
    scattered = traitlets.Bytes().tag(sync=True)
    incident_rms = traitlets.Bytes().tag(sync=True)
    receive = traitlets.Bytes().tag(sync=True)
    field_shape = traitlets.List(traitlets.Int()).tag(sync=True)
    receive_shape = traitlets.List(traitlets.Int()).tag(sync=True)
    field_times = traitlets.List(traitlets.Float()).tag(sync=True)
    receive_times = traitlets.List(traitlets.Float()).tag(sync=True)
    extent = traitlets.List(traitlets.Float()).tag(sync=True)
    elements = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
    scatterers = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
    coefficients = traitlets.List(traitlets.Float()).tag(sync=True)
    focus = traitlets.List(traitlets.Float(), allow_none=True, default_value=None).tag(sync=True)
    component = traitlets.Unicode("Total").tag(sync=True)
    rms_visible = traitlets.Bool(True).tag(sync=True)
    rms_dynamic_range = traitlets.Float(20.0).tag(sync=True)
    waveform_dynamic_range = traitlets.Float(60.0).tag(sync=True)
    propagation_speed = traitlets.Float(1540.0).tag(sync=True)
    time_index = traitlets.Int(0).tag(sync=True)

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
        **display,
    ) -> ScatteringViewer:
        """Create a widget from normalized NumPy arrays and physical metadata."""
        field_times = np.asarray(field_times, dtype=float)
        time_index = int(display.get("time_index", 0))
        time_index = max(0, min(time_index, field_times.size - 1)) if field_times.size else 0
        display["time_index"] = time_index
        return cls(
            incident=data.incident.tobytes(),
            scattered=data.scattered.tobytes(),
            incident_rms=data.incident_rms.tobytes(),
            receive=data.receive.tobytes(),
            field_shape=list(data.field_shape),
            receive_shape=list(data.rf_shape),
            field_times=field_times.tolist(),
            receive_times=np.asarray(receive_times, dtype=float).tolist(),
            extent=list(map(float, extent)),
            elements=np.asarray(elements, dtype=float).tolist(),
            scatterers=np.asarray(scatterers, dtype=float).reshape((-1, 2)).tolist(),
            coefficients=np.asarray(coefficients, dtype=float).tolist(),
            propagation_speed=float(propagation_speed),
            focus=None if focus is None else np.asarray(focus, dtype=float).tolist(),
            **display,
        )
