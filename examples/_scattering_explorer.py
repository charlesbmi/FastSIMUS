"""Deterministic scenes and coordinate conversions for the scattering explorer."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi, sqrt

import numpy as np

_DRAWING_LABELS = ("a", "b", "c", "d")


@dataclass(frozen=True)
class GridSpec:
    """Derived grid dimensions and effective physical spacing."""

    nx: int
    nz: int
    dx_mm: float
    dz_mm: float
    x_min_mm: float
    x_max_mm: float
    z_min_mm: float
    z_max_mm: float


@dataclass(frozen=True)
class WorkloadEstimate:
    """Spatial work and field-movie storage estimate for one configuration."""

    effective_dx_mm: float
    effective_dz_mm: float
    spatial_pairs: int
    record_samples: int
    field_movie_bytes: int


def picmus_point_targets() -> tuple[np.ndarray, np.ndarray]:
    """Return the 20-point PICMUS resolution phantom in millimetres."""
    axial = [(0.0, float(z)) for z in range(10, 50, 5)]
    lateral = [(float(x), float(z)) for z in (20, 40) for x in range(-15, 20, 5) if x != 0]
    positions = np.asarray(axial + lateral, dtype=np.float64)
    return positions, np.full(positions.shape[0], 0.005, dtype=np.float64)


def speckle_lesion(
    count: int,
    *,
    seed: int = 0,
    extent_mm: tuple[float, float, float, float] = (-20.0, 20.0, 5.0, 55.0),
    lesion_center_mm: tuple[float, float] = (0.0, 35.0),
    lesion_radius_mm: float = 7.0,
    lesion_amplitude: float = 0.2,
    background_mean: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic Rayleigh-amplitude speckle with a lesion."""
    if count < 0:
        raise ValueError("Scatterer count must be non-negative")
    x_min, x_max, z_min, z_max = extent_mm
    rng = np.random.default_rng(seed)
    positions = np.column_stack(
        [
            rng.uniform(x_min, x_max, count),
            rng.uniform(z_min, z_max, count),
        ]
    )
    rayleigh_scale = background_mean / sqrt(pi / 2.0)
    reflection_coefficients = rng.rayleigh(rayleigh_scale, count)
    distance = np.sqrt((positions[:, 0] - lesion_center_mm[0]) ** 2 + (positions[:, 1] - lesion_center_mm[1]) ** 2)
    reflection_coefficients = np.where(
        distance < lesion_radius_mm,
        lesion_amplitude * reflection_coefficients,
        reflection_coefficients,
    )
    return positions, reflection_coefficients


def canvas_to_physical(
    points: np.ndarray,
    extent_mm: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert drawdata's bottom-up canvas coordinates to physical millimetres."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    x_min, x_max, z_min, z_max = extent_mm
    x = x_min + points[:, 0] / width * (x_max - x_min)
    z = z_min + points[:, 1] / height * (z_max - z_min)
    return np.column_stack([x, z])


def physical_to_canvas(
    points_mm: np.ndarray,
    extent_mm: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert physical millimetres to drawdata's bottom-up canvas coordinates."""
    points_mm = np.asarray(points_mm, dtype=np.float64)
    if points_mm.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    x_min, x_max, z_min, z_max = extent_mm
    x = (points_mm[:, 0] - x_min) / (x_max - x_min) * width
    y = (points_mm[:, 1] - z_min) / (z_max - z_min) * height
    return np.column_stack([x, y])


def drawing_class_coefficients(labels: list[str], class_values: tuple[float, float, float, float]) -> np.ndarray:
    """Map drawdata class labels to relative scatterer amplitudes."""
    mapping = dict(zip(_DRAWING_LABELS, class_values, strict=True))
    default = class_values[2]
    return np.asarray([mapping.get(label, default) for label in labels], dtype=np.float64)


def normalize_custom_rows(rows) -> list[dict[str, float]]:
    """Return editor rows as plain physical-coordinate dictionaries."""
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict(orient="records")
    elif isinstance(rows, dict):
        columns = (rows.get("x_mm", []), rows.get("z_mm", []), rows.get("rc", []))
        rows = [dict(zip(("x_mm", "z_mm", "rc"), values, strict=True)) for values in zip(*columns, strict=True)]
    normalized = [
        {
            "x_mm": float(row["x_mm"]),
            "z_mm": float(row["z_mm"]),
            "rc": float(row["rc"]),
        }
        for row in rows
    ]
    if any(row["rc"] < 0.0 for row in normalized):
        raise ValueError("Custom reflectivity must be nonnegative")
    return normalized


def snapshot_custom_rows(rows) -> list[dict[str, float]]:
    """Copy a normalized Custom draft into an independent applied scene."""
    return normalize_custom_rows(rows)


def custom_rows_are_dirty(draft_rows, applied_rows) -> bool:
    """Return whether a Custom draft differs from the last applied scene."""
    return normalize_custom_rows(draft_rows) != normalize_custom_rows(applied_rows)


def wavelength_mm(speed_of_sound: float, center_frequency_mhz: float) -> float:
    """Return the center wavelength in millimetres."""
    if speed_of_sound <= 0.0 or center_frequency_mhz <= 0.0:
        raise ValueError("Sound speed and center frequency must be positive")
    return speed_of_sound / (center_frequency_mhz * 1e6) * 1e3


def spacing_in_mm(value: float, wavelength: float, *, wavelength_units: bool) -> float:
    """Convert a positive wavelength-relative or millimetre spacing to mm."""
    if value <= 0.0 or wavelength <= 0.0:
        raise ValueError("Grid spacing and wavelength must be positive")
    return value * wavelength if wavelength_units else value


def grid_from_spacing(
    x_limits_mm: tuple[float, float],
    z_limits_mm: tuple[float, float],
    spacing_mm: float,
) -> GridSpec:
    """Fit an endpoint-inclusive grid whose spacing does not exceed a request."""
    if spacing_mm <= 0.0:
        raise ValueError("Grid spacing must be positive")
    x_min, x_max = map(float, x_limits_mm)
    z_min, z_max = map(float, z_limits_mm)
    if x_max <= x_min or z_max <= z_min:
        raise ValueError("ROI limits must have positive extent")
    nx = ceil((x_max - x_min) / spacing_mm) + 1
    nz = ceil((z_max - z_min) / spacing_mm) + 1
    return GridSpec(
        nx=nx,
        nz=nz,
        dx_mm=(x_max - x_min) / (nx - 1),
        dz_mm=(z_max - z_min) / (nz - 1),
        x_min_mm=x_min,
        x_max_mm=x_max,
        z_min_mm=z_min,
        z_max_mm=z_max,
    )


def tukey_apodization(n_elements: int, roll: float) -> np.ndarray:
    """Return a symmetric Tukey window where 0 is uniform and 1 is Hann."""
    if n_elements <= 0:
        raise ValueError("Element count must be positive")
    if not 0.0 <= roll <= 1.0:
        raise ValueError("Tukey roll must be between 0 and 1")
    if n_elements == 1 or roll == 0.0:
        return np.ones(n_elements)
    x = np.linspace(0.0, 1.0, n_elements)
    weights = np.ones(n_elements)
    leading = x < roll / 2.0
    trailing = x >= 1.0 - roll / 2.0
    weights[leading] = 0.5 * (1.0 + np.cos(2.0 * pi / roll * (x[leading] - roll / 2.0)))
    weights[trailing] = 0.5 * (1.0 + np.cos(2.0 * pi / roll * (x[trailing] - 1.0 + roll / 2.0)))
    return weights


def apodization_svg(weights: np.ndarray, *, width: int = 250, height: int = 54) -> str:
    """Render apodization weights as a compact responsive SVG profile."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Apodization preview requires a nonempty vector")
    x = np.linspace(4.0, width - 4.0, values.size)
    y = height - 4.0 - np.clip(values, 0.0, 1.0) * (height - 12.0)
    points = " ".join(f"{x_value:.2f},{y_value:.2f}" for x_value, y_value in zip(x, y, strict=True))
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Transmit apodization profile" '
        'style="display:block;width:100%;height:auto">'
        f'<polyline points="{points}" fill="none" stroke="currentColor" stroke-width="2"/>'
        f'<line x1="4" y1="{height - 4}" x2="{width - 4}" y2="{height - 4}" '
        'stroke="currentColor" stroke-opacity="0.25"/>'
        "</svg>"
    )


def estimate_field_movie_bytes(
    grid_shape: tuple[int, int],
    n_times: int,
    *,
    bytes_per_value: int = 8,
    temporal_oversampling: int = 1,
) -> int:
    """Estimate storage for incident and scattered time-domain frames."""
    nz, nx = grid_shape
    return nz * nx * n_times * temporal_oversampling * bytes_per_value * 2


def estimate_spatial_pairs(grid_shape: tuple[int, int], n_scatterers: int) -> int:
    """Return observation-scatterer pairs evaluated per frequency."""
    if n_scatterers < 0:
        raise ValueError("Scatterer count must be non-negative")
    nz, nx = grid_shape
    return nz * nx * n_scatterers


def estimate_simulation_workload(
    *,
    grid_shape: tuple[int, int],
    x_limits_mm: tuple[float, float],
    z_limits_mm: tuple[float, float],
    scatterers_mm: np.ndarray,
    n_scatterers: int,
    propagation_speed: float,
    center_frequency_mhz: float,
    pulse_wavelengths: float,
    time_oversampling: int,
) -> WorkloadEstimate:
    """Estimate sample count and storage from physical extents and pulse duration."""
    nx, nz = grid_shape
    if nx < 2 or nz < 2:
        raise ValueError("Grid dimensions must each contain at least two samples")
    if propagation_speed <= 0.0 or center_frequency_mhz <= 0.0:
        raise ValueError("Sound speed and center frequency must be positive")

    scatterers = np.asarray(scatterers_mm, dtype=float).reshape((-1, 2))
    lateral_values = [abs(x_limits_mm[0]), abs(x_limits_mm[1])]
    axial_values = [abs(z_limits_mm[0]), abs(z_limits_mm[1])]
    if scatterers.size:
        lateral_values.extend(np.abs(scatterers[:, 0]).tolist())
        axial_values.extend(np.abs(scatterers[:, 1]).tolist())
    maximum_range_mm = float(np.hypot(max(lateral_values), max(axial_values)))
    record_seconds = 2.0 * maximum_range_mm * 1e-3 / propagation_speed + pulse_wavelengths / (
        center_frequency_mhz * 1e6
    )
    record_samples = max(1, int(np.ceil(4.0 * center_frequency_mhz * 1e6 * record_seconds)))
    return WorkloadEstimate(
        effective_dx_mm=(x_limits_mm[1] - x_limits_mm[0]) / (nx - 1),
        effective_dz_mm=(z_limits_mm[1] - z_limits_mm[0]) / (nz - 1),
        spatial_pairs=estimate_spatial_pairs((nz, nx), n_scatterers),
        record_samples=record_samples,
        field_movie_bytes=estimate_field_movie_bytes(
            (nz, nx),
            record_samples,
            temporal_oversampling=time_oversampling,
        ),
    )


def append_drawn_points(
    rows,
    drawing_data: list[dict],
    extent_mm: tuple[float, float, float, float],
    class_values: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> list[dict[str, float]]:
    """Append staged drawdata points to the canonical physical table."""
    existing = normalize_custom_rows(rows)
    if not drawing_data:
        return existing
    canvas = np.asarray([[item["x"], item["y"]] for item in drawing_data], dtype=np.float64)
    physical = canvas_to_physical(canvas, extent_mm, width=width, height=height)
    coefficients = drawing_class_coefficients([str(item.get("label", "c")) for item in drawing_data], class_values)
    additions = [
        {"x_mm": float(point[0]), "z_mm": float(point[1]), "rc": float(rc)}
        for point, rc in zip(physical, coefficients, strict=True)
    ]
    return [*existing, *additions]


def status_text(n_scatterers: int, n_channels: int, field_shape: tuple[int, int]) -> str:
    """Format the compact notebook simulation summary."""
    scatterer_label = "scatterer" if n_scatterers == 1 else "scatterers"
    channel_label = "receive channel" if n_channels == 1 else "receive channels"
    nz, nx = field_shape
    return (
        f"{n_scatterers} {scatterer_label} · {n_channels} {channel_label} · "
        f"{nx} \N{MULTIPLICATION SIGN} {nz} grid · first-order scattering"
    )
