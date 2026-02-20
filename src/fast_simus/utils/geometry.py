"""Geometry calculation utilities."""

from __future__ import annotations

from math import inf
from typing import TYPE_CHECKING, Any

from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

if TYPE_CHECKING:
    from fast_simus.utils._array_api import _ArrayNamespace

# Type alias for Array API objects (until protocol is standardized)
Array = Any


@jaxtyped(typechecker=typechecker)
def element_positions(
    n_elements: int,
    pitch: float,
    radius: float,
    xp: _ArrayNamespace,
) -> tuple[
    Float[Array, "n_elements 2"],
    Float[Array, " n_elements"] | None,
    float,
]:
    """Calculate transducer element positions.

    Computes the (x, z) positions of transducer elements for both linear
    and convex arrays. For linear arrays, elements are evenly spaced along
    the x-axis. For convex arrays, elements are positioned along an arc
    defined by the radius of curvature.

    Args:
        n_elements: Number of transducer elements.
        pitch: Element pitch (center-to-center spacing) in meters.
        radius: Curvature radius in meters. Use inf for linear arrays.
        xp: Array namespace for creating arrays in the desired backend.

    Returns:
        Tuple of (positions, theta_rad, apex_offset_m):
        - positions: Array of (x, z) coordinates in meters. Shape (n_elements, 2).
          positions[:, 0] is lateral (x), positions[:, 1] is axial (z).
        - theta_rad: Array of angular positions in radians for convex arrays,
          None for linear arrays. Shape (n_elements,) or None.
        - apex_offset_m: Distance from array center to arc apex in meters.
          Zero for linear arrays.
    """
    is_linear = radius == inf

    if is_linear:
        # Linear array: elements evenly spaced along x-axis
        indices = xp.arange(n_elements)
        x = (indices - (n_elements - 1) / 2) * pitch
        z = xp.zeros(n_elements)
        theta = None
        apex_offset = 0.0
    else:
        # Convex array: elements positioned along arc
        # Compute chord length subtended by the array
        # Each element subtends an angle: pitch / (2 * radius)
        half_angle_per_element = xp.asin(xp.asarray(pitch / 2 / radius))
        total_angle = half_angle_per_element * (n_elements - 1)
        chord = xp.asarray(2 * radius) * xp.sin(total_angle)

        # Compute apex offset (distance from center to arc apex)
        # h = sqrt(radius^2 - (chord/2)^2)
        apex_offset_arr = xp.sqrt(xp.asarray(radius**2) - (chord / 2) ** 2)
        apex_offset = float(apex_offset_arr)

        # Compute angular positions using linspace if available, otherwise manual
        theta_start_arr = xp.atan2(-chord / 2, apex_offset_arr)
        theta_start = float(theta_start_arr)
        theta_end_arr = xp.atan2(chord / 2, apex_offset_arr)
        theta_end = float(theta_end_arr)

        theta = xp.linspace(theta_start, theta_end, n_elements)

        # Convert angular positions to (x, z) coordinates
        # z = radius * cos(theta) - h (where h is apex_offset)
        z = xp.asarray(radius) * xp.cos(theta) - apex_offset
        x = xp.asarray(radius) * xp.sin(theta)

    # Stack x and z into a single position array
    positions = xp.stack([x, z], axis=-1)
    return positions, theta, apex_offset
