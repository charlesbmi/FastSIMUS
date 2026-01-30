"""Geometry calculation utilities."""

from math import inf

from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

from fast_simus.utils._array_api import _ArrayNamespace


@jaxtyped(typechecker=typechecker)
def element_positions(
    n_elements: int,
    pitch: float,
    radius: float,
    xp: _ArrayNamespace,
) -> tuple[
    Float[ArrayAPIObj, "n_elements"],
    Float[ArrayAPIObj, "n_elements"],
    Float[ArrayAPIObj, "n_elements"] | None,
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
        Tuple of (x_positions_m, z_positions_m, theta_rad, apex_offset_m):
        - x_positions_m: Array of x-coordinates in meters. Shape (n_elements,).
        - z_positions_m: Array of z-coordinates in meters. Shape (n_elements,).
        - theta_rad: Array of angular positions in radians for convex arrays,
          None for linear arrays. Shape (n_elements,) or None.
        - apex_offset_m: Distance from array center to arc apex in meters.
          Zero for linear arrays.
    """
    is_linear = radius == inf

    if is_linear:
        # Linear array: elements evenly spaced along x-axis
        indices = xp.arange(n_elements, dtype=xp.float64)
        x = (indices - (n_elements - 1) / 2) * pitch
        z = xp.zeros(n_elements, dtype=xp.float64)
        theta = None
        apex_offset = 0.0
    else:
        # Convex array: elements positioned along arc
        # Compute chord length subtended by the array
        # Each element subtends an angle: pitch / (2 * radius)
        half_angle_per_element = xp.arcsin(pitch / 2 / radius)
        total_angle = half_angle_per_element * (n_elements - 1)
        chord = 2 * radius * xp.sin(total_angle)

        # Compute apex offset (distance from center to arc apex)
        # h = sqrt(radius^2 - (chord/2)^2)
        apex_offset = float(xp.sqrt(radius**2 - (chord / 2) ** 2))

        # Compute angular positions using linspace if available, otherwise manual
        theta_start = float(xp.arctan2(-chord / 2, apex_offset))
        theta_end = float(xp.arctan2(chord / 2, apex_offset))

        # Try to use linspace if available (most backends support it)
        if hasattr(xp, "linspace"):
            theta = xp.linspace(theta_start, theta_end, n_elements, dtype=xp.float64)
        else:
            # Manual linspace implementation for Array API compliance
            indices = xp.arange(n_elements, dtype=xp.float64)
            if n_elements > 1:
                theta = theta_start + (theta_end - theta_start) * indices / (n_elements - 1)
            else:
                theta = xp.asarray([theta_start], dtype=xp.float64)

        # Convert angular positions to (x, z) coordinates
        # z = radius * cos(theta) - h (where h is apex_offset)
        z = radius * xp.cos(theta) - apex_offset
        x = radius * xp.sin(theta)

    return x, z, theta, apex_offset
