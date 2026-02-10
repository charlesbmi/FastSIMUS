"""Transmit delay calculations for ultrasound transducer arrays.

This module provides functions to compute transmit time delays for different
beam patterns (focused/diverging, plane wave, circular wave) with linear and
convex arrays.

All functions are Array API compliant and work with NumPy, JAX, CuPy backends.

Coordinate System Convention:
    - x-axis: Lateral (perpendicular to beam direction), in meters
    - z-axis: Axial (along beam direction, positive = into tissue), in meters
    - Origin: Center of transducer face

Sign Convention:
    - Positive z: Points into the imaging medium (tissue)
    - Delays are relative to minimum (all non-negative)
    - For focused beams: positive z0 = focus in front, negative z0 = diverging
"""

from __future__ import annotations

from math import cos, fmod, inf, pi, sin, sqrt, tan

from array_api_compat import array_namespace
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

from fast_simus.utils._array_api import Array


@jaxtyped(typechecker=typechecker)
def focused(
    element_positions: Float[Array, "n_elements dim"],
    focus: Float[Array, " dim"],
    *,
    speed_of_sound: float,
    radius: float = inf,
    apex_offset: float = 0.0,
) -> Float[Array, " n_elements"]:
    """Compute transmit time delays for focused or diverging spherical waves.

    Spherical waves propagate like a collapsing sphere focusing onto a point
    (positive z), or an expanding sphere diverging from a virtual source
    (negative z).

    Args:
        element_positions:
            Element positions in meters. Shape (n_elements, dim).
            For 2D: (x, z) where x is lateral, z is axial.
            For 3D: (x, y, z) where z is axial (depth).
        focus:
            Focal position in meters. Shape (dim,).
            Coordinates match element_positions dimensions.
            - Last coordinate (z): Axial position
              - Positive z: Focused wave (focus in front of transducer)
              - Negative z: Diverging wave (virtual source behind transducer)
              - z=0: Treated as focusing just in front (negative delays)
        speed_of_sound:
            Speed of sound in m/s.
        radius:
            Curvature radius in meters. Use inf for linear arrays.
            Convex arrays require dim=2 (runtime check).
        apex_offset:
            Distance from array center to arc apex in meters. Zero for linear arrays.
            Defaults to 0.0.

    Returns:
        Transmit time delays in seconds. Shape (n_elements,).
        Delays are relative to minimum (all non-negative).

    Notes:
        Implementation matches PyMUST reference with identical sign conventions.

        For linear arrays, sign is based on z-coordinate of focus.
        For convex arrays, sign is based on whether focus is inside or outside
        the arc (comparing squared distance to squared radius).

        Reference:
            Perrot et al. 2021, "So you think you can DAS?"
            https://www.biomecardio.com/publis/ultrasonics21.pdf
            (Equation 5, extended to support virtual sources)
    """
    xp = array_namespace(element_positions, focus)

    # Compute distance from each element to focus
    # element_positions: (n_elements, dim), focus: (dim,)
    # Broadcasting: (n_elements, dim) - (dim,) -> (n_elements, dim)
    diff = element_positions - focus
    distances = xp.sqrt(xp.sum(diff**2, axis=-1))  # (n_elements,)

    if radius == inf:
        # Linear array: sign based on axial position (last coordinate)
        z_focus = focus[-1]
        sgn = xp.sign(z_focus)
        sgn = xp.where(sgn == 0, -1, sgn)
        delays = -distances * sgn / speed_of_sound
    else:
        # Convex array: requires 2D coordinates
        if focus.shape[0] != 2:
            msg = f"Convex arrays require 2D coordinates, got dim={focus.shape[0]}"
            raise ValueError(msg)

        # Sign based on inside/outside arc
        x_focus = focus[0]
        z_focus = focus[-1]
        sgn = xp.sign(x_focus**2 + (z_focus + apex_offset) ** 2 - radius**2)
        sgn = xp.where(sgn == 0, -1, sgn)
        delays = -distances * sgn / speed_of_sound

    # Make all delays non-negative by subtracting minimum
    delays = delays - xp.min(delays)

    return delays


@jaxtyped(typechecker=typechecker)
def plane_wave(
    element_positions: Float[Array, "n_elements xz=2"],
    tilt_rad: float,
    *,
    speed_of_sound: float,
    radius: float = inf,
    apex_offset: float = 0.0,
) -> Float[Array, " n_elements"]:
    """Compute transmit time delays for plane wave transmission.

    Plane waves have a flat wavefront propagating in a specified direction,
    defined by the tilt angle from the array normal (z-axis).

    Args:
        element_positions:
            Element (x, z) positions in meters. Shape (n_elements, 2).
            - element_positions[:, 0]: Lateral positions (x)
            - element_positions[:, 1]: Axial positions (z)
        tilt_rad:
            Tilt angle in radians.
            - tilt_rad=0: Straight ahead (perpendicular to array)
            - Positive tilt_rad: Beam steers right (positive x direction)
            - Negative tilt_rad: Beam steers left (negative x direction)
            Must satisfy |tilt_rad| < π/2.
        speed_of_sound:
            Speed of sound in m/s.
        radius:
            Curvature radius in meters. Use inf for linear arrays.
        apex_offset:
            Distance from array center to arc apex in meters. Zero for linear arrays.
            Defaults to 0.0.

    Returns:
        Transmit time delays in seconds. Shape (n_elements,).
        Delays are relative to minimum (all non-negative).

    Raises:
        ValueError: If |tilt_rad| >= π/2 (non-physical angle).

    Notes:
        For linear arrays, delays follow simple sinusoidal pattern based on
        lateral element position.

        For convex arrays, the computation accounts for the curved geometry
        by computing distance to the plane perpendicular to the tilt direction.
    """
    # Validate tilt angle
    if abs(tilt_rad) >= pi / 2:
        msg = "Tilt angles must satisfy |tilt_rad| < π/2"
        raise ValueError(msg)

    xp = array_namespace(element_positions)

    # Extract element positions
    x_elem = element_positions[:, 0]  # Shape (n_elements,)
    z_elem = element_positions[:, 1]  # Shape (n_elements,)

    if radius == inf:
        # Linear array: delay proportional to lateral position
        delays = x_elem * sin(tilt_rad) / speed_of_sound
    else:
        # Convex array: geometric calculation
        xn = radius * sin(tilt_rad)
        zn = radius * cos(tilt_rad) - apex_offset
        numerator = xp.abs(z_elem + xn / (zn + apex_offset) * x_elem - xn**2 / (zn + apex_offset) - zn)
        denominator = sqrt(1 + xn**2 / (zn + apex_offset) ** 2)
        d = numerator / denominator
        delays = -d / speed_of_sound

    # Make all delays non-negative
    delays = delays - xp.min(delays)

    return delays


@jaxtyped(typechecker=typechecker)
def diverging_wave(
    element_positions: Float[Array, "n_elements xz=2"],
    tilt_rad: float,
    width_rad: float,
    *,
    aperture_length: float,
    speed_of_sound: float,
) -> Float[Array, " n_elements"]:
    """Compute transmit time delays for diverging circular wave transmission.

    Circular waves originate from a virtual point source positioned such that
    the wavefront spans a specified angular width across the array.
    Only supported for linear arrays.

    Args:
        element_positions:
            Element (x, z) positions in meters. Shape (n_elements, 2).
            - element_positions[:, 0]: Lateral positions (x)
            - element_positions[:, 1]: Axial positions (z)
        tilt_rad:
            Tilt angle of the wave center in radians.
        width_rad:
            Angular width of the wave in radians.
            Must satisfy 0 < width_rad < π.
        aperture_length:
            Array aperture length in meters (typically (n_elements - 1) * pitch).
        speed_of_sound:
            Speed of sound in m/s.

    Returns:
        Transmit time delays in seconds. Shape (n_elements,).
        Delays are relative to minimum (all non-negative).

    Raises:
        ValueError: If width_rad is outside (0, π).

    Notes:
        The virtual source position is computed from the tilt and width angles
        using geometric constraints. This creates a diverging spherical wave
        that appears to originate from behind the transducer.

        Only supported for linear arrays (not convex arrays).
    """
    if width_rad <= 0 or width_rad >= pi:
        msg = "Width angles must satisfy 0 < width_rad < π"
        raise ValueError(msg)

    x0, z0 = _angles_to_virtual_source(aperture_length, tilt_rad, width_rad)

    xp = array_namespace(element_positions)
    focus = xp.asarray([x0, z0])

    # Delegate to focused() for delay computation
    return focused(element_positions, focus, speed_of_sound=speed_of_sound)


def _angles_to_virtual_source(
    aperture_length: float,
    tilt_rad: float,
    width_rad: float,
) -> tuple[float, float]:
    """Convert tilt and width angles to virtual source position.

    Args:
        aperture_length: Array aperture length in meters.
        tilt_rad: Tilt angle in radians.
        width_rad: Angular width in radians.

    Returns:
        Tuple of (x0_m, z0_m) virtual source position in meters.
    """
    # Normalize tilt to [-π/2, π/2]
    tilt_norm = fmod(-tilt_rad + pi / 2, 2 * pi) - pi / 2
    sign_correction = 1.0

    if abs(tilt_norm) > pi / 2:
        tilt_norm = pi - tilt_norm
        sign_correction = -1.0

    denominator = tan(tilt_norm - width_rad / 2) - tan(tilt_norm + width_rad / 2)
    z0 = sign_correction * aperture_length / denominator
    x0 = sign_correction * z0 * tan(width_rad / 2 - tilt_norm) + aperture_length / 2

    return x0, z0
