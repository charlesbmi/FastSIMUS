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

from math import inf, pi
from typing import TYPE_CHECKING, Any

import array_api_extra as xpx
from array_api_compat import array_namespace
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

if TYPE_CHECKING:
    pass

# Type alias for Array API objects (until protocol is standardized)
ArrayAPIObj = Any


@jaxtyped(typechecker=typechecker)
def compute_focused_delays(
    element_positions: Float[ArrayAPIObj, "n_elements xz=2"],
    speed_of_sound: float,
    radius: float,
    focus: Float[ArrayAPIObj, "*batch xz=2"] | float,
    apex_offset: float = 0.0,
) -> Float[ArrayAPIObj, "*batch n_elements"]:
    """Compute transmit time delays for focused or diverging spherical waves.

    Spherical waves propagate like a collapsing sphere focusing onto a point
    (positive z), or an expanding sphere diverging from a virtual source
    (negative z).

    Args:
        element_positions:
            Element (x, z) positions in meters. Shape (n_elements, 2).
            - element_positions[:, 0]: Lateral positions (x)
            - element_positions[:, 1]: Axial positions (z)
        speed_of_sound:
            Speed of sound in m/s.
        radius:
            Curvature radius in meters. Use inf for linear arrays.
        focus:
            Focal position(s) in meters as [..., (x, z)] coordinates.
            Shape (*batch, 2) or (2,) for single focus.
            - focus[..., 0]: Lateral position (x), center of array at 0
            - focus[..., 1]: Axial position (z)
              - Positive z: Focused wave (focus in front of transducer)
              - Negative z: Diverging wave (virtual source behind transducer)
              - z=0: Treated as focusing just in front (negative delays)
        apex_offset:
            Distance from array center to arc apex in meters. Zero for linear arrays.
            Defaults to 0.0.

    Returns:
        Transmit time delays in seconds. Shape (*batch, n_elements).
        Delays are relative to minimum (all non-negative).
        Each row corresponds to one beam configuration.

    Examples:
        >>> from fast_simus import compute_focused_delays
        >>> from fast_simus.utils.geometry import element_positions
        >>> from fast_simus.transducer_presets import P4_2v
        >>> from array_api_compat import array_namespace
        >>> import numpy as np
        >>> from math import inf
        >>>
        >>> # Single focused beam at 5cm depth, centered laterally
        >>> params = P4_2v()
        >>> xp = array_namespace(np.array([1.0]))
        >>> x, z, _, apex = element_positions(
        ...     params.n_elements, params.pitch, params.radius, xp
        ... )
        >>> elem_pos = np.stack([x, z], axis=-1)
        >>> focus = np.array([0.0, 0.05])  # [x, z]
        >>> delays = compute_focused_delays(
        ...     elem_pos, params.speed_of_sound, params.radius, focus, apex
        ... )
        >>> delays.shape
        (64,)  # One delay per element
        >>>
        >>> # Vectorized: multiple focal points
        >>> focus = np.array([[0.0, 0.04], [0.01, 0.05], [0.02, 0.06]])
        >>> delays = compute_focused_delays(
        ...     elem_pos, params.speed_of_sound, params.radius, focus, apex
        ... )
        >>> delays.shape
        (3, 64)  # 3 beams x 64 elements
        >>>
        >>> # Diverging wave (virtual source behind array)
        >>> focus = np.array([0.0, -0.03])
        >>> delays = compute_focused_delays(
        ...     elem_pos, params.speed_of_sound, params.radius, focus, apex
        ... )
        >>> # Creates expanding wavefront

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
    # Convert to array and get namespace
    xp = array_namespace(focus, element_positions)

    # Ensure focus has shape (*batch, 2)
    focus_arr: Float[ArrayAPIObj, "batch xz=2"] = xpx.atleast_nd(focus, ndim=2, xp=xp)

    x0_arr = focus_arr[..., :1]
    z0_arr = focus_arr[..., 1:2]

    # Extract element positions
    elem_pos_arr = xp.asarray(element_positions)
    x_elem = elem_pos_arr[:, 0:1]  # Shape (n_elements, 1)
    z_elem = elem_pos_arr[:, 1:2]  # Shape (n_elements, 1)

    c = speed_of_sound

    # Euclidean distance from each element to each focal point
    distances = xp.sqrt((x_elem - x0_arr) ** 2 + (z_elem - z0_arr) ** 2)

    if radius == inf:
        # Linear array: sign based on axial position
        sgn = xp.sign(z0_arr)
        sgn = xp.where(sgn == 0, -1, sgn)
        delays = -distances * sgn / c
    else:
        # Convex array: sign based on inside/outside arc
        sgn = xp.sign(x0_arr**2 + (z0_arr + apex_offset) ** 2 - radius**2)
        sgn = xp.where(sgn == 0, -1, sgn)
        delays = -distances * sgn / c

    # Make all delays non-negative by subtracting minimum
    delays = delays - xp.min(delays, axis=-1, keepdims=True)
    return delays


@jaxtyped(typechecker=typechecker)
def plane_wave(
    element_positions: Float[ArrayAPIObj, "n_elements xz=2"],
    speed_of_sound: float,
    radius: float,
    tilt_rad: Float[ArrayAPIObj, "*batch"] | float,
    apex_offset: float = 0.0,
) -> Float[ArrayAPIObj, "*batch n_elements"]:
    """Compute transmit time delays for plane wave transmission.

    Plane waves have a flat wavefront propagating in a specified direction,
    defined by the tilt angle from the array normal (z-axis).

    Args:
        element_positions:
            Element (x, z) positions in meters. Shape (n_elements, 2).
            - element_positions[:, 0]: Lateral positions (x)
            - element_positions[:, 1]: Axial positions (z)
        speed_of_sound:
            Speed of sound in m/s.
        radius:
            Curvature radius in meters. Use inf for linear arrays.
        tilt_rad:
            Tilt angle(s) in radians. Shape (*batch,).
            - tilt_rad=0: Straight ahead (perpendicular to array)
            - Positive tilt_rad: Beam steers right (positive x direction)
            - Negative tilt_rad: Beam steers left (negative x direction)
            Must satisfy |tilt_rad| < π/2.
        apex_offset:
            Distance from array center to arc apex in meters. Zero for linear arrays.
            Defaults to 0.0.

    Returns:
        Transmit time delays in seconds. Shape (*batch, n_elements).
        Delays are relative to minimum (all non-negative).
        Each row corresponds to one beam angle.

    Raises:
        ValueError: If any |tilt_rad| >= π/2 (non-physical angle).

    Examples:
        >>> from fast_simus import plane_wave
        >>> from fast_simus.utils.geometry import element_positions
        >>> from fast_simus.transducer_presets import P4_2v
        >>> from array_api_compat import array_namespace
        >>> import numpy as np
        >>> from math import inf
        >>>
        >>> # Single plane wave at 10 degrees
        >>> params = P4_2v()
        >>> xp = array_namespace(np.array([1.0]))
        >>> x, z, _, apex = element_positions(
        ...     params.n_elements, params.pitch, params.radius, xp
        ... )
        >>> elem_pos = np.stack([x, z], axis=-1)
        >>> delays = plane_wave(
        ...     elem_pos, params.speed_of_sound, params.radius,
        ...     np.radians(10), apex
        ... )
        >>> delays.shape
        (64,)
        >>>
        >>> # Compound angle imaging: -20°, -10°, 0°, 10°, 20°
        >>> angles = np.radians([-20, -10, 0, 10, 20])
        >>> delays = plane_wave(
        ...     elem_pos, params.speed_of_sound, params.radius, angles, apex
        ... )
        >>> delays.shape
        (5, 64)  # 5 angles x 64 elements
        >>>
        >>> # Zero angle gives zero delays
        >>> delays_zero = plane_wave(
        ...     elem_pos, params.speed_of_sound, params.radius, 0.0, apex
        ... )
        >>> np.allclose(delays_zero, 0.0)
        True

    Notes:
        For linear arrays, delays follow simple sinusoidal pattern based on
        lateral element position.

        For convex arrays, the computation accounts for the curved geometry
        by computing distance to the plane perpendicular to the tilt direction.
    """
    # Convert to array and get namespace
    xp = array_namespace(tilt_rad, element_positions)

    # Ensure tilt_rad has shape (*batch, 1) for broadcasting
    tilt_arr = xp.asarray(tilt_rad)
    if tilt_arr.ndim == 0:
        tilt_arr = xp.reshape(tilt_arr, (1,))
    tilt_arr = xp.reshape(tilt_arr, (-1, 1))

    if xp.any(xp.abs(tilt_arr) >= pi / 2):
        msg = "Tilt angles must satisfy |tilt_rad| < π/2"
        raise ValueError(msg)

    # Extract element positions
    elem_pos_arr = xp.asarray(element_positions)
    x_elem = elem_pos_arr[:, 0:1]  # Shape (n_elements, 1)
    z_elem = elem_pos_arr[:, 1:2]  # Shape (n_elements, 1)

    c = speed_of_sound

    if radius == inf:
        # Linear array: delay proportional to lateral position
        delays = x_elem * xp.sin(tilt_arr) / c
    else:
        # Convex array: geometric calculation
        xn = radius * xp.sin(tilt_arr)
        zn = radius * xp.cos(tilt_arr) - apex_offset
        numerator = xp.abs(z_elem + xn / (zn + apex_offset) * x_elem - xn**2 / (zn + apex_offset) - zn)
        denominator = xp.sqrt(1 + xn**2 / (zn + apex_offset) ** 2)
        d = numerator / denominator
        delays = -d / c

    # Make all delays non-negative
    delays = delays - xp.min(delays, axis=-1, keepdims=True)
    return delays


@jaxtyped(typechecker=typechecker)
def diverging_wave(
    element_positions: Float[ArrayAPIObj, "n_elements xz=2"],
    speed_of_sound: float,
    radius: float,
    angles: Float[ArrayAPIObj, "*batch angles=2"] | float,
    aperture_length: float,
) -> Float[ArrayAPIObj, "*batch n_elements"]:
    """Compute transmit time delays for diverging circular wave transmission.

    Circular waves originate from a virtual point source positioned such that
    the wavefront spans a specified angular width across the array.
    Only supported for linear arrays.

    Args:
        element_positions:
            Element (x, z) positions in meters. Shape (n_elements, 2).
            - element_positions[:, 0]: Lateral positions (x)
            - element_positions[:, 1]: Axial positions (z)
        speed_of_sound:
            Speed of sound in m/s.
        radius:
            Curvature radius in meters. Must be inf for linear arrays.
        angles:
            Wave angles as [..., (tilt, width)] in radians. Shape (*batch, 2) or (2,).
            - angles[..., 0]: Tilt angle of the wave center
            - angles[..., 1]: Angular width of the wave
              Must satisfy 0 < width < π.
        aperture_length:
            Array aperture length in meters (typically (n_elements - 1) * pitch).

    Returns:
        Transmit time delays in seconds. Shape (*batch, n_elements).
        Delays are relative to minimum (all non-negative).

    Raises:
        ValueError: If radius is not inf (convex arrays not supported).
        ValueError: If any width is outside (0, π).

    Examples:
        >>> from fast_simus import diverging_wave
        >>> from fast_simus.utils.geometry import element_positions
        >>> from fast_simus.transducer_presets import P4_2v
        >>> from array_api_compat import array_namespace
        >>> import numpy as np
        >>> from math import inf
        >>>
        >>> # Single circular wave: 30° width, 10° tilt
        >>> params = P4_2v()
        >>> xp = array_namespace(np.array([1.0]))
        >>> x, z, _, _ = element_positions(
        ...     params.n_elements, params.pitch, params.radius, xp
        ... )
        >>> elem_pos = np.stack([x, z], axis=-1)
        >>> angles = np.array([np.radians(10), np.radians(30)])  # [tilt, width]
        >>> L = (params.n_elements - 1) * params.pitch
        >>> delays = diverging_wave(
        ...     elem_pos, params.speed_of_sound, params.radius, angles, L
        ... )
        >>> delays.shape
        (64,)
        >>>
        >>> # Multiple configurations
        >>> angles = np.array([
        ...     [0.0, np.radians(30)],
        ...     [np.radians(10), np.radians(45)]
        ... ])
        >>> delays = diverging_wave(
        ...     elem_pos, params.speed_of_sound, params.radius, angles, L
        ... )
        >>> delays.shape
        (2, 64)

    Notes:
        The virtual source position is computed from the tilt and width angles
        using geometric constraints. This creates a diverging spherical wave
        that appears to originate from behind the transducer.
    """
    if radius != inf:
        msg = "Diverging wave delays are not supported for convex arrays"
        raise ValueError(msg)

    # Convert to array and get namespace
    xp = array_namespace(angles, element_positions)

    # Ensure angles has shape (*batch, 2)
    angles_arr: Float[ArrayAPIObj, "batch angles=2"] = xpx.atleast_nd(angles, ndim=2, xp=xp)

    tilt_arr = angles_arr[..., :1]
    width_arr = angles_arr[..., 1:2]

    if xp.any(width_arr <= 0) or xp.any(width_arr >= pi):
        msg = "Width angles must satisfy 0 < width_rad < π"
        raise ValueError(msg)

    # Extract element positions
    elem_pos_arr = xp.asarray(element_positions)
    x_elem = elem_pos_arr[:, 0:1]  # Shape (n_elements, 1)

    x0, z0 = _angles_to_origin(aperture_length, tilt_arr, width_arr)
    c = speed_of_sound

    distances = xp.sqrt((x_elem - x0) ** 2 + z0**2)
    delays = -distances * xp.sign(z0) / c
    delays = delays - xp.min(delays, axis=-1, keepdims=True)
    return delays


def _angles_to_origin(
    L: float,
    tilt_rad: Float[ArrayAPIObj, "*batch 1"],
    width_rad: Float[ArrayAPIObj, "*batch 1"],
) -> tuple[Float[ArrayAPIObj, "*batch 1"], Float[ArrayAPIObj, "*batch 1"]]:
    """Convert tilt and width angles to virtual source position.

    Args:
        L: Array aperture length in meters.
        tilt_rad: Tilt angles in radians. Shape (*batch, 1).
        width_rad: Angular width in radians. Shape (*batch, 1).

    Returns:
        Tuple of (x0_m, z0_m) virtual source positions. Each has shape (*batch, 1).
    """
    xp = array_namespace(tilt_rad, width_rad)

    # Normalize tilt to [-π/2, π/2]
    tilt_norm = xp.fmod(-tilt_rad + pi / 2, 2 * pi) - pi / 2
    sign_correction = xp.ones_like(tilt_norm)
    idx = xp.abs(tilt_norm) > pi / 2
    tilt_norm = xp.where(idx, pi - tilt_norm, tilt_norm)
    sign_correction = xp.where(idx, -1, sign_correction)

    denominator = xp.tan(tilt_norm - width_rad / 2) - xp.tan(tilt_norm + width_rad / 2)
    z0 = sign_correction * L / denominator
    x0 = sign_correction * z0 * xp.tan(width_rad / 2 - tilt_norm) + L / 2

    return x0, z0
