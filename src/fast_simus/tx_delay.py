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

import numpy as np
from array_api_compat import array_namespace
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped

from fast_simus.transducer_params import TransducerParams

if TYPE_CHECKING:
    pass

# Type alias for Array API objects (until protocol is standardized)
ArrayAPIObj = Any


def _compute_element_positions(
    params: TransducerParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """Compute transducer element positions.

    Args:
        params: Transducer configuration.

    Returns:
        Tuple of (x_positions_m, z_positions_m, theta_rad, apex_offset_m)
        where theta and apex_offset are None/0.0 for linear arrays.
    """
    n = params.n_elements
    pitch = params.pitch
    radius = params.radius
    is_linear = radius == inf

    if is_linear:
        x = ((np.arange(n) - (n - 1) / 2) * pitch).astype(np.float64)
        z = np.zeros(n, dtype=np.float64)
        theta = None
        apex_offset = 0.0
    else:
        chord = 2 * radius * np.sin(np.arcsin(pitch / 2 / radius) * (n - 1))
        apex_offset = np.sqrt(radius**2 - chord**2 / 4)
        theta = np.linspace(
            np.arctan2(-chord / 2, apex_offset),
            np.arctan2(chord / 2, apex_offset),
            n,
        )
        z = radius * np.cos(theta) - apex_offset
        x = radius * np.sin(theta)

    return x, z, theta, apex_offset


@jaxtyped(typechecker=typechecker)
def compute_focused_delays(
    params: TransducerParams,
    focus: Float[ArrayAPIObj, "*batch xz=2"] | float,
) -> Float[ArrayAPIObj, "*batch n_elements"]:
    """Compute transmit time delays for focused or diverging spherical waves.

    Spherical waves propagate like a collapsing sphere focusing onto a point
    (positive z), or an expanding sphere diverging from a virtual source
    (negative z).

    Args:
        params:
            Transducer configuration including geometry and sound speed.
        focus:
            Focal position(s) in meters as [..., (x, z)] coordinates.
            Shape (*batch, 2) or (2,) for single focus.
            - focus[..., 0]: Lateral position (x), center of array at 0
            - focus[..., 1]: Axial position (z)
              - Positive z: Focused wave (focus in front of transducer)
              - Negative z: Diverging wave (virtual source behind transducer)
              - z=0: Treated as focusing just in front (negative delays)

    Returns:
        Transmit time delays in seconds. Shape (*batch, n_elements).
        Delays are relative to minimum (all non-negative).
        Each row corresponds to one beam configuration.

    Examples:
        >>> from fast_simus import compute_focused_delays
        >>> from fast_simus.transducer_presets import P4_2v
        >>> import numpy as np
        >>>
        >>> # Single focused beam at 5cm depth, centered laterally
        >>> params = P4_2v()
        >>> focus = np.array([0.0, 0.05])  # [x, z]
        >>> delays = compute_focused_delays(params, focus=focus)
        >>> delays.shape
        (64,)  # One delay per element
        >>>
        >>> # Vectorized: multiple focal points
        >>> focus = np.array([[0.0, 0.04], [0.01, 0.05], [0.02, 0.06]])
        >>> delays = compute_focused_delays(params, focus=focus)
        >>> delays.shape
        (3, 64)  # 3 beams x 64 elements
        >>>
        >>> # Diverging wave (virtual source behind array)
        >>> focus = np.array([0.0, -0.03])
        >>> delays = compute_focused_delays(params, focus=focus)
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
    focus_arr = np.atleast_1d(np.asarray(focus, dtype=np.float64))
    xp = array_namespace(focus_arr)

    # Ensure focus has shape (*batch, 2)
    if focus_arr.ndim == 1:
        focus_arr = focus_arr[np.newaxis, :]

    x0_arr = focus_arr[..., :1]
    z0_arr = focus_arr[..., 1:2]

    x_elem, z_elem, _, apex_offset = _compute_element_positions(params)
    x_elem = xp.asarray(x_elem)
    z_elem = xp.asarray(z_elem)
    c = params.speed_of_sound

    # Euclidean distance from each element to each focal point
    distances = xp.sqrt((x_elem - x0_arr) ** 2 + (z_elem - z0_arr) ** 2)

    if params.radius == inf:
        # Linear array: sign based on axial position
        sgn = xp.sign(z0_arr)
        sgn = xp.where(sgn == 0, -1, sgn)
        delays = -distances * sgn / c
    else:
        # Convex array: sign based on inside/outside arc
        radius = params.radius
        sgn = xp.sign(x0_arr**2 + (z0_arr + apex_offset) ** 2 - radius**2)
        sgn = xp.where(sgn == 0, -1, sgn)
        delays = -distances * sgn / c

    # Make all delays non-negative by subtracting minimum
    delays = delays - xp.min(delays, axis=-1, keepdims=True)
    return delays


@jaxtyped(typechecker=typechecker)
def compute_plane_wave_delays(
    params: TransducerParams,
    tilt_rad: Float[ArrayAPIObj, "*batch"] | float,
) -> Float[ArrayAPIObj, "*batch n_elements"]:
    """Compute transmit time delays for plane wave transmission.

    Plane waves have a flat wavefront propagating in a specified direction,
    defined by the tilt angle from the array normal (z-axis).

    Args:
        params:
            Transducer configuration including geometry and sound speed.
        tilt_rad:
            Tilt angle(s) in radians. Shape (*batch,).
            - tilt_rad=0: Straight ahead (perpendicular to array)
            - Positive tilt_rad: Beam steers right (positive x direction)
            - Negative tilt_rad: Beam steers left (negative x direction)
            Must satisfy |tilt_rad| < π/2.

    Returns:
        Transmit time delays in seconds. Shape (*batch, n_elements).
        Delays are relative to minimum (all non-negative).
        Each row corresponds to one beam angle.

    Raises:
        ValueError: If any |tilt_rad| >= π/2 (non-physical angle).

    Examples:
        >>> from fast_simus import compute_plane_wave_delays
        >>> from fast_simus.transducer_presets import P4_2v
        >>> import numpy as np
        >>>
        >>> # Single plane wave at 10 degrees
        >>> params = P4_2v()
        >>> delays = compute_plane_wave_delays(params, tilt_rad=np.radians(10))
        >>> delays.shape
        (64,)
        >>>
        >>> # Compound angle imaging: -20°, -10°, 0°, 10°, 20°
        >>> angles = np.radians([-20, -10, 0, 10, 20])
        >>> delays = compute_plane_wave_delays(params, tilt_rad=angles)
        >>> delays.shape
        (5, 64)  # 5 angles x 64 elements
        >>>
        >>> # Zero angle gives zero delays
        >>> delays_zero = compute_plane_wave_delays(params, tilt_rad=0.0)
        >>> np.allclose(delays_zero, 0.0)
        True

    Notes:
        For linear arrays, delays follow simple sinusoidal pattern based on
        lateral element position.

        For convex arrays, the computation accounts for the curved geometry
        by computing distance to the plane perpendicular to the tilt direction.
    """
    # Convert scalars to arrays
    tilt_arr = np.atleast_1d(np.asarray(tilt_rad, dtype=np.float64)).reshape(-1, 1)

    # Get the appropriate namespace from the arrays
    xp = array_namespace(tilt_arr)

    if xp.any(xp.abs(tilt_arr) >= pi / 2):
        msg = "Tilt angles must satisfy |tilt_rad| < π/2"
        raise ValueError(msg)

    x_elem, z_elem, _, apex_offset = _compute_element_positions(params)
    x_elem = xp.asarray(x_elem)
    z_elem = xp.asarray(z_elem)
    c = params.speed_of_sound

    if params.radius == inf:
        # Linear array: delay proportional to lateral position
        delays = x_elem * xp.sin(tilt_arr) / c
    else:
        # Convex array: geometric calculation
        radius = params.radius
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
def compute_circular_wave_delays(
    params: TransducerParams,
    angles: Float[ArrayAPIObj, "*batch angles=2"] | float,
) -> Float[ArrayAPIObj, "*batch n_elements"]:
    """Compute transmit time delays for circular wave transmission.

    Circular waves originate from a virtual point source positioned such that
    the wavefront spans a specified angular width across the array.
    Only supported for linear arrays.

    Args:
        params:
            Transducer configuration. Must be a linear array (radius=inf).
        angles:
            Wave angles as [..., (tilt, width)] in radians. Shape (*batch, 2) or (2,).
            - angles[..., 0]: Tilt angle of the wave center
            - angles[..., 1]: Angular width of the wave
              Must satisfy 0 < width < π.

    Returns:
        Transmit time delays in seconds. Shape (*batch, n_elements).
        Delays are relative to minimum (all non-negative).

    Raises:
        ValueError: If params describes a convex array (not supported).
        ValueError: If any width is outside (0, π).

    Examples:
        >>> from fast_simus import compute_circular_wave_delays
        >>> from fast_simus.transducer_presets import P4_2v
        >>> import numpy as np
        >>>
        >>> # Single circular wave: 30° width, 10° tilt
        >>> params = P4_2v()
        >>> angles = np.array([np.radians(10), np.radians(30)])  # [tilt, width]
        >>> delays = compute_circular_wave_delays(params, angles=angles)
        >>> delays.shape
        (64,)
        >>>
        >>> # Multiple configurations
        >>> angles = np.array([
        ...     [0.0, np.radians(30)],
        ...     [np.radians(10), np.radians(45)]
        ... ])
        >>> delays = compute_circular_wave_delays(params, angles=angles)
        >>> delays.shape
        (2, 64)

    Notes:
        The virtual source position is computed from the tilt and width angles
        using geometric constraints. This creates a diverging spherical wave
        that appears to originate from behind the transducer.
    """
    if params.radius != inf:
        msg = "Circular wave delays are not supported for convex arrays"
        raise ValueError(msg)

    # Convert to array and get namespace
    angles_arr = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    xp = array_namespace(angles_arr)

    # Ensure angles has shape (*batch, 2)
    if angles_arr.ndim == 1:
        angles_arr = angles_arr[np.newaxis, :]

    tilt_arr = angles_arr[..., :1]
    width_arr = angles_arr[..., 1:2]

    if xp.any(width_arr <= 0) or xp.any(width_arr >= pi):
        msg = "Width angles must satisfy 0 < width_rad < π"
        raise ValueError(msg)

    x_elem, _, _, _ = _compute_element_positions(params)
    x_elem = xp.asarray(x_elem)
    L = (params.n_elements - 1) * params.pitch
    x0, z0 = _angles_to_origin(L, tilt_arr, width_arr)
    c = params.speed_of_sound

    distances = xp.sqrt((x_elem - x0) ** 2 + z0**2)
    delays = -distances * xp.sign(z0) / c
    delays = delays - xp.min(delays, axis=-1, keepdims=True)
    return delays


def _angles_to_origin(
    L: float,
    tilt_rad: np.ndarray,
    width_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert tilt and width angles to virtual source position.

    Args:
        L: Array aperture length in meters.
        tilt_rad: Tilt angles in radians.
        width_rad: Angular width in radians.

    Returns:
        Tuple of (x0_m, z0_m) virtual source positions.
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
