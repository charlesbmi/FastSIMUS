"""Transmit delay calculations for ultrasound transducer arrays.

This module provides functions to compute transmit time delays for different
beam patterns (focused, plane wave, circular wave) with linear and convex arrays.

All functions are Array API compliant and work with NumPy, JAX, CuPy backends.
"""

from __future__ import annotations

from math import inf, pi
from typing import TYPE_CHECKING

from fast_simus.transducer_params import TransducerParams

if TYPE_CHECKING:
    from array_api_compat import Array


def _compute_element_positions(
    params: TransducerParams,
) -> tuple[Array, Array, Array | None, float]:
    """Compute transducer element positions."""
    import numpy as np

    n = params.n_elements
    pitch = params.pitch
    radius = params.radius
    is_linear = radius == inf

    if is_linear:
        x = ((np.arange(n) - (n - 1) / 2) * pitch).astype(np.float64)
        z = np.zeros(n, dtype=np.float64)
        theta = None
        h = 0.0
    else:
        chord = 2 * radius * np.sin(np.arcsin(pitch / 2 / radius) * (n - 1))
        h = np.sqrt(radius**2 - chord**2 / 4)
        theta = np.linspace(np.arctan2(-chord / 2, h), np.arctan2(chord / 2, h), n)
        z = radius * np.cos(theta) - h
        x = radius * np.sin(theta)

    return x, z, theta, h


def compute_focused_delays(
    params: TransducerParams,
    x0: float | Array,
    z0: float | Array,
) -> Array:
    """Compute transmit delays for focused beam(s)."""
    import numpy as np

    x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64)).reshape(-1, 1)
    z0_arr = np.atleast_1d(np.asarray(z0, dtype=np.float64)).reshape(-1, 1)

    if x0_arr.shape != z0_arr.shape:
        msg = "x0 and z0 must have the same length"
        raise ValueError(msg)

    x_elem, z_elem, _, h = _compute_element_positions(params)
    x_elem = np.asarray(x_elem)
    z_elem = np.asarray(z_elem)
    c = params.speed_of_sound

    distances = np.sqrt((x_elem - x0_arr) ** 2 + (z_elem - z0_arr) ** 2)

    if params.radius == inf:
        sgn = np.sign(z0_arr)
        sgn = np.where(sgn == 0, -1, sgn)
        delays = -distances * sgn / c
    else:
        radius = params.radius
        sgn = np.sign(x0_arr**2 + (z0_arr + h) ** 2 - radius**2)
        sgn = np.where(sgn == 0, -1, sgn)
        delays = -distances * sgn / c

    delays = delays - np.min(delays, axis=-1, keepdims=True)
    return delays


def compute_plane_wave_delays(
    params: TransducerParams,
    tilt: float | Array,
) -> Array:
    """Compute transmit delays for plane wave(s)."""
    import numpy as np

    tilt_arr = np.atleast_1d(np.asarray(tilt, dtype=np.float64)).reshape(-1, 1)

    if np.any(np.abs(tilt_arr) >= pi / 2):
        msg = "Tilt angles must satisfy |tilt| < π/2"
        raise ValueError(msg)

    x_elem, z_elem, _, h = _compute_element_positions(params)
    x_elem = np.asarray(x_elem)
    z_elem = np.asarray(z_elem)
    c = params.speed_of_sound

    if params.radius == inf:
        delays = x_elem * np.sin(tilt_arr) / c
    else:
        radius = params.radius
        xn = radius * np.sin(tilt_arr)
        zn = radius * np.cos(tilt_arr) - h
        numerator = np.abs(z_elem + xn / (zn + h) * x_elem - xn**2 / (zn + h) - zn)
        denominator = np.sqrt(1 + xn**2 / (zn + h) ** 2)
        d = numerator / denominator
        delays = -d / c

    delays = delays - np.min(delays, axis=-1, keepdims=True)
    return delays


def compute_circular_wave_delays(
    params: TransducerParams,
    tilt: float | Array,
    width: float | Array,
) -> Array:
    """Compute transmit delays for circular wave(s)."""
    import numpy as np

    if params.radius != inf:
        msg = "Circular wave delays are not supported for convex arrays"
        raise ValueError(msg)

    tilt_arr = np.atleast_1d(np.asarray(tilt, dtype=np.float64)).reshape(-1, 1)
    width_arr = np.atleast_1d(np.asarray(width, dtype=np.float64)).reshape(-1, 1)

    if tilt_arr.shape != width_arr.shape:
        msg = "tilt and width must have the same length"
        raise ValueError(msg)

    if np.any(width_arr <= 0) or np.any(width_arr >= pi):
        msg = "Width angles must satisfy 0 < width < π"
        raise ValueError(msg)

    x_elem, _, _, _ = _compute_element_positions(params)
    x_elem = np.asarray(x_elem)
    L = (params.n_elements - 1) * params.pitch
    x0, z0 = _angles_to_origin(L, tilt_arr, width_arr)
    c = params.speed_of_sound

    distances = np.sqrt((x_elem - x0) ** 2 + z0**2)
    delays = -distances * np.sign(z0) / c
    delays = delays - np.min(delays, axis=-1, keepdims=True)
    return delays


def _angles_to_origin(L: float, tilt: Array, width: Array) -> tuple[Array, Array]:
    """Convert tilt and width angles to virtual source position."""
    import numpy as np

    tilt_norm = np.mod(-tilt + pi / 2, 2 * pi) - pi / 2
    sign_correction = np.ones_like(tilt_norm)
    idx = np.abs(tilt_norm) > pi / 2
    tilt_norm[idx] = pi - tilt_norm[idx]
    sign_correction[idx] = -1

    denominator = np.tan(tilt_norm - width / 2) - np.tan(tilt_norm + width / 2)
    z0 = sign_correction * L / denominator
    x0 = sign_correction * z0 * np.tan(width / 2 - tilt_norm) + L / 2

    return x0, z0
