"""Prepared strip geometry, independent of field and pulse-echo outputs."""

from math import inf
from typing import NamedTuple

from fast_simus._pfield_math import _distances_and_angles, _obliquity_factor, _subelement_centroids
from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace
from fast_simus.utils.geometry import element_positions


class _StripGeometry(NamedTuple):
    """Point-to-subelement data; final axes retain element and subdivision identity."""

    distances: Array
    sin_theta: Array
    obliquity: Array
    is_out: Array


def _prepare_strip_geometry(
    positions: Array, n_sub: int, params: TransducerParams, medium: MediumParams, xp: _ArrayNamespace
) -> _StripGeometry:
    """Prepare the legacy 2D strip model, including its physical exclusion mask."""
    element_pos, theta_elements, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(params.n_elements)

    speed_of_sound = medium.speed_of_sound

    subelement_offsets = _subelement_centroids(params.element_width, n_sub, theta_elements, xp)

    x = positions[..., 0]
    z = positions[..., 1]
    is_out = z < 0
    if params.radius != inf:
        is_out = is_out | ((x**2 + (z + apex_offset) ** 2) <= params.radius**2)

    distances, sin_theta, theta_arr = _distances_and_angles(
        positions, subelement_offsets, element_pos, theta_elements, speed_of_sound, params.freq_center, xp
    )
    obliquity_factor = _obliquity_factor(theta_arr, params.baffle, xp)

    return _StripGeometry(distances, sin_theta, obliquity_factor, is_out)
