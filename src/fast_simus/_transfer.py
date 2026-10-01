"""Element-to-point transfer setup shared by pressure and pulse-echo sweeps.

The strip model retains cylindrical spreading and its existing normalization.
Output contractions decide whether to preserve element identity or flatten it.
"""

from math import inf, pi
from typing import NamedTuple

import array_api_extra as xpx

from fast_simus._pfield_math import _distances_and_angles, _init_exponentials, _obliquity_factor, _subelement_centroids
from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace
from fast_simus.utils.geometry import element_positions


class _TransferPlan(NamedTuple):
    """Frequency samples and strip subdivision used by either output path."""

    selected_freqs: Array
    n_sub: int
    seg_length: float
    freq_start: float
    freq_step: float


class _Transfer(NamedTuple):
    """Geometric and transmit progressions on a uniform frequency grid."""

    phase: Array
    phase_step: Array
    delay_apod: Array
    delay_apod_step: Array
    sin_theta: Array
    is_out: Array
    wavenumbers: Array


def _prepare_strip_transfer(
    positions: Array,
    delays: Array,
    apodization: Array,
    plan: _TransferPlan,
    params: TransducerParams,
    medium: MediumParams,
    *,
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> _Transfer:
    """Build one transfer for both TX and reciprocal RX, without conjugation."""
    element_pos, theta_elements, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(params.n_elements)
    offsets = _subelement_centroids(params.element_width, plan.n_sub, theta_elements, xp)
    x, z = positions[..., 0], positions[..., 1]
    is_out = z < 0
    if params.radius != inf:
        is_out = is_out | ((x**2 + (z + apex_offset) ** 2) <= params.radius**2)
    distances, sin_theta, theta = _distances_and_angles(
        positions, offsets, element_pos, theta_elements, medium.speed_of_sound, params.freq_center, xp
    )
    obliquity = _obliquity_factor(theta, params.baffle, xp)
    freq_start = plan.freq_start
    freq_step = plan.freq_step
    phase, phase_step = _init_exponentials(
        freq_start,
        medium.speed_of_sound,
        medium.attenuation,
        distances,
        obliquity,
        freq_step,
        xp,
    )
    if not full_frequency_directivity:
        center_wavenumber = 2.0 * pi * params.freq_center / medium.speed_of_sound
        sinc_arg = xp.asarray(center_wavenumber * plan.seg_length / 2.0) * sin_theta / pi
        phase = phase * xpx.sinc(sinc_arg, xp=xp)
    delay_apod = xp.exp(xp.asarray(1j * 2.0 * pi) * freq_start * delays) * apodization
    delay_apod_step = xp.exp(xp.asarray(1j * 2.0 * pi) * freq_step * delays)
    wavenumbers = xp.asarray(2.0 * pi) * plan.selected_freqs / medium.speed_of_sound
    return _Transfer(phase, phase_step, delay_apod, delay_apod_step, sin_theta, is_out, wavenumbers)
