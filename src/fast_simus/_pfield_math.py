"""Physics helpers for pfield computation.

Pure Array API functions for geometry, phase initialization, frequency
selection, and obliquity. No loop structure or backend-specific code.
"""

from __future__ import annotations

from math import ceil, log, pi
from typing import NamedTuple

from beartype import beartype as typechecker
from jaxtyping import Complex, Float, jaxtyped

from fast_simus.spectrum import probe_spectrum, pulse_spectrum
from fast_simus.transducer_params import BaffleType
from fast_simus.utils._array_api import Array, _ArrayNamespace

# Conversion factor: Nepers to dB -- 20/log(10) ≈ 8.6859
# Shared by Array API path and Metal kernel wrapper.
NEPER_TO_DB = 20.0 / log(10.0)


class _FrequencyPlan(NamedTuple):
    """Frequency sampling plan for pfield computation.

    Attributes:
        selected_freqs: Selected frequencies, shape
        pulse_spectrum: Pulse spectrum at selected frequencies, shape
        probe_spectrum: Probe spectrum at selected frequencies, shape
        freq_step: Frequency step in Hz
    """

    selected_freqs: Float[Array, " n_frequencies"]
    pulse_spectrum: Complex[Array, " n_frequencies"]
    probe_spectrum: Float[Array, " n_frequencies"]
    freq_step: float


@jaxtyped(typechecker=typechecker)
def _subelement_centroids(
    element_width: float,
    n_sub: int,
    theta_e: Float[Array, " n_elements"],
    xp: _ArrayNamespace,
) -> Float[Array, "n_elements n_sub 2"]:
    """Compute sub-element centroid positions relative to element centers.

    Args:
        element_width: Element width in meters.
        n_sub: Number of sub-elements per element.
        theta_e: Element angular positions in radians.
        xp: Array namespace.

    Returns:
        Sub-element offsets with shape (n_elements, n_sub, 2) where [..., 0]
        is lateral (x) and [..., 1] is axial (z).
    """
    seg_length = element_width / n_sub
    seg_offsets = xp.asarray([-element_width / 2.0 + seg_length / 2.0 + i * seg_length for i in range(n_sub)])
    # Broadcasting: (n_sub,) -> (1, n_sub), (n_elements,) -> (n_elements, 1)
    seg_offsets_2d = xp.reshape(seg_offsets, (1, n_sub))
    cos_theta = xp.cos(theta_e)[:, None]
    sin_neg_theta = xp.sin(-theta_e)[:, None]
    subelement_dx = seg_offsets_2d * cos_theta
    subelement_dz = seg_offsets_2d * sin_neg_theta
    return xp.stack([subelement_dx, subelement_dz], axis=-1)


@jaxtyped(typechecker=typechecker)
def _distances_and_angles(
    points: Float[Array, "*batch 2"],
    subelement_offsets: Float[Array, "n_elements n_sub 2"],
    element_pos: Float[Array, "n_elements 2"],
    theta_e: Float[Array, " n_elements"],
    speed_of_sound: float,
    freq_center: float,
    xp: _ArrayNamespace,
) -> tuple[
    Float[Array, "*batch n_elements n_sub"],
    Float[Array, "*batch n_elements n_sub"],
    Float[Array, "*batch n_elements n_sub"],
]:
    """Compute distances and angles from grid points to sub-elements.

    Args:
        points: Grid point positions.
        subelement_offsets: Sub-element offsets.
        element_pos: Element positions.
        theta_e: Element angular positions.
        speed_of_sound: Speed of sound in m/s.
        freq_center: Center frequency in Hz.
        xp: Array namespace.

    Returns:
        Tuple of (distances, sin_theta, theta_arr):
        - distances: Distances from grid points to sub-elements.
        - sin_theta: Sine of angles relative to element normal.
        - theta_arr: Angles relative to element normal.
    """
    delta: Float[Array, "*batch n_elements n_sub xz=2"] = (
        points[..., None, None, :] - subelement_offsets - element_pos[:, None, :]
    )
    delta_x = delta[..., 0]
    delta_z = delta[..., 1]
    dist_squared = delta_x**2 + delta_z**2
    distances = xp.sqrt(dist_squared)

    # Distances with clipping (use unclipped sqrt for angle computation)
    min_distance = xp.asarray(speed_of_sound / freq_center / 2.0)
    distances_clipped = xp.where(distances < min_distance, min_distance, distances)

    # Angle relative to element normal
    _div_eps = xp.asarray(1e-16)  # Numerical stability for division
    theta_arr = xp.asin((delta_x + _div_eps) / (distances + _div_eps)) - theta_e[:, None]
    sin_theta = xp.sin(theta_arr)

    return distances_clipped, sin_theta, theta_arr


def _select_frequencies(
    fc: float,
    bandwidth: float,
    tx_n_wavelengths: float,
    db_thresh: float,
    max_freq_step: float,
    xp: _ArrayNamespace,
) -> _FrequencyPlan:
    """Select frequency samples for pfield computation.

    Args:
        fc: Center frequency in Hz.
        bandwidth: Fractional bandwidth.
        tx_n_wavelengths: Number of wavelengths in TX pulse.
        db_thresh: Threshold in dB for frequency component selection.
        max_freq_step: Upper bound for frequency step.
        xp: Array namespace.

    Returns:
        FrequencyPlan with selected frequencies and spectra.
    """
    # Frequency samples
    n_freq = int(2 * ceil(fc / max_freq_step) + 1)
    frequencies = xp.linspace(0, 2 * fc, n_freq)
    freq_step = float(frequencies[1])

    # Keep only significant components (dB threshold)
    angular_freqs_all = xp.asarray(2.0 * pi) * frequencies
    spectrum_magnitude = xp.abs(
        pulse_spectrum(angular_freqs_all, fc, tx_n_wavelengths) * probe_spectrum(angular_freqs_all, fc, bandwidth)
    )
    gain_db = 20.0 * xp.log10(xp.asarray(1e-200) + spectrum_magnitude / xp.max(spectrum_magnitude))
    above_threshold = gain_db > db_thresh
    idx_first, idx_last = _first_last_true(xp, above_threshold)

    selected_freqs = frequencies[idx_first : idx_last + 1]

    angular_freqs_sel = xp.asarray(2.0 * pi) * selected_freqs
    pulse_spect = pulse_spectrum(angular_freqs_sel, fc, tx_n_wavelengths)
    probe_spect = probe_spectrum(angular_freqs_sel, fc, bandwidth)

    return _FrequencyPlan(selected_freqs, pulse_spect, probe_spect, freq_step)


@jaxtyped(typechecker=typechecker)
def _obliquity_factor(
    theta_arr: Float[Array, "*batch n_elements n_sub"],
    baffle: BaffleType | float,
    xp: _ArrayNamespace,
) -> Float[Array, "*batch n_elements n_sub"]:
    """Compute obliquity factor based on baffle type.

    Args:
        theta_arr: Angles relative to element normal.
        baffle: Baffle type or impedance ratio.
        xp: Array namespace.

    Returns:
        Obliquity factor.
    """
    non_rigid_baffle = baffle != BaffleType.RIGID
    _horizon_floor = xp.asarray(1e-16)  # Near-zero for beyond-hemisphere angles

    if non_rigid_baffle:
        if baffle == BaffleType.SOFT:
            obliquity_factor = xp.cos(theta_arr)
        else:
            cos_th = xp.cos(theta_arr)
            obliquity_factor = cos_th / (cos_th + float(baffle))
    else:
        obliquity_factor = xp.ones(theta_arr.shape)

    obliquity_factor = xp.where(
        xp.abs(theta_arr) >= xp.asarray(pi / 2),
        _horizon_floor,
        obliquity_factor,
    )

    return obliquity_factor


@jaxtyped(typechecker=typechecker)
def _init_exponentials(
    freq_start: float,
    speed_of_sound: float,
    attenuation: float,
    distances: Float[Array, "*batch n_elements n_sub"],
    obliquity_factor: Float[Array, "*batch n_elements n_sub"],
    freq_step: float,
    xp: _ArrayNamespace,
) -> tuple[
    Complex[Array, "*batch n_elements n_sub"],
    Complex[Array, "*batch n_elements n_sub"],
]:
    """Initialize exponential arrays for frequency loop.

    Args:
        freq_start: Initial frequency in Hz.
        speed_of_sound: Speed of sound in m/s.
        attenuation: Attenuation coefficient in dB/cm/MHz.
        distances: Distances.
        obliquity_factor: Obliquity factor.
        freq_step: Frequency step in Hz.
        xp: Array namespace.

    Returns:
        Tuple of (phase_decay, phase_decay_step):
        - phase_decay: Initial complex exponential array.
        - phase_decay_step: Complex exponential increment per frequency step.
    """
    wavenumber_init = 2.0 * pi * freq_start / speed_of_sound
    attenuation_wavenum = attenuation / NEPER_TO_DB * freq_start / 1e6 * 1e2

    # exp(-kwa*distances + 1j*mod(kw*distances, 2pi))
    kw0_r = xp.asarray(wavenumber_init) * distances
    two_pi = xp.asarray(2.0 * pi)
    phase_mod = kw0_r - two_pi * xp.floor(kw0_r / two_pi)
    phase_decay = xp.exp(xp.asarray(-attenuation_wavenum) * distances + xp.asarray(1j) * phase_mod)

    wavenumber_step = 2.0 * pi * freq_step / speed_of_sound
    attenuation_step = attenuation / NEPER_TO_DB * freq_step / 1e6 * 1e2
    phase_decay_step = xp.exp(xp.asarray(-attenuation_step + 1j * wavenumber_step) * distances)

    # Incorporate obliquity / sqrt(distances) (2D, no elevation)
    phase_decay = phase_decay * obliquity_factor / xp.sqrt(distances)

    return phase_decay, phase_decay_step


def _first_last_true(xp: _ArrayNamespace, mask: Array) -> tuple[int, int]:
    """Find first and last True index in 1D boolean array. JAX-compatible (no nonzero)."""
    n = mask.shape[0]
    if n == 0:
        return 0, 0
    # Cast to int: argmax on bool not allowed by array_api_strict
    mask_int = xp.asarray(mask, dtype=xp.int32)
    first = int(xp.argmax(mask_int))
    if int(xp.max(mask_int)) == 0:
        return 0, 0
    last = n - 1 - int(xp.argmax(mask_int[::-1]))
    return first, last
