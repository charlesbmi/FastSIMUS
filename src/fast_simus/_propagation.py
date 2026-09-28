"""Shared homogeneous-medium phase and frequency-linear attenuation primitives."""

from math import pi


def propagation_exponential(distance, wavenumber, attenuation_wavenumber, xp, *, phase_offset=0.0, wrap=False):
    """Complex propagation factor; spreading and visibility belong to element models."""
    phase = xp.asarray(wavenumber) * distance
    if wrap:
        two_pi = xp.asarray(2 * pi)
        phase = phase - two_pi * xp.floor(phase / two_pi)
    if not isinstance(phase_offset, (int, float)) or phase_offset != 0:
        phase = phase + phase_offset
    return xp.exp(-xp.asarray(attenuation_wavenumber) * distance + xp.asarray(1j) * phase)
