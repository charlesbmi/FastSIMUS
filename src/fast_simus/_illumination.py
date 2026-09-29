"""Incident pressure and scattering strength shared by RF and point observers."""

from fast_simus._field import transmit_at_frequency


def scatterer_illumination(points, coefficients, delays, apodization, plan, k, full_directivity, xp):
    """Return rc times incident pressure, including exactly one transmit response."""
    frequency = (plan.freq_idx_start + k) * plan.freq_step
    pressure = transmit_at_frequency(points, delays, apodization, plan, frequency, full_directivity, xp)
    return coefficients * pressure * plan._pulse[k] * plan._probe[k]
