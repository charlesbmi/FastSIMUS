"""Finite-aperture pressure orchestration over the shared element response."""

from math import pi

from fast_simus._compat import _clean_transmit_inputs
from fast_simus._contractions import _transmit_pressure
from fast_simus._transfer_3d import rectangle_response
from fast_simus.plans import response_medium
from fast_simus.utils._array_api import array_namespace


def field_spectrum(positions, delays, plan, params, medium, apodization, full_directivity, strategy=None):
    """Compute raw selected pressure samples with preserved spatial axes."""
    plan.check_static(positions, delays, params, medium)
    if strategy in ("metal", "cuda"):
        raise NotImplementedError("Native kernels do not support finite 3D apertures")
    xp = array_namespace(positions, delays, apodization)
    if apodization is not None and apodization.shape != delays.shape:
        raise ValueError("Apodization must have shape (E,)")
    delays, apodization = _clean_transmit_inputs(delays, apodization, params.n_elements, xp)
    points = xp.reshape(positions, (-1, 3))
    physics = response_medium(plan)
    samples = []
    for k in range(plan.selected_freqs.shape[0]):
        frequency = (plan.freq_idx_start + k) * plan.freq_step
        h = rectangle_response(
            points, params.aperture, plan._counts, frequency, params.freq_center, physics, full_directivity, xp
        )
        excitation = xp.exp(2j * pi * frequency * delays) * apodization
        pressure = _transmit_pressure(h, excitation, plan._pulse[k] * plan._probe[k], xp.zeros(points.shape[0]) < 0, xp)
        samples.append(pressure)
    return xp.reshape(xp.stack(samples, axis=-1), (*positions.shape[:-1], len(samples)))
