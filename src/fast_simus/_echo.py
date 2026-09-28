"""Pulse-echo orchestration using the same one-way transfer on both legs."""

from math import pi

from fast_simus._compat import _clean_transmit_inputs
from fast_simus._contractions import _receive_spectrum, _transmit_pressure
from fast_simus._transfer_3d import rectangle_response
from fast_simus.plans import response_medium
from fast_simus.utils._array_api import array_namespace


def echo_spectrum(points, rc, delays, plan, params, medium, apodization, full_directivity, strategy):
    """Compute selected unscaled channel spectra before event finalization."""
    plan.check_static(points, delays, params, medium)
    if rc.shape != points.shape[:-1]:
        raise ValueError("reflectivity must exactly match scatterer shape")
    if strategy in ("metal", "cuda"):
        raise NotImplementedError("Native kernels do not support finite 3D apertures")
    xp = array_namespace(points, rc, delays, apodization)
    if apodization is not None and apodization.shape != delays.shape:
        raise ValueError("Apodization must have shape (E,)")
    delays, apodization = _clean_transmit_inputs(delays, apodization, params.n_elements, xp)
    flat = xp.reshape(points, (-1, 3))
    rc = xp.reshape(rc, (-1,))
    physics = response_medium(plan)
    samples = []
    for k in range(plan.selected_freqs.shape[0]):
        f = (plan.freq_idx_start + k) * plan.freq_step
        h = rectangle_response(
            flat, params.aperture, plan._counts, f, params.freq_center, physics, full_directivity, xp
        )
        excitation = xp.exp(2j * pi * f * delays) * apodization
        pressure = _transmit_pressure(h, excitation, plan._pulse[k] * plan._probe[k], xp.zeros(flat.shape[0]) < 0, xp)
        samples.append(plan._probe[k] * _receive_spectrum(h, rc * pressure))
    return xp.stack(samples, axis=0)
