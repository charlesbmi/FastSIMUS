"""Two-pass pulse-echo orchestration with bounded point/element/patch workspace."""

import array_api_extra as xpx

from fast_simus._blocking import block_count, run_loop, scatterer_block
from fast_simus._compat import _clean_transmit_inputs
from fast_simus._contractions import _receive_spectrum
from fast_simus._field import incident_at_frequency
from fast_simus._transfer_3d import element_response
from fast_simus.plans import response_medium
from fast_simus.utils._array_api import array_namespace


def echo_spectrum(points, rc, delays, plan, params, medium, apodization, full_directivity):
    """Accumulate the complete event spectrum before applying RF thresholding."""
    plan.check_static(points, delays, params, medium)
    if rc.shape != points.shape[:-1]:
        raise ValueError("reflectivity must exactly match scatterer shape")
    xp = array_namespace(points, rc, delays, apodization)
    if apodization is not None and apodization.shape != delays.shape:
        raise ValueError("Apodization must have shape (E,)")
    delays, apodization = _clean_transmit_inputs(delays, apodization, params.n_elements, xp)
    flat = xp.reshape(points, (-1, 3))
    rc = xp.reshape(rc, (-1,))
    physics = response_medium(plan)
    counts = xp.asarray(plan._counts, dtype=xp.int32)
    size = plan._tiles.points
    n_freq = plan.selected_freqs.shape[0]
    output = xp.zeros((n_freq, params.n_elements), dtype=points.dtype) + 0j

    def frequency_step(k, spectrum):
        f = (plan.freq_idx_start + k) * plan.freq_step

        def scatterer_step(i, channels):
            block, coefficients = scatterer_block(flat, rc, i, size, xp)
            weighted = coefficients * incident_at_frequency(block, delays, apodization, plan, k, full_directivity, xp)

            def receive(e, result):
                h = element_response(
                    block, params.aperture, counts, e, f, params.freq_center, physics, full_directivity, plan._tiles, xp
                )
                value = _receive_spectrum(h, weighted) * plan._probe[k]
                return xpx.at(result)[e, :].add(value)  # type: ignore[attr-defined]

            return run_loop(block_count(params.n_elements, plan._tiles.elements), receive, channels, xp)

        initial = (
            xp.zeros((block_count(params.n_elements, plan._tiles.elements), plan._tiles.elements), dtype=points.dtype)
            + 0j
        )
        channels = run_loop(block_count(flat.shape[0], size), scatterer_step, initial, xp)
        channels = xp.reshape(channels, (-1,))[: params.n_elements]
        return xpx.at(spectrum)[k, :].set(channels)  # type: ignore[attr-defined]

    return run_loop(n_freq, frequency_step, output, xp)
