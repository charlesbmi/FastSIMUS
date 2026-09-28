"""Bounded pressure orchestration over the shared element response."""

from math import pi

import array_api_extra as xpx

from fast_simus._blocking import block_count, element_block, point_block, run_loop
from fast_simus._capabilities import _require_strategy
from fast_simus._compat import _clean_transmit_inputs
from fast_simus._contractions import _transmit_pressure
from fast_simus._transfer_3d import element_response
from fast_simus.plans import response_medium
from fast_simus.utils._array_api import array_namespace


def transmit_at_frequency(points, delays, apodization, plan, frequency, full_directivity, xp):
    """Complete the coherent element sum before any output reduction."""
    params = plan._params
    physics = response_medium(plan)
    counts = xp.asarray(plan._counts, dtype=xp.int32)

    def add_element(e, pressure):
        h = element_response(
            points,
            params.aperture,
            counts,
            e,
            frequency,
            params.freq_center,
            physics,
            full_directivity,
            plan._tiles,
            xp,
        )
        indices, valid = element_block(e, plan._tiles.elements, params.n_elements, xp)
        weights = xp.take(apodization, indices, axis=0)
        weights = xp.where(valid, weights, xp.zeros_like(weights))
        excitation = xp.exp(2j * pi * frequency * xp.take(delays, indices, axis=0)) * weights
        return pressure + _transmit_pressure(h, excitation, 1.0, None, xp)

    return run_loop(
        block_count(params.n_elements, plan._tiles.elements), add_element, xp.zeros_like(points[:, 0]) + 0j, xp
    )


def field_block(points, delays, apodization, plan, full_directivity, xp, rms=False):
    """Compute one fixed point block, accumulating energy only after coherent TX."""
    n = plan.selected_freqs.shape[0]
    shape = (points.shape[0],) if rms else (points.shape[0], n)
    output = xp.zeros(shape, dtype=points.dtype) if rms else xp.zeros(shape, dtype=points.dtype) + 0j

    def frequency_step(k, result):
        f = (plan.freq_idx_start + k) * plan.freq_step
        pressure = (
            transmit_at_frequency(points, delays, apodization, plan, f, full_directivity, xp)
            * plan._pulse[k]
            * plan._probe[k]
        )
        if rms:
            return result + xp.real(pressure * xp.conj(pressure))
        return xpx.at(result)[:, k].set(pressure)  # type: ignore[attr-defined]

    output = run_loop(n, frequency_step, output, xp)
    return xp.sqrt(output * plan.correction_factor) if rms else output


def field_spectrum(positions, delays, plan, params, medium, apodization, full_directivity, strategy=None, rms=False):
    """Compute raw samples or RMS with preserved spatial axes and bounded work."""
    plan.check_static(positions, delays, params, medium)
    if strategy in ("metal", "cuda"):
        raise NotImplementedError("Native kernels do not support finite 3D apertures")
    xp = array_namespace(positions, delays, apodization)
    if strategy is not None:
        _require_strategy(strategy, xp, params.baffle, full_directivity)
    if apodization is not None and apodization.shape != delays.shape:
        raise ValueError("Apodization must have shape (E,)")
    delays, apodization = _clean_transmit_inputs(delays, apodization, params.n_elements, xp)
    points = xp.reshape(positions, (-1, 3))
    size = plan._tiles.points
    blocks = block_count(points.shape[0], size)
    tail = () if rms else (plan.selected_freqs.shape[0],)
    output = xp.zeros((blocks, size, *tail), dtype=positions.dtype)
    if not rms:
        output = output + 0j

    def compute_block(i, result):
        block, _valid = point_block(points, i, size, xp)
        values = field_block(block, delays, apodization, plan, full_directivity, xp, rms)
        return xpx.at(result)[i, ...].set(values)  # type: ignore[attr-defined]

    output = run_loop(blocks, compute_block, output, xp)
    output = xp.reshape(output, (blocks * size, *tail))[: points.shape[0], ...]
    return xp.reshape(output, (*positions.shape[:-1], *tail))
