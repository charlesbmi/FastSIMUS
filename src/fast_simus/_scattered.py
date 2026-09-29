"""Bounded isotropic observation of single-scattered pressure."""

from math import pi

import array_api_extra as xpx

from fast_simus._blocking import block_count, point_block, run_loop
from fast_simus._contractions import _receive_spectrum
from fast_simus._field import transmit_at_frequency
from fast_simus._illumination import scatterer_illumination
from fast_simus._pfield_math import NEPER_TO_DB
from fast_simus._propagation import propagation_exponential


def point_response(sources, observers, frequency, fc, medium, xp):
    """Isotropic G(source,observer); half-wavelength regularization matches element paths.

    Point pressure has no receiver baffle, lens, aperture average or probe filter.
    Distances below c/(2*fc) use that distance for both phase and spreading.
    """
    delta = sources[:, None, :] - observers[None, :, :]
    distance = xp.sqrt(xp.sum(delta * delta, axis=-1))
    distance = xp.maximum(distance, xp.asarray(medium.speed_of_sound / (2 * fc), dtype=distance.dtype))
    return (
        propagation_exponential(
            distance,
            2 * pi * frequency / medium.speed_of_sound,
            medium.attenuation / NEPER_TO_DB * frequency * 1e-4,
            xp,
        )
        / distance
    )


def illumination_cache(scatterers, rc, delays, apodization, plan, full, xp, cancelled=None):
    """Cache illumination only when its complete allocation fits the reserved budget."""
    if not plan._cache_bytes or not scatterers.shape[0]:
        return None
    n_freq = plan.selected_freqs.shape[0]
    size = plan._source_size
    n_blocks = block_count(scatterers.shape[0], size)
    result = xp.zeros((n_blocks, size, n_freq), dtype=scatterers.dtype) + 0j

    def source_step(i, result):
        points, valid = point_block(scatterers, i, size, xp)
        coefficients, _ = point_block(rc[:, None], i, size, xp)
        coefficients = xp.where(valid, coefficients[:, 0], xp.zeros_like(coefficients[:, 0]))

        def frequency_step(k, block):
            values = scatterer_illumination(points, coefficients, delays, apodization, plan._field, k, full, xp)
            return xpx.at(block)[:, k].set(values)  # type: ignore[attr-defined]

        block = run_loop(n_freq, frequency_step, xp.zeros((size, n_freq), dtype=scatterers.dtype) + 0j, xp)
        return xpx.at(result)[i, :, :].set(block)  # type: ignore[attr-defined]

    for i in range(n_blocks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Simulation cancelled during scatterer illumination")
        result = source_step(i, result)
        if hasattr(xp, "eval"):
            xp.eval(result)
    return xp.reshape(result, (-1, n_freq))


def scattered_block(observers, scatterers, rc, delays, apodization, plan, component, full, cache, xp):
    """Compute a spatial block on the original common spectral grid."""
    base = plan._field
    n_freq = plan.selected_freqs.shape[0]
    source_size = plan._source_size
    result = xp.zeros((observers.shape[0], n_freq), dtype=observers.dtype) + 0j

    def frequency_step(k, output):
        frequency = (plan.freq_idx_start + k) * plan.freq_step
        values = xp.zeros_like(observers[:, 0]) + 0j
        if component != "scattered":
            values = transmit_at_frequency(observers, delays, apodization, base, frequency, full, xp)
            values = values * base._pulse[k] * base._probe[k]
        if component != "incident" and scatterers.shape[0]:

            def source_step(i, pressure):
                points, valid = point_block(scatterers, i, source_size, xp)
                if cache is None:
                    coefficients, _ = point_block(rc[:, None], i, source_size, xp)
                    coefficients = xp.where(valid, coefficients[:, 0], xp.zeros_like(coefficients[:, 0]))
                    weighted = scatterer_illumination(points, coefficients, delays, apodization, base, k, full, xp)
                else:
                    indices = i * source_size + xp.arange(source_size)
                    weighted = xp.take(cache[:, k], indices, axis=0)
                response = point_response(points, observers, frequency, base._params.freq_center, base._medium, xp)
                return pressure + _receive_spectrum(response, weighted)

            values = run_loop(block_count(scatterers.shape[0], source_size), source_step, values, xp)
        return xpx.at(output)[:, k].set(values)  # type: ignore[attr-defined]

    return run_loop(n_freq, frequency_step, result, xp)
