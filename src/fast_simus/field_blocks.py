"""Iterators over flattened spatial blocks on one common spectral plan."""

from dataclasses import dataclass
from math import prod

from fast_simus._blocking import legacy_point_count, point_block
from fast_simus._compat import _clean_transmit_inputs, _validate_apodization
from fast_simus._field import field_block
from fast_simus.medium_params import MediumParams
from fast_simus.pfield import pfield_spectrum_compute
from fast_simus.plans import FieldPlan
from fast_simus.utils._array_api import Array, array_namespace
from fast_simus.wavefield import spectrum_to_wavefield, wavefield_times

_DEFAULT_MEDIUM = MediumParams()


@dataclass(frozen=True, eq=False)
class FieldBlock:
    """Flat spatial interval [start,stop) and values (point,frequency or time)."""

    start: int
    stop: int
    values: Array


def iter_pfield_spectrum(
    positions,
    delays,
    plan,
    params,
    medium=_DEFAULT_MEDIUM,
    *,
    tx_apodization=None,
    full_frequency_directivity=False,
    execution=None,
):
    """Validate original inputs once, then yield selected pressure spectra by point block."""
    xp = array_namespace(positions, delays, tx_apodization)
    count = prod(positions.shape[:-1])
    if count == 0:
        raise ValueError("Grid has no points")
    flat = xp.reshape(positions, (count, positions.shape[-1]))
    if isinstance(plan, FieldPlan):
        if type(plan) is not FieldPlan:
            raise TypeError("Field spectrum requires a FieldPlan, not an EchoPlan")
        _validate_apodization(tx_apodization, delays)
        plan.check_static(positions, delays, params, medium)
        plan.validate_inputs(positions, delays)
        if execution is not None and execution != plan.execution:
            raise ValueError("execution differs from plan")
        clean, apod = _clean_transmit_inputs(delays, tx_apodization, params.n_elements, xp)
        size = plan._tiles.points
        for start in range(0, count, size):
            block, _ = point_block(flat, start // size, size, xp)
            values = field_block(block, clean, apod, plan, full_frequency_directivity, xp)
            stop = min(start + size, count)
            yield FieldBlock(start, stop, values[: stop - start, :])
    else:
        size = min(count, 1024) if execution is None else legacy_point_count(execution, params.n_elements, plan.n_sub)
        for start in range(0, count, size):
            stop = min(start + size, count)
            values = pfield_spectrum_compute(
                flat[start:stop, :],
                delays,
                plan,
                params,
                medium,
                tx_apodization=tx_apodization,
                full_frequency_directivity=full_frequency_directivity,
                execution=execution,
            )
            yield FieldBlock(start, stop, values)


def iter_wavefield(
    positions,
    delays,
    plan,
    params,
    medium=_DEFAULT_MEDIUM,
    *,
    tx_apodization=None,
    full_frequency_directivity=False,
    execution=None,
):
    """Yield transient spatial blocks sharing wavefield_times(plan)."""
    wavefield_times(plan)
    for block in iter_pfield_spectrum(
        positions,
        delays,
        plan,
        params,
        medium,
        tx_apodization=tx_apodization,
        full_frequency_directivity=full_frequency_directivity,
        execution=execution,
    ):
        yield FieldBlock(block.start, block.stop, spectrum_to_wavefield(block.values, plan).frames)
