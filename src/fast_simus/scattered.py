"""Single-scattered pressure at arbitrary points in a homogeneous 3D medium."""

from dataclasses import dataclass, replace
from math import prod

from fast_simus._blocking import choose_tiles
from fast_simus._compat import _clean_transmit_inputs, _validate_apodization
from fast_simus._frequency import frequency_grid
from fast_simus._scattered import illumination_cache, scattered_block
from fast_simus.aperture import _same_arrays
from fast_simus.execution import ExecutionOptions
from fast_simus.field_blocks import FieldBlock
from fast_simus.medium_params import MediumParams
from fast_simus.plans import FieldPlan, FieldSpectrumInfo, _path_bound, _validate_arrays, prepare_field
from fast_simus.transducer import Transducer

_DEFAULT_MEDIUM = MediumParams()
_COMPONENTS = ("incident", "scattered", "total")


def _validate_cloud(observers, scatterers, rc, delays, params):
    xp = _validate_arrays(observers, delays, params)
    _same_arrays(observers, scatterers, rc)
    if scatterers.shape[-1:] != (3,) or rc.shape != scatterers.shape[:-1]:
        raise ValueError("Scatterers require (*shape,3) coordinates and matching reflectivity")
    flat = xp.reshape(scatterers, (-1, 3))
    coefficients = xp.reshape(rc, (-1,))
    for start in range(0, flat.shape[0], 256):
        stop = min(start + 256, flat.shape[0])
        if not bool(xp.all(xp.isfinite(flat[start:stop, :]))) or not bool(
            xp.all(xp.isfinite(coefficients[start:stop]))
        ):
            raise ValueError("Scatterers and reflectivity must be finite")
    return xp


def _support(observers, scatterers, params, medium, xp):
    minimum = medium.speed_of_sound / (2 * params.freq_center)
    direct = max(minimum, _path_bound(observers, params.aperture, xp))
    if not prod(scatterers.shape[:-1]):
        return direct
    # AABB extrema bound all source-observer distances without an S*O array.
    obs = xp.reshape(observers, (-1, 3))
    src = xp.reshape(scatterers, (-1, 3))
    low = xp.minimum(xp.min(obs, axis=0), xp.min(src, axis=0))
    high = xp.maximum(xp.max(obs, axis=0), xp.max(src, axis=0))
    second = max(minimum, float(xp.sqrt(xp.sum((high - low) ** 2))))
    first = max(minimum, _path_bound(scatterers, params.aperture, xp))
    return max(direct, first + second)


@dataclass(frozen=True, eq=False)
class ScatteredFieldPlan(FieldSpectrumInfo):
    """Common incident/scattered spectral grid; input arrays are borrowed, never mutated.

    Coordinates are meters, delays seconds, and rc is the same effective scattering
    strength as SIMUS (arbitrary amplitude convention). Single scattering only.
    Runtime arrays must retain shape, backend, dtype and the validated path bounds.
    """

    _field: FieldPlan
    _scatterer_shape: tuple
    _support: float
    _observer_size: int
    _source_size: int
    _cache_bytes: int
    execution: ExecutionOptions

    @property
    def estimated_workspace_bytes(self):
        """Conservative live workspace, excluding inputs and returned spectra."""
        return self._cache_bytes + max(
            self._field.estimated_workspace_bytes, 256 * self._observer_size * self._source_size
        )

    @property
    def lens_reference_delay(self):
        """Transmit-only common lens delay in seconds."""
        return self._field.lens_reference_delay

    def validate_inputs(self, observers, scatterers, rc, delays, params, medium=_DEFAULT_MEDIUM):
        """Validate eagerly before numerical computation or compiled reuse."""
        xp = _validate_cloud(observers, scatterers, rc, delays, params)
        self._field.check_static(observers, delays, params, medium)
        if scatterers.shape != self._scatterer_shape:
            raise ValueError("Scatterer shape differs from plan")
        tolerance = 1e-6 if observers.dtype == xp.float32 else 1e-12
        delay = float(xp.max(xp.where(xp.isnan(delays), xp.zeros_like(delays), delays)))
        if (
            _support(observers, scatterers, params, medium, xp) > self._support * (1 + tolerance)
            or delay > self._field._delay
        ):
            raise ValueError("Inputs exceed scattered-field plan bounds; replan")


def scattered_field_precompute(
    observers,
    scatterers,
    rc,
    delays,
    params,
    medium=_DEFAULT_MEDIUM,
    *,
    tx_n_wavelengths=1.0,
    db_thresh=-60.0,
    element_splitting=None,
    frequency_step=0.5,
    execution=None,
):
    """Plan arbitrary-point pressure on one grid covering direct and scattered paths.

    observers: (*grid,3); scatterers: (*cloud,3); rc: (*cloud,); delays: (E,).
    Empty clouds are valid. Use finite 3D Transducer descriptions. Plans support
    CW spectra; spectrum_to_wavefield requires a finite pulse. ExecutionOptions
    includes any optional illumination cache; dense returned outputs are excluded.
    """
    if not isinstance(params, Transducer):
        raise ValueError("Scattered pressure requires a finite 3D Transducer")
    xp = _validate_cloud(observers, scatterers, rc, delays, params)
    execution = execution or ExecutionOptions()
    base = prepare_field(
        observers,
        delays,
        params,
        medium,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
        execution=execution,
    )
    support = _support(observers, scatterers, params, medium, xp)
    duration = 0 if tx_n_wavelengths == float("inf") else tx_n_wavelengths / params.freq_center
    max_step = frequency_step / (
        2 * (support / medium.speed_of_sound + base._delay + duration + base.lens_reference_delay)
    )
    grid, pulse, probe = frequency_grid(
        params.freq_center, params.bandwidth, tx_n_wavelengths, db_thresh, max_step, xp, observers.dtype
    )
    n_sources = prod(scatterers.shape[:-1])
    # Reserve half for functional update copies of a small illumination cache.
    source_size = min(max(1, n_sources), 128, max(1, execution.workspace_bytes // 4096))
    item = 8 if observers.dtype == xp.float32 else 16
    # Twice the padded cache size covers functional-update copies. The upper
    # padding bound also remains valid if aperture tiling reduces source_size.
    cache_bytes = 2 * (n_sources + source_size - 1) * grid.selected_freqs.shape[0] * item if n_sources else 0
    if cache_bytes > execution.workspace_bytes // 2:
        cache_bytes = 0
    available = execution.workspace_bytes - cache_bytes
    observer_size = min(prod(observers.shape[:-1]), 256, max(1, available // (256 * source_size)))
    tiles = choose_tiles(max(observer_size, source_size), base._counts, ExecutionOptions(available))
    source_size = min(source_size, tiles.points)
    observer_size = min(observer_size, tiles.points)
    base = replace(base, _grid=grid, _pulse=pulse, _probe=probe, _tiles=tiles)
    return ScatteredFieldPlan(grid, base, scatterers.shape, support, observer_size, source_size, cache_bytes, execution)


def iter_scattered_pfield_spectrum(
    observers,
    scatterers,
    rc,
    delays,
    plan,
    params,
    medium=_DEFAULT_MEDIUM,
    *,
    component="scattered",
    tx_apodization=None,
    full_frequency_directivity=False,
    cancelled=None,
):
    """Yield FieldBlock(start,stop,values) with flat observer indices and complex (O,F).

    Runs outside JIT. Each yield is a cancellation/progress boundary. Components
    use identical frequency samples; total is summed before temporal conversion.
    """
    if component not in _COMPONENTS:
        raise ValueError(f"component must be one of {_COMPONENTS}")
    plan.validate_inputs(observers, scatterers, rc, delays, params, medium)
    _validate_apodization(tx_apodization, delays)
    xp = _same_arrays(observers, scatterers, rc, delays)
    clean, apod = _clean_transmit_inputs(delays, tx_apodization, params.n_elements, xp)
    points = xp.reshape(observers, (-1, 3))
    sources = xp.reshape(scatterers, (-1, 3))
    coefficients = xp.reshape(rc, (-1,))
    cache = (
        None
        if component == "incident"
        else illumination_cache(sources, coefficients, clean, apod, plan, full_frequency_directivity, xp, cancelled)
    )
    for start in range(0, points.shape[0], plan._observer_size):
        if cancelled is not None and cancelled():
            raise InterruptedError("Simulation cancelled between observation blocks")
        stop = min(start + plan._observer_size, points.shape[0])
        values = scattered_block(
            points[start:stop, :],
            sources,
            coefficients,
            clean,
            apod,
            plan,
            component,
            full_frequency_directivity,
            cache,
            xp,
        )
        yield FieldBlock(start, stop, values)


def scattered_pfield_spectrum(
    observers,
    scatterers,
    rc,
    delays,
    params,
    medium=_DEFAULT_MEDIUM,
    *,
    plan=None,
    component="scattered",
    tx_apodization=None,
    full_frequency_directivity=False,
    tx_n_wavelengths=1.0,
    db_thresh=-60.0,
    element_splitting=None,
    frequency_step=0.5,
    execution=None,
):
    """Return dense (*grid,F) pressure and common-grid metadata.

    Supply plan to reuse prepared physics; spectral/planning keywords apply only
    when creating a plan. Use the iterator to bound output storage for large grids.
    """
    if plan is None:
        plan = scattered_field_precompute(
            observers,
            scatterers,
            rc,
            delays,
            params,
            medium,
            tx_n_wavelengths=tx_n_wavelengths,
            db_thresh=db_thresh,
            element_splitting=element_splitting,
            frequency_step=frequency_step,
            execution=execution,
        )
    elif execution is not None and execution != plan.execution:
        raise ValueError("execution differs from plan")
    blocks = list(
        iter_scattered_pfield_spectrum(
            observers,
            scatterers,
            rc,
            delays,
            plan,
            params,
            medium,
            component=component,
            tx_apodization=tx_apodization,
            full_frequency_directivity=full_frequency_directivity,
        )
    )
    xp = _same_arrays(observers, scatterers)
    spectrum = xp.concat([block.values for block in blocks], axis=0)
    return xp.reshape(spectrum, (*observers.shape[:-1], plan.selected_freqs.shape[0])), plan
