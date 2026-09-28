"""Opaque prepared plans for finite apertures; validation is eager, outside JIT."""

from dataclasses import dataclass
from math import ceil, isfinite, prod
from types import SimpleNamespace

from fast_simus._frequency import FrequencyGrid, SamplingInfo, _two_way_pulse_duration, frequency_grid
from fast_simus.aperture import _same_arrays
from fast_simus.transducer import Transducer
from fast_simus.utils._array_api import Array, array_namespace


def _validate_arrays(positions, delays, params):
    xp = _same_arrays(params.aperture.centers, positions, delays)
    if positions.shape[-1:] != (3,) or prod(positions.shape[:-1]) == 0:
        raise ValueError("3D points require nonempty (*shape,3) coordinates")
    if delays.shape != (params.n_elements,):
        raise ValueError("Delays must have shape (E,)")
    if not bool(xp.all(xp.isfinite(positions))):
        raise ValueError("Positions must be finite")
    active = xp.where(xp.isnan(delays), xp.zeros_like(delays), delays)
    if not bool(xp.all(xp.isfinite(active))) or bool(xp.any(active < 0)):
        raise ValueError("Active delays must be finite and nonnegative")
    return xp


def _path_bound(points, aperture, xp):
    # Triangle inequality encloses every patch while avoiding P*E*Q allocation.
    flat = xp.reshape(points, (-1, 3))
    maximum = 0.0
    for e in range(aperture.centers.shape[0]):
        distances = xp.sqrt(xp.sum((flat - aperture.centers[e]) ** 2, axis=-1))
        radius = xp.sqrt(xp.sum(aperture.sizes[e] ** 2)) / 2
        maximum = max(maximum, float(xp.max(distances) + radius))
    return maximum


@dataclass(frozen=True, eq=False)
class FieldSpectrumInfo:
    """Read-only selected-band metadata, with explicit CW semantics."""

    _grid: FrequencyGrid

    @property
    def selected_freqs(self):
        """Selected frequency samples in Hz."""
        return self._grid.selected_freqs

    @property
    def freq_step(self):
        """Canonical spacing in Hz."""
        return self._grid.freq_step

    @property
    def n_freq_full(self):
        """Number of bins on the full grid."""
        return self._grid.n_freq_full

    @property
    def freq_idx_start(self):
        """Offset of the selected band."""
        return self._grid.freq_idx_start

    @property
    def is_cw(self):
        """Whether this is a continuous-wave calculation."""
        return self._grid.is_cw

    @property
    def correction_factor(self):
        """Energy integration weight for raw pressure samples."""
        return 1.0 if self.is_cw else self.freq_step


@dataclass(frozen=True, eq=False)
class FieldPlan(FieldSpectrumInfo):
    """Prepared finite-aperture calculation, bound to an immutable description.

    Runtime arrays may vary within the original shape and path/delay bounds.
    Call validate_inputs eagerly before reusing a plan with different arrays.
    """

    _params: Transducer
    _medium: object
    _shape: tuple
    _dtype: object
    _counts: tuple
    _path: float
    _delay: float
    _pulse: Array
    _probe: Array

    def validate_inputs(self, positions, delays):
        """Eagerly enforce shapes, physical validity and original planning bounds."""
        xp = _validate_arrays(positions, delays, self._params)
        self.check_static(positions, delays, self._params, self._medium)
        delay = float(xp.max(xp.where(xp.isnan(delays), xp.zeros_like(delays), delays)))
        tolerance = 1e-6 if positions.dtype == xp.float32 else 1e-12
        if _path_bound(positions, self._params.aperture, xp) > self._path * (1 + tolerance) or delay > self._delay:
            raise ValueError("Inputs exceed plan bounds; replan")

    def check_static(self, positions, delays, params, medium):
        """Check immutable configuration and array metadata without tracing values."""
        _same_arrays(params.aperture.centers, positions, delays)
        if (
            params is not self._params
            or medium != self._medium
            or positions.shape != self._shape
            or positions.dtype != self._dtype
            or delays.shape != (params.n_elements,)
        ):
            raise ValueError("Inputs are incompatible with plan configuration")


def prepare_field(positions, delays, params, medium, *, tx_n_wavelengths, db_thresh, element_splitting, frequency_step):
    """Resolve topology, support bounds and the uniform spectral grid."""
    xp = _validate_arrays(positions, delays, params)
    if not isfinite(frequency_step) or frequency_step <= 0:
        raise ValueError("frequency_step must be positive and finite")
    sizes = params.aperture.sizes
    wavelength = medium.speed_of_sound / (params.freq_center * (1 + params.bandwidth / 2))
    if element_splitting is None:
        counts = tuple(
            (max(1, ceil(float(size[0]) / wavelength)), max(1, ceil(float(size[1]) / wavelength))) for size in sizes
        )
    else:
        if (
            not isinstance(element_splitting, tuple)
            or len(element_splitting) != 2
            or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in element_splitting)
        ):
            raise ValueError("3D element_splitting must be a positive (nu,nv) tuple")
        counts = (element_splitting,) * params.n_elements
    path = max(_path_bound(positions, params.aperture, xp), wavelength / 2)
    delay = float(xp.max(xp.where(xp.isnan(delays), xp.zeros_like(delays), delays)))
    duration = 0 if tx_n_wavelengths == float("inf") else tx_n_wavelengths / params.freq_center
    max_step = frequency_step / (2 * (path / medium.speed_of_sound + delay + duration))
    grid, pulse, probe = frequency_grid(
        params.freq_center, params.bandwidth, tx_n_wavelengths, db_thresh, max_step, xp, positions.dtype
    )
    return FieldPlan(grid, params, medium, positions.shape, positions.dtype, counts, path, delay, pulse, probe)


def response_medium(plan):
    """Resolve response constants once before entering a driver."""
    return SimpleNamespace(
        speed_of_sound=plan._medium.speed_of_sound, attenuation=plan._medium.attenuation, baffle=plan._params.baffle
    )


@dataclass(frozen=True, eq=False)
class EchoPlan(FieldPlan):
    """Finite-aperture pulse-echo plan with explicit sampling metadata."""

    _sampling: SamplingInfo

    @property
    def correction_factor(self):
        """Raw receive spectra use unscaled inverse-DFT normalization."""
        return 1.0

    @property
    def requested_sampling_frequency(self):
        """Requested sample rate in Hz."""
        return self._sampling.requested_sampling_frequency

    @property
    def sampling_frequency(self):
        """Effective sample rate in Hz."""
        return self._sampling.sampling_frequency

    @property
    def n_fft(self):
        """Full inverse transform length."""
        return self._sampling.n_fft

    @property
    def time_origin(self):
        """Trigger-relative origin in seconds."""
        return self._sampling.time_origin

    @property
    def sample_times(self):
        """Causal RF sample times in seconds."""
        return self._sampling.times(array_namespace(self.selected_freqs), self._dtype)


def prepare_echo(
    positions, rc, delays, params, medium, *, fs, tx_n_wavelengths, db_thresh, element_splitting, frequency_step
):
    """Prepare round-trip support using the common spectral grid builder."""
    if not isfinite(tx_n_wavelengths) or tx_n_wavelengths <= 0:
        raise ValueError("RF requires a finite positive pulse duration")
    fs = 4 * params.freq_center if fs is None else fs
    if not isfinite(fs) or fs < 4 * params.freq_center:
        raise ValueError("RF sampling frequency must be at least 4*fc")
    xp = _validate_arrays(positions, delays, params)
    _same_arrays(positions, rc)
    if rc.shape != positions.shape[:-1] or not bool(xp.all(xp.isfinite(rc))):
        raise ValueError("Finite reflectivity must exactly match scatterer shape")
    base = prepare_field(
        positions,
        delays,
        params,
        medium,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
    )
    duration = _two_way_pulse_duration(params.freq_center, params.bandwidth, tx_n_wavelengths, xp)
    step = frequency_step / (2 * (2 * (base._path / medium.speed_of_sound + duration) + base._delay))
    grid, pulse, probe = frequency_grid(
        params.freq_center, params.bandwidth, tx_n_wavelengths, db_thresh, step, xp, positions.dtype
    )
    sampling = SamplingInfo(fs, ceil(fs / (2 * params.freq_center) * (grid.n_freq_full - 1)), grid.freq_step)
    return EchoPlan(
        grid, params, medium, base._shape, base._dtype, base._counts, base._path, base._delay, pulse, probe, sampling
    )
