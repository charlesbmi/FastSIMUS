"""Independent transmit events with a shared causal receive sampling grid."""

from dataclasses import dataclass
from typing import NamedTuple

from fast_simus._frequency import SamplingInfo
from fast_simus.aperture import _same_arrays
from fast_simus.execution import ExecutionOptions
from fast_simus.medium_params import MediumParams
from fast_simus.plans import EchoPlan
from fast_simus.simus import SimusPlan, SimusResult, simus_compute, simus_precompute
from fast_simus.utils._array_api import Array, array_namespace

_DEFAULT_MEDIUM = MediumParams()


@dataclass(frozen=True, eq=False)
class TransmitSequence:
    """Borrowed (event,element) delays in seconds and optional finite apodization.

    NaN delay disables transmission on that channel for that event. Event rows
    are independent emissions; no PRF or acquisition clock is implied.
    """

    delays: Array
    apodization: Array | None = None

    def __post_init__(self):
        """Validate the event definition eagerly."""
        xp = array_namespace(self.delays)
        if self.delays.ndim != 2 or min(self.delays.shape) < 1:
            raise ValueError("Sequence delays require nonempty (event,element) shape")
        active = xp.where(xp.isnan(self.delays), xp.zeros_like(self.delays), self.delays)
        if not bool(xp.all(xp.isfinite(active))) or not bool(xp.all(active >= 0)):
            raise ValueError("Sequence delays must be nonnegative finite or NaN")
        if self.apodization is not None:
            _same_arrays(self.delays, self.apodization)
            if self.apodization.shape != self.delays.shape or not bool(xp.all(xp.isfinite(self.apodization))):
                raise ValueError("Sequence apodization must be finite and match delays")


class _SequenceTiming:
    _sampling: SamplingInfo

    @property
    def requested_sampling_frequency(self):
        """Requested sample rate in Hz."""
        return self._sampling.requested_sampling_frequency

    @property
    def sampling_frequency(self):
        """Effective sample rate in Hz."""
        return self._sampling.sampling_frequency

    @property
    def time_origin(self):
        """Time origin relative to each independent trigger, in seconds."""
        return self._sampling.time_origin


@dataclass(frozen=True, eq=False)
class SequencePlan(_SequenceTiming):
    """Common echo plan and trigger-relative timing for a fixed event count."""

    echo_plan: SimusPlan | EchoPlan
    execution: ExecutionOptions | None
    _sampling: SamplingInfo
    sample_times: Array
    lens_reference_delay: float
    _params: object
    _medium: MediumParams
    _shape: tuple
    _delay_bound: float
    _events: int


class SequenceEvent(NamedTuple):
    """One independent event result and its zero-based input row index."""

    index: int
    result: SimusResult


@dataclass(frozen=True, eq=False)
class SequenceResult(_SequenceTiming):
    """Stacked RF (A,T,E) and spectra (A,F,E), with the original event definition."""

    rf: Array
    spectrum: Array
    sample_times: Array
    sequence: TransmitSequence
    _sampling: SamplingInfo
    lens_reference_delay: float


def sequence_precompute(
    scatterers,
    rc,
    sequence,
    params,
    medium=_DEFAULT_MEDIUM,
    *,
    fs=None,
    execution=None,
    tx_n_wavelengths=1.0,
    db_thresh=-60.0,
    element_splitting=None,
    frequency_step=1.0,
) -> SequencePlan:
    """Plan all events against their largest active delay without simulating an envelope event."""
    xp = array_namespace(scatterers, rc, sequence.delays)
    if sequence.delays.shape[1] != params.n_elements or rc.shape != scatterers.shape[:-1]:
        raise ValueError("Sequence channels and reflectivity must match the scene")
    cleaned = xp.where(xp.isnan(sequence.delays), xp.zeros_like(sequence.delays), sequence.delays)
    envelope = xp.max(cleaned, axis=0)
    echo = simus_precompute(
        scatterers,
        rc,
        envelope,
        params,
        medium,
        fs=fs,
        execution=execution,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
    )
    requested = fs if fs is not None else 4 * params.freq_center
    if isinstance(echo, EchoPlan):
        sampling = echo._sampling
        lens_delay = echo.lens_reference_delay
    else:
        sampling = SamplingInfo(requested, echo.n_fft, 2 * params.freq_center / (echo.n_freq_full - 1))
        lens_delay = 0.0
    return SequencePlan(
        echo,
        execution,
        sampling,
        sampling.times(xp, scatterers.dtype),
        lens_delay,
        params,
        medium,
        scatterers.shape,
        float(xp.max(cleaned)),
        sequence.delays.shape[0],
    )


def iter_simus_sequence(
    scatterers, rc, sequence, plan, params, medium=_DEFAULT_MEDIUM, *, backend=None, full_frequency_directivity=False
):
    """Yield one event at a time; memory for all event outputs is never allocated."""
    xp = array_namespace(scatterers, sequence.delays)
    if (
        params is not plan._params
        or medium != plan._medium
        or scatterers.shape != plan._shape
        or sequence.delays.shape != (plan._events, params.n_elements)
    ):
        raise ValueError("Sequence inputs do not match plan")
    cleaned = xp.where(xp.isnan(sequence.delays), xp.zeros_like(sequence.delays), sequence.delays)
    if float(xp.max(cleaned)) > plan._delay_bound:
        raise ValueError("Sequence exceeds planned delay bound")
    if isinstance(plan.echo_plan, EchoPlan):
        plan.echo_plan.validate_inputs(scatterers, xp.max(cleaned, axis=0))
    for index in range(plan._events):
        apodization = None if sequence.apodization is None else sequence.apodization[index, :]
        result = simus_compute(
            scatterers,
            rc,
            sequence.delays[index, :],
            plan.echo_plan,
            params,
            medium,
            tx_apodization=apodization,
            full_frequency_directivity=full_frequency_directivity,
            backend=backend,
            execution=plan.execution,
        )
        yield SequenceEvent(index, result)


def simus_sequence(
    scatterers,
    rc,
    sequence,
    params,
    medium=_DEFAULT_MEDIUM,
    *,
    fs=None,
    execution=None,
    backend=None,
    tx_n_wavelengths=1.0,
    db_thresh=-60.0,
    element_splitting=None,
    frequency_step=1.0,
    full_frequency_directivity=False,
) -> SequenceResult:
    """Simulate and stack independent events; outputs scale with event count."""
    plan = sequence_precompute(
        scatterers,
        rc,
        sequence,
        params,
        medium,
        fs=fs,
        execution=execution,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
    )
    events = list(
        iter_simus_sequence(
            scatterers,
            rc,
            sequence,
            plan,
            params,
            medium,
            backend=backend,
            full_frequency_directivity=full_frequency_directivity,
        )
    )
    xp = array_namespace(scatterers)
    return SequenceResult(
        xp.stack([e.result.rf for e in events]),
        xp.stack([e.result.spectrum for e in events]),
        plan.sample_times,
        sequence,
        plan._sampling,
        plan.lens_reference_delay,
    )
