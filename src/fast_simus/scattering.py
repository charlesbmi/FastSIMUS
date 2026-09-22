"""First-order scattered pressure fields at ideal observation points."""

from __future__ import annotations

from math import ceil, isinf, pi, sqrt
from typing import NamedTuple

from jaxtyping import Complex, Float

from fast_simus._pfield_math import NEPER_TO_DB, _distances_and_angles, _subelement_centroids
from fast_simus._scattering_math import _scatter_and_sum
from fast_simus.medium_params import MediumParams
from fast_simus.pfield import (
    PfieldPlan,
    PfieldSpectrumInfo,
    _pfield_plan_for_travel_time,
    pfield_spectrum_compute,
)
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace, array_namespace
from fast_simus.utils.geometry import element_positions
from fast_simus.wavefield import spectrum_to_wavefield

_DEFAULT_MEDIUM = MediumParams()
_MAX_PAIR_ELEMENTS = 262_144


class ScatteringSpectrumResult(NamedTuple):
    """Incident and first-order scattered pressure on one frequency grid."""

    incident: Complex[Array, "*grid_shape n_freq_selected"]
    scattered: Complex[Array, "*grid_shape n_freq_selected"]
    info: PfieldSpectrumInfo

    @property
    def total(self) -> Complex[Array, "*grid_shape n_freq_selected"]:
        """Return incident plus scattered pressure."""
        return self.incident + self.scattered


class ScatteringWavefieldResult(NamedTuple):
    """Incident and first-order scattered pressure over time."""

    incident: Float[Array, "*grid_shape n_times"]
    scattered: Float[Array, "*grid_shape n_times"]
    times: Float[Array, " n_times"]

    @property
    def total(self) -> Float[Array, "*grid_shape n_times"]:
        """Return incident plus scattered pressure."""
        return self.incident + self.scattered


def _validate_inputs(
    positions: Array,
    scatterers: Array,
    reflection_coefficients: Array,
    delays: Array,
    params: TransducerParams,
) -> None:
    if positions.ndim < 2 or positions.shape[-1] != 2 or positions.size == 0:
        raise ValueError("Observation positions must have shape (*grid_shape, 2) and contain at least one point")
    if scatterers.ndim < 2 or scatterers.shape[-1] != 2:
        raise ValueError("Scatterers must have shape (*batch, 2)")
    if reflection_coefficients.shape != scatterers.shape[:-1]:
        raise ValueError("Reflection coefficients must match scatterers.shape[:-1]")
    if delays.shape != (params.n_elements,):
        raise ValueError(f"Delays must have shape ({params.n_elements},)")


def _element_splitting(params: TransducerParams, medium: MediumParams, requested: int | None) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("Element splitting must be positive")
        return requested
    lambda_min = medium.speed_of_sound / (params.freq_center * (1.0 + params.bandwidth / 2.0))
    return ceil(params.element_width / lambda_min)


def _maximum_pair_distance(
    first: Float[Array, "n_first 2"],
    second: Float[Array, "n_second 2"],
    xp: _ArrayNamespace,
) -> float:
    if first.shape[0] == 0 or second.shape[0] == 0:
        return 0.0
    chunk_size = max(1, _MAX_PAIR_ELEMENTS // max(second.shape[0], 1))
    maximum = 0.0
    for start in range(0, first.shape[0], chunk_size):
        end = min(start + chunk_size, first.shape[0])
        delta = first[start:end, None, :] - second[None, :, :]
        distances = xp.sqrt(xp.sum(delta * delta, axis=-1))
        maximum = max(maximum, float(xp.max(distances)))
    return maximum


def _scattering_plan(
    positions: Float[Array, "*grid_shape 2"],
    scatterers: Float[Array, "*batch 2"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams,
    medium: MediumParams,
    *,
    tx_n_wavelengths: float | int,
    db_thresh: float | int,
    element_splitting: int | None,
    frequency_step: float | int,
    xp: _ArrayNamespace,
) -> PfieldPlan:
    n_sub = _element_splitting(params, medium, element_splitting)
    observation_flat = xp.reshape(positions, (-1, 2))
    scatterer_flat = xp.reshape(scatterers, (-1, 2))

    element_pos, theta_elements, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(params.n_elements)
    offsets = _subelement_centroids(params.element_width, n_sub, theta_elements, xp)
    observation_distances, _, _ = _distances_and_angles(
        observation_flat,
        offsets,
        element_pos,
        theta_elements,
        medium.speed_of_sound,
        params.freq_center,
        xp,
    )
    max_direct_distance = float(xp.max(observation_distances))
    max_two_leg_distance = 0.0
    if scatterer_flat.shape[0] > 0:
        scatterer_distances, _, _ = _distances_and_angles(
            scatterer_flat,
            offsets,
            element_pos,
            theta_elements,
            medium.speed_of_sound,
            params.freq_center,
            xp,
        )
        max_tx_distance = float(xp.max(scatterer_distances))
        max_rx_distance = _maximum_pair_distance(scatterer_flat, observation_flat, xp)
        max_two_leg_distance = max_tx_distance + max_rx_distance

    delays_clean = xp.where(xp.isnan(delays), xp.asarray(0.0), delays)
    pulse_duration = 0.0 if isinf(float(tx_n_wavelengths)) else float(tx_n_wavelengths) / params.freq_center
    max_travel_time = (
        max(max_direct_distance, max_two_leg_distance) / medium.speed_of_sound
        + float(xp.max(delays_clean))
        + pulse_duration
    )
    return _pfield_plan_for_travel_time(
        max_travel_time,
        params,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        n_sub=n_sub,
        frequency_step=frequency_step,
        xp=xp,
    )


def _point_observer_spectrum(
    positions: Float[Array, "n_observers 2"],
    scatterers: Float[Array, "n_scatterers 2"],
    reflection_coefficients: Float[Array, " n_scatterers"],
    incident_at_scatterers: Complex[Array, "n_scatterers n_freq_selected"],
    frequencies: Float[Array, " n_freq_selected"],
    params: TransducerParams,
    medium: MediumParams,
    xp: _ArrayNamespace,
) -> Complex[Array, "n_observers n_freq_selected"]:
    n_observers = positions.shape[0]
    n_scatterers = scatterers.shape[0]
    n_frequencies = frequencies.shape[0]
    if n_scatterers == 0:
        return xp.zeros((n_observers, n_frequencies), dtype=incident_at_scatterers.dtype)

    scatterer_chunk = min(n_scatterers, max(1, int(sqrt(_MAX_PAIR_ELEMENTS))))
    observer_chunk = max(1, _MAX_PAIR_ELEMENTS // scatterer_chunk)
    minimum_distance = xp.asarray(medium.speed_of_sound / params.freq_center / 2.0)
    two_pi = xp.asarray(2.0 * pi)
    output_chunks = []

    for observer_start in range(0, n_observers, observer_chunk):
        observer_end = min(observer_start + observer_chunk, n_observers)
        observer_block = positions[observer_start:observer_end, ...]
        block_spectrum = xp.zeros((observer_block.shape[0], n_frequencies), dtype=incident_at_scatterers.dtype)
        for scatterer_start in range(0, n_scatterers, scatterer_chunk):
            scatterer_end = min(scatterer_start + scatterer_chunk, n_scatterers)
            scatterer_block = scatterers[scatterer_start:scatterer_end, ...]
            delta = scatterer_block[:, None, :] - observer_block[None, :, :]
            distances = xp.sqrt(xp.sum(delta * delta, axis=-1))
            distances = xp.where(distances < minimum_distance, minimum_distance, distances)
            inverse_spreading = 1.0 / xp.sqrt(distances)
            contributions = []
            for frequency_index in range(n_frequencies):
                frequency = frequencies[frequency_index]
                wavenumber = two_pi * frequency / medium.speed_of_sound
                phase_distance = wavenumber * distances
                phase = phase_distance - two_pi * xp.floor(phase_distance / two_pi)
                attenuation_wavenumber = medium.attenuation / NEPER_TO_DB * frequency / 1e6 * 1e2
                transfer = xp.exp(-attenuation_wavenumber * distances + xp.asarray(1j) * phase) * inverse_spreading
                contribution = _scatter_and_sum(
                    incident_at_scatterers[
                        scatterer_start:scatterer_end,
                        frequency_index,
                    ],
                    reflection_coefficients[scatterer_start:scatterer_end],
                    transfer,
                )
                contributions.append(contribution)
            block_spectrum = block_spectrum + xp.stack(contributions, axis=-1)
        output_chunks.append(block_spectrum)

    return output_chunks[0] if len(output_chunks) == 1 else xp.concat(output_chunks, axis=0)


def scattering_pfield_spectrum(
    positions: Float[Array, "*grid_shape 2"],
    scatterers: Float[Array, "*batch 2"],
    reflection_coefficients: Float[Array, " *batch"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    full_frequency_directivity: bool = False,
    element_splitting: int | None = None,
    frequency_step: float | int = 0.5,
) -> ScatteringSpectrumResult:
    """Compute incident and first-order scattered pressure spectra.

    Observation positions are ideal omnidirectional points. Scatterers reradiate
    their local incident pressure using an attenuating 2-D cylindrical Green's
    function; scatterer-to-scatterer interactions are not included. Reflection
    coefficients are signed relative point-scatterer strengths, not calibrated
    material-interface pressure reflection coefficients.
    """
    xp = array_namespace(positions, scatterers, reflection_coefficients, delays, tx_apodization)
    _validate_inputs(positions, scatterers, reflection_coefficients, delays, params)
    plan = _scattering_plan(
        positions,
        scatterers,
        delays,
        params,
        medium,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
        xp=xp,
    )
    incident = pfield_spectrum_compute(
        positions,
        delays,
        plan,
        params,
        medium,
        tx_apodization=tx_apodization,
        full_frequency_directivity=full_frequency_directivity,
    )

    scatterer_flat = xp.reshape(scatterers, (-1, 2))
    coefficient_flat = xp.reshape(reflection_coefficients, (-1,))
    if scatterer_flat.shape[0] == 0:
        incident_at_scatterers = xp.zeros((0, plan.selected_freqs.shape[0]), dtype=incident.dtype)
    else:
        incident_at_scatterers = pfield_spectrum_compute(
            scatterer_flat,
            delays,
            plan,
            params,
            medium,
            tx_apodization=tx_apodization,
            full_frequency_directivity=full_frequency_directivity,
        )

    observation_flat = xp.reshape(positions, (-1, 2))
    scattered_flat = _point_observer_spectrum(
        observation_flat,
        scatterer_flat,
        coefficient_flat,
        incident_at_scatterers,
        plan.selected_freqs,
        params,
        medium,
        xp,
    )
    scattered = xp.reshape(scattered_flat, (*positions.shape[:-1], plan.selected_freqs.shape[0]))
    info = PfieldSpectrumInfo(
        selected_freqs=plan.selected_freqs,
        freq_idx_start=plan.freq_idx_start,
        n_freq_full=plan.n_freq_full,
        freq_step=plan.freq_step,
        correction_factor=plan.correction_factor,
    )
    return ScatteringSpectrumResult(incident=incident, scattered=scattered, info=info)


def scattering_wavefield(
    positions: Float[Array, "*grid_shape 2"],
    scatterers: Float[Array, "*batch 2"],
    reflection_coefficients: Float[Array, " *batch"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    full_frequency_directivity: bool = False,
    element_splitting: int | None = None,
    frequency_step: float | int = 0.5,
    time_oversampling: int = 1,
) -> ScatteringWavefieldResult:
    """Compute incident and first-order scattered pressure over time.

    ``time_oversampling`` is forwarded to the inverse FFT for denser,
    phase-faithful time samples without changing the simulated spectrum.
    """
    spectrum = scattering_pfield_spectrum(
        positions,
        scatterers,
        reflection_coefficients,
        delays,
        params,
        medium,
        tx_apodization=tx_apodization,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        full_frequency_directivity=full_frequency_directivity,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
    )
    incident = spectrum_to_wavefield(spectrum.incident, spectrum.info, time_oversampling=time_oversampling)
    scattered = spectrum_to_wavefield(spectrum.scattered, spectrum.info, time_oversampling=time_oversampling)
    return ScatteringWavefieldResult(
        incident=incident.frames,
        scattered=scattered.frames,
        times=incident.times,
    )
