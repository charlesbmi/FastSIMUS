"""Typed simulation boundary for the interactive scattering explorer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from math import inf, radians, tan

import numpy as np

import fast_simus as fs
from fast_simus.transducer_params import TransducerParams
from fast_simus.transducer_presets import C5_2v, L11_5v, L12_3v, P4_2v
from fast_simus.utils import as_numpy
from fast_simus.utils._array_api import _ArrayNamespace

PROBE_PRESETS: dict[str, Callable[[], TransducerParams]] = {
    "P4-2v phased": P4_2v,
    "L12-3v linear": L12_3v,
    "C5-2v convex": C5_2v,
    "L11-5v linear": L11_5v,
}
DEFAULT_TIME_OVERSAMPLING = 2


def probe_with_center_frequency(params: TransducerParams, center_frequency_mhz: float) -> TransducerParams:
    """Override frequency while preserving a preset's geometry and bandwidth."""
    if not 1.0 <= center_frequency_mhz <= 15.0:
        raise ValueError("Center frequency must be between 1 and 15 MHz")
    return params.model_copy(update={"freq_center": center_frequency_mhz * 1e6})


@dataclass(frozen=True)
class SimulationConfig:
    """Hashable physical inputs for one explorer simulation."""

    probe_name: str
    transmit_name: str
    center_frequency_mhz: float
    apodization: tuple[float, ...]
    focus_depth_mm: float
    steering_deg: float
    diverging_width_deg: float
    pulse_wavelengths: float
    propagation_speed: float
    focusing_speed: float
    attenuation: float
    x_limits_mm: tuple[float, float]
    z_limits_mm: tuple[float, float]
    grid_shape: tuple[int, int]
    scatterers_mm: tuple[tuple[float, float], ...]
    reflection_coefficients: tuple[float, ...]
    frequency_step: float
    time_oversampling: int = DEFAULT_TIME_OVERSAMPLING


@dataclass(frozen=True)
class ScatteringSimulationResult:
    """NumPy display data produced at the notebook simulation boundary."""

    incident: np.ndarray
    scattered: np.ndarray
    incident_rms: np.ndarray
    times: np.ndarray
    receive: np.ndarray
    receive_times: np.ndarray
    elements_mm: np.ndarray
    scatterers_mm: np.ndarray
    reflection_coefficients: np.ndarray
    extent_mm: tuple[float, float, float, float]
    focus_mm: np.ndarray | None
    probe: TransducerParams
    propagation_speed: float


def _transmit_delays(
    config: SimulationConfig,
    params: TransducerParams,
    elements,
    apex,
    xp: _ArrayNamespace,
):
    steering = radians(config.steering_deg)
    if config.transmit_name == "Focused":
        focus = xp.asarray([config.focus_depth_mm * 1e-3 * tan(steering), config.focus_depth_mm * 1e-3])
        delays = fs.focused(
            elements,
            focus,
            speed_of_sound=config.focusing_speed,
            radius=params.radius,
            apex_offset=apex,
        )
    elif config.transmit_name == "Plane wave":
        focus = None
        delays = fs.plane_wave(
            elements,
            steering,
            speed_of_sound=config.focusing_speed,
            radius=params.radius,
            apex_offset=apex,
        )
    elif config.transmit_name == "Diverging" and params.radius == inf:
        focus = None
        delays = fs.diverging_wave(
            elements,
            steering,
            radians(config.diverging_width_deg),
            aperture_length=(params.n_elements - 1) * params.pitch,
            speed_of_sound=config.focusing_speed,
        )
    elif config.transmit_name == "Diverging":
        virtual_depth = -config.focus_depth_mm * 1e-3
        focus = xp.asarray([abs(virtual_depth) * tan(steering), virtual_depth])
        delays = fs.focused(
            elements,
            focus,
            speed_of_sound=config.focusing_speed,
            radius=params.radius,
            apex_offset=apex,
        )
    else:
        raise ValueError(f"Unknown transmit mode: {config.transmit_name}")
    return delays, focus


def run_simulation(config: SimulationConfig, xp: _ArrayNamespace) -> ScatteringSimulationResult:
    """Run field and physical receive simulations, then cross into NumPy for display."""
    try:
        preset = PROBE_PRESETS[config.probe_name]
    except KeyError as error:
        raise ValueError(f"Unknown probe preset: {config.probe_name}") from error

    params = probe_with_center_frequency(preset(), config.center_frequency_mhz)
    if len(config.apodization) != params.n_elements:
        raise ValueError(f"Apodization must contain {params.n_elements} element weights")
    if len(config.scatterers_mm) != len(config.reflection_coefficients):
        raise ValueError("Reflection coefficients must match the scatterer count")

    medium = fs.MediumParams(speed_of_sound=config.propagation_speed, attenuation=config.attenuation)
    elements, _theta, apex = fs.element_positions(params.n_elements, params.pitch, params.radius, xp)
    delays, focus = _transmit_delays(config, params, elements, apex, xp)

    nx, nz = config.grid_shape
    x_axis = xp.linspace(config.x_limits_mm[0] * 1e-3, config.x_limits_mm[1] * 1e-3, nx)
    z_axis = xp.linspace(config.z_limits_mm[0] * 1e-3, config.z_limits_mm[1] * 1e-3, nz)
    x_grid, z_grid = xp.meshgrid(x_axis, z_axis)
    positions = xp.stack([x_grid, z_grid], axis=-1)
    scatterers = xp.reshape(xp.asarray(config.scatterers_mm), (-1, 2)) * 1e-3
    reflection_coefficients = xp.asarray(config.reflection_coefficients)
    apodization = xp.asarray(config.apodization)

    spectrum = fs.scattering_pfield_spectrum(
        positions,
        scatterers,
        reflection_coefficients,
        delays,
        params,
        medium,
        tx_apodization=apodization,
        tx_n_wavelengths=config.pulse_wavelengths,
        frequency_step=config.frequency_step,
    )
    incident_movie = fs.spectrum_to_wavefield(
        spectrum.incident,
        spectrum.info,
        time_oversampling=config.time_oversampling,
    )
    scattered_movie = fs.spectrum_to_wavefield(
        spectrum.scattered,
        spectrum.info,
        time_oversampling=config.time_oversampling,
    )

    sampling_frequency = 4.0 * params.freq_center
    if reflection_coefficients.shape[0] == 0:
        receive = xp.zeros((1, params.n_elements))
        receive_times = xp.zeros(1)
    else:
        receive_result = fs.simus(
            scatterers,
            reflection_coefficients,
            delays,
            params,
            medium,
            fs=sampling_frequency,
            tx_apodization=apodization,
            tx_n_wavelengths=config.pulse_wavelengths,
            frequency_step=config.frequency_step,
        )
        receive = receive_result.rf
        receive_times = xp.arange(receive.shape[0], dtype=receive.dtype) / sampling_frequency

    return ScatteringSimulationResult(
        incident=as_numpy(incident_movie.frames),
        scattered=as_numpy(scattered_movie.frames),
        incident_rms=as_numpy(fs.rms_from_spectrum(spectrum.incident, spectrum.info)),
        times=as_numpy(incident_movie.times),
        receive=as_numpy(receive),
        receive_times=as_numpy(receive_times),
        elements_mm=as_numpy(elements) * 1e3,
        scatterers_mm=np.asarray(config.scatterers_mm, dtype=float).reshape((-1, 2)),
        reflection_coefficients=np.asarray(config.reflection_coefficients, dtype=float),
        extent_mm=(*config.x_limits_mm, *config.z_limits_mm),
        focus_mm=None if focus is None else as_numpy(focus) * 1e3,
        probe=params,
        propagation_speed=config.propagation_speed,
    )


@lru_cache(maxsize=8)
def cached_simulation(config: SimulationConfig, xp: _ArrayNamespace) -> ScatteringSimulationResult:
    """Cache completed simulations by immutable physical inputs and backend."""
    return run_simulation(config, xp)
