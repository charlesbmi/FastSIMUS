"""Tests for the notebook's simulation orchestration boundary."""

from dataclasses import replace
from typing import cast

import numpy as np

from examples._scattering_simulation import SimulationConfig, run_simulation
from fast_simus.utils._array_api import _ArrayNamespace

NP_NAMESPACE = cast(_ArrayNamespace, np)


def _config(**changes) -> SimulationConfig:
    base = SimulationConfig(
        probe_name="P4-2v phased",
        transmit_name="Focused",
        center_frequency_mhz=2.72,
        apodization=tuple(np.ones(64)),
        focus_depth_mm=30.0,
        steering_deg=0.0,
        diverging_width_deg=70.0,
        pulse_wavelengths=2.0,
        propagation_speed=1540.0,
        focusing_speed=1540.0,
        attenuation=0.5,
        x_limits_mm=(-1.0, 1.0),
        z_limits_mm=(20.0, 22.0),
        grid_shape=(3, 2),
        scatterers_mm=((0.0, 21.0),),
        reflection_coefficients=(0.005,),
        frequency_step=4.0,
        time_oversampling=1,
    )
    return replace(base, **changes)


def test_run_simulation_returns_typed_display_boundary() -> None:
    """One small run preserves physical coordinates and component shapes."""
    result = run_simulation(_config(), NP_NAMESPACE)

    assert result.incident.shape[:2] == (2, 3)
    assert result.scattered.shape == result.incident.shape
    assert result.incident_rms.shape == (2, 3)
    assert result.times.shape == (result.incident.shape[-1],)
    assert result.receive.shape[1] == result.probe.n_elements
    assert result.receive_times.shape == (result.receive.shape[0],)
    np.testing.assert_array_equal(result.scatterers_mm, [[0.0, 21.0]])
    np.testing.assert_array_equal(result.reflection_coefficients, [0.005])
    assert result.focus_mm is not None
    np.testing.assert_allclose(result.focus_mm, [0.0, 30.0])
    assert result.extent_mm == (-1.0, 1.0, 20.0, 22.0)


def test_transmit_modes_expose_only_meaningful_focus_geometry() -> None:
    """Focused, plane, and linear diverging modes retain their UI semantics."""
    focused = run_simulation(_config(transmit_name="Focused", steering_deg=10.0), NP_NAMESPACE)
    plane = run_simulation(_config(transmit_name="Plane wave", steering_deg=10.0), NP_NAMESPACE)
    diverging = run_simulation(_config(transmit_name="Diverging", steering_deg=10.0), NP_NAMESPACE)

    assert focused.focus_mm is not None
    np.testing.assert_allclose(focused.focus_mm[1], 30.0)
    assert plane.focus_mm is None
    assert diverging.focus_mm is None


def test_empty_scene_returns_a_finite_receive_placeholder() -> None:
    """The viewer contract stays valid when no receive simulation is needed."""
    result = run_simulation(
        _config(scatterers_mm=(), reflection_coefficients=()),
        NP_NAMESPACE,
    )

    assert result.scatterers_mm.shape == (0, 2)
    assert result.receive.shape == (1, result.probe.n_elements)
    np.testing.assert_array_equal(result.receive, 0.0)
    np.testing.assert_array_equal(result.receive_times, 0.0)
