"""Behavioral tests for first-order scattered pressure fields."""

from __future__ import annotations

from typing import Any, cast

import array_api_strict
import numpy as np
import pytest

from fast_simus.medium_params import MediumParams
from fast_simus.scattering import ScatteringSpectrumResult, scattering_pfield_spectrum, scattering_wavefield
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils._array_api import Array, _ArrayNamespace
from tests.conftest import to_numpy


def _array(value: Any) -> Array:
    return cast(Array, np.asarray(value))


def _case(xp: _ArrayNamespace):
    params = P4_2v()
    positions = xp.asarray([[0.0, 0.025], [0.003, 0.032]])
    scatterers = xp.asarray([[0.0, 0.018], [0.002, 0.022]])
    rc = xp.asarray([1.0, -0.5])
    delays = xp.zeros(params.n_elements)
    return params, positions, scatterers, rc, delays


def test_scattering_spectrum_contract_and_total(xp: _ArrayNamespace) -> None:
    """Incident, scattered, and total fields share spatial and frequency axes."""
    params, positions, scatterers, rc, delays = _case(xp)

    result = scattering_pfield_spectrum(
        positions,
        scatterers,
        rc,
        delays,
        params,
        frequency_step=2.0,
    )

    assert isinstance(result, ScatteringSpectrumResult)
    assert result.incident.shape == result.scattered.shape
    assert result.incident.shape[:-1] == positions.shape[:-1]
    assert result.incident.shape[-1] == result.info.selected_freqs.shape[0]
    np.testing.assert_allclose(to_numpy(result.total), to_numpy(result.incident + result.scattered))


def test_zero_and_empty_scatterers_produce_zero_echo(xp: _ArrayNamespace) -> None:
    """No reflecting strength means no scattered pressure."""
    params, positions, scatterers, rc, delays = _case(xp)

    zero = scattering_pfield_spectrum(
        positions,
        scatterers,
        xp.zeros(rc.shape),
        delays,
        params,
        frequency_step=2.0,
    )
    empty = scattering_pfield_spectrum(
        positions,
        xp.zeros((0, 2)),
        xp.zeros((0,)),
        delays,
        params,
        frequency_step=2.0,
    )

    np.testing.assert_array_equal(to_numpy(zero.scattered), 0.0)
    np.testing.assert_array_equal(to_numpy(empty.scattered), 0.0)
    assert empty.scattered.shape == empty.incident.shape


def test_scattering_is_linear_in_reflectivity() -> None:
    """Signed point-target contributions obey first-order superposition."""
    params = P4_2v()
    positions = _array([[0.001, 0.035]])
    scatterers = _array([[-0.002, 0.02], [0.003, 0.024]])
    delays = _array(np.zeros(params.n_elements))

    both = scattering_pfield_spectrum(positions, scatterers, _array([1.0, -0.4]), delays, params, frequency_step=2.0)
    first = scattering_pfield_spectrum(positions, scatterers, _array([1.0, 0.0]), delays, params, frequency_step=2.0)
    second = scattering_pfield_spectrum(positions, scatterers, _array([0.0, -0.4]), delays, params, frequency_step=2.0)

    np.testing.assert_allclose(
        to_numpy(both.scattered),
        to_numpy(first.scattered + second.scattered),
        rtol=2e-6,
        atol=1e-9,
    )


def test_point_propagation_has_cylindrical_spreading() -> None:
    """One target decays as inverse square-root distance without attenuation."""
    params = P4_2v()
    medium = MediumParams(attenuation=0.0)
    scatterers = _array([[0.0, 0.02]])
    positions = _array([[0.0, 0.03], [0.0, 0.06]])

    result = scattering_pfield_spectrum(
        positions,
        scatterers,
        _array(np.ones(1)),
        _array(np.zeros(params.n_elements)),
        params,
        medium,
        frequency_step=2.0,
    )
    magnitude = np.abs(to_numpy(result.scattered))
    populated = magnitude[0] > np.max(magnitude[0]) * 1e-8
    ratio = magnitude[0, populated] / magnitude[1, populated]

    np.testing.assert_allclose(ratio, np.sqrt(0.04 / 0.01), rtol=2e-5)


def test_attenuation_reduces_scattered_field() -> None:
    """Positive attenuation lowers echo magnitude at every selected frequency."""
    params = P4_2v()
    positions = _array([[0.0, 0.05]])
    scatterers = _array([[0.0, 0.02]])
    delays = _array(np.zeros(params.n_elements))
    rc = _array(np.ones(1))

    lossless = scattering_pfield_spectrum(
        positions, scatterers, rc, delays, params, MediumParams(attenuation=0.0), frequency_step=2.0
    )
    lossy = scattering_pfield_spectrum(
        positions, scatterers, rc, delays, params, MediumParams(attenuation=0.5), frequency_step=2.0
    )

    assert np.max(np.abs(to_numpy(lossy.scattered))) < np.max(np.abs(to_numpy(lossless.scattered)))


def test_coincident_observer_is_finite() -> None:
    """The half-wavelength distance floor regularizes a target on a grid point."""
    params = P4_2v()
    point = _array([[0.0, 0.02]])
    result = scattering_pfield_spectrum(
        point, point, _array(np.ones(1)), _array(np.zeros(params.n_elements)), params, frequency_step=2.0
    )

    assert np.all(np.isfinite(to_numpy(result.scattered)))


def test_scattering_wavefield_components_and_arrival() -> None:
    """Time conversion preserves components and return-path travel time."""
    params = P4_2v()
    medium = MediumParams()
    scatterers = _array([[0.0, 0.02]])
    positions = _array([[0.0, 0.03], [0.0, 0.04]])

    result = scattering_wavefield(
        positions,
        scatterers,
        _array(np.ones(1)),
        _array(np.zeros(params.n_elements)),
        params,
        medium,
        frequency_step=0.5,
    )
    scattered = to_numpy(result.scattered)
    times = to_numpy(result.times)
    arrival_gap = times[np.argmax(np.abs(scattered[1]))] - times[np.argmax(np.abs(scattered[0]))]

    assert result.incident.shape == result.scattered.shape
    np.testing.assert_allclose(to_numpy(result.total), to_numpy(result.incident + result.scattered))
    assert arrival_gap == pytest.approx(0.01 / medium.speed_of_sound, rel=0.2)


def test_scattering_wavefield_time_oversampling_preserves_original_samples() -> None:
    """The convenience API exposes phase-faithful inverse-FFT oversampling."""
    params = P4_2v()
    positions = _array([[0.0, 0.03]])
    scatterers = _array([[0.0, 0.02]])
    reflection_coefficients = _array([0.5])
    delays = _array(np.zeros(params.n_elements))

    base = scattering_wavefield(
        positions,
        scatterers,
        reflection_coefficients,
        delays,
        params,
        frequency_step=2.0,
    )
    dense = scattering_wavefield(
        positions,
        scatterers,
        reflection_coefficients,
        delays,
        params,
        frequency_step=2.0,
        time_oversampling=2,
    )

    assert dense.incident.shape[-1] == 2 * base.incident.shape[-1]
    np.testing.assert_allclose(to_numpy(dense.times[::2]), to_numpy(base.times))
    np.testing.assert_allclose(to_numpy(dense.incident[..., ::2]), to_numpy(base.incident), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(to_numpy(dense.scattered[..., ::2]), to_numpy(base.scattered), rtol=1e-5, atol=1e-8)


def test_inputs_are_not_mutated() -> None:
    """Public scattering calls leave all caller-owned arrays unchanged."""
    params = P4_2v()
    positions = np.asarray([[0.0, 0.03]])
    scatterers = np.asarray([[0.0, 0.02]])
    rc = np.asarray([-1.0])
    delays = np.zeros(params.n_elements)
    originals = [value.copy() for value in (positions, scatterers, rc, delays)]

    scattering_pfield_spectrum(
        cast(Array, positions),
        cast(Array, scatterers),
        cast(Array, rc),
        cast(Array, delays),
        params,
        frequency_step=2.0,
    )

    for value, original in zip((positions, scatterers, rc, delays), originals, strict=True):
        np.testing.assert_array_equal(value, original)


def test_array_api_strict_preserves_backend() -> None:
    """The point-observer path uses only operations in the Array API contract."""
    strict_xp = cast(_ArrayNamespace, array_api_strict)
    params, positions, scatterers, rc, delays = _case(strict_xp)

    result = scattering_pfield_spectrum(
        positions,
        scatterers,
        rc,
        delays,
        params,
        frequency_step=4.0,
    )

    assert type(result.incident).__module__.startswith("array_api_strict")
    assert type(result.scattered).__module__.startswith("array_api_strict")


def test_chunking_does_not_change_public_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pair chunk boundaries do not alter the accumulated scattered field."""
    import fast_simus.scattering as scattering_module

    params = P4_2v()
    positions = _array([[0.0, 0.03], [0.004, 0.04], [-0.003, 0.05]])
    scatterers = _array([[-0.002, 0.018], [0.0, 0.023], [0.003, 0.027]])
    rc = _array([1.0, -0.5, 0.25])
    delays = _array(np.zeros(params.n_elements))
    expected = scattering_pfield_spectrum(positions, scatterers, rc, delays, params, frequency_step=4.0).scattered

    monkeypatch.setattr(scattering_module, "_MAX_PAIR_ELEMENTS", 2)
    actual = scattering_pfield_spectrum(positions, scatterers, rc, delays, params, frequency_step=4.0).scattered

    np.testing.assert_allclose(to_numpy(actual), to_numpy(expected), rtol=1e-12, atol=1e-12)
