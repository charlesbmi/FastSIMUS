"""Independent single-scattering pressure and common-grid contracts."""

from typing import Any

import numpy as _np
import pytest

from fast_simus import ExecutionOptions, MediumParams, Transducer, matrix_aperture, spectrum_to_wavefield
from fast_simus.scattered import iter_scattered_pfield_spectrum, scattered_field_precompute, scattered_pfield_spectrum
from tests._reference_3d import rectangular_transfer
from tests.conftest import to_numpy

np: Any = _np


def scene(xp=np):
    """Small off-axis observation and scatterer geometry."""
    probe = Transducer(matrix_aperture(shape=(2, 1), pitch=(0.0004, 0.0004), size=(0.0002, 0.0002), xp=xp), "3d", 2e6)
    observers = xp.asarray([[0.001, 0.002, 0.012], [-0.001, 0.001, 0.016]], dtype=xp.float32)
    scatterers = xp.asarray([[0.0, 0.0, 0.01], [0.002, 0.0, 0.018], [-0.001, 0.002, 0.015]], dtype=xp.float32)
    return (
        probe,
        observers,
        scatterers,
        xp.asarray([0.001, -0.0004, 0.0002], dtype=xp.float32),
        xp.zeros(2, dtype=xp.float32),
    )


@pytest.mark.parametrize("budget", [8192, 1048576])
def test_scattered_reference_and_components(xp, budget):
    """Point pressure includes one transmit probe response and isotropic return propagation."""
    probe, observers, scatterers, rc, delays = scene(xp)
    medium = MediumParams(attenuation=0.4)
    plan = scattered_field_precompute(
        observers, scatterers, rc, delays, probe, medium, execution=ExecutionOptions(budget)
    )
    options = dict(plan=plan, full_frequency_directivity=True)
    scattered, info = scattered_pfield_spectrum(observers, scatterers, rc, delays, probe, medium, **options)
    incident, _ = scattered_pfield_spectrum(
        observers, scatterers, rc, delays, probe, medium, component="incident", **options
    )
    total, _ = scattered_pfield_spectrum(observers, scatterers, rc, delays, probe, medium, component="total", **options)
    np.testing.assert_allclose(to_numpy(total), to_numpy(incident) + to_numpy(scattered), rtol=1e-5, atol=1e-6)
    f = to_numpy(info.selected_freqs).astype(np.float64)
    aperture = probe.aperture
    h = rectangular_transfer(
        to_numpy(scatterers),
        to_numpy(aperture.centers),
        to_numpy(aperture.width_axes),
        to_numpy(aperture.height_axes),
        to_numpy(aperture.sizes),
        f,
        attenuation=0.4,
    )
    from fast_simus.spectrum import probe_spectrum, pulse_spectrum

    illumination = (
        h.sum(axis=-1)
        * pulse_spectrum(2 * np.pi * f, 2e6, 1)[:, None]
        * probe_spectrum(2 * np.pi * f, 2e6, 0.75)[:, None]
    )
    distance = np.linalg.norm(to_numpy(scatterers)[:, None, :] - to_numpy(observers)[None, :, :], axis=-1)
    expected = np.stack(
        [
            (illumination[k] * to_numpy(rc))
            @ (np.exp((2j * np.pi * frequency / 1540 - 0.4 * np.log(10) / 20 * frequency * 1e-4) * distance) / distance)
            for k, frequency in enumerate(f)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(to_numpy(scattered), expected, rtol=0, atol=1e-4 * np.max(np.abs(expected)))
    blocks = list(
        iter_scattered_pfield_spectrum(
            observers, scatterers, rc, delays, plan, probe, medium, full_frequency_directivity=True
        )
    )
    np.testing.assert_allclose(np.concatenate([to_numpy(b.values) for b in blocks]), to_numpy(scattered), rtol=1e-5)


def test_empty_and_linearity():
    """Empty clouds, zero strengths and coefficient scaling preserve pressure conventions."""
    probe, observers, scatterers, rc, delays = scene()
    plan = scattered_field_precompute(observers, scatterers, rc, delays, probe)
    original = observers.copy(), scatterers.copy(), rc.copy(), delays.copy()
    a, _ = scattered_pfield_spectrum(observers, scatterers, rc, delays, probe, plan=plan)
    b, _ = scattered_pfield_spectrum(observers, scatterers, 2 * rc, delays, probe, plan=plan)
    np.testing.assert_allclose(b, 2 * a)
    zero, _ = scattered_pfield_spectrum(observers, scatterers[:0], rc[:0], delays, probe)
    np.testing.assert_array_equal(zero, 0)
    for value, before in zip((observers, scatterers, rc, delays), original, strict=True):
        np.testing.assert_array_equal(value, before)
    zero, _ = scattered_pfield_spectrum(observers, scatterers, np.zeros_like(rc), delays, probe, plan=plan)
    np.testing.assert_array_equal(zero, 0)
    wave = spectrum_to_wavefield(a, plan)
    assert np.isfinite(wave.frames).all()


def test_point_arrival_and_rigid_transform():
    """Two-leg propagation arrives causally and is invariant under a rigid translation."""
    from fast_simus import transform_aperture

    probe, observers, scatterers, rc, delays = scene()
    observers = np.array([[0, 0, 0.005]], dtype=np.float32)
    scatterers = np.array([[0, 0, 0.01]], dtype=np.float32)
    rc = np.array([0.001], dtype=np.float32)
    spectrum, plan = scattered_pfield_spectrum(observers, scatterers, rc, delays, probe)
    wave = spectrum_to_wavefield(spectrum, plan)
    arrival = np.asarray(wave.times)[np.argmax(np.abs(wave.frames[0]))]
    assert abs(arrival - 0.015 / 1540) < 1 / 2e6
    shift = np.array([0.003, -0.004, 0.002], dtype=np.float32)
    moved = Transducer(transform_aperture(probe.aperture, np.eye(3, dtype=np.float32), shift), "3d", 2e6)
    other, other_plan = scattered_pfield_spectrum(observers + shift, scatterers + shift, rc, delays, moved)
    np.testing.assert_allclose(other_plan.selected_freqs, plan.selected_freqs, rtol=1e-6)
    np.testing.assert_allclose(other, spectrum, rtol=0, atol=1e-4 * np.max(np.abs(spectrum)))


def test_orthoslice_sampling():
    """Deduplicated planes recover the same coordinates and spectra as a tiny volume."""
    from examples._wavefield3d import orthoslices

    x, y, z = np.array([-0.001, 0, 0.001]), np.array([-0.002, 0, 0.002]), np.array([0.012, 0.014, 0.016])
    slices = orthoslices(x, y, z)
    probe, _, scatterers, rc, delays = scene()
    volume = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).astype(np.float32)
    dense, plan = scattered_pfield_spectrum(volume, scatterers, rc, delays, probe)
    # The bounding boxes coincide, so the two observation sets share the grid.
    sparse, info = scattered_pfield_spectrum(slices.points, scatterers, rc, delays, probe)
    np.testing.assert_array_equal(plan.selected_freqs, info.selected_freqs)
    for point, value in zip(slices.points, sparse, strict=True):
        index = np.argwhere(np.all(volume == point, axis=-1))[0]
        np.testing.assert_allclose(value, dense[tuple(index)], rtol=1e-5)


def test_plan_bounds_and_cancellation():
    """Changed support, incompatible shapes and cancellation fail before yielding data."""
    probe, observers, scatterers, rc, delays = scene()
    plan = scattered_field_precompute(observers, scatterers, rc, delays, probe)
    with pytest.raises(ValueError, match="bounds"):
        scattered_pfield_spectrum(observers * 2, scatterers, rc, delays, probe, plan=plan)
    with pytest.raises(ValueError, match="reflectivity"):
        scattered_field_precompute(observers, scatterers, rc[:1], delays, probe)
    with pytest.raises(InterruptedError):
        list(iter_scattered_pfield_spectrum(observers, scatterers, rc, delays, plan, probe, cancelled=lambda: True))


def test_coincident_point_regularization():
    """The documented distance floor applies to phase, attenuation and spreading."""
    from fast_simus import pfield_spectrum

    fc, speed, attenuation = 2e6, 1480.0, 0.6
    medium = MediumParams(speed_of_sound=speed, attenuation=attenuation)
    minimum = speed / (2 * fc)
    distances = np.array([0, minimum / 4, minimum, 2 * minimum])
    source = np.array([[0, 0, 0.01]])
    observers = source + np.column_stack([distances, np.zeros((4, 2))])
    probe = Transducer(
        matrix_aperture(shape=(1, 1), pitch=(0.0003, 0.0003), size=(0.0002, 0.0002), xp=np, dtype=np.float64), "3d", fc
    )
    delays, coefficients = np.zeros(1), np.array([0.001])
    incident, _ = pfield_spectrum(source, delays, probe, medium, tx_n_wavelengths=np.inf)
    actual, _ = scattered_pfield_spectrum(
        observers, source, coefficients, delays, probe, medium, tx_n_wavelengths=np.inf
    )
    safe = np.maximum(distances, minimum)
    response = np.exp((2j * np.pi * fc / speed - attenuation * np.log(10) / 20 * fc * 1e-4) * safe) / safe
    expected = coefficients[0] * incident[0, 0] * response
    np.testing.assert_allclose(actual[:, 0], expected, rtol=1e-12)


def test_scattered_rotation_invariance():
    """A general rigid rotation preserves CW pressure with oriented elements."""
    from fast_simus import transform_aperture

    probe, observers, scatterers, rc, delays = scene()
    angle = 0.63
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32
    )
    shift = np.array([0.003, -0.004, 0.002], dtype=np.float32)
    moved = Transducer(transform_aperture(probe.aperture, rotation, shift), "3d", 2e6)
    expected, _ = scattered_pfield_spectrum(observers, scatterers, rc, delays, probe, tx_n_wavelengths=float("inf"))
    actual, _ = scattered_pfield_spectrum(
        observers @ rotation.T + shift,
        scatterers @ rotation.T + shift,
        rc,
        delays,
        moved,
        tx_n_wavelengths=float("inf"),
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4 * np.max(np.abs(expected)))
