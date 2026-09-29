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


def test_compiled_point_observation():
    """Numerical block execution supports JAX device loops after eager planning."""
    import jax
    import jax.numpy as xp

    from fast_simus._scattered import scattered_block

    probe, observers, scatterers, rc, delays = scene(xp)
    plan = scattered_field_precompute(observers, scatterers, rc, delays, probe, execution=ExecutionOptions(8192))
    compute = jax.jit(lambda p, s, r, d: scattered_block(p, s, r, d, xp.ones_like(d), plan, "total", True, None, xp))
    actual = compute(observers, scatterers, rc, delays)
    expected, _ = scattered_pfield_spectrum(
        observers, scatterers, rc, delays, probe, plan=plan, component="total", full_frequency_directivity=True
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4 * np.max(np.abs(expected)))


def test_orthoslice_sampling_and_simulation():
    """Deduplicated planes recover the same coordinates and spectra as a tiny volume."""
    from examples._wavefield3d import orthoslices, simulate

    x, y, z = np.array([-0.001, 0, 0.001]), np.array([-0.002, 0, 0.002]), np.array([0.012, 0.014, 0.016])
    slices = orthoslices(x, y, z)
    assert len(slices.points) == 19
    probe, _, scatterers, rc, delays = scene()
    volume = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).astype(np.float32)
    dense, plan = scattered_pfield_spectrum(volume, scatterers, rc, delays, probe)
    # The bounding boxes coincide, so the two observation sets share the grid.
    sparse, info = scattered_pfield_spectrum(slices.points, scatterers, rc, delays, probe)
    np.testing.assert_array_equal(plan.selected_freqs, info.selected_freqs)
    for point, value in zip(slices.points, sparse, strict=True):
        index = np.argwhere(np.all(volume == point, axis=-1))[0]
        np.testing.assert_allclose(value, dense[tuple(index)], rtol=1e-5)
    result = simulate(
        dict(scene="Point", count=2, backend="NumPy", transmit="Plane wave", steer=0, focus=18, side=2, smoke=True)
    )
    assert result.incident.shape == result.scattered.shape
    assert np.isfinite(result.scattered).all()


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
