"""Thin-lens phase and global time-reference verification."""

from typing import Any

import numpy as _np

from fast_simus import pfield_spectrum
from fast_simus.aperture import matrix_aperture
from fast_simus.lens import ElevationLens
from fast_simus.spectrum import probe_spectrum, pulse_spectrum
from fast_simus.transducer import Transducer
from tests._reference_3d import rectangular_transfer

np: Any = _np


def test_lens_reference_and_infinite_focus():
    """Patch phase uses one causal offset and actual temporal frequency."""
    a = matrix_aperture(shape=(1, 1), pitch=(0.001, 0.001), size=(0.0002, 0.002), xp=np, dtype=np.float64)
    p = np.array([[0.0, 0.0, 0.025], [0.001, 0.001, 0.035]])
    d = np.zeros(1)
    focus = 0.025
    t = Transducer(a, "3d", 2e6, lens=ElevationLens(np.array([focus])))
    spec, info = pfield_spectrum(p, d, t, element_splitting=(1, 64), full_frequency_directivity=True)
    f = np.asarray(info.selected_freqs)
    tau = 0.002**2 / (8 * 1540 * focus)
    h = rectangular_transfer(
        p,
        a.centers,
        a.width_axes,
        a.height_axes,
        a.sizes,
        f,
        subdivision=(1, 64),
        lens_focus=focus,
        lens_reference_delay=tau,
    )
    expected = h[:, :, 0].T * pulse_spectrum(2 * np.pi * f, 2e6, 1.0) * probe_spectrum(2 * np.pi * f, 2e6, 0.75)
    np.testing.assert_allclose(spec, expected, rtol=0, atol=1e-8 * np.max(np.abs(expected)))
    unlensed, _ = pfield_spectrum(p, d, Transducer(a, "3d", 2e6))
    infinite, _ = pfield_spectrum(p, d, Transducer(a, "3d", 2e6, lens=ElevationLens(np.array([np.inf]))))
    np.testing.assert_array_equal(infinite, unlensed)


def test_lens_quadrature_converges():
    """Refinement meets the independent successive-quadrature acceptance gate."""
    a = matrix_aperture(shape=(1, 1), pitch=(0.001, 0.001), size=(0.0002, 0.002), xp=np, dtype=np.float64)
    points = np.array([[0.0001, 0.0008, 0.025]])
    focus = 0.025
    tau = 0.002**2 / (8 * 1540 * focus)
    previous = None
    for level in range(8):
        count = 8 * 2**level
        reference = rectangular_transfer(
            points,
            a.centers,
            a.width_axes,
            a.height_axes,
            a.sizes,
            [2e6],
            subdivision=(1, count),
            lens_focus=focus,
            lens_reference_delay=tau,
        )
        if previous is not None and np.max(np.abs(reference - previous)) <= 2.5e-5 * np.max(np.abs(reference)):
            break
        previous = reference
    else:
        raise AssertionError("Lens quadrature failed to converge in eight doublings")
    t = Transducer(a, "3d", 2e6, lens=ElevationLens(np.array([focus])))
    actual, _ = pfield_spectrum(
        points, np.zeros(1), t, tx_n_wavelengths=np.inf, element_splitting=(1, count), full_frequency_directivity=True
    )
    np.testing.assert_allclose(actual, reference[:, :, 0].T, rtol=0, atol=1e-4 * np.max(np.abs(reference)))


def test_lens_jax_compiles():
    """Lens delay support is resolved eagerly, never converted from a tracer."""
    import pytest

    jax = pytest.importorskip("jax")
    xp = pytest.importorskip("jax.numpy")
    from fast_simus import pfield_compute, pfield_precompute, simus_compute, simus_precompute

    a = matrix_aperture(shape=(1, 1), pitch=(0.001, 0.001), size=(0.0002, 0.0002), xp=xp)
    t = Transducer(a, "3d", 2e6, lens=ElevationLens(xp.asarray([0.02])))
    p = xp.asarray([[0.0, 0.0, 0.02]])
    d = xp.zeros(1)
    rc = xp.ones(1)
    field = pfield_precompute(p, d, t)
    echo = simus_precompute(p, rc, d, t)
    pressure = jax.jit(lambda p, d: pfield_compute(p, d, field, t))(p, d)
    rf = jax.jit(lambda p, rc, d: simus_compute(p, rc, d, echo, t))(p, rc, d).rf
    assert np.all(np.isfinite(np.asarray(pressure))) and np.all(np.isfinite(np.asarray(rf)))


def test_lens_echo_uses_phase_on_both_legs():
    """Independent quadrature catches omitted or conjugated receive lens phase."""
    from fast_simus import simus_compute, simus_precompute

    aperture = matrix_aperture(shape=(2, 1), pitch=(0.0004, 0.0004), size=(0.0002, 0.002), xp=np, dtype=np.float64)
    focus = 0.025
    probe = Transducer(aperture, "3d", 2e6, lens=ElevationLens(np.full(2, focus)))
    points = np.array([[0.001, 0.0008, 0.02], [-0.001, 0.001, 0.03]])
    delays = np.array([0.0, 1e-7])
    rc = np.array([1.0, -0.4])
    plan = simus_precompute(points, rc, delays, probe, element_splitting=(1, 32))
    result = simus_compute(points, rc, delays, plan, probe, full_frequency_directivity=True)
    frequencies = np.asarray(plan.selected_freqs)
    transfer = rectangular_transfer(
        points,
        aperture.centers,
        aperture.width_axes,
        aperture.height_axes,
        aperture.sizes,
        frequencies,
        subdivision=(1, 32),
        lens_focus=focus,
        lens_reference_delay=plan.lens_reference_delay,
    )
    pressure = np.einsum("fpe,fe->fp", transfer, np.exp(2j * np.pi * frequencies[:, None] * delays))
    pressure *= plan._pulse[:, None] * plan._probe[:, None]
    expected = np.einsum("fpe,fp->fe", transfer, pressure * rc) * plan._probe[:, None]
    observed = np.asarray(result.spectrum)[plan.freq_idx_start : plan.freq_idx_start + len(frequencies)]
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-8 * np.max(np.abs(expected)))
