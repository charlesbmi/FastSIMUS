"""Complex 3D field verification against independent direct quadrature."""

from typing import Any

import numpy as _np

np: Any = _np
import pytest

from fast_simus import pfield, pfield_precompute, pfield_spectrum, pfield_spectrum_compute, rms_from_spectrum
from fast_simus.aperture import matrix_aperture, transform_aperture
from fast_simus.medium_params import MediumParams
from fast_simus.plans import FieldPlan, FieldSpectrumInfo
from fast_simus.spectrum import probe_spectrum, pulse_spectrum
from fast_simus.transducer import Transducer
from tests._reference_3d import rectangular_transfer


@pytest.mark.parametrize("baffle", ["soft", "rigid", 0.3])
@pytest.mark.parametrize("full", [True, False])
def test_complex_oracle(baffle, full):
    """Off-axis phase and amplitude agree without independent normalization."""
    a = matrix_aperture(shape=(2, 2), pitch=(0.0004, 0.0005), size=(0.0003, 0.0004), xp=np, dtype=np.float64)
    t = Transducer(a, "3d", 2e6, baffle=baffle)
    p = np.array([[0.001, -0.002, 0.015], [0.003, 0.001, 0.024]])
    d = np.array([0.0, 2e-7, 4e-7, 1e-7])
    m = MediumParams(attenuation=0.4)
    spec, info = pfield_spectrum(p, d, t, m, element_splitting=(2, 3), full_frequency_directivity=full)
    f = np.asarray(info.selected_freqs)
    h = rectangular_transfer(
        p,
        a.centers,
        a.width_axes,
        a.height_axes,
        a.sizes,
        f,
        subdivision=(2, 3),
        baffle=baffle,
        attenuation=0.4,
        full_frequency_directivity=full,
    )
    expected = np.einsum("fpe,fe->pf", h, np.exp(2j * np.pi * f[:, None] * d))
    expected *= pulse_spectrum(2 * np.pi * f, 2e6, 1.0) * probe_spectrum(2 * np.pi * f, 2e6, 0.75)
    np.testing.assert_allclose(spec, expected, rtol=0, atol=1e-8 * np.max(np.abs(expected)))
    np.testing.assert_allclose(
        pfield(p, d, t, m, element_splitting=(2, 3), full_frequency_directivity=full), rms_from_spectrum(spec, info)
    )


def test_transform_cw_and_plan_bounds():
    """Rigid transforms retain local visibility, including negative global z."""
    a = matrix_aperture(shape=(1, 1), pitch=(0.001, 0.001), size=(0.0002, 0.0002), xp=np, dtype=np.float64)
    t = Transducer(a, "3d", 2e6)
    p = np.array([0.001, 0.002, 0.02])
    d = np.zeros(1)
    spec, info = pfield_spectrum(p, d, t, tx_n_wavelengths=np.inf)
    assert isinstance(info, FieldSpectrumInfo)
    assert info.is_cw and spec.shape == (1,)
    r = np.diag([1.0, -1.0, -1.0])
    shift = np.array([0.01, 0.0, -0.03])
    transformed = Transducer(transform_aperture(a, r, shift), "3d", 2e6)
    rotated, _ = pfield_spectrum(p @ r.T + shift, d, transformed, tx_n_wavelengths=np.inf)
    np.testing.assert_allclose(rotated, spec, rtol=1e-10)
    plan = pfield_precompute(p, d, t)
    assert isinstance(plan, FieldPlan)
    with pytest.raises(ValueError, match="bound"):
        plan.validate_inputs(p * 2, d)
    with pytest.raises(ValueError, match="plan"):
        pfield_spectrum_compute(p, d, plan, transformed)
    with pytest.raises(ValueError):
        pfield(p, -np.ones(1), t)


def test_sparse_nonplanar_geometry_and_permutation():
    """Unequal rotated elements share the solver and preserve channel permutations."""
    from fast_simus.aperture import RectangularAperture

    centers = np.array([[-0.001, 0.0, 0.0002], [0.002, 0.001, -0.0001]])
    u = np.array([[1.0, 0.0, 0.0], [np.cos(0.2), 0.0, -np.sin(0.2)]])
    v = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    size = np.array([[0.0002, 0.0004], [0.0003, 0.0002]])
    a = RectangularAperture(centers, u, v, size)
    p = np.array([[0.001, -0.001, 0.02], [0.0, 0.001, 0.03]])
    delays = np.array([0.0, 1e-7])
    t = Transducer(a, "3d", 2e6)
    spectrum, info = pfield_spectrum(p, delays, t, element_splitting=(2, 2), full_frequency_directivity=True)
    f = np.asarray(info.selected_freqs)
    h = rectangular_transfer(p, centers, u, v, size, f, subdivision=(2, 2))
    expected = np.einsum("fpe,fe->pf", h, np.exp(2j * np.pi * f[:, None] * delays))
    expected *= pulse_spectrum(2 * np.pi * f, 2e6, 1.0) * probe_spectrum(2 * np.pi * f, 2e6, 0.75)
    np.testing.assert_allclose(spectrum, expected, rtol=0, atol=1e-8 * np.max(np.abs(expected)))
    swapped = Transducer(RectangularAperture(centers[::-1], u[::-1], v[::-1], size[::-1]), "3d", 2e6)
    other, _ = pfield_spectrum(p, delays[::-1], swapped, element_splitting=(2, 2), full_frequency_directivity=True)
    np.testing.assert_allclose(other, spectrum, rtol=1e-10)
