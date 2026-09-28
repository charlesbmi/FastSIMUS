"""Volumetric pulse-echo contractions and sampling contracts."""

from typing import Any

import numpy as _np
import pytest

from fast_simus import simus, simus_compute, simus_precompute
from fast_simus.aperture import matrix_aperture
from fast_simus.plans import EchoPlan
from fast_simus.transducer import Transducer
from tests._reference_3d import rectangular_transfer

np: Any = _np


def test_echo_complex_oracle_and_shapes():
    """Reciprocal receive uses an ordinary transpose and preserves channels."""
    a = matrix_aperture(shape=(2, 2), pitch=(0.0004, 0.0005), size=(0.0002, 0.0003), xp=np, dtype=np.float64)
    t = Transducer(a, "3d", 2e6)
    p = np.array([[[0.001, 0.002, 0.015], [-0.002, 0.001, 0.02]]])
    d = np.array([0.0, 1e-7, 2e-7, 0.0])
    rc = np.array([[1.0, -0.3]])
    plan = simus_precompute(p, rc, d, t, element_splitting=(2, 2))
    assert isinstance(plan, EchoPlan)
    result = simus_compute(p, rc, d, plan, t, full_frequency_directivity=True)
    h = rectangular_transfer(
        p.reshape(-1, 3), a.centers, a.width_axes, a.height_axes, a.sizes, plan.selected_freqs, subdivision=(2, 2)
    )
    f = np.asarray(plan.selected_freqs)
    tx = np.einsum("fpe,fe->fp", h, np.exp(2j * np.pi * f[:, None] * d)) * plan._pulse[:, None] * plan._probe[:, None]
    expected = np.einsum("fpe,fp->fe", h, tx * rc.reshape(-1)) * plan._probe[:, None]
    observed = np.asarray(result.spectrum)[plan.freq_idx_start : plan.freq_idx_start + len(f)]
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-8 * np.max(np.abs(expected)))
    flat = simus(p.reshape(-1, 3), rc.reshape(-1), d, t, element_splitting=(2, 2), full_frequency_directivity=True)
    np.testing.assert_allclose(flat.rf, result.rf)
    assert len(plan.sample_times) == result.rf.shape[0]
    assert plan.sampling_frequency == plan.n_fft * plan.freq_step
    zeros = simus(p, rc * 0, d, t)
    np.testing.assert_array_equal(zeros.rf, 0)
    disabled = simus(p, rc, d * np.nan, t)
    np.testing.assert_array_equal(disabled.spectrum, 0)


def test_echo_invalid_and_single_point():
    """Single points are valid; mismatched reflectivity and unsafe sampling fail."""
    t = Transducer(
        matrix_aperture(shape=(1, 1), pitch=(0.001, 0.001), size=(0.0002, 0.0002), xp=np, dtype=np.float64), "3d", 2e6
    )
    p = np.array([0.0, 0.0, 0.02])
    d = np.zeros(1)
    assert simus(p, np.asarray(1.0), d, t).rf.shape[1] == 1
    with pytest.raises(ValueError):
        simus(p, np.asarray(1.0), d, t, fs=1e6)
    with pytest.raises(ValueError):
        simus(p, np.asarray(1.0), d, t, tx_n_wavelengths=np.inf)
    with pytest.raises(ValueError, match="reflectivity"):
        simus(p, np.ones(1), d, t)
