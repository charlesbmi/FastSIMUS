"""Unified backend requests preserve finite-aperture and workspace semantics."""

import numpy as np
import pytest

from fast_simus import ExecutionOptions, Transducer, TransducerParams, matrix_aperture, simus
from fast_simus.backends._selection import _namespace_kind
from tests.conftest import to_numpy


def test_finite_aperture_backend_request(xp):
    """Auto and explicit portable requests agree; required native 3D fails clearly."""
    probe = Transducer(matrix_aperture(shape=(2, 1), pitch=(0.0003, 0.0003), size=(0.0002, 0.0002), xp=xp), "3d", 2e6)
    points = xp.asarray([[0.001, 0.002, 0.015]], dtype=xp.float32)
    rc, delays = xp.ones(1, dtype=xp.float32), xp.zeros(2, dtype=xp.float32)
    kind = _namespace_kind(xp)
    automatic = simus(points, rc, delays, probe)
    portable = simus(points, rc, delays, probe, backend=kind)
    np.testing.assert_allclose(to_numpy(automatic.rf), to_numpy(portable.rf), rtol=1e-5)
    native = {"mlx": "metal", "cupy": "cuda"}.get(kind)
    if native:
        with pytest.raises(NotImplementedError, match="finite 3D"):
            simus(points, rc, delays, probe, backend=native)


def test_bounded_strip_backend_request(xp):
    """Bounded portable echo accumulation matches the complete strip calculation."""
    probe = TransducerParams(freq_center=2e6, pitch=0.0003, width=0.0002, n_elements=2)
    points = xp.asarray([[0.001, 0.015], [-0.002, 0.02], [0, 0.018]], dtype=xp.float32)
    rc, delays = xp.ones(3, dtype=xp.float32), xp.zeros(2, dtype=xp.float32)
    kind = _namespace_kind(xp)
    expected = simus(points, rc, delays, probe, backend=kind)
    actual = simus(points, rc, delays, probe, execution=ExecutionOptions(4096))
    for value, reference in zip(actual, expected, strict=True):
        ref = to_numpy(reference)
        np.testing.assert_allclose(to_numpy(value), ref, rtol=0, atol=1e-4 * np.max(np.abs(ref)))
    native = {"mlx": "metal", "cupy": "cuda"}.get(kind)
    if native:
        with pytest.raises(NotImplementedError, match="workspace"):
            simus(points, rc, delays, probe, backend=native, execution=ExecutionOptions(4096))
