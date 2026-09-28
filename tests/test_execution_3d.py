"""Workspace-independent numerical results and portable device execution."""

from typing import Any

import numpy as _np

np: Any = _np
import pytest

from fast_simus import Transducer, matrix_aperture, pfield, simus
from fast_simus.execution import ExecutionOptions
from tests.conftest import to_numpy


@pytest.mark.parametrize("budget", [4096, 16384, 1048576])
def test_budgets_and_backends(xp, budget):
    """Uneven point/patch tails produce the same field and raw RF on each device."""
    a = matrix_aperture(shape=(2, 1), pitch=(0.0004, 0.0004), size=(0.0003, 0.0002), xp=xp)
    t = Transducer(a, "3d", 2e6)
    p = xp.asarray([[0.001, 0.002, 0.015], [-0.002, 0.001, 0.018], [0.0, 0.0, 0.02]], dtype=xp.float32)
    d = xp.zeros(2, dtype=xp.float32)
    rc = xp.asarray([1.0, -0.2, 0.4], dtype=xp.float32)
    field = pfield(
        p, d, t, execution=ExecutionOptions(budget), element_splitting=(3, 5), full_frequency_directivity=True
    )
    echo = simus(
        p, rc, d, t, execution=ExecutionOptions(budget), element_splitting=(3, 5), full_frequency_directivity=True
    )
    a_ref = matrix_aperture(shape=(2, 1), pitch=(0.0004, 0.0004), size=(0.0003, 0.0002), xp=np, dtype=np.float64)
    t_ref = Transducer(a_ref, "3d", 2e6)
    pr = to_numpy(p).astype(np.float64)
    dr = to_numpy(d).astype(np.float64)
    ref_field = pfield(pr, dr, t_ref, element_splitting=(3, 5), full_frequency_directivity=True)
    ref_echo = simus(
        pr, to_numpy(rc).astype(np.float64), dr, t_ref, element_splitting=(3, 5), full_frequency_directivity=True
    )
    for result, reference in ((field, ref_field), (echo.spectrum, ref_echo.spectrum), (echo.rf, ref_echo.rf)):
        reference = to_numpy(reference)
        np.testing.assert_allclose(to_numpy(result), reference, rtol=0, atol=1e-4 * np.max(np.abs(reference)))


def test_workspace_rejected():
    """Impossible budgets fail clearly."""
    with pytest.raises(ValueError):
        ExecutionOptions(1)


def test_jax_compiled_tiles():
    """Closed-over plans compile with fixed-size device loop carries."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from fast_simus import pfield_compute, pfield_precompute, simus_compute, simus_precompute

    a = matrix_aperture(shape=(2, 1), pitch=(0.0004, 0.0004), size=(0.0003, 0.0002), xp=jnp)
    t = Transducer(a, "3d", 2e6)
    p = jnp.asarray([[0.0, 0.0, 0.02], [0.002, 0.001, 0.017]], dtype=jnp.float32)
    d = jnp.zeros(2)
    rc = jnp.ones(2)
    options = ExecutionOptions(4096)
    field_plan = pfield_precompute(p, d, t, execution=options)
    echo_plan = simus_precompute(p, rc, d, t, execution=options)
    field = jax.jit(lambda p, d: pfield_compute(p, d, field_plan, t))
    echo = jax.jit(lambda p, rc, d: simus_compute(p, rc, d, echo_plan, t))
    np.testing.assert_allclose(to_numpy(field(p, d)), to_numpy(pfield_compute(p, d, field_plan, t)), rtol=1e-5)
    np.testing.assert_allclose(
        to_numpy(echo(p, rc, d).rf), to_numpy(simus_compute(p, rc, d, echo_plan, t).rf), rtol=1e-4, atol=1e-8
    )
