"""SIMUS phase-grid regressions against a double-precision portable reference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_simus import jit, simus_compute, simus_precompute
from tests._picmus_phantom_shared import (
    _fastsimus_plane_wave_delays,
    _make_picmus_resolution_phantom,
    make_l11_picmus_matched_params,
)


def _inputs(angle_deg: float, disabled: bool = False):
    params = make_l11_picmus_matched_params()
    phantom = _make_picmus_resolution_phantom()
    positions = np.stack([phantom.x, phantom.z], axis=-1)
    delays = _fastsimus_plane_wave_delays(
        params.fastsimus_transducer, params.fastsimus_medium, np.deg2rad([angle_deg])
    )[0]
    if disabled:
        delays[::3] = np.nan
    return params, (positions, phantom.rc, delays)


def _plan(inputs, params, frequency_step: float = 1.0):
    return simus_precompute(
        *inputs,
        params=params.fastsimus_transducer,
        medium=params.fastsimus_medium,
        fs=params.sampling_frequency_hz,
        tx_n_wavelengths=params.tx_n_wavelengths,
        frequency_step=frequency_step,
    )


def _single_bin(plan):
    index = plan.selected_freqs.shape[0] // 2
    return plan._replace(
        selected_freqs=plan.selected_freqs[index : index + 1],
        pulse_spectrum=plan.pulse_spectrum[index : index + 1],
        probe_spectrum=plan.probe_spectrum[index : index + 1],
        freq_idx_start=plan.freq_idx_start + index,
    )


def _assert_result_close(actual, reference, atol_peak: float = 1e-4):
    for name in ("rf", "spectrum"):
        expected = np.asarray(getattr(reference, name))
        observed = np.asarray(getattr(actual, name))
        peak = np.max(np.abs(expected))
        np.testing.assert_allclose(observed, expected, rtol=0, atol=atol_peak * peak, err_msg=name)


@pytest.mark.parametrize(
    "backend,angle_deg,frequency_step,disabled,singleton",
    [
        (backend, angle, 1.0, False, False)
        for backend in ("rounded-numpy-plan", "mlx-python", "metal")
        for angle in (-16.0, 0.0, 16.0)
    ]
    + [
        ("metal", 16.0, 0.5, False, False),
        ("metal", 16.0, 2.0, True, False),
        ("metal", 16.0, 1.0, False, True),
    ],
)
def test_float32_frequency_grid_matches_portable_reference(
    angle_deg: float, frequency_step: float, disabled: bool, singleton: bool, backend: str
):
    """Rounding stored frequency samples must not accumulate phase-step error."""
    params, inputs = _inputs(angle_deg, disabled)
    plan = _plan(inputs, params, frequency_step)
    if singleton:
        plan = _single_bin(plan)
    expected = simus_compute(*inputs, plan=plan, params=params.fastsimus_transducer, medium=params.fastsimus_medium)
    if backend == "rounded-numpy-plan":
        candidate_inputs = inputs
        candidate_plan = plan._replace(
            selected_freqs=plan.selected_freqs.astype(np.float32),
            pulse_spectrum=plan.pulse_spectrum.astype(np.complex64),
            probe_spectrum=plan.probe_spectrum.astype(np.float32),
        )
        execution_backend = "numpy"
    else:
        mx = pytest.importorskip("mlx.core")
        from fast_simus.backends.mlx import ensure_compat

        ensure_compat(mx)
        candidate_inputs = tuple(mx.asarray(value) for value in inputs)
        candidate_plan = _plan(candidate_inputs, params, frequency_step)
        if singleton:
            candidate_plan = _single_bin(candidate_plan)
        execution_backend = "metal" if backend == "metal" else "mlx"
    snapshots = [np.array(value) for value in candidate_inputs]
    for _ in range(2):  # Reusing the plan must preserve the same numerical result.
        actual = simus_compute(
            *candidate_inputs,
            plan=candidate_plan,
            params=params.fastsimus_transducer,
            medium=params.fastsimus_medium,
            backend=execution_backend,
        )
        _assert_result_close(actual, expected)
    for value, before in zip(candidate_inputs, snapshots, strict=True):
        np.testing.assert_array_equal(np.asarray(value), before)


def test_jax_grid_compiles_with_closed_over_plan():
    """Canonical grid scalars remain static under the supported JIT calling pattern."""
    jnp = pytest.importorskip("jax.numpy")
    params, inputs = _inputs(16)
    candidate = tuple(jnp.asarray(value) for value in inputs)
    plan = _plan(candidate, params)

    def compute(*values: Any):
        return simus_compute(*values, plan=plan, params=params.fastsimus_transducer, medium=params.fastsimus_medium)

    compiled = jit(compute, xp=jnp)
    actual = compiled(*candidate)
    _assert_result_close(actual, compute(*candidate))
    reference = simus_compute(
        *inputs, plan=_plan(inputs, params), params=params.fastsimus_transducer, medium=params.fastsimus_medium
    )
    _assert_result_close(actual, reference)
