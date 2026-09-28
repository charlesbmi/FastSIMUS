"""Public 2D contracts retained by the shared simulation core."""

import numpy as np
import pytest

from fast_simus import (
    PfieldPlan,
    PfieldStrategy,
    SimusPlan,
    SimusStrategy,
    TransducerParams,
    pfield,
    pfield_precompute,
    simus,
    simus_precompute,
)
from fast_simus.transducer_params import BaffleType
from tests.conftest import to_numpy


def _inputs(xp, baffle=BaffleType.SOFT):
    params = TransducerParams(freq_center=2e6, pitch=3e-4, width=2.5e-4, n_elements=16, baffle=baffle)
    points = xp.asarray([[0.003, 0.012], [-0.002, 0.025]], dtype=xp.float32)
    return params, points, xp.ones(2), xp.zeros(16)


def test_legacy_plan_tuple_contracts():
    """Plans retain positional construction, named fields and replacement."""
    params, points, rc, delays = _inputs(np)
    common = ("selected_freqs", "pulse_spectrum", "probe_spectrum", "n_sub", "seg_length", "correction_factor")
    field = pfield_precompute(points, delays, params=params)
    echo = simus_precompute(points, rc, delays, params=params)
    assert isinstance(field, PfieldPlan)
    assert field._fields == (*common, "freq_step", "n_freq_full", "freq_idx_start")
    assert isinstance(echo, SimusPlan)
    assert echo._fields == (*common, "n_freq_full", "freq_idx_start", "n_fft")
    assert isinstance(field, PfieldPlan)
    assert isinstance(echo, SimusPlan)
    restored_plans = (PfieldPlan(*field), SimusPlan(*echo))
    for plan, restored in zip((field, echo), restored_plans, strict=True):
        restored = restored._replace(correction_factor=plan.correction_factor)
        assert restored.selected_freqs is plan.selected_freqs
        assert restored.correction_factor == plan.correction_factor


def test_disabled_transmit_is_zero_without_mutation(xp):
    """Disabling all transmitters yields zero fields and echoes on each backend."""
    params, points, rc, delays = _inputs(xp)
    delays = delays + xp.asarray(float("nan"))
    apodization = xp.ones(params.n_elements)
    before = [to_numpy(a).copy() for a in (points, rc, delays, apodization)]
    pressure = pfield(points, delays, params, tx_apodization=apodization)
    result = simus(points, rc, delays, params, tx_apodization=apodization)
    for result_array in (pressure, result.rf, result.spectrum):
        np.testing.assert_array_equal(to_numpy(result_array), 0)
    for original, saved in zip((points, rc, delays, apodization), before, strict=True):
        np.testing.assert_array_equal(to_numpy(original), saved)


@pytest.mark.parametrize("baffle,full_directivity", [("rigid", False), (0.5, False), ("soft", True)])
def test_auto_echo_honors_requested_physics(xp, baffle, full_directivity):
    """Auto dispatch preserves physics unsupported by the native kernels."""
    params, points, rc, delays = _inputs(xp, baffle)
    options = {"full_frequency_directivity": full_directivity}
    expected = simus(points, rc, delays, params, strategy=SimusStrategy.PYTHON, **options)
    actual = simus(points, rc, delays, params, **options)
    for observed, reference in zip(actual, expected, strict=True):
        peak = np.max(np.abs(to_numpy(reference)))
        np.testing.assert_allclose(to_numpy(observed), to_numpy(reference), rtol=0, atol=1e-4 * peak)


@pytest.mark.parametrize("strategy", [SimusStrategy.METAL, SimusStrategy.CUDA])
def test_explicit_native_echo_rejects_unsupported_physics(strategy):
    """Unsupported requests fail before importing or launching a native kernel."""
    params, points, rc, delays = _inputs(np, "rigid")
    with pytest.raises(NotImplementedError, match="baffle"):
        simus(points, rc, delays, params, strategy=strategy)


def test_explicit_metal_field_rejects_wrong_backend():
    """A native strategy on incompatible arrays has an actionable error."""
    params, points, _rc, delays = _inputs(np)
    with pytest.raises(NotImplementedError, match="MLX"):
        pfield(points, delays, params, strategy=PfieldStrategy.METAL)


def test_supported_metal_calls_skip_portable_preparation(monkeypatch):
    """Supported native calls do not allocate the portable geometry tensors."""
    import importlib

    mx = pytest.importorskip("mlx.core")
    from fast_simus.backends.mlx import ensure_compat

    ensure_compat(mx)
    params, points, rc, delays = _inputs(mx)

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Native execution entered portable geometry preparation")

    for module_name, helper in (("pfield", "_prepare_frequency_sweep"), ("simus", "_prepare_simus_sweep")):
        monkeypatch.setattr(importlib.import_module(f"fast_simus.{module_name}"), helper, unexpected_preparation)
    pressure = pfield(points, delays, params, strategy=PfieldStrategy.METAL)
    echo = simus(points, rc, delays, params, strategy=SimusStrategy.METAL)
    for result in (pressure, echo.rf, echo.spectrum):
        assert np.all(np.isfinite(to_numpy(result)))
