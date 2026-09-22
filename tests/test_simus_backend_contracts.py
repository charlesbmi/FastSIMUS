"""Public behavioral contracts for SIMUS backend requests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import array_api_strict as strict
import numpy as np
import pytest

from fast_simus import BackendKind, get_backend
from fast_simus.simus import SimusPlan, simus, simus_compute, simus_precompute
from fast_simus.transducer_params import BaffleType
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils._array_api import Array, _ArrayNamespace, as_numpy


@dataclass(frozen=True)
class _AcceleratedCase:
    custom_kind: BackendKind
    portable_kind: BackendKind
    xp: _ArrayNamespace
    kernel_module: str
    kernel_name: str
    eligibility_name: str


@pytest.fixture(params=["metal", "cuda"])
def accelerated_case(request: pytest.FixtureRequest) -> _AcceleratedCase:
    """Provide each available custom kernel and its same-library baseline."""
    if request.param == "metal":
        mx = pytest.importorskip("mlx.core")
        from fast_simus.backends.mlx import ensure_compat

        ensure_compat(mx)
        return _AcceleratedCase(
            BackendKind.METAL,
            BackendKind.MLX,
            cast(_ArrayNamespace, mx),
            "fast_simus.kernels.metal_simus",
            "simus_metal",
            "metal_simus_unsupported_reason",
        )

    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CuPy CUDA device not available")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy CUDA device not available")
    return _AcceleratedCase(
        BackendKind.CUDA,
        BackendKind.CUPY,
        cast(_ArrayNamespace, cp),
        "fast_simus.kernels.cuda_simus",
        "simus_cuda",
        "cuda_simus_unsupported_reason",
    )


def _accelerated_inputs(case: _AcceleratedCase):
    params = P4_2v()
    scatterers = case.xp.asarray([[0.0, 2e-2], [0.0, 4e-2]], dtype=case.xp.float32)
    rc = case.xp.ones(2, dtype=case.xp.float32)
    delays = case.xp.zeros(params.n_elements, dtype=case.xp.float32)
    return params, scatterers, rc, delays


def _assert_peak_close(actual: Any, expected: Any, *, fraction: float = 2e-2) -> None:
    actual_np = np.asarray(as_numpy(actual))
    expected_np = np.asarray(as_numpy(expected))
    peak = max(float(np.max(np.abs(actual_np))), float(np.max(np.abs(expected_np))))
    np.testing.assert_allclose(actual_np, expected_np, rtol=0.0, atol=fraction * peak)


def _numpy_inputs():
    params = P4_2v()
    scatterers = cast(Array, np.asarray([[0.0, 3e-2]], dtype=np.float32))
    rc = cast(Array, np.ones(1, dtype=np.float32))
    delays = cast(Array, np.zeros(params.n_elements, dtype=np.float32))
    return params, scatterers, rc, delays


def test_backend_context_is_not_a_simus_execution_request() -> None:
    """Discovery contexts and SIMUS execution requests have distinct roles."""
    params, scatterers, rc, delays = _numpy_inputs()

    with pytest.raises(TypeError, match=r"pass backend\.kind, or omit backend"):
        simus(scatterers, rc, delays, params, backend=get_backend("numpy"))  # type: ignore[arg-type]


def test_auto_name_infers_from_inputs() -> None:
    """The explicit auto name has the same behavior as an omitted request."""
    params, scatterers, rc, delays = _numpy_inputs()

    inferred = simus(scatterers, rc, delays, params)
    automatic = simus(scatterers, rc, delays, params, backend=BackendKind.AUTO)

    np.testing.assert_array_equal(automatic.rf, inferred.rf)
    np.testing.assert_array_equal(automatic.spectrum, inferred.spectrum)


@pytest.mark.parametrize("kind", [BackendKind.METAL, BackendKind.CUDA])
def test_accelerated_backend_labels_are_conditional(kind: BackendKind, monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery labels do not promise that every workload is kernel-eligible."""
    import fast_simus.backends._selection as selection

    monkeypatch.setattr(selection, "_mlx_namespace" if kind is BackendKind.METAL else "_cupy_namespace", lambda: np)

    assert "when supported" in get_backend(kind).label


def test_precompute_rejects_mixed_reflectivity_namespace() -> None:
    """Every array supplied to precompute participates in namespace validation."""
    params, scatterers, _, delays = _numpy_inputs()
    strict_rc = cast(Array, strict.ones((1,)))

    with pytest.raises(TypeError):
        simus_precompute(scatterers, strict_rc, delays, params)


def test_compute_rejects_mixed_apodization_namespace() -> None:
    """Apodization cannot be copied implicitly into the input namespace."""
    params, scatterers, rc, delays = _numpy_inputs()
    plan = simus_precompute(scatterers, rc, delays, params)
    strict_apodization = cast(Array, strict.ones((params.n_elements,)))

    with pytest.raises(TypeError):
        simus_compute(
            scatterers,
            rc,
            delays,
            plan,
            params,
            tx_apodization=strict_apodization,
        )


@pytest.mark.parametrize("field_name", ["selected_freqs", "pulse_spectrum", "probe_spectrum"])
def test_compute_rejects_mixed_plan_namespace(field_name: str) -> None:
    """Every array-valued plan field must share the input namespace."""
    params, scatterers, rc, delays = _numpy_inputs()
    plan = simus_precompute(scatterers, rc, delays, params)
    values = plan._asdict()
    values[field_name] = strict.asarray(values[field_name])
    mixed_plan = SimusPlan(**values)

    with pytest.raises(TypeError):
        simus_compute(scatterers, rc, delays, mixed_plan, params)


def test_inferred_acceleration_matches_explicit_and_preserves_namespace(
    accelerated_case: _AcceleratedCase,
) -> None:
    """Inference selects the matching custom kernel without moving outputs."""
    params, scatterers, rc, delays = _accelerated_inputs(accelerated_case)

    inferred = simus(scatterers, rc, delays, params, element_splitting=1)
    explicit = simus(
        scatterers,
        rc,
        delays,
        params,
        element_splitting=1,
        backend=accelerated_case.custom_kind,
    )

    assert type(inferred.rf) is type(scatterers)
    assert type(inferred.spectrum) is type(scatterers)
    np.testing.assert_array_equal(as_numpy(inferred.rf), as_numpy(explicit.rf))


def test_accelerated_path_matches_same_library_portable_path(accelerated_case: _AcceleratedCase) -> None:
    """A custom kernel agrees with the portable implementation on its device."""
    params, scatterers, rc, delays = _accelerated_inputs(accelerated_case)

    accelerated = simus(
        scatterers,
        rc,
        delays,
        params,
        element_splitting=1,
        backend=accelerated_case.custom_kind,
    )
    portable = simus(
        scatterers,
        rc,
        delays,
        params,
        element_splitting=1,
        backend=accelerated_case.portable_kind,
    )

    _assert_peak_close(accelerated.rf, portable.rf)


def test_portable_request_never_invokes_custom_kernel(
    accelerated_case: _AcceleratedCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-library portable request authoritatively disables acceleration."""
    params, scatterers, rc, delays = _accelerated_inputs(accelerated_case)
    kernel_module = import_module(accelerated_case.kernel_module)

    def fail_launch(*args, **kwargs):
        raise AssertionError("custom kernel should not run")

    monkeypatch.setattr(kernel_module, accelerated_case.kernel_name, fail_launch)

    portable = simus(
        scatterers,
        rc,
        delays,
        params,
        element_splitting=1,
        backend=accelerated_case.portable_kind,
    )
    assert type(portable.rf) is type(scatterers)


@pytest.mark.parametrize("unsupported", ["directivity", "baffle"])
def test_known_unsupported_workload_falls_back_only_for_inference(
    accelerated_case: _AcceleratedCase,
    unsupported: str,
) -> None:
    """Known limitations fall back opportunistically but strict requests fail."""
    params, scatterers, rc, delays = _accelerated_inputs(accelerated_case)
    kwargs: dict[str, Any] = {"element_splitting": 1}
    if unsupported == "directivity":
        kwargs["full_frequency_directivity"] = True
    else:
        params = params.model_copy(update={"baffle": BaffleType.RIGID})

    inferred = simus(scatterers, rc, delays, params, **kwargs)
    portable = simus(scatterers, rc, delays, params, backend=accelerated_case.portable_kind, **kwargs)
    np.testing.assert_allclose(as_numpy(inferred.rf), as_numpy(portable.rf), rtol=1e-5, atol=1e-7)

    with pytest.raises(NotImplementedError, match=unsupported if unsupported == "directivity" else "baffle"):
        simus(scatterers, rc, delays, params, backend=accelerated_case.custom_kind, **kwargs)


def test_simus_compute_honors_backend_contract(accelerated_case: _AcceleratedCase) -> None:
    """The split precompute/compute interface uses the same execution policy."""
    params, scatterers, rc, delays = _accelerated_inputs(accelerated_case)
    plan = simus_precompute(scatterers, rc, delays, params, element_splitting=1)

    inferred = simus_compute(scatterers, rc, delays, plan, params)
    explicit = simus_compute(
        scatterers,
        rc,
        delays,
        plan,
        params,
        backend=accelerated_case.custom_kind,
    )

    np.testing.assert_array_equal(as_numpy(inferred.rf), as_numpy(explicit.rf))


def test_resource_limit_falls_back_only_for_inference(
    accelerated_case: _AcceleratedCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter resource limits obey the same prefer-versus-require policy."""
    params, scatterers, rc, delays = _accelerated_inputs(accelerated_case)
    kernel_module = import_module(accelerated_case.kernel_module)
    monkeypatch.setattr(
        kernel_module,
        accelerated_case.eligibility_name,
        lambda *args: "synthetic device resource limit",
    )

    inferred = simus(scatterers, rc, delays, params, element_splitting=1)
    portable = simus(
        scatterers,
        rc,
        delays,
        params,
        element_splitting=1,
        backend=accelerated_case.portable_kind,
    )
    np.testing.assert_array_equal(as_numpy(inferred.rf), as_numpy(portable.rf))

    with pytest.raises(NotImplementedError, match="synthetic device resource limit"):
        simus(
            scatterers,
            rc,
            delays,
            params,
            element_splitting=1,
            backend=accelerated_case.custom_kind,
        )


def test_unexpected_kernel_failure_propagates(
    accelerated_case: _AcceleratedCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference never disguises compilation or launch failures as fallback."""
    params, scatterers, rc, delays = _accelerated_inputs(accelerated_case)
    kernel_module = import_module(accelerated_case.kernel_module)

    def fail_launch(*args, **kwargs):
        raise RuntimeError("synthetic kernel launch failure")

    monkeypatch.setattr(kernel_module, accelerated_case.kernel_name, fail_launch)

    with pytest.raises(RuntimeError, match="synthetic kernel launch failure"):
        simus(scatterers, rc, delays, params, element_splitting=1)
