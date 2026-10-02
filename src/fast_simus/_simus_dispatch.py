"""Deep execution seam for the SIMUS selected-frequency spectrum."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from types import ModuleType
from typing import TYPE_CHECKING, cast

from array_api_compat import is_jax_namespace
from jaxtyping import Complex, Float

from fast_simus._blocking import legacy_point_count
from fast_simus._pfield_math import (
    _canonical_frequency_grid,
)
from fast_simus._transfer import _prepare_strip_transfer, _TransferPlan
from fast_simus.backends._selection import Backend, BackendKind, _coerce_backend_kind, _namespace_kind
from fast_simus.execution import ExecutionOptions
from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import BaffleType, TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace

if TYPE_CHECKING:
    from fast_simus.simus import SimusPlan


class _KernelPolicy(StrEnum):
    PREFER = "prefer"
    REQUIRE = "require"
    DISABLE = "disable"


@dataclass(frozen=True)
class _ResolvedBackendRequest:
    namespace_kind: BackendKind | None
    kernel_kind: BackendKind | None
    kernel_policy: _KernelPolicy


@dataclass(frozen=True)
class _SimusSpectrumRequest:
    """All data needed to compute the selected-frequency SIMUS spectrum."""

    scatterers: Float[Array, "n_scatterers 2"]
    rc: Float[Array, " n_scatterers"]
    delays_clean: Float[Array, " n_elements"]
    tx_apodization: Float[Array, " n_elements"]
    plan: SimusPlan
    params: TransducerParams
    medium: MediumParams
    full_frequency_directivity: bool
    xp: _ArrayNamespace
    backend: BackendKind | str | None
    execution: ExecutionOptions | None = None


def _expected_namespace_kind(kind: BackendKind) -> BackendKind:
    if kind is BackendKind.METAL:
        return BackendKind.MLX
    if kind is BackendKind.CUDA:
        return BackendKind.CUPY
    return kind


def _inferred_kernel_kind(namespace_kind: BackendKind | None) -> BackendKind | None:
    if namespace_kind is BackendKind.MLX:
        return BackendKind.METAL
    if namespace_kind is BackendKind.CUPY:
        return BackendKind.CUDA
    return None


def _resolve_backend_request(
    xp: _ArrayNamespace,
    backend: BackendKind | str | None,
) -> _ResolvedBackendRequest:
    """Resolve namespace and acceleration policy without probing hardware."""
    namespace_kind = _namespace_kind(xp)
    if isinstance(backend, Backend):
        raise TypeError("backend must be a backend name or BackendKind; pass backend.kind, or omit backend to infer")

    requested = BackendKind.AUTO if backend is None else _coerce_backend_kind(backend)
    if requested is BackendKind.AUTO:
        return _ResolvedBackendRequest(
            namespace_kind=namespace_kind,
            kernel_kind=_inferred_kernel_kind(namespace_kind),
            kernel_policy=_KernelPolicy.PREFER,
        )

    expected = _expected_namespace_kind(requested)
    if namespace_kind is not expected:
        namespace_name = getattr(xp, "__name__", type(xp).__name__)
        raise ValueError(f"Backend {requested.value!r} does not match the input array namespace {namespace_name!r}")

    if requested in (BackendKind.METAL, BackendKind.CUDA):
        return _ResolvedBackendRequest(namespace_kind, requested, _KernelPolicy.REQUIRE)
    return _ResolvedBackendRequest(namespace_kind, None, _KernelPolicy.DISABLE)


def _common_unsupported_reasons(request: _SimusSpectrumRequest) -> list[str]:
    reasons = []
    if request.execution is not None:
        reasons.append("an execution workspace budget")
    if request.full_frequency_directivity:
        reasons.append("full_frequency_directivity=True")
    if request.params.baffle != BaffleType.SOFT:
        reasons.append(f"baffle={request.params.baffle!r} (only SOFT is supported)")
    return reasons


def _metal_unsupported_reason(request: _SimusSpectrumRequest) -> str | None:
    from fast_simus.kernels.metal_simus import metal_simus_unsupported_reason

    return metal_simus_unsupported_reason(request.params.n_elements)


def _cuda_unsupported_reason(request: _SimusSpectrumRequest) -> str | None:
    from fast_simus.kernels.cuda_simus import cuda_simus_unsupported_reason

    return cuda_simus_unsupported_reason(request.params.n_elements, request.plan.n_sub)


def _run_metal(request: _SimusSpectrumRequest) -> Array:
    import mlx.core as mx

    from fast_simus.kernels.metal_simus import simus_metal

    return cast(
        Array,
        simus_metal(
            scatterers=cast(mx.array, request.scatterers),
            rc=cast(mx.array, request.rc),
            params=request.params,
            plan=request.plan,
            medium=request.medium,
            delays_clean=cast(mx.array, request.delays_clean),
            tx_apodization=cast(mx.array, request.tx_apodization),
        ),
    )


def _run_cuda(request: _SimusSpectrumRequest) -> Array:
    from fast_simus.kernels.cuda_simus import simus_cuda

    return cast(
        Array,
        simus_cuda(
            scatterers=request.scatterers,
            rc=request.rc,
            params=request.params,
            plan=request.plan,
            medium=request.medium,
            delays_clean=request.delays_clean,
            tx_apodization=request.tx_apodization,
        ),
    )


_AdapterReason = Callable[[_SimusSpectrumRequest], str | None]
_AdapterRun = Callable[[_SimusSpectrumRequest], Array]
_ADAPTERS: dict[BackendKind, tuple[_AdapterReason, _AdapterRun]] = {
    BackendKind.METAL: (_metal_unsupported_reason, _run_metal),
    BackendKind.CUDA: (_cuda_unsupported_reason, _run_cuda),
}


def _prepare_simus_sweep(request: _SimusSpectrumRequest) -> dict:
    """Prepare portable geometry and phase arrays for the frequency sweep."""
    xp = request.xp
    params = request.params
    plan = request.plan
    medium = request.medium
    freq_start, freq_step = _canonical_frequency_grid(params.freq_center, plan.n_freq_full, plan.freq_idx_start)
    transfer = _prepare_strip_transfer(
        request.scatterers,
        request.delays_clean,
        request.tx_apodization,
        _TransferPlan(plan.selected_freqs, plan.n_sub, plan.seg_length, freq_start, freq_step),
        params,
        medium,
        full_frequency_directivity=request.full_frequency_directivity,
        xp=xp,
    )
    return dict(
        phase_init=transfer.phase,
        phase_step=transfer.phase_step,
        delay_apod_init=transfer.delay_apod,
        delay_apod_step=transfer.delay_apod_step,
        is_out=transfer.is_out,
        wavenumbers=transfer.wavenumbers,
        pulse_spect=plan.pulse_spectrum,
        probe_spect=plan.probe_spectrum,
        seg_length=plan.seg_length,
        sin_theta=transfer.sin_theta,
        full_frequency_directivity=request.full_frequency_directivity,
    )


def _run_portable_block(request: _SimusSpectrumRequest) -> Array:
    sweep = _prepare_simus_sweep(request)
    if is_jax_namespace(cast(ModuleType, request.xp)):
        from fast_simus._simus_strategies import _simus_freq_outer_scan

        return _simus_freq_outer_scan(rc=request.rc, xp=request.xp, **sweep)

    from fast_simus._simus_strategies import _simus_freq_outer_python

    return _simus_freq_outer_python(rc=request.rc, xp=request.xp, **sweep)


def require_portable_backend(xp, backend, feature):
    """Validate explicit namespace requests and reject required unsupported kernels."""
    resolved = _resolve_backend_request(xp, backend)
    if resolved.kernel_policy is _KernelPolicy.REQUIRE and resolved.kernel_kind is not None:
        raise NotImplementedError(f"Backend {resolved.kernel_kind.value!r} does not support {feature}")


def _run_portable(request: _SimusSpectrumRequest) -> Array:
    if request.execution is None:
        return _run_portable_block(request)
    size = legacy_point_count(request.execution, request.params.n_elements, request.plan.n_sub)
    result = request.xp.zeros(
        (request.plan.selected_freqs.shape[0], request.params.n_elements), dtype=request.plan.pulse_spectrum.dtype
    )
    for start in range(0, request.scatterers.shape[0], size):
        block = replace(
            request, scatterers=request.scatterers[start : start + size], rc=request.rc[start : start + size]
        )
        result = result + _run_portable_block(block)
    return result


def compute_simus_spectrum(
    request: _SimusSpectrumRequest,
) -> Complex[Array, "n_frequencies n_elements"]:
    """Compute a selected-frequency spectrum using the requested policy."""
    resolved = _resolve_backend_request(request.xp, request.backend)
    if resolved.kernel_kind is None:
        return _run_portable(request)

    reason_fn, run = _ADAPTERS[resolved.kernel_kind]
    reasons = _common_unsupported_reasons(request)
    adapter_reason = reason_fn(request)
    if adapter_reason is not None:
        reasons.append(adapter_reason)

    if reasons:
        if resolved.kernel_policy is _KernelPolicy.REQUIRE:
            detail = ", ".join(reasons)
            raise NotImplementedError(f"Backend {resolved.kernel_kind.value!r} does not support: {detail}")
        return _run_portable(request)

    return run(request)
