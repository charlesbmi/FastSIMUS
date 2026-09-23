"""Deep execution seam for the SIMUS selected-frequency spectrum."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import inf, pi
from types import ModuleType
from typing import TYPE_CHECKING, cast

import array_api_extra as xpx
from array_api_compat import is_jax_namespace
from jaxtyping import Complex, Float

from fast_simus._pfield_math import (
    _canonical_frequency_grid,
    _distances_and_angles,
    _init_exponentials,
    _obliquity_factor,
    _subelement_centroids,
)
from fast_simus.backends._selection import Backend, BackendKind, _coerce_backend_kind, _namespace_kind
from fast_simus.medium_params import MediumParams
from fast_simus.transducer_params import BaffleType, TransducerParams
from fast_simus.utils._array_api import Array, _ArrayNamespace
from fast_simus.utils.geometry import element_positions

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
    element_pos, theta_elements, apex_offset = element_positions(params.n_elements, params.pitch, params.radius, xp)
    if theta_elements is None:
        theta_elements = xp.zeros(params.n_elements)

    subelement_offsets = _subelement_centroids(params.element_width, plan.n_sub, theta_elements, xp)
    x = request.scatterers[..., 0]
    z = request.scatterers[..., 1]
    is_out = z < 0
    if params.radius != inf:
        is_out = is_out | ((x**2 + (z + apex_offset) ** 2) <= params.radius**2)

    distances, sin_theta, theta_arr = _distances_and_angles(
        request.scatterers,
        subelement_offsets,
        element_pos,
        theta_elements,
        medium.speed_of_sound,
        params.freq_center,
        xp,
    )
    obliquity_factor = _obliquity_factor(theta_arr, params.baffle, xp)
    freq_start, freq_step = _canonical_frequency_grid(params.freq_center, plan.n_freq_full, plan.freq_idx_start)
    phase_init, phase_step = _init_exponentials(
        freq_start,
        medium.speed_of_sound,
        medium.attenuation,
        distances,
        obliquity_factor,
        freq_step,
        xp,
    )

    if not request.full_frequency_directivity:
        center_wavenumber = 2.0 * pi * params.freq_center / medium.speed_of_sound
        sinc_arg = xp.asarray(center_wavenumber * plan.seg_length / 2.0) * sin_theta / pi
        phase_init = phase_init * xpx.sinc(sinc_arg, xp=xp)

    delay_apod_init = xp.exp(xp.asarray(1j * 2.0 * pi) * freq_start * request.delays_clean) * request.tx_apodization
    delay_apod_step = xp.exp(xp.asarray(1j * 2.0 * pi) * freq_step * request.delays_clean)
    wavenumbers = xp.asarray(2.0 * pi) * plan.selected_freqs / medium.speed_of_sound

    return {
        "phase_init": phase_init,
        "phase_step": phase_step,
        "delay_apod_init": delay_apod_init,
        "delay_apod_step": delay_apod_step,
        "is_out": is_out,
        "wavenumbers": wavenumbers,
        "pulse_spect": plan.pulse_spectrum,
        "probe_spect": plan.probe_spectrum,
        "seg_length": plan.seg_length,
        "sin_theta": sin_theta,
        "full_frequency_directivity": request.full_frequency_directivity,
    }


def _run_portable(request: _SimusSpectrumRequest) -> Array:
    sweep = _prepare_simus_sweep(request)
    if is_jax_namespace(cast(ModuleType, request.xp)):
        from fast_simus._simus_strategies import _simus_freq_outer_scan

        return _simus_freq_outer_scan(rc=request.rc, xp=request.xp, **sweep)

    from fast_simus._simus_strategies import _simus_freq_outer_python

    return _simus_freq_outer_python(rc=request.rc, xp=request.xp, **sweep)


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
