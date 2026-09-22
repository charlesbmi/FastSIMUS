"""Backend policy for SIMUS execution."""

from __future__ import annotations

from enum import StrEnum
from types import ModuleType
from typing import cast

from array_api_compat import is_cupy_namespace, is_jax_namespace, is_numpy_namespace

from fast_simus.backends._selection import Backend, BackendKind, backend_kind_from_namespace, get_backend
from fast_simus.transducer_params import BaffleType, TransducerParams
from fast_simus.utils._array_api import ArrayNamespace, is_mlx_namespace


class SimusImplementation(StrEnum):
    """Private implementation selected after public backend resolution."""

    PYTHON = "python"
    SCAN = "scan"
    METAL = "metal"
    CUDA = "cuda"


def _namespace_matches(kind: BackendKind, xp: ArrayNamespace) -> bool:
    if kind in (BackendKind.METAL, BackendKind.MLX):
        return is_mlx_namespace(xp)
    if kind in (BackendKind.CUDA, BackendKind.CUPY):
        return is_cupy_namespace(xp)
    if kind is BackendKind.JAX:
        return is_jax_namespace(cast(ModuleType, xp))
    if kind is BackendKind.NUMPY:
        return is_numpy_namespace(xp)
    return False


def _portable_implementation(xp: ArrayNamespace) -> SimusImplementation:
    if is_jax_namespace(cast(ModuleType, xp)):
        return SimusImplementation.SCAN
    return SimusImplementation.PYTHON


def _unsupported_reasons(
    kind: BackendKind,
    params: TransducerParams,
    *,
    n_sub: int,
    full_frequency_directivity: bool,
) -> list[str]:
    reasons = []
    if full_frequency_directivity:
        reasons.append("full_frequency_directivity=True")
    if params.baffle != BaffleType.SOFT:
        reasons.append(f"baffle={params.baffle!r} (only SOFT is supported)")

    if kind is BackendKind.METAL and params.n_elements * 2 > 1024:
        reasons.append(f"n_elements={params.n_elements} exceeds the Metal receive-kernel limit")
    if kind is BackendKind.CUDA:
        from fast_simus.kernels._cuda_capabilities import cuda_shared_memory_unsupported_reason

        shared_memory_reason = cuda_shared_memory_unsupported_reason(params.n_elements, n_sub)
        if shared_memory_reason is not None:
            reasons.append(shared_memory_reason)
    return reasons


def _requested_backend(
    backend: Backend | BackendKind | str | None,
    xp: ArrayNamespace,
) -> tuple[BackendKind | None, bool]:
    """Return concrete kind and whether acceleration must be honored."""
    if backend is None:
        return backend_kind_from_namespace(xp), False

    if isinstance(backend, Backend):
        kind = backend.kind
        strict = backend._strict_acceleration
    else:
        try:
            requested = BackendKind(backend)
        except ValueError as error:
            choices = ", ".join(item.value for item in BackendKind)
            raise ValueError(f"Unknown backend {backend!r}; expected one of: {choices}") from error
        if requested is BackendKind.AUTO:
            return backend_kind_from_namespace(xp), False
        resolved = get_backend(requested)
        kind = resolved.kind
        strict = True

    if not _namespace_matches(kind, xp):
        raise ValueError(f"Backend {kind.value!r} does not match the input array namespace {xp.__name__!r}")
    return kind, strict


def select_simus_implementation(
    xp: ArrayNamespace,
    backend: Backend | BackendKind | str | None,
    params: TransducerParams,
    *,
    n_sub: int,
    full_frequency_directivity: bool,
) -> SimusImplementation:
    """Resolve custom-kernel or portable execution without moving arrays."""
    kind, strict = _requested_backend(backend, xp)
    if kind not in (BackendKind.METAL, BackendKind.CUDA):
        return _portable_implementation(xp)

    unsupported = _unsupported_reasons(
        kind,
        params,
        n_sub=n_sub,
        full_frequency_directivity=full_frequency_directivity,
    )
    if unsupported:
        if strict:
            raise NotImplementedError(f"Backend {kind.value!r} does not support: {', '.join(unsupported)}")
        return _portable_implementation(xp)

    return SimusImplementation(kind.value)
