"""Array backend discovery for FastSIMUS public entry points."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType
from typing import cast

from array_api_compat import is_cupy_namespace, is_jax_namespace, is_numpy_namespace

from fast_simus.utils._array_api import ArrayNamespace, is_mlx_namespace


class BackendKind(StrEnum):
    """Selectable numerical backends.

    ``METAL`` and ``CUDA`` use the custom SIMUS kernels. ``MLX`` and
    ``CUPY`` use the portable Array API implementation on the same devices.
    """

    AUTO = "auto"
    METAL = "metal"
    CUDA = "cuda"
    MLX = "mlx"
    CUPY = "cupy"
    JAX = "jax"
    NUMPY = "numpy"


@dataclass(frozen=True)
class Backend:
    """Resolved numerical backend and its Array API namespace."""

    kind: BackendKind
    xp: ArrayNamespace
    label: str


_LABELS = {
    BackendKind.METAL: "Metal (MLX; custom SIMUS when supported)",
    BackendKind.CUDA: "CUDA (CuPy; custom SIMUS when supported)",
    BackendKind.MLX: "MLX (portable Array API SIMUS)",
    BackendKind.CUPY: "CuPy (portable Array API SIMUS)",
    BackendKind.JAX: "JAX",
    BackendKind.NUMPY: "NumPy",
}


def _coerce_backend_kind(kind: BackendKind | str) -> BackendKind:
    try:
        return BackendKind(kind)
    except ValueError as error:
        choices = ", ".join(item.value for item in BackendKind)
        raise ValueError(f"Unknown backend {kind!r}; expected one of: {choices}") from error


def _cupy_namespace() -> ArrayNamespace | None:
    try:
        import cupy as cp
    except ImportError:
        return None
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return None
    except cp.cuda.runtime.CUDARuntimeError:
        return None
    return cast(ArrayNamespace, cp)


def _mlx_namespace() -> ArrayNamespace | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None

    from fast_simus.backends.mlx import ensure_compat

    ensure_compat(mx)
    return cast(ArrayNamespace, mx)


def _jax_namespace() -> ArrayNamespace | None:
    try:
        import jax.numpy as jnp
    except ImportError:
        return None
    return cast(ArrayNamespace, jnp)


def _numpy_namespace() -> ArrayNamespace:
    import array_api_compat.numpy as np

    return cast(ArrayNamespace, np)


def _resolved_backend(kind: BackendKind) -> Backend:
    if kind in (BackendKind.CUDA, BackendKind.CUPY):
        xp = _cupy_namespace()
        requirement = "CuPy and a visible CUDA device"
    elif kind in (BackendKind.METAL, BackendKind.MLX):
        xp = _mlx_namespace()
        requirement = "MLX on Apple Silicon"
    elif kind is BackendKind.JAX:
        xp = _jax_namespace()
        requirement = "JAX"
    elif kind is BackendKind.NUMPY:
        xp = _numpy_namespace()
        requirement = "NumPy"
    else:
        raise AssertionError(f"Cannot resolve backend kind {kind!r}")

    if xp is None:
        raise RuntimeError(f"Backend {kind.value!r} requires {requirement}, but it is not available")
    return Backend(kind=kind, xp=xp, label=_LABELS[kind])


def get_backend(kind: BackendKind | str = BackendKind.AUTO) -> Backend:
    """Return an available backend context.

    Automatic selection prefers CUDA, Metal, JAX, then NumPy. Explicit
    requests are authoritative and raise when their runtime is unavailable.
    """
    requested = _coerce_backend_kind(kind)
    if requested is not BackendKind.AUTO:
        return _resolved_backend(requested)

    for candidate, loader in (
        (BackendKind.CUDA, _cupy_namespace),
        (BackendKind.METAL, _mlx_namespace),
        (BackendKind.JAX, _jax_namespace),
    ):
        xp = loader()
        if xp is not None:
            return Backend(
                kind=candidate,
                xp=xp,
                label=_LABELS[candidate],
            )
    return Backend(
        kind=BackendKind.NUMPY,
        xp=_numpy_namespace(),
        label=_LABELS[BackendKind.NUMPY],
    )


def _namespace_kind(xp: ArrayNamespace) -> BackendKind | None:
    """Return the portable kind represented by an Array API namespace."""
    if is_cupy_namespace(xp):
        return BackendKind.CUPY
    if is_mlx_namespace(xp):
        return BackendKind.MLX
    if is_jax_namespace(cast(ModuleType, xp)):
        return BackendKind.JAX
    if is_numpy_namespace(xp):
        return BackendKind.NUMPY
    return None
