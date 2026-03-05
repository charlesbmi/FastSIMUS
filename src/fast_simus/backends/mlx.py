"""Array API compatibility shim for MLX.

Temporary until array_api_compat gains native MLX support.
Tracking: https://github.com/data-apis/array-api-compat/issues/162
"""

from __future__ import annotations

import contextlib
from typing import Any

import array_api_compat
import array_api_compat.common._helpers as _helpers

_MLX_ARRAY_API_ALIASES: dict[str, str] = {
    "asin": "arcsin",
    "acos": "arccos",
    "atan2": "arctan2",
    "bool": "bool_",
}

_MLX_ISDTYPE_KIND_MAP: dict[str, str] = {
    "bool": "bool_",
    "signed integer": "signedinteger",
    "unsigned integer": "unsignedinteger",
    "integral": "integer",
    "real floating": "floating",
    "complex floating": "complexfloating",
    "numeric": "number",
}


def _make_isdtype(xp: Any) -> Any:
    def isdtype(dtype: Any, kind: Any) -> bool:
        if isinstance(kind, str):
            category = _MLX_ISDTYPE_KIND_MAP.get(kind)
            if category is None:
                msg = f"Unrecognized dtype kind: {kind!r}"
                raise ValueError(msg)
            return bool(xp.issubdtype(dtype, getattr(xp, category)))
        if isinstance(kind, tuple):
            return any(isdtype(dtype, k) for k in kind)
        return dtype == kind

    return isdtype


def _patch_namespace(xp: Any) -> None:
    """Add Array API aliases to mlx.core (idempotent)."""
    for standard_name, mlx_name in _MLX_ARRAY_API_ALIASES.items():
        if not hasattr(xp, standard_name) and hasattr(xp, mlx_name):
            setattr(xp, standard_name, getattr(xp, mlx_name))

    if not hasattr(xp, "isdtype") and hasattr(xp, "issubdtype"):
        xp.isdtype = _make_isdtype(xp)

    if not hasattr(xp, "astype"):

        def _astype(x: Any, dtype: Any, /, *, copy: bool = False) -> Any:
            return x.astype(dtype)

        xp.astype = _astype

    if not getattr(xp.asarray, "_fastsimus_wrapped", False):
        _original = xp.asarray

        def _asarray(a: Any, *, dtype: Any = None, **_kwargs: Any) -> Any:
            if dtype is not None:
                return _original(a, dtype=dtype)
            return _original(a)

        _asarray._fastsimus_wrapped = True  # type: ignore[attr-defined]
        xp.asarray = _asarray


def _patch_device(xp: Any) -> None:
    """Patch array_api_compat device() for MLX unified memory."""
    _original = _helpers.device
    if getattr(_original, "_fastsimus_mlx", False):
        return

    def _device_with_mlx(x: Any, /) -> Any:
        if type(x).__module__.startswith("mlx"):
            return xp.default_device()
        return _original(x)

    _device_with_mlx._fastsimus_mlx = True  # type: ignore[attr-defined]

    _helpers.device = _device_with_mlx  # type: ignore[assignment]
    array_api_compat.device = _device_with_mlx  # type: ignore[assignment]

    with contextlib.suppress(ImportError):
        import array_api_extra._lib._utils._compat as _xpx_compat  # noqa: PLC0415  # type: ignore[import-untyped]

        _xpx_compat.device = _device_with_mlx


def ensure_compat(xp: Any) -> None:
    """Apply all MLX compatibility patches (idempotent)."""
    _patch_namespace(xp)
    _patch_device(xp)
