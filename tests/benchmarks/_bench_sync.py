"""Block until JAX / MLX arrays are materialized (wall-clock benchmarks)."""

from __future__ import annotations

import contextlib
from types import ModuleType
from typing import TYPE_CHECKING, cast

from array_api_compat import is_jax_namespace

from fast_simus.utils._array_api import Array, is_mlx_namespace

if TYPE_CHECKING:
    from fast_simus.utils._array_api import _ArrayNamespace

_jax = None
with contextlib.suppress(ImportError):
    import jax as _jax

_mx_eval = None
with contextlib.suppress(ImportError):
    import mlx.core as _mlx_core

    _mx_eval = _mlx_core.eval


def sync_benchmark_array(value: Array, xp: _ArrayNamespace) -> None:
    """Ensure *value* is evaluated on async backends before timing ends."""
    if is_jax_namespace(cast(ModuleType, xp)):
        assert _jax is not None
        _jax.block_until_ready(value)
        return
    if _mx_eval is not None and is_mlx_namespace(xp):
        _mx_eval(value)
