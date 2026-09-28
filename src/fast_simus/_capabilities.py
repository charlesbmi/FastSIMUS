"""Capability checks performed before native imports or portable preparation."""

from types import ModuleType
from typing import cast

from array_api_compat import is_jax_namespace

from fast_simus.transducer_params import BaffleType
from fast_simus.utils._array_api import _ArrayNamespace, is_cupy_namespace, is_mlx_namespace


def _unsupported(strategy: str, xp: _ArrayNamespace, baffle: BaffleType | float, full_directivity: bool) -> list[str]:
    """Describe unsupported physics and array backends for an execution path."""
    reasons = []
    if strategy in ("metal", "cuda"):
        if baffle != BaffleType.SOFT:
            reasons.append(f"baffle={baffle!r} (only SOFT supported)")
        if full_directivity:
            reasons.append("full_frequency_directivity=True")
        if strategy == "metal" and not is_mlx_namespace(xp):
            reasons.append("requires MLX arrays")
        if strategy == "cuda" and not is_cupy_namespace(xp):
            reasons.append("requires CuPy arrays")
    if strategy == "scan" and not is_jax_namespace(cast(ModuleType, xp)):
        reasons.append("requires JAX arrays")
    return reasons


def _require_strategy(strategy: str, xp: _ArrayNamespace, baffle: BaffleType | float, full_directivity: bool) -> None:
    """Reject explicit unsupported requests before allocating or launching."""
    reasons = _unsupported(strategy, xp, baffle, full_directivity)
    if reasons:
        raise NotImplementedError(
            f"{strategy} strategy does not support: {', '.join(reasons)}. Use strategy=None for auto-selection."
        )
