"""Backend-neutral JIT compilation for Array API callables."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import ParamSpec, TypeVar, cast

from array_api_compat import is_jax_namespace

from fast_simus.utils._array_api import ArrayNamespace, is_mlx_namespace

_P = ParamSpec("_P")
_R = TypeVar("_R")


def jit(function: Callable[_P, _R], *, xp: ArrayNamespace) -> Callable[_P, _R]:
    """Compile an Array API callable when its backend provides a JIT.

    JAX callables use :func:`jax.jit`, MLX callables use
    :func:`mlx.core.compile`, and other backends keep eager execution. Bind
    plans, parameter models, and other static configuration in a closure so
    the returned callable accepts arrays only::

        compute = jit(
            lambda positions, delays: pfield_compute(
                positions, delays, plan, params
            ),
            xp=xp,
        )

    The backend remains an explicit choice so compilation never silently
    differs based on which arguments happen to be passed first.

    Args:
        function: Callable to compile. For JAX and MLX, its runtime arguments
            should be arrays or containers of arrays.
        xp: Array API namespace used by ``function``.

    Returns:
        A compiled callable for JAX or MLX, otherwise ``function`` unchanged.
    """
    if is_jax_namespace(cast(ModuleType, xp)):
        import jax  # noqa: PLC0415

        return cast("Callable[_P, _R]", jax.jit(function))

    if is_mlx_namespace(xp):
        import mlx.core as mx  # noqa: PLC0415

        return mx.compile(function)

    return function
