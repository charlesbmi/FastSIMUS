"""Shared helpers for simus benchmarks (JIT wrap, backend sync)."""

from __future__ import annotations

import contextlib
from types import ModuleType
from typing import TYPE_CHECKING, cast

from array_api_compat import is_jax_namespace

from fast_simus.simus import SimusResult, simus_compute

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_simus.simus import SimusPlan
    from fast_simus.transducer_params import TransducerParams
    from fast_simus.utils._array_api import _ArrayNamespace

_jax = None
_eqx = None
with contextlib.suppress(ImportError):
    import equinox as _eqx
    import jax as _jax

_mx = None
with contextlib.suppress(ImportError):
    import mlx.core as _mx


def make_simus_compute(plan: SimusPlan, params: TransducerParams, xp: _ArrayNamespace) -> Callable:
    """Return (scatterers, rc, delays) -> SimusResult with JAX JIT when applicable."""
    if is_jax_namespace(cast(ModuleType, xp)):
        assert _eqx is not None
        jitted = _eqx.filter_jit(simus_compute)
        return lambda scat, rc, dl: jitted(scat, rc, dl, plan, params)

    return lambda scat, rc, dl: simus_compute(scat, rc, dl, plan, params)


def sync_simus_result(result: SimusResult, xp: _ArrayNamespace) -> None:
    """Block until RF data is ready for async backends."""
    if is_jax_namespace(cast(ModuleType, xp)):
        assert _jax is not None
        _jax.block_until_ready(result.rf)
        return
    if _mx is not None and xp is _mx:
        _mx.eval(result.rf)
