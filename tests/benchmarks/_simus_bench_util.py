"""Backend-neutral JIT wrapper for simus benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_simus import BackendKind, jit
from fast_simus.simus import simus_compute
from fast_simus.utils._array_api import is_mlx_namespace

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_simus.simus import SimusPlan
    from fast_simus.transducer_params import TransducerParams
    from fast_simus.utils._array_api import _ArrayNamespace


def make_simus_compute(
    plan: SimusPlan,
    params: TransducerParams,
    xp: _ArrayNamespace,
    *,
    backend: BackendKind | str | None = None,
) -> Callable:
    """Return (scatterers, rc, delays) -> SimusResult with backend JIT when available."""

    def compute(scat, rc, dl):
        return simus_compute(scat, rc, dl, plan, params, backend=backend)

    if is_mlx_namespace(xp) and backend in (None, BackendKind.AUTO, BackendKind.METAL, "auto", "metal"):
        return compute
    return jit(compute, xp=xp)
