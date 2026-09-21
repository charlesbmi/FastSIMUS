"""Backend-neutral JIT wrapper for simus benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_simus import jit
from fast_simus.simus import simus_compute

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_simus.simus import SimusPlan
    from fast_simus.transducer_params import TransducerParams
    from fast_simus.utils._array_api import _ArrayNamespace


def make_simus_compute(plan: SimusPlan, params: TransducerParams, xp: _ArrayNamespace) -> Callable:
    """Return (scatterers, rc, delays) -> SimusResult with backend JIT when available."""
    return jit(lambda scat, rc, dl: simus_compute(scat, rc, dl, plan, params), xp=xp)
