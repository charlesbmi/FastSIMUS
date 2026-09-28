"""Translate legacy public plans into the portable core's setup inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float

from fast_simus._transfer import _TransferPlan
from fast_simus.utils._array_api import Array, _ArrayNamespace

if TYPE_CHECKING:
    from fast_simus.pfield import PfieldPlan
    from fast_simus.simus import SimusPlan


def _transfer_plan(plan: PfieldPlan | SimusPlan) -> _TransferPlan:
    """Adapt either public tuple without changing its layout or allocating arrays."""
    return _TransferPlan(plan.selected_freqs, plan.n_sub, plan.seg_length)


def _clean_transmit_inputs(
    delays: Float[Array, " n_elements"],
    tx_apodization: Float[Array, " n_elements"] | None,
    n_elements: int,
    xp: _ArrayNamespace,
) -> tuple[Float[Array, " n_elements"], Float[Array, " n_elements"]]:
    """Zero disabled elements and replace their NaN delays."""
    if tx_apodization is None:
        tx_apodization = xp.ones(n_elements)
    nan_mask = xp.isnan(delays)
    return (
        xp.where(nan_mask, xp.asarray(0.0), delays),
        xp.where(nan_mask, xp.asarray(0.0), tx_apodization),
    )
