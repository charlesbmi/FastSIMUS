"""Translate legacy public plans into the portable core's setup inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float

from fast_simus._transfer import _TransferPlan
from fast_simus.utils._array_api import Array, _ArrayNamespace

if TYPE_CHECKING:
    from fast_simus.pfield import PfieldPlan
    from fast_simus.simus import SimusPlan


def _transfer_plan(plan: PfieldPlan | SimusPlan, freq_center: float) -> _TransferPlan:
    """Adapt either public tuple without changing its layout or allocating arrays."""
    step = 2 * freq_center / (plan.n_freq_full - 1)
    return _TransferPlan(plan.selected_freqs, plan.n_sub, plan.seg_length, plan.freq_idx_start * step, step)


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


def _validate_apodization(apodization, delays):
    """Check eager public excitation input before entering traceable compute."""
    if apodization is None:
        return
    from fast_simus.aperture import _same_arrays  # noqa: PLC0415

    xp = _same_arrays(delays, apodization)
    if apodization.shape != delays.shape or not bool(xp.all(xp.isfinite(apodization))):
        raise ValueError("Apodization must be finite and match delays")
