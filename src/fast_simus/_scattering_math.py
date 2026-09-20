"""Shared Array API contractions for first-order scattering."""

from __future__ import annotations

from jaxtyping import Complex, Float

from fast_simus.utils._array_api import Array, _ArrayNamespace


def _scatter_and_sum(
    incident: Complex[Array, " n_scatterers"],
    reflection_coefficients: Float[Array, " n_scatterers"],
    receive_transfer: Complex[Array, "n_scatterers n_observers"],
    xp: _ArrayNamespace,
) -> Complex[Array, " n_observers"]:
    """Weight incident pressure by reflectivity and sum it at observers."""
    return (reflection_coefficients * incident) @ receive_transfer
