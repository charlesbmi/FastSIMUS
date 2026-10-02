"""Shared Array API contractions for first-order scattering."""

from __future__ import annotations

from jaxtyping import Complex, Float

from fast_simus._contractions import _receive_spectrum
from fast_simus.utils._array_api import Array


def _scatter_and_sum(
    incident: Complex[Array, " n_scatterers"],
    reflection_coefficients: Float[Array, " n_scatterers"],
    receive_transfer: Complex[Array, "n_scatterers n_observers"],
) -> Complex[Array, " n_observers"]:
    """Weight incident pressure by reflectivity and sum it at observers."""
    return _receive_spectrum(receive_transfer, reflection_coefficients * incident)
