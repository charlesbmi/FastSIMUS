"""Array-only reductions for pressure and reciprocal pulse-echo propagation."""

from jaxtyping import Complex, Float

from fast_simus.utils._array_api import Array, _ArrayNamespace


def _pressure_at_freq(
    phase: Complex[Array, " *grid n_sources"],
    spectrum_k: complex | Array,
    xp: _ArrayNamespace,
    *,
    directivity_k: Float[Array, " *grid n_sources"] | None = None,
) -> Complex[Array, " *grid"]:
    """Contract source phases and apply the spectrum weight at one frequency."""
    phase_weighted = phase if directivity_k is None else phase * directivity_k
    return spectrum_k * xp.sum(phase_weighted, axis=-1)


def _element_response(phase: Array, directivity: Array | None, xp: _ArrayNamespace) -> Array:
    """Average strip subdivisions while preserving the receive element axis."""
    return xp.mean(phase if directivity is None else phase * directivity, axis=-1)


def _transmit_pressure(
    response: Array, excitation: Array, spectrum: complex | Array, is_out: Array | None, xp: _ArrayNamespace
) -> Array:
    """Contract elements into pressure at each scattering point."""
    pressure = spectrum * (response @ excitation[..., None])[..., 0]
    return pressure if is_out is None else xp.where(is_out, xp.asarray(0.0 + 0j), pressure)


def _receive_spectrum(response: Array, weighted_pressure: Array) -> Array:
    """Contract scattering points into receive channels using reciprocity.

    The same transfer is used on both legs; it must not be conjugated.
    """
    return weighted_pressure @ response
