"""Loop drivers for pfield frequency sweep (Layer 3).

Each driver iterates the same _freq_step_body() using a different mechanism:

- _pfield_freq_vectorized: tensor broadcast (NumPy/CuPy small grids, MLX)
- _freq_outer_scan: JAX lax.scan for O(1) compilation cost
"""

from __future__ import annotations

from math import pi

import array_api_extra as xpx
from beartype import beartype as typechecker
from jaxtyping import Bool, Complex, Float, jaxtyped

from fast_simus.utils._array_api import Array, _ArrayNamespace


def _freq_step_body(
    phase: Complex[Array, " *grid n_sources"],
    phase_step: Complex[Array, " *grid n_sources"],
    spectrum_k: complex | Array,
    xp: _ArrayNamespace,
    *,
    directivity_k: Float[Array, " *grid n_sources"] | None = None,
) -> tuple[Complex[Array, " *grid n_sources"], Float[Array, " *grid"]]:
    """One frequency step: geometric update, source contraction, spectrum weight.

    Single source of truth for per-frequency math. Source points are
    already flattened (n_elements * n_sub) with 1/n_sub absorbed.

    Args:
        phase: Current phase state (geometric progression).
        phase_step: Per-step multiplier for geometric progression.
        spectrum_k: Combined pulse*probe spectrum weight for this frequency.
        xp: Array namespace.
        directivity_k: Per-source directivity for this frequency (optional).

    Returns:
        Tuple of (updated_phase, rp_k) where rp_k = |P_k|^2 at this frequency.
    """
    if directivity_k is not None:
        phase_weighted = phase * directivity_k
    else:
        phase_weighted = phase
    pressure_k = spectrum_k * xp.sum(phase_weighted, axis=-1)
    phase = phase * phase_step
    return phase, xp.real(pressure_k * xp.conj(pressure_k))


@jaxtyped(typechecker=typechecker)
def _pfield_freq_vectorized(
    phase_decay_init: Complex[Array, " *grid n_sources"],
    phase_decay_step: Complex[Array, " *grid n_sources"],
    is_out: Bool[Array, " *grid"],
    wavenumbers: Float[Array, " n_freq"],
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    seg_length: float,
    sin_theta: Float[Array, " *grid n_sources"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> Float[Array, " *grid"]:
    """Vectorized frequency sweep: broadcast all frequencies at once.

    Uses the geometric progression: phase_decay[k] = init * step^k.
    Source points are pre-flattened with 1/n_sub absorbed.

    Best for small grids where the (*grid, n_sources, n_freq) tensor fits
    in memory. For large grids, use an iterative driver instead.
    """
    n_freq = wavenumbers.shape[0]
    exponents = xp.arange(n_freq, dtype=wavenumbers.dtype)

    # (*grid, n_sources, n_freq) via geometric progression
    phase_k = phase_decay_init[..., None] * phase_decay_step[..., None] ** exponents

    if full_frequency_directivity:
        sinc_arg = wavenumbers * seg_length / 2.0 * sin_theta[..., None] / pi
        phase_k = xpx.sinc(sinc_arg, xp=xp) * phase_k

    # Contract over sources: (*grid, n_sources, n_freq) -> (*grid, n_freq)
    pressure_all = xp.sum(phase_k, axis=-2)

    pressure_all = pulse_spect * probe_spect * pressure_all

    pressure_all = xp.where(is_out[..., None], xp.asarray(0.0 + 0j), pressure_all)

    return xp.sum(xp.abs(pressure_all) ** 2, axis=-1)


def _freq_outer_scan(
    phase_decay_init: Complex[Array, " *grid n_sources"],
    phase_decay_step: Complex[Array, " *grid n_sources"],
    is_out: Bool[Array, " *grid"],
    wavenumbers: Float[Array, " n_freq"],
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    seg_length: float,
    sin_theta: Float[Array, " *grid n_sources"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> Float[Array, " *grid"]:
    """JAX lax.scan loop driver for pfield frequency sweep.

    Compiles the frequency loop into XLA's while_loop, giving constant
    compilation cost regardless of n_freq and enabling automatic
    differentiation through the loop.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    spectra = pulse_spect * probe_spect
    zero = xp.asarray(0.0)

    if full_frequency_directivity:

        def scan_fn(carry, k):
            phase, rp = carry
            sinc_arg = wavenumbers[k] * seg_length / 2.0 * sin_theta / pi
            directivity_k = xpx.sinc(sinc_arg, xp=xp)
            phase, rp_k = _freq_step_body(phase, phase_decay_step, spectra[k], xp, directivity_k=directivity_k)
            rp = rp + xp.where(is_out, zero, rp_k)
            return (phase, rp), None

        (_, rp), _ = jax.lax.scan(
            scan_fn, (phase_decay_init, jnp.zeros(phase_decay_init.shape[:-1])), jnp.arange(spectra.shape[0])
        )
    else:

        def scan_fn_no_dir(carry, spectrum_k):
            phase, rp = carry
            phase, rp_k = _freq_step_body(phase, phase_decay_step, spectrum_k, xp)
            rp = rp + xp.where(is_out, zero, rp_k)
            return (phase, rp), None

        (_, rp), _ = jax.lax.scan(scan_fn_no_dir, (phase_decay_init, jnp.zeros(phase_decay_init.shape[:-1])), spectra)

    return rp
