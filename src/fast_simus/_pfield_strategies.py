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
    phase: Complex[Array, "*grid n_elements n_sub"],
    phase_step: Complex[Array, "*grid n_elements n_sub"],
    spectrum_k: complex | Array,
    n_sub: int,
    xp: _ArrayNamespace,
    *,
    directivity_k: Float[Array, "*grid n_elements n_sub"] | None = None,
) -> tuple[Complex[Array, "*grid n_elements n_sub"], Float[Array, " *grid"]]:
    """One frequency step: geometric update, element contraction, spectrum weight.

    This is the single source of truth for per-frequency math in the
    frequency-outer loop architecture. All loop drivers (Python for-loop,
    JAX lax.scan) call this function.

    Args:
        phase: Current phase state (geometric progression).
        phase_step: Per-step multiplier for geometric progression.
        spectrum_k: Combined pulse*probe spectrum weight for this frequency.
        n_sub: Number of sub-elements per element.
        xp: Array namespace.
        directivity_k: Per-element directivity for this frequency (optional,
            used when full_frequency_directivity=True).

    Returns:
        Tuple of (updated_phase, rp_k) where rp_k = |P_k|^2 at this frequency.
    """
    if directivity_k is not None:
        phase_weighted = phase * directivity_k
    else:
        phase_weighted = phase
    pressure_k = spectrum_k * xp.sum(xp.mean(phase_weighted, axis=-1), axis=-1)
    phase = phase * phase_step
    return phase, xp.real(pressure_k * xp.conj(pressure_k))


@jaxtyped(typechecker=typechecker)
def _pfield_freq_vectorized(
    phase_decay_init: Complex[Array, "*grid n_elements n_sub"],
    phase_decay_step: Complex[Array, "*grid n_elements n_sub"],
    is_out: Bool[Array, " *grid"],
    wavenumbers: Float[Array, " n_freq"],
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    n_sub: int,
    seg_length: float,
    sin_theta: Float[Array, "*grid n_elements n_sub"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> Float[Array, " *grid"]:
    """Vectorized frequency sweep: broadcast all frequencies at once.

    Uses the geometric progression: phase_decay[k] = init * step^k.
    The sub-element loop (n_sub iterations) avoids materializing the full
    (*grid, n_elements, n_sub, n_freq) tensor.

    Best for small grids where the (*grid, n_elements, n_freq) tensor fits
    in memory. For large grids, use an iterative driver instead.
    """
    n_freq = wavenumbers.shape[0]
    exponents = xp.arange(n_freq, dtype=wavenumbers.dtype)

    pressure_all = xp.asarray(0.0 + 0j)

    for i in range(n_sub):
        phase_k = phase_decay_init[..., i, None] * phase_decay_step[..., i, None] ** exponents

        if full_frequency_directivity:
            sinc_arg = wavenumbers * seg_length / 2.0 * sin_theta[..., i, None] / pi
            phase_k = xpx.sinc(sinc_arg, xp=xp) * phase_k

        pressure_all = pressure_all + xp.sum(phase_k, axis=-2)

    pressure_all = pressure_all / n_sub

    pressure_all = pulse_spect * probe_spect * pressure_all

    pressure_all = xp.where(is_out[..., None], xp.asarray(0.0 + 0j), pressure_all)

    return xp.sum(xp.abs(pressure_all) ** 2, axis=-1)


def _freq_outer_scan(
    phase_decay_init: Complex[Array, "*grid n_elements n_sub"],
    phase_decay_step: Complex[Array, "*grid n_elements n_sub"],
    is_out: Bool[Array, " *grid"],
    wavenumbers: Float[Array, " n_freq"],
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    n_sub: int,
    seg_length: float,
    sin_theta: Float[Array, "*grid n_elements n_sub"],
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
            phase, rp_k = _freq_step_body(phase, phase_decay_step, spectra[k], n_sub, xp, directivity_k=directivity_k)
            rp = rp + xp.where(is_out, zero, rp_k)
            return (phase, rp), None

        (_, rp), _ = jax.lax.scan(
            scan_fn, (phase_decay_init, jnp.zeros(phase_decay_init.shape[:-2])), jnp.arange(spectra.shape[0])
        )
    else:

        def scan_fn_no_dir(carry, spectrum_k):
            phase, rp = carry
            phase, rp_k = _freq_step_body(phase, phase_decay_step, spectrum_k, n_sub, xp)
            rp = rp + xp.where(is_out, zero, rp_k)
            return (phase, rp), None

        (_, rp), _ = jax.lax.scan(scan_fn_no_dir, (phase_decay_init, jnp.zeros(phase_decay_init.shape[:-2])), spectra)

    return rp
