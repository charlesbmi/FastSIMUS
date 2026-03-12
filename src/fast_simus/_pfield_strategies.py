"""Backend-specific loop drivers for pfield frequency sweep.

These functions implement Layer 3 (loop driver) of the three-layer pfield
architecture. Each wraps the same _freq_step_body() but uses a
backend-specific iteration mechanism:

- _freq_outer_scan: JAX lax.scan for O(1) compilation cost
"""

from __future__ import annotations

from math import pi

import array_api_extra as xpx
from jaxtyping import Bool, Complex, Float

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
