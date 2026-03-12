"""Backend-specific loop drivers for pfield frequency sweep.

These functions implement Layer 3 (loop driver) of the three-layer pfield
architecture. Each wraps the same _freq_step_body() from pfield.py but
uses a backend-specific iteration mechanism:

- _freq_outer_scan: JAX lax.scan for O(1) compilation cost
- _freq_outer_mlx: MLX with per-iteration eval to bound graph memory

See design doc: thoughts/shared/plans/2026-03-07-backend-strategy-architecture.md
"""

from __future__ import annotations

from math import pi

import array_api_extra as xpx
from jaxtyping import Bool, Complex, Float

from fast_simus.pfield import _freq_step_body
from fast_simus.utils._array_api import Array, _ArrayNamespace


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
    import jax
    import jax.numpy as jnp

    spectra = pulse_spect * probe_spect

    if full_frequency_directivity:

        def scan_fn(carry, k):
            phase, rp = carry
            sinc_arg = wavenumbers[k] * seg_length / 2.0 * sin_theta / pi
            directivity_k = xpx.sinc(sinc_arg, xp=xp)
            phase, rp_k = _freq_step_body(
                phase, phase_decay_step, spectra[k], n_sub, xp, directivity_k=directivity_k
            )
            rp = rp + jnp.where(is_out, 0.0, rp_k)
            return (phase, rp), None

        (_, rp), _ = jax.lax.scan(
            scan_fn, (phase_decay_init, jnp.zeros(phase_decay_init.shape[:-2])), jnp.arange(spectra.shape[0])
        )
    else:

        def scan_fn_no_dir(carry, spectrum_k):
            phase, rp = carry
            phase, rp_k = _freq_step_body(phase, phase_decay_step, spectrum_k, n_sub, xp)
            rp = rp + jnp.where(is_out, 0.0, rp_k)
            return (phase, rp), None

        (_, rp), _ = jax.lax.scan(
            scan_fn_no_dir, (phase_decay_init, jnp.zeros(phase_decay_init.shape[:-2])), spectra
        )

    return rp


def _freq_outer_mlx(
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
    """MLX loop driver with per-iteration computation trigger.

    Identical to _pfield_freq_outer except it triggers MLX computation
    after each iteration. Without this, MLX builds an n_freq-deep lazy
    computation graph consuming unbounded memory before executing anything.
    """
    import mlx.core as mx

    # MLX array evaluation trigger -- forces lazy graph execution
    _trigger_compute = getattr(mx, "eval")

    n_freq = wavenumbers.shape[0]
    spectra = pulse_spect * probe_spect

    phase = phase_decay_init
    rp = xp.zeros(phase_decay_init.shape[:-2])
    zero = xp.asarray(0.0)

    for k in range(n_freq):
        directivity_k = None
        if full_frequency_directivity:
            sinc_arg = wavenumbers[k] * seg_length / 2.0 * sin_theta / pi
            directivity_k = xpx.sinc(sinc_arg, xp=xp)

        phase, rp_k = _freq_step_body(
            phase, phase_decay_step, spectra[k], n_sub, xp, directivity_k=directivity_k
        )
        rp = rp + xp.where(is_out, zero, rp_k)
        _trigger_compute(rp, phase)

    return rp
