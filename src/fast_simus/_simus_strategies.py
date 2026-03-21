"""Loop drivers for simus frequency sweep (Layer 3).

Each driver iterates _simus_freq_step_body() using a different mechanism:

- _simus_freq_outer_python: Python for-loop (NumPy/CuPy, constant memory)
- _simus_freq_outer_scan: JAX lax.scan for O(1) compilation cost

The simus step body differs from pfield's: instead of accumulating |P_k|^2
per grid point, it computes the full TX->scatter->RX chain and accumulates
complex RF spectrum per element.
"""

from __future__ import annotations

from math import pi

import array_api_extra as xpx
from jaxtyping import Bool, Complex, Float

from fast_simus.utils._array_api import Array, _ArrayNamespace


def _simus_freq_step_body(
    phase: Complex[Array, "n_scat n_elem n_sub"],
    phase_step: Complex[Array, "n_scat n_elem n_sub"],
    delay_apod_phase: Complex[Array, " n_elem"],
    delay_apod_step: Complex[Array, " n_elem"],
    rc: Float[Array, " n_scat"],
    pulse_probe_k: complex | Array,
    probe_k: float | Array,
    is_out: Bool[Array, " n_scat"],
    xp: _ArrayNamespace,
    *,
    directivity_k: Float[Array, "n_scat n_elem n_sub"] | None = None,
) -> tuple[
    Complex[Array, "n_scat n_elem n_sub"],
    Complex[Array, " n_elem"],
    Complex[Array, " n_elem"],
]:
    """One frequency step: TX forward, scatter, RX backprop.

    Args:
        phase: Geometric progression state (n_scat, n_elem, n_sub).
        phase_step: Per-step multiplier for geometric progression.
        delay_apod_phase: Current delay+apodization phase per element.
        delay_apod_step: Per-step delay+apodization multiplier.
        rc: Reflection coefficients per scatterer.
        pulse_probe_k: Combined pulse*probe spectrum weight for this frequency.
        probe_k: Probe-only spectrum weight for RX.
        is_out: Boolean mask for out-of-field scatterers.
        xp: Array namespace.
        directivity_k: Per-source directivity (optional, for full_frequency_directivity).

    Returns:
        Tuple of (updated_phase, updated_delay_apod, spect_k) where
        spect_k is the complex RF spectrum contribution for this frequency,
        shape (n_elements,).
    """
    if directivity_k is not None:
        rp_mono = xp.mean(phase * directivity_k, axis=-1)
    else:
        rp_mono = xp.mean(phase, axis=-1)

    # TX: contract over elements -> pressure at each scatterer
    p_k = pulse_probe_k * (rp_mono @ delay_apod_phase[..., None])[..., 0]
    p_k = xp.where(is_out, xp.asarray(0.0 + 0j), p_k)

    # RX: contract over scatterers -> spectrum per element
    # (rc * p_k)^T @ rp_mono = sum_i(rc_i * p_k_i * rp_mono[i, e])
    weighted = rc * p_k
    spect_k = weighted @ rp_mono
    spect_k = probe_k * spect_k

    phase = phase * phase_step
    delay_apod_phase = delay_apod_phase * delay_apod_step

    return phase, delay_apod_phase, spect_k


def _simus_freq_outer_python(
    phase_init: Complex[Array, "n_scat n_elem n_sub"],
    phase_step: Complex[Array, "n_scat n_elem n_sub"],
    delay_apod_init: Complex[Array, " n_elem"],
    delay_apod_step: Complex[Array, " n_elem"],
    rc: Float[Array, " n_scat"],
    is_out: Bool[Array, " n_scat"],
    wavenumbers: Float[Array, " n_freq"],
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    seg_length: float,
    sin_theta: Float[Array, "n_scat n_elem n_sub"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> Complex[Array, "n_freq n_elem"]:
    """Python for-loop driver: iterates one frequency at a time.

    Accumulates the complex RF spectrum (n_freq, n_elements).
    Peak memory is O(n_scat * n_elem * n_sub), independent of n_freq.
    """
    spectra = pulse_spect * probe_spect
    n_freq = int(wavenumbers.shape[0])
    n_elem = phase_init.shape[1]

    spect_accum = xp.zeros((n_freq, n_elem), dtype=phase_init.dtype)
    phase = phase_init
    delay_apod_phase = delay_apod_init

    if full_frequency_directivity:
        for k in range(n_freq):
            sinc_arg = wavenumbers[k] * seg_length / 2.0 * sin_theta / pi
            directivity_k = xpx.sinc(sinc_arg, xp=xp)
            phase, delay_apod_phase, spect_k = _simus_freq_step_body(
                phase,
                phase_step,
                delay_apod_phase,
                delay_apod_step,
                rc,
                spectra[k],
                probe_spect[k],
                is_out,
                xp,
                directivity_k=directivity_k,
            )
            spect_accum = _set_row(spect_accum, k, spect_k)
    else:
        for k in range(n_freq):
            phase, delay_apod_phase, spect_k = _simus_freq_step_body(
                phase,
                phase_step,
                delay_apod_phase,
                delay_apod_step,
                rc,
                spectra[k],
                probe_spect[k],
                is_out,
                xp,
            )
            spect_accum = _set_row(spect_accum, k, spect_k)

    return spect_accum


def _set_row(
    arr: Complex[Array, "n_freq n_elem"],
    k: int,
    row: Complex[Array, " n_elem"],
) -> Complex[Array, "n_freq n_elem"]:
    """Set row k of arr to row, Array API compatible."""
    return xpx.at(arr)[k, :].set(row)  # type: ignore[attr-defined]


def _simus_freq_outer_scan(
    phase_init: Complex[Array, "n_scat n_elem n_sub"],
    phase_step: Complex[Array, "n_scat n_elem n_sub"],
    delay_apod_init: Complex[Array, " n_elem"],
    delay_apod_step: Complex[Array, " n_elem"],
    rc: Float[Array, " n_scat"],
    is_out: Bool[Array, " n_scat"],
    wavenumbers: Float[Array, " n_freq"],
    pulse_spect: Complex[Array, " n_freq"],
    probe_spect: Float[Array, " n_freq"],
    seg_length: float,
    sin_theta: Float[Array, "n_scat n_elem n_sub"],
    full_frequency_directivity: bool,
    xp: _ArrayNamespace,
) -> Complex[Array, "n_freq n_elem"]:
    """JAX lax.scan driver: scan over frequencies with full tensor carry.

    The carry holds (phase, delay_apod_phase) with shapes
    (n_scat, n_elem, n_sub) and (n_elem,). Each step outputs
    spect_k with shape (n_elem,), stacked by scan into (n_freq, n_elem).
    """
    import jax

    spectra = pulse_spect * probe_spect

    if full_frequency_directivity:

        def scan_fn(carry, xs):
            phase, delay_apod = carry
            spectrum_k, probe_k, wavenum_k = xs
            sinc_arg = wavenum_k * seg_length / 2.0 * sin_theta / pi
            directivity_k = xpx.sinc(sinc_arg, xp=xp)
            rp_mono = xp.mean(phase * directivity_k, axis=-1)
            p_k = spectrum_k * (rp_mono @ delay_apod[..., None])[..., 0]
            p_k = xp.where(is_out, xp.asarray(0.0 + 0j), p_k)
            spect_k = probe_k * (rc * p_k) @ rp_mono
            phase = phase * phase_step
            delay_apod = delay_apod * delay_apod_step
            return (phase, delay_apod), spect_k

        init_carry = (phase_init, delay_apod_init)
        _, spect_all = jax.lax.scan(scan_fn, init_carry, (spectra, probe_spect, wavenumbers))
    else:

        def scan_fn_no_dir(carry, xs):
            phase, delay_apod = carry
            spectrum_k, probe_k = xs
            rp_mono = xp.mean(phase, axis=-1)
            p_k = spectrum_k * (rp_mono @ delay_apod[..., None])[..., 0]
            p_k = xp.where(is_out, xp.asarray(0.0 + 0j), p_k)
            spect_k = probe_k * (rc * p_k) @ rp_mono
            phase = phase * phase_step
            delay_apod = delay_apod * delay_apod_step
            return (phase, delay_apod), spect_k

        init_carry = (phase_init, delay_apod_init)
        _, spect_all = jax.lax.scan(scan_fn_no_dir, init_carry, (spectra, probe_spect))

    return spect_all
