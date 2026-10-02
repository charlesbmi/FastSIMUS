"""Shared RF spectrum placement, inverse transform and event threshold."""

from typing import cast

import array_api_extra as xpx
from jaxtyping import Complex, Float

from fast_simus.utils._array_api import Array, _ArrayNamespace, _ArrayNamespaceWithFFT


def _irfft_and_threshold(
    spect_selected: Complex[Array, "n_freq_sel n_elem"],
    plan,
    n_elements: int,
    xp: _ArrayNamespace,
) -> tuple[Float[Array, "n_samples n_elem"], Complex[Array, "n_freq_full n_elem"]]:
    """Place selected spectrum, IFFT to time domain, apply smooth thresholding."""
    if not hasattr(xp, "fft"):
        msg = "simus requires an array backend with FFT support (e.g. numpy, jax, cupy)"
        raise RuntimeError(msg)
    xp_fft = cast(_ArrayNamespaceWithFFT, xp)

    n_freq_sel = spect_selected.shape[0]
    full_spectrum = xp.zeros((plan.n_freq_full, n_elements), dtype=spect_selected.dtype)
    full_spectrum = xpx.at(full_spectrum)[plan.freq_idx_start : plan.freq_idx_start + n_freq_sel, :].set(  # type: ignore[attr-defined]
        spect_selected
    )

    rf = xp_fft.fft.irfft(xp.conj(full_spectrum), n=plan.n_fft, axis=0)

    n_keep = (plan.n_fft + 1) // 2
    rf = rf[:n_keep, ...]

    # Smooth thresholding of small values (-100 dB)
    rel_thresh = 1e-5
    rf_peak = xp.max(xp.abs(rf))
    rel_rf = xp.abs(rf) / (rf_peak + xp.asarray(1e-30))
    smooth_gate = 0.5 * (1.0 + xp.tanh((rel_rf - rel_thresh) / (rel_thresh / 10.0)))  # type: ignore[attr-defined]

    rf = rf * smooth_gate

    return rf, full_spectrum
