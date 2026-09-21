"""Time-domain pressure fields for transducer arrays.

The inverse transform is applied only along temporal frequency; spatial axes
are preserved.

References:
    Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging.
    Part I: theory & examples. CMPB, 2022;218:106726.
"""

from __future__ import annotations

from math import prod
from typing import NamedTuple

from jaxtyping import Complex, Float

from fast_simus.medium_params import MediumParams
from fast_simus.pfield import PfieldPlan, PfieldSpectrumInfo, pfield_spectrum
from fast_simus.transducer_params import TransducerParams
from fast_simus.utils._array_api import Array, array_namespace

_DEFAULT_MEDIUM = MediumParams()

# Upper bound on the number of complex values in one inverse-FFT batch. The
# padded spectrum, its conjugate, the real output, and the FFT plan's work area
# are all live at once, so an unchunked transform of a large grid can need
# several GB. Chunking bounds that regardless of grid or bandwidth.
_MAX_FFT_BATCH_ELEMENTS = 4_000_000


class WavefieldResult(NamedTuple):
    """Time-resolved pressure field.

    Attributes:
        frames: Real pressure in arbitrary units, shape ``(*grid_shape, n_times)``.
        times: Frame times in seconds, starting at the transmit reference.
    """

    frames: Float[Array, "*grid_shape n_times"]
    times: Float[Array, " n_times"]


def spectrum_to_wavefield(
    spectrum: Complex[Array, "*grid_shape n_freq_selected"],
    info: PfieldPlan | PfieldSpectrumInfo,
) -> WavefieldResult:
    """Transform a complex pressure spectrum into a propagating wave over time.

    The selected band is placed into its full frequency grid, inverse
    transformed, and cropped to the causal half. A finer frequency grid makes
    the record longer without changing the frame spacing.

    Args:
        spectrum: Complex pressure spectrum from
            :func:`fast_simus.pfield.pfield_spectrum`, shape
            ``(*grid_shape, n_freq_selected)``.
        info: Frequency-grid metadata from the same call, or the
            ``PfieldPlan`` used to compute ``spectrum``.

    Returns:
        WavefieldResult with real frames and their times in seconds.
    """
    xp = array_namespace(spectrum)

    n_selected = spectrum.shape[-1]
    n_before = info.freq_idx_start
    n_after = info.n_freq_full - n_before - n_selected
    if n_after < 0:
        raise ValueError(
            f"Selected band ({n_selected} bins at offset {n_before}) does not fit in a grid of {info.n_freq_full} bins."
        )

    grid_shape = spectrum.shape[:-1]
    n_points = prod(grid_shape)
    n_time = 2 * (info.n_freq_full - 1)
    n_keep = n_time // 2
    # Scale so the result approximates the inverse Fourier integral rather than
    # a bare DFT, making amplitudes independent of the frequency-grid spacing.
    scale = n_time * info.freq_step

    xp_fft = _fft_namespace(xp)
    flat = xp.reshape(spectrum, (n_points, n_selected))

    chunk = max(1, _MAX_FFT_BATCH_ELEMENTS // max(info.n_freq_full, 1))
    blocks = []
    for start in range(0, n_points, chunk):
        block = flat[start : start + chunk, :]
        n_rows = block.shape[0]
        # Conjugating before the inverse transform yields a time-forward
        # signal, matching the convention used by simus.
        padded = xp.concat(
            [
                xp.zeros((n_rows, n_before), dtype=block.dtype),
                xp.conj(block),
                xp.zeros((n_rows, n_after), dtype=block.dtype),
            ],
            axis=-1,
        )
        blocks.append(xp_fft.fft.irfft(padded, n=n_time, axis=-1)[:, :n_keep] * scale)

    frames_flat = blocks[0] if len(blocks) == 1 else xp.concat(blocks, axis=0)
    frames = xp.reshape(frames_flat, (*grid_shape, n_keep))

    dt = 1.0 / scale
    times = xp.arange(n_keep, dtype=frames.dtype) * dt

    return WavefieldResult(frames=frames, times=times)


def _fft_namespace(xp):
    """Return the namespace exposing ``fft.irfft`` for this backend."""
    if not hasattr(xp, "fft"):
        msg = "wavefield requires an array backend with FFT support (e.g. numpy, jax, cupy, mlx)"
        raise RuntimeError(msg)
    return xp


def wavefield(
    positions: Float[Array, "*grid_shape 2"],
    delays: Float[Array, " n_elements"],
    params: TransducerParams,
    medium: MediumParams = _DEFAULT_MEDIUM,
    *,
    tx_apodization: Float[Array, " n_elements"] | None = None,
    tx_n_wavelengths: float | int = 1.0,
    db_thresh: float | int = -60.0,
    full_frequency_directivity: bool = False,
    element_splitting: int | None = None,
    frequency_step: float | int = 0.5,
) -> WavefieldResult:
    """Simulate a transmitted wave propagating through a grid over time.

    This is the time-domain counterpart to :func:`fast_simus.pfield.pfield` and
    is equivalent to MUST's ``mkmovie``. The default frequency step provides a
    longer record than ``pfield`` to reduce time-domain wrap-around.

    Args:
        positions: Grid positions in meters. Shape ``(*grid_shape, 2)`` where
            ``positions[..., 0]`` is lateral (x) and ``positions[..., 1]`` is
            axial (z, into tissue).
        delays: Transmit time delays in seconds. Shape ``(n_elements,)``.
        params: Transducer parameters.
        medium: Medium parameters.
        tx_apodization: Transmit apodization weights. Shape ``(n_elements,)``.
        tx_n_wavelengths: Number of wavelengths in the TX pulse.
        db_thresh: Threshold in dB for frequency component selection.
        full_frequency_directivity: If True, compute element directivity at
            every frequency.
        element_splitting: Number of sub-elements per element. MUST's
            ``mkmovie`` forces 1; None selects the automatic value.
        frequency_step: Scaling factor for the frequency step. Smaller values
            lengthen the time record.

    Returns:
        WavefieldResult with frames of shape ``(*grid_shape, n_times)``.
    """
    spectrum, info = pfield_spectrum(
        positions,
        delays,
        params,
        medium,
        tx_apodization=tx_apodization,
        tx_n_wavelengths=tx_n_wavelengths,
        db_thresh=db_thresh,
        full_frequency_directivity=full_frequency_directivity,
        element_splitting=element_splitting,
        frequency_step=frequency_step,
    )
    return spectrum_to_wavefield(spectrum, info)
