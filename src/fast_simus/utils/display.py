"""Display mappings for bipolar pressure fields.

These are numerical transforms, not plotting. They turn a pressure array into
values a diverging colormap can show: white at zero, saturated color at the
chosen limits. Callers still choose a colormap and a widget.

The signed-dB map is the ultrasound convention used by delay-and-sum.com
("Display in dB"): amplitude is log-scaled against a fixed peak so weak
wavefronts stay visible while the color limits stay constant over time.
"""

from __future__ import annotations

from jaxtyping import Float

from fast_simus.utils._array_api import Array, array_namespace


def signed_db(
    pressure: Float[Array, "*shape"],
    peak: float,
    dynamic_range: float,
) -> Float[Array, "*shape"]:
    """Map bipolar pressure onto a signed dB scale in ``[-DR, +DR]``.

    Zero stays zero. ``|p| == peak`` maps to ``±dynamic_range``. Amplitudes at
    or below ``peak * 10**(-DR/20)`` collapse to zero. The peak is a caller-
    supplied reference (typically the movie-wide maximum), so the scale does
    not change from frame to frame.

    Args:
        pressure: Real pressure. Shape is preserved.
        peak: Positive reference amplitude in the same units as ``pressure``.
        dynamic_range: Positive dynamic range in dB.

    Returns:
        Signed dB values in ``[-dynamic_range, dynamic_range]``.

    Raises:
        ValueError: If ``peak`` or ``dynamic_range`` is not positive.
    """
    if peak <= 0.0:
        raise ValueError("peak must be positive")
    if dynamic_range <= 0.0:
        raise ValueError("dynamic_range must be positive")

    xp = array_namespace(pressure)
    floor = xp.asarray(1e-12, dtype=pressure.dtype)
    mag = 20.0 * xp.log10(xp.abs(pressure) / peak + floor)
    shifted = mag + dynamic_range
    zero = xp.asarray(0.0, dtype=shifted.dtype)
    hi = xp.asarray(dynamic_range, dtype=shifted.dtype)
    clipped = xp.where(shifted < zero, zero, shifted)
    clipped = xp.where(clipped > hi, hi, clipped)
    return xp.sign(pressure) * clipped
