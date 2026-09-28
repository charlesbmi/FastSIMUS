"""Thin quadratic elevation lens in each element's local height coordinate."""

from dataclasses import dataclass

from fast_simus.utils._array_api import Array, array_namespace


@dataclass(frozen=True, eq=False)
class ElevationLens:
    """Positive focal lengths in meters, shape (E,); +inf means no local curvature."""

    focal_lengths: Array

    def __post_init__(self):
        """Validate focal lengths without changing their namespace or buffers."""
        xp = array_namespace(self.focal_lengths)
        if self.focal_lengths.ndim != 1 or self.focal_lengths.shape[0] == 0 or not bool(xp.all(self.focal_lengths > 0)):
            raise ValueError("Lens focal lengths must be positive finite or +inf, shape (E,)")


def lens_reference_delay(transducer, speed_of_sound):
    """One aperture-wide causal offset, in seconds, shared by all elements."""
    if transducer.lens is None:
        return 0.0
    xp = array_namespace(transducer.aperture.sizes)
    return float(xp.max(transducer.aperture.sizes[:, 1] ** 2 / (8 * speed_of_sound * transducer.lens.focal_lengths)))
