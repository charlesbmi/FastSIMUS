"""Acoustic response attached to an explicit finite three-dimensional aperture."""

from dataclasses import dataclass
from math import inf, isfinite
from typing import Literal

from fast_simus.aperture import RectangularAperture
from fast_simus.transducer_params import BaffleType
from fast_simus.utils.geometry import element_positions


@dataclass(frozen=True, eq=False)
class Transducer:
    """Finite rectangular radiator with scalar probe response.

    Geometry buffers are borrowed and must not change during plan use.
    Frequency is in Hz, bandwidth is fractional, model must explicitly be '3d'.
    """

    aperture: RectangularAperture
    model: Literal["3d"]
    freq_center: float
    bandwidth: float = 0.75
    baffle: BaffleType | float = BaffleType.SOFT

    def __post_init__(self):
        """Validate the physical description eagerly."""
        if self.model != "3d":
            raise ValueError("Transducer requires model='3d'")
        if not isfinite(self.freq_center) or self.freq_center <= 0 or not 0 < self.bandwidth <= 2:
            raise ValueError("Require positive finite center frequency and bandwidth in (0,2]")
        if self.baffle not in (BaffleType.SOFT, BaffleType.RIGID) and (
            not isinstance(self.baffle, (int, float)) or not isfinite(self.baffle) or self.baffle < 0
        ):
            raise ValueError("Invalid baffle")

    @property
    def n_elements(self):
        """Electrical element count."""
        return self.aperture.centers.shape[0]


def transducer_from_params(params, *, model="3d", xp, dtype=None, device=None) -> Transducer:
    """Explicitly convert a finite-height conventional probe, preserving its origin."""
    if not isfinite(params.height):
        raise ValueError("Finite height is required for 3D conversion")
    if params.elev_focus != inf:
        raise ValueError("Finite elevation focus requires the lens response implementation")
    pos, theta, _ = element_positions(params.n_elements, params.pitch, params.radius, xp)
    kw = dict(dtype=dtype or xp.float32, device=device)
    pos = xp.asarray(pos, **kw)
    zeros = xp.zeros(params.n_elements, **kw)
    theta = zeros if theta is None else xp.asarray(theta, **kw)
    centers = xp.stack((pos[:, 0], zeros, pos[:, 1]), axis=-1)
    u = xp.stack((xp.cos(theta), zeros, -xp.sin(theta)), axis=-1)
    v = xp.stack((zeros, xp.ones_like(zeros), zeros), axis=-1)
    sizes = xp.broadcast_to(xp.asarray([params.element_width, params.height], **kw), (params.n_elements, 2))
    return Transducer(
        RectangularAperture(centers, u, v, sizes), model, params.freq_center, params.bandwidth, params.baffle
    )
