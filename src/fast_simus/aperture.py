"""Finite rectangular element geometry in physical coordinates (meters)."""

from dataclasses import dataclass
from math import isfinite

import array_api_compat

from fast_simus.utils._array_api import Array, array_namespace


def _same_arrays(*arrays):
    """Require real floating arrays on one namespace, device and precision."""
    xp = array_namespace(*arrays)
    first = arrays[0]
    for a in arrays:
        actual_device, expected_device = array_api_compat.device(a), array_api_compat.device(first)
        # JAX tracers have no concrete device; eager plan validation checked it.
        if a.dtype != first.dtype or (
            actual_device is not None and expected_device is not None and actual_device != expected_device
        ):
            raise ValueError("Arrays must share dtype and device")
        if a.dtype not in (xp.float32, getattr(xp, "float64", None)):
            raise ValueError("Coordinates must be real floating arrays")
    return xp


def _cross(a, b, xp):
    return xp.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


@dataclass(frozen=True, eq=False)
class RectangularAperture:
    """Oriented rectangles; buffers must remain unchanged while a plan uses them.

    Centers/width_axes/height_axes have shape (E,3); sizes has shape (E,2).
    Axes are orthonormal. Their cross product points into the radiating half-space.
    """

    centers: Array
    width_axes: Array
    height_axes: Array
    sizes: Array

    def __post_init__(self):
        """Validate the physical description eagerly."""
        xp = _same_arrays(self.centers, self.width_axes, self.height_axes, self.sizes)
        e = self.centers.shape[0]
        if (
            e < 1
            or self.centers.shape != (e, 3)
            or self.width_axes.shape != (e, 3)
            or self.height_axes.shape != (e, 3)
            or self.sizes.shape != (e, 2)
        ):
            raise ValueError("Expected nonempty (E,3) centers/axes and (E,2) sizes")
        for a in (self.centers, self.width_axes, self.height_axes, self.sizes):
            if not bool(xp.all(xp.isfinite(a))):
                raise ValueError("Geometry must be finite")
        if not bool(xp.all(self.sizes > 0)):
            raise ValueError("Element sizes must be positive")
        tol = 1e-5 if self.centers.dtype == xp.float32 else 1e-10
        u, v = self.width_axes, self.height_axes
        if any(
            not bool(xp.all(xp.abs(value) < tol))
            for value in (xp.sum(u * u, axis=-1) - 1, xp.sum(v * v, axis=-1) - 1, xp.sum(u * v, axis=-1))
        ):
            raise ValueError("Element axes must be unit and orthogonal")

    @property
    def normals(self):
        """Right-handed radiating normals, shape (E,3)."""
        return _cross(self.width_axes, self.height_axes, array_namespace(self.centers))


def matrix_aperture(*, shape, pitch, size, xp, dtype=None, device=None) -> RectangularAperture:
    """Construct a +z-facing matrix with x-fastest order iy*nx+ix.

    Shape is (nx,ny); pitch and size are (x,y) lengths in meters.
    """
    if len(shape) != 2 or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape):
        raise ValueError("shape must contain two positive integers")
    if len(pitch) != 2 or len(size) != 2 or any(not isfinite(v) or v <= 0 for v in (*pitch, *size)):
        raise ValueError("pitch and size must be finite positive pairs")
    kw = dict(dtype=dtype or xp.float32)
    if device is not None:
        kw["device"] = device
    nx, ny = shape
    idx = xp.arange(nx * ny, **kw)
    x = (idx % nx - (nx - 1) / 2) * pitch[0]
    y = (xp.floor(idx / nx) - (ny - 1) / 2) * pitch[1]
    centers = xp.stack((x, y, xp.zeros_like(x)), axis=-1)
    u = xp.broadcast_to(xp.asarray([1.0, 0.0, 0.0], **kw), (nx * ny, 3))
    v = xp.broadcast_to(xp.asarray([0.0, 1.0, 0.0], **kw), (nx * ny, 3))
    sizes = xp.broadcast_to(xp.asarray(size, **kw), (nx * ny, 2))
    return RectangularAperture(centers, u, v, sizes)


def transform_aperture(aperture, rotation, translation) -> RectangularAperture:
    """Apply a proper (3,3) rotation and (3,) translation in meters."""
    xp = _same_arrays(aperture.centers, rotation, translation)
    tol = 1e-5 if rotation.dtype == xp.float32 else 1e-10
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Expected (3,3) rotation and (3,) translation")
    if not bool(xp.all(xp.isfinite(rotation))) or not bool(xp.all(xp.isfinite(translation))):
        raise ValueError("Transform must be finite")
    determinant = xp.sum(rotation[0] * _cross(rotation[1], rotation[2], xp))
    if (
        not bool(xp.all(xp.abs(rotation @ rotation.T - xp.eye(3, dtype=rotation.dtype)) < tol))
        or abs(float(determinant) - 1) > tol
    ):
        raise ValueError("rotation must be orthogonal with determinant +1")
    return RectangularAperture(
        aperture.centers @ rotation.T + translation,
        aperture.width_axes @ rotation.T,
        aperture.height_axes @ rotation.T,
        aperture.sizes,
    )
