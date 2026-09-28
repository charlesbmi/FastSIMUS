"""Portable finite rectangular response with bounded patch integration."""

from math import pi

import array_api_extra as xpx

from fast_simus._blocking import block_count, element_block, run_loop
from fast_simus._pfield_math import NEPER_TO_DB
from fast_simus._propagation import propagation_exponential
from fast_simus.aperture import _cross
from fast_simus.transducer_params import BaffleType


def element_response(points, aperture, counts, e, frequency, fc, medium, full_directivity, tiles, xp):
    """Return H(point,element) for one static element tile, normalized over patches."""
    indices, element_valid = element_block(e, tiles.elements, counts.shape[0], xp)
    subdivision = xp.take(counts, indices, axis=0)
    nu, nv = subdivision[:, 0, None], subdivision[:, 1, None]
    count_u, count_v = xp.astype(nu, points.dtype), xp.astype(nv, points.dtype)
    axes_u = xp.take(aperture.width_axes, indices, axis=0)
    axes_v = xp.take(aperture.height_axes, indices, axis=0)
    normal = _cross(axes_u, axes_v, xp)
    sizes = xp.take(aperture.sizes, indices, axis=0)
    width, height = sizes[:, 0, None], sizes[:, 1, None]
    element_centers = xp.take(aperture.centers, indices, axis=0)

    def integrate(block, total):
        q = block * tiles.patches + xp.arange(tiles.patches)
        valid = (q[None, :] < nu * nv) & element_valid[:, None]
        # Inactive patches use the element center, avoiding extreme fake geometry.
        u = xp.where(
            valid,
            ((xp.astype(q // nv, points.dtype) + 0.5) / count_u - 0.5) * width,
            xp.asarray(0.0, dtype=points.dtype),
        )
        v = xp.where(
            valid,
            ((xp.astype(q % nv, points.dtype) + 0.5) / count_v - 0.5) * height,
            xp.asarray(0.0, dtype=points.dtype),
        )
        centers = element_centers[:, None, :] + u[:, :, None] * axes_u[:, None, :] + v[:, :, None] * axes_v[:, None, :]
        delta = points[:, None, None, :] - centers[None, :, :, :]
        r = xp.sqrt(xp.sum(delta * delta, axis=-1))
        safe = xp.maximum(r, xp.asarray(medium.speed_of_sound / (2 * fc), dtype=r.dtype))
        denom = xp.where(r > 0, r, xp.ones_like(r))
        cos_u = xp.sum(delta * axes_u[None, :, None, :], axis=-1) / denom
        cos_v = xp.sum(delta * axes_v[None, :, None, :], axis=-1) / denom
        cos_n = xp.sum(delta * normal[None, :, None, :], axis=-1) / denom
        visible = (cos_n > 0) & valid[None, :, :]
        cosine = xp.where(visible, cos_n, xp.ones_like(cos_n))
        obliquity = aperture_baffle(cosine, medium.baffle, xp)
        f_dir = frequency if full_directivity else fc
        directivity = xpx.sinc(f_dir / medium.speed_of_sound * width / count_u * cos_u, xp=xp)
        directivity = directivity * xpx.sinc(f_dir / medium.speed_of_sound * height / count_v * cos_v, xp=xp)
        delay = medium.lens_reference_delay
        if medium.lens is not None:
            delay = delay - v * v / (
                2 * medium.speed_of_sound * xp.take(medium.lens.focal_lengths, indices, axis=0)[:, None]
            )
        attenuation = medium.attenuation / NEPER_TO_DB * frequency * 1e-4
        response = (
            propagation_exponential(
                safe,
                2 * pi * frequency / medium.speed_of_sound,
                attenuation,
                xp,
                phase_offset=2 * pi * frequency * delay,
            )
            * directivity
            * obliquity
            / safe
        )
        return (
            total
            + xp.sum(xp.where(visible, response, xp.zeros_like(response)), axis=-1)
            / (count_u[:, 0] * count_v[:, 0])[None, :]
        )

    return run_loop(
        block_count(tiles.max_patches, tiles.patches),
        integrate,
        xp.zeros((points.shape[0], tiles.elements), dtype=points.dtype) + 0j,
        xp,
    )


def aperture_baffle(cosine, baffle, xp):
    """Evaluate a baffle on positive local direction cosines."""
    if baffle == BaffleType.SOFT:
        return cosine
    if baffle == BaffleType.RIGID:
        return xp.ones_like(cosine)
    return cosine / (cosine + baffle)
