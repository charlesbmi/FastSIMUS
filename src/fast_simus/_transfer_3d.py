"""Portable finite rectangular response with bounded patch integration."""

from math import pi

import array_api_extra as xpx

from fast_simus._blocking import block_count, run_loop
from fast_simus._pfield_math import NEPER_TO_DB
from fast_simus.transducer_params import BaffleType


def element_response(points, aperture, counts, e, frequency, fc, medium, full_directivity, tiles, xp):
    """Return one electrical element's H(point), normalized over its patches."""
    nu, nv = counts[e, 0], counts[e, 1]
    normal = aperture.normals[e]
    width, height = aperture.sizes[e, 0], aperture.sizes[e, 1]

    def integrate(block, total):
        q = block * tiles.patches + xp.arange(tiles.patches)
        valid = q < nu * nv
        # Inactive patches use the element center, avoiding extreme fake geometry.
        u = xp.where(valid, ((q // nv + 0.5) / nu - 0.5) * width, 0.0)
        v = xp.where(valid, ((q % nv + 0.5) / nv - 0.5) * height, 0.0)
        centers = aperture.centers[e] + u[:, None] * aperture.width_axes[e] + v[:, None] * aperture.height_axes[e]
        delta = points[:, None, :] - centers
        r = xp.sqrt(xp.sum(delta * delta, axis=-1))
        safe = xp.maximum(r, xp.asarray(medium.speed_of_sound / (2 * fc), dtype=r.dtype))
        denom = xp.where(r > 0, r, xp.ones_like(r))
        cos_u = xp.sum(delta * aperture.width_axes[e], axis=-1) / denom
        cos_v = xp.sum(delta * aperture.height_axes[e], axis=-1) / denom
        cos_n = xp.sum(delta * normal, axis=-1) / denom
        visible = (cos_n > 0) & valid[None, :]
        cosine = xp.where(visible, cos_n, xp.ones_like(cos_n))
        obliquity = aperture_baffle(cosine, medium.baffle, xp)
        f_dir = frequency if full_directivity else fc
        directivity = xpx.sinc(f_dir / medium.speed_of_sound * width / nu * cos_u, xp=xp)
        directivity = directivity * xpx.sinc(f_dir / medium.speed_of_sound * height / nv * cos_v, xp=xp)
        delay = medium.lens_reference_delay
        if medium.lens is not None:
            delay = delay - v * v / (2 * medium.speed_of_sound * medium.lens.focal_lengths[e])
        phase = 2 * pi * frequency * (safe / medium.speed_of_sound + delay)
        attenuation = medium.attenuation / NEPER_TO_DB * frequency * 1e-4
        response = xp.exp(-attenuation * safe + 1j * phase) * directivity * obliquity / safe
        return total + xp.sum(xp.where(visible, response, xp.zeros_like(response)), axis=-1) / (nu * nv)

    return run_loop(block_count(tiles.max_patches, tiles.patches), integrate, xp.zeros_like(points[:, 0]) + 0j, xp)


def aperture_baffle(cosine, baffle, xp):
    """Evaluate a baffle on positive local direction cosines."""
    if baffle == BaffleType.SOFT:
        return cosine
    if baffle == BaffleType.RIGID:
        return xp.ones_like(cosine)
    return cosine / (cosine + baffle)
