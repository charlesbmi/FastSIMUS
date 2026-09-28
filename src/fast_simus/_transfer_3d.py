"""Portable finite rectangular element response with local half-space visibility."""

from math import pi

import array_api_extra as xpx

from fast_simus._pfield_math import NEPER_TO_DB
from fast_simus.transducer_params import BaffleType


def rectangle_response(points, aperture, counts, frequency, fc, medium, full_directivity, xp):
    """Return H(point,element), normalized to unit strength per element.

    Counts are resolved eagerly. Source integration keeps electrical elements
    distinct for reciprocal receive contraction.
    """
    responses = []
    for e, (nu, nv) in enumerate(counts):
        u = ((xp.arange(nu, dtype=points.dtype) + 0.5) / nu - 0.5) * aperture.sizes[e, 0]
        v = ((xp.arange(nv, dtype=points.dtype) + 0.5) / nv - 0.5) * aperture.sizes[e, 1]
        centers = (
            aperture.centers[e] + u[:, None, None] * aperture.width_axes[e] + v[None, :, None] * aperture.height_axes[e]
        )
        delta = points[:, None, :] - xp.reshape(centers, (-1, 3))
        r = xp.sqrt(xp.sum(delta * delta, axis=-1))
        safe = xp.maximum(r, xp.asarray(medium.speed_of_sound / (2 * fc), dtype=r.dtype))
        denom = xp.where(r > 0, r, xp.ones_like(r))
        cos_u = xp.sum(delta * aperture.width_axes[e], axis=-1) / denom
        cos_v = xp.sum(delta * aperture.height_axes[e], axis=-1) / denom
        cos_n = xp.sum(delta * aperture.normals[e], axis=-1) / denom
        visible = cos_n > 0
        cosine = xp.where(visible, cos_n, xp.ones_like(cos_n))
        baffle = aperture_baffle(cosine, medium.baffle, xp)
        f_dir = frequency if full_directivity else fc
        directivity = xpx.sinc(f_dir / medium.speed_of_sound * aperture.sizes[e, 0] / nu * cos_u, xp=xp)
        directivity = directivity * xpx.sinc(f_dir / medium.speed_of_sound * aperture.sizes[e, 1] / nv * cos_v, xp=xp)
        phase = 2 * pi * frequency / medium.speed_of_sound * safe
        attenuation = medium.attenuation / NEPER_TO_DB * frequency * 1e-4
        response = xp.exp(-attenuation * safe + 1j * phase) * directivity * baffle / safe
        responses.append(xp.sum(xp.where(visible, response, xp.zeros_like(response)), axis=-1) / (nu * nv))
    return xp.stack(responses, axis=-1)


def aperture_baffle(cosine, baffle, xp):
    """Evaluate a baffle on positive local direction cosines."""
    if baffle == BaffleType.SOFT:
        return cosine
    if baffle == BaffleType.RIGID:
        return xp.ones_like(cosine)
    return cosine / (cosine + baffle)
