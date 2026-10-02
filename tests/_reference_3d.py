"""Small independent double-precision rectangular aperture oracle.

Direct exponentiation at every frequency intentionally avoids production helpers
and frequency recurrences. Source strength is normalized per electrical element.
"""

import numpy as np


def rectangular_transfer(
    points,
    centers,
    width_axes,
    height_axes,
    sizes,
    frequencies,
    *,
    speed_of_sound=1540.0,
    freq_center=2e6,
    subdivision=(1, 1),
    baffle="soft",
    attenuation=0.0,
    full_frequency_directivity=True,
    lens_focus=None,
    lens_reference_delay=0.0,
):
    """Return H with shape (frequency, point, element), without pulse response."""
    points, centers, width_axes, height_axes, sizes = (
        np.asarray(value, dtype=np.float64) for value in (points, centers, width_axes, height_axes, sizes)
    )
    frequencies = np.asarray(frequencies, dtype=np.float64)
    counts = np.broadcast_to(np.asarray(subdivision, dtype=int), (len(centers), 2))
    output = np.zeros((len(frequencies), len(points), len(centers)), dtype=np.complex128)
    for element, (center, u_axis, v_axis, size, count) in enumerate(
        zip(centers, width_axes, height_axes, sizes, counts, strict=True)
    ):
        normal = np.cross(u_axis, v_axis)
        for iu in range(count[0]):
            for iv in range(count[1]):
                u = ((iu + 0.5) / count[0] - 0.5) * size[0]
                v = ((iv + 0.5) / count[1] - 0.5) * size[1]
                delta = points - (center + u * u_axis + v * v_axis)
                distance = np.linalg.norm(delta, axis=-1)
                safe_distance = np.maximum(distance, speed_of_sound / freq_center / 2)
                direction_distance = np.maximum(distance, np.finfo(float).tiny)
                cosine = (delta @ normal) / direction_distance
                visible = cosine > 0
                if baffle == "soft":
                    obliquity = cosine
                elif baffle == "rigid":
                    obliquity = np.ones_like(cosine)
                else:
                    obliquity = cosine / (cosine + float(baffle))
                obliquity = np.where(visible, obliquity, 0)
                for index, frequency in enumerate(frequencies):
                    directivity_frequency = frequency if full_frequency_directivity else freq_center
                    directivity = np.sinc(
                        directivity_frequency
                        / speed_of_sound
                        * size[0]
                        / count[0]
                        * (delta @ u_axis)
                        / direction_distance
                    ) * np.sinc(
                        directivity_frequency
                        / speed_of_sound
                        * size[1]
                        / count[1]
                        * (delta @ v_axis)
                        / direction_distance
                    )
                    decay = attenuation / (20 / np.log(10)) * frequency / 1e6 * 100
                    delay = lens_reference_delay
                    if lens_focus is not None:
                        delay -= v * v / (2 * speed_of_sound * lens_focus)
                    response = np.exp(
                        -decay * safe_distance + 2j * np.pi * frequency * (safe_distance / speed_of_sound + delay)
                    )
                    output[index, :, element] += response * directivity * obliquity / safe_distance / np.prod(count)
    return output
