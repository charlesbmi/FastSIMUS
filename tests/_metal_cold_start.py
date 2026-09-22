"""Standalone Metal cold-start regression scenario for subprocess tests."""

from typing import cast

import mlx.core as mx
import numpy as np

from fast_simus import (
    MediumParams,
    SimusStrategy,
    element_positions,
    focused,
    rms_from_spectrum,
    scattering_pfield_spectrum,
    simus,
    spectrum_to_wavefield,
)
from fast_simus.transducer_presets import P4_2v
from fast_simus.utils._array_api import Array, _ArrayNamespace


def main() -> None:
    """Run wavefield work followed by the first Metal receive simulation."""
    params = P4_2v()
    medium = MediumParams(attenuation=0.5)
    xp = cast(_ArrayNamespace, mx)
    elements, _, apex = element_positions(params.n_elements, params.pitch, params.radius, xp)
    focus = cast(Array, mx.array([0.0, 0.03]))
    delays = focused(elements, focus, speed_of_sound=1540.0, radius=params.radius, apex_offset=apex)
    x_axis = mx.linspace(-0.02, 0.02, 12)
    z_axis = mx.linspace(0.0, 0.055, 16)
    x_grid, z_grid = mx.meshgrid(x_axis, z_axis)
    positions = cast(Array, mx.stack([x_grid, z_grid], axis=-1))
    scatterers = cast(Array, mx.array([[0.0, 0.03]]))
    coefficients = cast(Array, mx.array([0.005]))
    apodization = cast(Array, mx.ones(params.n_elements))
    spectrum = scattering_pfield_spectrum(
        positions,
        scatterers,
        coefficients,
        delays,
        params,
        medium,
        tx_apodization=apodization,
        frequency_step=2.0,
    )
    np.asarray(spectrum_to_wavefield(spectrum.incident, spectrum.info).frames)
    np.asarray(rms_from_spectrum(spectrum.incident, spectrum.info))

    result = simus(
        scatterers,
        coefficients,
        delays,
        params,
        medium,
        fs=4.0 * params.freq_center,
        tx_apodization=apodization,
        frequency_step=0.5,
        strategy=SimusStrategy.METAL,
    )
    rf = np.asarray(result.rf)
    if not np.all(np.isfinite(rf)) or np.max(np.abs(rf)) <= 0.0:
        raise RuntimeError("First Metal receive result must be finite and nonzero")


if __name__ == "__main__":
    main()
