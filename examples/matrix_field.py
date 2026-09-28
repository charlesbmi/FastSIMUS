"""Off-axis matrix field and transient slice; run with uv run python examples/matrix_field.py."""

import argparse
from typing import cast

import numpy as np

from fast_simus import Transducer, matrix_aperture, pfield_spectrum, plane_wave_delays, spectrum_to_wavefield
from fast_simus.utils._array_api import Array


def main():
    """Evaluate a small physical slice and report finite output dimensions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    n = 4 if args.smoke else 24
    aperture = matrix_aperture(shape=(3, 3), pitch=(0.0003, 0.0003), size=(0.0002, 0.0002), xp=np, dtype=np.float64)
    transducer = Transducer(aperture, "3d", 2e6)
    x, z = np.meshgrid(np.linspace(-0.005, 0.005, n), np.linspace(0.01, 0.03, n))
    points = np.stack((x, np.full_like(x, 0.001), z), axis=-1)
    direction = np.array([0.1, 0.05, np.sqrt(1 - 0.1**2 - 0.05**2)])
    delays = plane_wave_delays(aperture.centers, direction)
    spectrum, info = pfield_spectrum(cast(Array, points), delays, transducer)
    result = spectrum_to_wavefield(spectrum, info)
    if not np.all(np.isfinite(np.asarray(result.frames))):
        raise RuntimeError("Nonfinite wavefield")
    print(f"pressure spectrum {spectrum.shape}; transient slice {result.frames.shape}")


if __name__ == "__main__":
    main()
