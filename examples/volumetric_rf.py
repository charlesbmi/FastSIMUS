"""Seeded 3D cloud with independent plane-wave acquisitions."""

import argparse
from typing import cast

import numpy as np

from fast_simus import Transducer, TransmitSequence, matrix_aperture, plane_wave_delays, simus_sequence
from fast_simus.utils._array_api import Array


def main():
    """Simulate independent events and expose effective receive timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    points = rng.uniform([-0.003, -0.003, 0.01], [0.003, 0.003, 0.03], size=(3 if args.smoke else 100, 3))
    rc = rng.normal(size=points.shape[0])
    aperture = matrix_aperture(shape=(2, 2), pitch=(0.0003, 0.0003), size=(0.0002, 0.0002), xp=np, dtype=np.float64)
    transducer = Transducer(aperture, "3d", 2e6)
    delays = np.stack([plane_wave_delays(aperture.centers, np.array([np.sin(a), 0.0, np.cos(a)])) for a in (-0.1, 0.1)])
    result = simus_sequence(points, rc, TransmitSequence(cast(Array, delays)), transducer)
    if not np.all(np.isfinite(np.asarray(result.rf))):
        raise RuntimeError("Nonfinite RF")
    print(f"RF {result.rf.shape}; effective sample rate {result.sampling_frequency:g} Hz")


if __name__ == "__main__":
    main()
