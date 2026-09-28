"""Raw pinned PyMUST3 parity on explicitly matched frequency grids."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as _np

from fast_simus import (
    Transducer,
    matrix_aperture,
    pfield_precompute,
    pfield_spectrum_compute,
    simus_compute,
    simus_precompute,
)
from fast_simus._frequency import FrequencyGrid, SamplingInfo
from fast_simus.spectrum import probe_spectrum, pulse_spectrum

np: Any = _np


def test_pinned_raw_field_and_echo():
    """Match raw complex amplitudes without per-output normalization."""
    data = np.load(Path(__file__).parent / "data/3d/planar_two_element.npz")
    a = matrix_aperture(shape=(2, 1), pitch=(0.0003, 0.0003), size=(0.0002, 0.0002), xp=np, dtype=np.float64)
    t = Transducer(a, "3d", 2e6)
    p, d, rc = data["points"], data["delays"], data["reflectivity"]
    field = pfield_precompute(p, d, t, element_splitting=(1, 1))
    frequencies = data["field_frequencies"]
    step = 4e6 / (len(data["field_mask"]) - 1)
    start = int(np.flatnonzero(data["field_mask"])[0])
    grid = FrequencyGrid(frequencies, step, len(data["field_mask"]), start, False)
    field = replace(
        field,
        _grid=grid,
        _pulse=pulse_spectrum(2 * np.pi * frequencies, 2e6, 1.0),
        _probe=probe_spectrum(2 * np.pi * frequencies, 2e6, 0.75),
    )
    actual = pfield_spectrum_compute(p, d, field, t, full_frequency_directivity=True)
    expected = data["field_spectrum"]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4 * np.max(np.abs(expected)))
    echo = simus_precompute(p, rc, d, t, element_splitting=(1, 1))
    frequencies = data["echo_frequencies"]
    step = 4e6 / (len(frequencies) - 1)
    grid = FrequencyGrid(frequencies, step, len(frequencies), 0, False)
    echo = replace(
        echo,
        _grid=grid,
        _pulse=pulse_spectrum(2 * np.pi * frequencies, 2e6, 1.0),
        _probe=probe_spectrum(2 * np.pi * frequencies, 2e6, 0.75),
        _sampling=SamplingInfo(8e6, 2 * (len(frequencies) - 1), step),
    )
    result = simus_compute(p, rc, d, echo, t, full_frequency_directivity=True)
    # PyMUST's threshold removes small spectral tails; compare its retained band.
    mask = np.max(np.abs(data["echo_spectrum"]), axis=-1) > 0
    expected = data["echo_spectrum"][mask]
    np.testing.assert_allclose(
        np.asarray(result.spectrum)[mask], expected, rtol=0, atol=1e-4 * np.max(np.abs(expected))
    )
