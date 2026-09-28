"""Public block/dense transient volume workflow."""

from typing import Any

import numpy as _np
import pytest

from fast_simus import Transducer, matrix_aperture, pfield_precompute, pfield_spectrum_compute, spectrum_to_wavefield
from fast_simus.execution import ExecutionOptions
from fast_simus.field_blocks import iter_pfield_spectrum, iter_wavefield
from fast_simus.wavefield import wavefield_times

np: Any = _np


def test_blocks_and_transient_times():
    """Uneven blocks retain the original common grid and time axis."""
    t = Transducer(
        matrix_aperture(shape=(2, 1), pitch=(0.0004, 0.0004), size=(0.0002, 0.0002), xp=np, dtype=np.float64), "3d", 2e6
    )
    p = np.array([[[0.0, 0.0, 0.015], [0.001, 0.002, 0.02], [0.003, -0.001, 0.018]]])
    d = np.zeros(2)
    plan = pfield_precompute(p, d, t, execution=ExecutionOptions(4096))
    dense = pfield_spectrum_compute(p, d, plan, t)
    blocks = list(iter_pfield_spectrum(p, d, plan, t))
    joined = np.concatenate([b.values for b in blocks])
    np.testing.assert_allclose(joined, np.asarray(dense).reshape(-1, dense.shape[-1]))
    assert blocks[0].start == 0 and blocks[-1].stop == 3
    wave = spectrum_to_wavefield(dense, plan)
    frames = np.concatenate([b.values for b in iter_wavefield(p, d, plan, t)])
    np.testing.assert_allclose(frames, np.asarray(wave.frames).reshape(3, -1))
    np.testing.assert_array_equal(wave.times, wavefield_times(plan))
    arrival = wave.times[np.argmax(np.abs(wave.frames[0, 0]))]
    assert abs(float(arrival) - 0.015 / 1540) < 1 / 2e6
    cw = pfield_precompute(p, d, t, tx_n_wavelengths=np.inf)
    with pytest.raises(ValueError, match="CW"):
        wavefield_times(cw)


@pytest.mark.parametrize("script", ["matrix_field.py", "volumetric_rf.py"])
def test_example_smoke(script):
    """The documented standalone scripts run with small finite fixtures."""
    import subprocess
    import sys
    from pathlib import Path

    subprocess.run(  # noqa: S603 - fixed local example paths
        [sys.executable, str(Path(__file__).parents[1] / "examples" / script), "--smoke"],
        check=True,
        capture_output=True,
    )
