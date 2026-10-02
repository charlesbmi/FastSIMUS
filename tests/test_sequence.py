"""Independent acquisitions reuse one grid without summing events."""

from typing import Any

import numpy as _np

from fast_simus import Transducer, matrix_aperture, simus_compute
from fast_simus.sequence import TransmitSequence, iter_simus_sequence, sequence_precompute, simus_sequence

np: Any = _np


def test_sequence_common_grid_and_disabled_event():
    """Iteration and stacking match single-event calls on the common plan."""
    t = Transducer(
        matrix_aperture(shape=(2, 1), pitch=(0.0004, 0.0004), size=(0.0002, 0.0002), xp=np, dtype=np.float64), "3d", 2e6
    )
    p = np.array([[0.0, 0.001, 0.02]])
    rc = np.ones(1)
    sequence = TransmitSequence(np.array([[0.0, 1e-7], [2e-6, 2.1e-6], [np.nan, np.nan]]))
    plan = sequence_precompute(p, rc, sequence, t)
    events = list(iter_simus_sequence(p, rc, sequence, plan, t))
    result = simus_sequence(p, rc, sequence, t)
    for index, event in enumerate(events):
        assert event.index == index
        individual = simus_compute(p, rc, sequence.delays[index], plan.echo_plan, t)
        np.testing.assert_allclose(event.result.rf, individual.rf)
        np.testing.assert_allclose(result.spectrum[index], individual.spectrum)
    np.testing.assert_array_equal(result.rf[-1], 0)
    assert result.rf.shape == (3, result.sample_times.shape[0], 2)


def test_legacy_sequence_preserves_event_offsets():
    """Legacy plans use one common grid and preserve common electronic delays."""
    from fast_simus import TransducerParams

    t = TransducerParams(freq_center=2e6, pitch=0.0004, width=0.0002, n_elements=2)
    p = np.array([[0.001, 0.02]])
    rc = np.ones(1)
    d = np.array([[0.0, 1e-7], [1e-6, 1.1e-6], [0.0, 1e-7]])
    result = simus_sequence(p, rc, TransmitSequence(d), t)
    np.testing.assert_array_equal(result.spectrum[0], result.spectrum[2])
    f = np.arange(result.spectrum.shape[1]) * result.sampling_frequency / (2 * result.rf.shape[1])
    expected = result.spectrum[0] * np.exp(2j * np.pi * f[:, None] * 1e-6)
    np.testing.assert_allclose(result.spectrum[1], expected, rtol=0, atol=1e-10 * np.max(np.abs(expected)))
    reordered = simus_sequence(p, rc, TransmitSequence(d[::-1]), t)
    np.testing.assert_allclose(reordered.rf, result.rf[::-1])
