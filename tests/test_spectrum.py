"""Tests for spectrum helper functions.

Compares FastSIMUS spectrum functions against PyMUST reference implementations.
"""

import sys

import numpy as np
import pytest

from fast_simus.spectrum import probe_spectrum_fn, pulse_spectrum_fn

# PyMUST may not be available (Python 3.14+ has syntax errors)
if sys.version_info >= (3, 14):
    PYMUST_AVAILABLE = False
else:
    try:
        from pymust import getparam

        PYMUST_AVAILABLE = True
    except ImportError:
        PYMUST_AVAILABLE = False

requires_pymust = pytest.mark.skipif(not PYMUST_AVAILABLE, reason="PyMUST not available")


@requires_pymust
class TestPulseSpectrumMatchesPyMUST:
    """Compare pulse_spectrum_fn against PyMUST Param.getPulseSpectrumFunction."""

    @pytest.mark.parametrize("probe_name", ["P4-2v", "L11-5v", "C5-2v"])
    def test_pulse_spectrum_matches(self, probe_name):
        """Pulse spectrum should match PyMUST for standard probes."""
        param = getparam(probe_name)

        # PyMUST reference (no chirp)
        pymust_fn = param.getPulseSpectrumFunction(None)

        # FastSIMUS implementation
        # PyMUST bandwidth is in %, FastSIMUS uses fraction - but pulse spectrum
        # only depends on fc and TXnow, not bandwidth
        our_fn = pulse_spectrum_fn(param.fc, tx_n_wavelengths=1.0)  # type: ignore[arg-type]

        # Evaluate at a range of angular frequencies
        f = np.linspace(0, 2 * param.fc, 500)  # type: ignore[operator]
        w = 2 * np.pi * f

        pymust_result = pymust_fn(w)
        our_result = our_fn(w)

        np.testing.assert_allclose(our_result, pymust_result, rtol=1e-10, atol=1e-14)  # type: ignore[arg-type]

    def test_pulse_spectrum_txnow_2(self):
        """Pulse spectrum with TXnow=2 should match PyMUST."""
        param = getparam("P4-2v")
        param.TXnow = 2

        pymust_fn = param.getPulseSpectrumFunction(None)
        our_fn = pulse_spectrum_fn(param.fc, tx_n_wavelengths=2.0)  # type: ignore[arg-type]

        w = 2 * np.pi * np.linspace(0, 2 * param.fc, 500)  # type: ignore[operator]

        np.testing.assert_allclose(our_fn(w), pymust_fn(w), rtol=1e-10)  # type: ignore[arg-type]


@requires_pymust
class TestProbeSpectrumMatchesPyMUST:
    """Compare probe_spectrum_fn against PyMUST Param.getProbeFunction."""

    @pytest.mark.parametrize("probe_name", ["P4-2v", "L11-5v", "C5-2v"])
    def test_probe_spectrum_matches(self, probe_name):
        """Probe spectrum should match PyMUST for standard probes."""
        param = getparam(probe_name)

        # PyMUST reference
        pymust_fn = param.getProbeFunction()

        # FastSIMUS: convert bandwidth from % to fraction
        our_fn = probe_spectrum_fn(param.fc, bandwidth=param.bandwidth / 100.0)  # type: ignore[arg-type, operator]

        # Evaluate at a range of angular frequencies
        f = np.linspace(0, 2 * param.fc, 500)  # type: ignore[operator]
        w = 2 * np.pi * f

        pymust_result = pymust_fn(w)
        our_result = our_fn(w)

        np.testing.assert_allclose(our_result, pymust_result, rtol=1e-10)  # type: ignore[arg-type]
