"""Tests for spectrum helper functions.

Compares FastSIMUS spectrum functions against PyMUST reference implementations.
"""

import numpy as np
import pymust
import pytest

from fast_simus.spectrum import probe_spectrum, pulse_spectrum


class TestPulseSpectrumMatchesPyMUST:
    """Compare pulse_spectrum against PyMUST Param.getPulseSpectrumFunction."""

    @pytest.mark.parametrize("probe_name", ["P4-2v", "L11-5v", "C5-2v"])
    @pytest.mark.parametrize("tx_n_wavelengths", [1.0, 2.0])
    def test_pulse_spectrum_matches(self, probe_name, tx_n_wavelengths):
        """Pulse spectrum should match PyMUST for standard probes."""
        pymust_param = pymust.getparam(probe_name)
        pymust_param.TXnow = tx_n_wavelengths

        # PyMUST reference (no chirp)
        pymust_fn = pymust_param.getPulseSpectrumFunction(None)

        # Evaluate at a range of angular frequencies
        freqs = np.linspace(0, 2 * pymust_param.fc, 500)  # type: ignore[operator]
        angular_freqs = 2 * np.pi * freqs

        pymust_result = pymust_fn(angular_freqs)
        # FastSIMUS implementation
        # PyMUST bandwidth is in %, FastSIMUS uses fraction - but pulse spectrum
        # only depends on fc and TXnow, not bandwidth
        our_result = pulse_spectrum(angular_freqs, pymust_param.fc, tx_n_wavelengths=tx_n_wavelengths)

        np.testing.assert_allclose(our_result, pymust_result, rtol=1e-10, atol=1e-14)


class TestProbeSpectrumMatchesPyMUST:
    """Compare probe_spectrum against PyMUST Param.getProbeFunction."""

    @pytest.mark.parametrize("probe_name", ["P4-2v", "L11-5v", "C5-2v"])
    def test_probe_spectrum_matches(self, probe_name):
        """Probe spectrum should match PyMUST for standard probes."""
        pymust_param = pymust.getparam(probe_name)

        # PyMUST reference
        pymust_fn = pymust_param.getProbeFunction()

        # Evaluate at a range of angular frequencies
        freqs = np.linspace(0, 2 * pymust_param.fc, 500)  # type: ignore[operator]
        angular_freqs = 2 * np.pi * freqs

        pymust_result = pymust_fn(angular_freqs)
        # FastSIMUS: convert bandwidth from % to fraction
        our_result = probe_spectrum(angular_freqs, pymust_param.fc, bandwidth=pymust_param.bandwidth / 100.0)  # type: ignore[arg-type, operator]

        np.testing.assert_allclose(our_result, pymust_result, rtol=1e-10)
