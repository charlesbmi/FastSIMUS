"""Tests for spectrum helper functions.

Compares FastSIMUS spectrum functions against PyMUST reference implementations.
"""

import numpy as np
import pytest

from fast_simus.spectrum import mysinc, probe_spectrum_fn, pulse_spectrum_fn

# PyMUST may not be available
try:
    from pymust import getparam

    PYMUST_AVAILABLE = True
except (SyntaxError, ImportError):
    PYMUST_AVAILABLE = False

requires_pymust = pytest.mark.skipif(
    not PYMUST_AVAILABLE, reason="PyMUST not available"
)


class TestMysinc:
    """Test the unnormalized sinc function."""

    def test_zero_returns_one(self):
        """sinc(0) should be approximately 1."""
        result = mysinc(np.array([0.0]))
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_pi_returns_zero(self):
        """sinc(pi) = sin(pi)/pi should be approximately 0."""
        result = mysinc(np.array([np.pi]))
        np.testing.assert_allclose(result, np.sin(np.pi) / np.pi, atol=1e-10)

    def test_known_values(self):
        """Test against known values of sin(x)/x."""
        x = np.array([0.5, 1.0, 2.0, 3.0])
        expected = np.sin(x) / x
        result = mysinc(x)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_negative_values(self):
        """sinc is an even function: sinc(-x) == sinc(x)."""
        x = np.array([0.5, 1.0, 2.0])
        np.testing.assert_allclose(mysinc(-x), mysinc(x), rtol=1e-14)

    def test_array_input(self):
        """Should handle multi-dimensional arrays."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = mysinc(x)
        assert result.shape == (2, 2)


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
        our_fn = pulse_spectrum_fn(param.fc, tx_n_wavelengths=1.0)

        # Evaluate at a range of angular frequencies
        f = np.linspace(0, 2 * param.fc, 500)
        w = 2 * np.pi * f

        pymust_result = pymust_fn(w)
        our_result = our_fn(w)

        np.testing.assert_allclose(our_result, pymust_result, rtol=1e-10, atol=1e-14)

    def test_pulse_spectrum_txnow_2(self):
        """Pulse spectrum with TXnow=2 should match PyMUST."""
        param = getparam("P4-2v")
        param.TXnow = 2

        pymust_fn = param.getPulseSpectrumFunction(None)
        our_fn = pulse_spectrum_fn(param.fc, tx_n_wavelengths=2.0)

        w = 2 * np.pi * np.linspace(0, 2 * param.fc, 500)

        np.testing.assert_allclose(our_fn(w), pymust_fn(w), rtol=1e-10)


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
        our_fn = probe_spectrum_fn(param.fc, bandwidth=param.bandwidth / 100.0)

        # Evaluate at a range of angular frequencies
        f = np.linspace(0, 2 * param.fc, 500)
        w = 2 * np.pi * f

        pymust_result = pymust_fn(w)
        our_result = our_fn(w)

        np.testing.assert_allclose(our_result, pymust_result, rtol=1e-10)
