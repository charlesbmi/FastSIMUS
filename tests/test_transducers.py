"""Tests for transducer parameter definitions."""

import pytest

from fast_simus import TransducerParams
from fast_simus.transducer_presets import C5_2v, L11_5v, L12_3v, P4_2v


class TestTransducerParams:
    """Tests for TransducerParams class."""

    def test_basic_instantiation(self):
        """Test that TransducerParams can be instantiated with valid parameters."""
        # With width
        _ = TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, width=0.00025)

        # With kerf
        _ = TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, kerf=0.00005)

    def test_width_kerf_missing(self):
        """Test that either width or kerf must be provided."""
        with pytest.raises(ValueError, match="Either width or kerf must be provided"):
            TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64)

    def test_width_kerf_exclusivity(self):
        """Test that width and kerf are mutually exclusive."""
        with pytest.raises(ValueError, match="Cannot specify both width and kerf"):
            TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, width=0.00025, kerf=0.00005)

    def test_width_kerf_computation(self):
        """Test that width and kerf are computed correctly from each other."""
        # Provide width, compute kerf
        params = TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, width=0.00025)
        assert params.element_width == 0.00025
        assert params.kerf_width == pytest.approx(0.00005)

        # Provide kerf, compute width
        params = TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, kerf=0.00005)
        assert params.element_width == pytest.approx(0.00025)
        assert params.kerf_width == 0.00005

    def test_physical_constraints(self):
        """Test custom physical constraint validation."""
        # Width cannot exceed pitch
        with pytest.raises(ValueError, match=r"Element width .* cannot exceed pitch"):
            TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, width=0.0004)

        # Kerf must be less than pitch
        with pytest.raises(ValueError, match=r"Kerf .* must be less than pitch"):
            TransducerParams(freq_center=2.5e6, pitch=0.0003, n_elements=64, kerf=0.0003)


class TestPresets:
    """Tests for preset transducer configurations."""

    def test_all_presets_instantiate(self):
        """Test that all presets create valid TransducerParams instances."""
        presets = [P4_2v(), L11_5v(), L12_3v(), C5_2v()]

        for params in presets:
            assert isinstance(params, TransducerParams)
