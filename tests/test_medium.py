"""Tests for medium parameter definitions."""

import pytest

from fast_simus import MediumParams


class TestMediumParams:
    """Tests for MediumParams class."""

    def test_default_instantiation(self):
        """Test that MediumParams can be instantiated with defaults."""
        medium = MediumParams()
        assert medium.speed_of_sound == 1540.0
        assert medium.attenuation == 0.0

    def test_custom_values(self):
        """Test that MediumParams can be instantiated with custom values."""
        medium = MediumParams(speed_of_sound=1480.0, attenuation=0.5)
        assert medium.speed_of_sound == 1480.0
        assert medium.attenuation == 0.5

    def test_speed_of_sound_validation(self):
        """Test that speed_of_sound must be positive."""
        with pytest.raises(ValueError):
            MediumParams(speed_of_sound=0.0)
        with pytest.raises(ValueError):
            MediumParams(speed_of_sound=-1540.0)

    def test_attenuation_validation(self):
        """Test that attenuation must be non-negative."""
        with pytest.raises(ValueError):
            MediumParams(attenuation=-0.5)
        # Zero is allowed
        _ = MediumParams(attenuation=0.0)
