"""Medium parameter definitions for ultrasound propagation."""

from pydantic import BaseModel, ConfigDict, Field


class MediumParams(BaseModel):
    """Medium parameters for ultrasound propagation.

    This class encapsulates physical properties of the propagation medium
    (e.g., soft tissue, water) that affect ultrasound wave propagation.
    """

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        frozen=True,
    )

    speed_of_sound: float = Field(default=1540.0, gt=0)
    """Speed of sound in m/s."""

    attenuation: float = Field(default=0.0, ge=0)
    """Attenuation coefficient in dB/cm/MHz."""
