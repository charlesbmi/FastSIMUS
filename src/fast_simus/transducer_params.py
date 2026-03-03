"""Transducer parameter definitions for FastSIMUS."""

from enum import StrEnum
from math import inf
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, model_validator


class BaffleType(StrEnum):
    """Baffle type enumeration for transducer acoustic boundary conditions.

    The baffle property affects the obliquity factor in the directivity of elements.

    Attributes:
        SOFT: Soft baffle (pressure-release boundary), obliquity factor = cos(theta)
        RIGID: Rigid baffle (pressure-doubling boundary), obliquity factor = 1
    """

    SOFT = "soft"
    RIGID = "rigid"


class TransducerParams(BaseModel):
    """Transducer parameters for ultrasound simulation.

    This class encapsulates all physical and geometric parameters needed for
    ultrasound transducer simulation, following SIMUS/MUST conventions.

    Examples:
        >>> # Linear array with element width
        >>> params = TransducerParams(
        ...     freq_center=2.5e6,
        ...     pitch=300e-6,
        ...     n_elements=128,
        ...     width=250e-6
        ... )
        >>> params.element_width
        0.00025
        >>> params.kerf_width
        5e-05
        >>>
        >>> # Convex array with kerf
        >>> params = TransducerParams(
        ...     freq_center=3.5e6,
        ...     pitch=400e-6,
        ...     n_elements=64,
        ...     kerf=50e-6,
        ...     radius=50e-3
        ... )
        >>> params.element_width
        0.00035
    """

    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)

    # === Required Fields ===
    freq_center: float = Field(..., gt=0)
    """Center frequency in Hz. Must be positive."""

    pitch: float = Field(..., gt=0)
    """Element pitch (center-to-center spacing) in meters. Must be positive."""

    n_elements: int = Field(..., gt=0)
    """Number of transducer elements. Must be positive integer."""

    # === Mutually Exclusive: Width or Kerf ===
    width: float | None = Field(None, gt=0)
    """Element width in meters. Mutually exclusive with kerf. Must be positive if provided."""

    kerf: float | None = Field(None, ge=0)
    """Kerf width (gap between elements) in meters. Mutually exclusive with width. Must be non-negative if provided."""

    # === Optional Geometric Fields ===
    height: float = Field(default=inf, gt=0)
    """Element height in meters. Defaults to infinity for 2D simulation."""

    elev_focus: float = Field(default=inf, gt=0)
    """Elevation focus distance in meters. Defaults to infinity (unfocused)."""

    radius: float = Field(default=inf, gt=0)
    """Curvature radius in meters for convex arrays. Defaults to infinity (linear array)."""

    # === Optional Acoustic Fields ===
    bandwidth: float = Field(default=0.75, gt=0, le=2.0)
    """Fractional bandwidth (0.75 = 75%). Must be in (0, 2.0]. Defaults to 0.75."""

    baffle: BaffleType | NonNegativeFloat = Field(default=BaffleType.SOFT)
    """Baffle type or acoustic impedance ratio.

    Input: "soft", "rigid" (case-insensitive), or impedance ratio (float >= 0).
    After validation, becomes BaffleType.SOFT, BaffleType.RIGID, or float.
    Defaults to "soft".
    """

    # === Computed properties ===
    @property
    def element_width(self) -> float:
        """Element width in meters.

        Computed from width (if provided) or pitch - kerf (if kerf provided).

        Returns:
            Element width in meters.

        Raises:
            ValueError: If neither width nor kerf is provided.
        """
        if self.width is not None:
            return self.width
        if self.kerf is not None:
            return self.pitch - self.kerf
        msg = "Either width or kerf must be provided"
        raise ValueError(msg)

    @property
    def kerf_width(self) -> float:
        """Kerf width in meters.

        Computed from kerf (if provided) or pitch - width (if width provided).

        Returns:
            Kerf width in meters.

        Raises:
            ValueError: If neither width nor kerf is provided.
        """
        if self.kerf is not None:
            return self.kerf
        if self.width is not None:
            return self.pitch - self.width
        msg = "Either width or kerf must be provided"
        raise ValueError(msg)

    @model_validator(mode="after")
    def validate_width_kerf(self) -> Self:
        """Validate that exactly one of width or kerf is provided and physically valid.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If validation fails.
        """
        if self.width is None and self.kerf is None:
            msg = "Either width or kerf must be provided"
            raise ValueError(msg)
        if self.width is not None and self.kerf is not None:
            msg = "Cannot specify both width and kerf. Provide only one."
            raise ValueError(msg)

        # Validate physical constraints
        if self.width is not None and self.width > self.pitch:
            msg = f"Element width ({self.width}) cannot exceed pitch ({self.pitch})"
            raise ValueError(msg)
        if self.kerf is not None and self.kerf >= self.pitch:
            msg = f"Kerf ({self.kerf}) must be less than pitch ({self.pitch})"
            raise ValueError(msg)

        return self
