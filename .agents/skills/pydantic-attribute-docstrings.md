# Pydantic Attribute Docstrings

## When to Use

Use this skill when:

- Creating or modifying Pydantic models in Python
- Writing dataclasses with field documentation
- Improving code readability and IDE integration
- Following Pythonic documentation patterns

## Overview

Pydantic v2+ supports attribute docstrings, which provide a more Pythonic and IDE-friendly way to document model fields
compared to `Field(description=...)`.

## Core Pattern

### Enable Attribute Docstrings

Add `ConfigDict` to your Pydantic model:

```python
from pydantic import BaseModel, ConfigDict, Field

class MyModel(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    field_name: str
    """This is the field documentation.

    Can be multi-line and include detailed information.
    """
```

### Benefits

1. **IDE Integration**: Better autocomplete and hover documentation
1. **Pythonic**: Follows standard Python docstring conventions
1. **Cleaner**: Separates validation constraints from documentation
1. **Readable**: Documentation appears directly below the field

## Field Documentation Patterns

### Basic Field with Docstring

```python
freq_center: float = Field(..., gt=0)
"""Center frequency in Hz. Must be positive."""
```

### Field with Default and Docstring

```python
bandwidth: float = Field(default=0.75, gt=0, le=2.0)
"""Fractional bandwidth (0.75 = 75%). Must be in (0, 2.0]. Defaults to 0.75."""
```

## Documentation Style Guidelines

### 1. First Line: Brief Description

Start with a concise one-line summary:

```python
speed_of_sound: float = Field(default=1540.0, gt=0)
"""Speed of sound in m/s. Defaults to 1540 m/s (soft tissue)."""
```

### 2. Include Units

Always specify physical units for scientific/engineering fields:

```python
pitch: float = Field(..., gt=0)
"""Element pitch (center-to-center spacing) in meters. Must be positive."""
```

### 3. Document Constraints

Mention validation constraints in plain language:

```python
bandwidth: float = Field(default=0.75, gt=0, le=2.0)
"""Fractional bandwidth (0.75 = 75%). Must be in (0, 2.0]. Defaults to 0.75."""
```

### 4. Explain Defaults

Clarify what default values mean:

```python
height: float = Field(default=inf, gt=0)
"""Element height in meters. Defaults to infinity for 2D simulation."""
```

### 5. Multi-line for Complex Fields

Use multi-line docstrings for fields requiring explanation.

## When Field Description Overrides Docstring

If both `Field(description=...)` and attribute docstring are present, the Field description takes precedence:

```python
x: str
"""Attribute docstring"""

y: int = Field(description="Field description overrides")
"""This will be ignored"""
```

**Best practice**: Use attribute docstrings OR Field descriptions, not both.

## Complete Example

```python
from enum import StrEnum
from math import inf
from pydantic import BaseModel, ConfigDict, Field


class BaffleType(StrEnum):
    """Baffle type enumeration."""
    SOFT = "soft"
    RIGID = "rigid"


class TransducerParams(BaseModel):
    """Transducer parameters for ultrasound simulation."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    # Required fields
    freq_center: float = Field(..., gt=0)
    """Center frequency in Hz. Must be positive."""

    pitch: float = Field(..., gt=0)
    """Element pitch (center-to-center spacing) in meters. Must be positive."""

    n_elements: int = Field(..., gt=0)
    """Number of transducer elements. Must be positive integer."""

    # Optional fields with defaults
    height: float = Field(default=inf, gt=0)
    """Element height in meters. Defaults to infinity for 2D simulation."""

    bandwidth: float = Field(default=0.75, gt=0, le=2.0)
    """Fractional bandwidth (0.75 = 75%). Must be in (0, 2.0]. Defaults to 0.75."""

    speed_of_sound: float = Field(default=1540.0, gt=0)
    """Speed of sound in m/s. Defaults to 1540 m/s (soft tissue)."""
```

## Migration from Field Descriptions

### Before (using Field descriptions)

```python
class MyModel(BaseModel):
    freq_center: float = Field(
        ...,
        gt=0,
        description="Center frequency in Hz. Must be positive.",
    )
    bandwidth: float = Field(
        default=0.75,
        gt=0,
        le=2.0,
        description="Fractional bandwidth (0.75 = 75%). Must be in (0, 2.0].",
    )
```

### After (using attribute docstrings)

```python
class MyModel(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    freq_center: float = Field(..., gt=0)
    """Center frequency in Hz. Must be positive."""

    bandwidth: float = Field(default=0.75, gt=0, le=2.0)
    """Fractional bandwidth (0.75 = 75%). Must be in (0, 2.0]. Defaults to 0.75."""
```

## Key Advantages

1. **Separation of Concerns**: Validation logic in `Field()`, documentation in docstring
1. **Better IDE Support**: Most IDEs show docstrings on hover
1. **Standard Python**: Follows PEP 257 docstring conventions
1. **Cleaner Code**: Less nesting, easier to read
1. **Flexibility**: Can write multi-paragraph documentation when needed

## References

- [Pydantic ConfigDict Documentation](https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.use_attribute_docstrings)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
