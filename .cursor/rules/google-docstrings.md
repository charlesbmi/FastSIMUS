# Google-Style Docstrings

## Rule: Do Not Repeat Default Values in Args Section

When writing Google-style docstrings, do NOT repeat default values in the `Args:` section. The function signature is the single source of truth for default values.

### Bad Example

```python
def compute(x: float, threshold: float = 0.5) -> float:
    """Compute something.
    
    Args:
        x: Input value.
        threshold: Threshold value. Defaults to 0.5.
    
    Returns:
        Computed result.
    """
```

### Good Example

```python
def compute(x: float, threshold: float = 0.5) -> float:
    """Compute something.
    
    Args:
        x: Input value.
        threshold: Threshold value.
    
    Returns:
        Computed result.
    """
```

## Rationale

1. **Single Source of Truth**: The function signature already documents default values
2. **Maintainability**: Avoids duplication that can drift out of sync during refactoring
3. **Readability**: Reduces noise in docstrings, keeping them focused on semantics
4. **Convention**: Aligns with Python Enhancement Proposal 257 and Google's internal style guide

## Application

This rule applies to:
- Function and method parameters with default values
- Class `__init__` methods
- Dataclass fields (defaults documented in field definition)

This rule does NOT apply to:
- Return type documentation
- Explanatory notes about when to use default vs. custom values
- Range restrictions or validation rules

## Related

See also: [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
