# Google-style Docstrings

## Overview

FastSIMUS uses [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for all public functions, classes, and modules.

## Function Docstring Template

```python
@jaxtyped(typechecker=typechecker)
def simus(
    x: Float[ArrayAPIObj, "n_scatter"],
    z: Float[ArrayAPIObj, "n_scatter"],
    rc: Float[ArrayAPIObj, "n_scatter"],
    delays: Float[ArrayAPIObj, "n_elements"],
    params: TransducerParams,
) -> Float[ArrayAPIObj, "n_samples n_elements"]:
    """Simulate ultrasound RF signals for a linear or convex array.

    Uses frequency-domain time-harmonic analysis with Fraunhofer (far-field)
    approximation for element directivity.

    Args:
        x: X-coordinates of scatterers in meters. Parallel to transducer face.
        z: Z-coordinates of scatterers in meters. Perpendicular to transducer,
            increases with depth. Must have same shape as `x`.
        rc: Reflection coefficients (dimensionless). Amplitude of reflected
            wave relative to incident wave.
        delays: Transmit delays per element in seconds. Use `txdelay()` to
            compute for focused or diverging waves.
        params: Transducer and medium parameters. See `TransducerParams`.

    Returns:
        RF signals received by each element.
        Sampling frequency is `params.fs`.

    Raises:
        ValueError: If `x` and `z` have different shapes.
        ValueError: If `delays` length doesn't match `params.n_elements`.

    Note:
        - Assumes linear wave propagation (no harmonics)
        - Single scattering approximation
        - Default sampling frequency is 4× center frequency

    References:
        Garcia D. SIMUS: an open-source simulator for medical ultrasound
        imaging. Part I: theory & examples. CMPB, 2022;218:106726.
        https://doi.org/10.1016/j.cmpb.2022.106726
    """
```

## Section Reference

### Args

Document each parameter. Type hints in signature, so focus on description:

```python
Args:
    param1: Description of first parameter. Can span multiple lines
        with proper indentation.
    param2: Second parameter. Defaults to None.
    *args: Variable positional arguments.
    **kwargs: Arbitrary keyword arguments.
```

### Returns

Describe the return value(s):

```python
Returns:
    Description of return value. Include shape for arrays.

# For multiple returns (tuple):
Returns:
    Tuple containing:
    - rf: RF signals with shape (n_samples, n_elements).
    - spectrum: Frequency spectrum with shape (n_freq, n_elements).
```

### Raises

List exceptions that may be raised:

```python
Raises:
    ValueError: If input arrays have incompatible shapes.
    TypeError: If `params` is not a TransducerParams instance.
```

### Note

Additional information, caveats, assumptions:

```python
Note:
    This function assumes single-precision (float32) inputs for
    performance. Double precision is supported but slower.
```

### References

Citations and links:

```python
References:
    Garcia D. SIMUS: an open-source simulator. CMPB, 2022.
    https://doi.org/10.1016/j.cmpb.2022.106726
```

## Class Docstring Template

```python
@dataclass(frozen=True)
class TransducerParams:
    """Parameters for ultrasound transducer and medium.

    Immutable dataclass containing all configuration for simulation.
    Use preset constructors like `P4_2v()` for standard transducers.

    Attributes:
        fc: Center frequency in Hz.
        pitch: Element pitch (center-to-center spacing) in meters.
        n_elements: Number of transducer elements.
        width: Element width in meters. Either `width` or `kerf` required.
        kerf: Kerf (gap between elements) in meters.
        height: Element height in meters. Default is inf (2D simulation).
        focus: Elevation focus distance in meters. Default is inf.
        radius: Radius of curvature for convex arrays in meters.
        bandwidth: Fractional bandwidth at -6dB in percent. Default 75.
        c: Speed of sound in m/s. Default 1540 (soft tissue).
        fs: Sampling frequency in Hz. Default is 4× center frequency.

    Example:
        >>> params = TransducerParams(
        ...     fc=2.7e6,
        ...     pitch=0.3e-3,
        ...     n_elements=64,
        ...     width=0.25e-3,
        ... )
        >>> params.fc
        2700000.0
    """

    fc: float
    pitch: float
    # ... etc
```

## Module Docstring

At top of file:

```python
"""Pressure field computation for ultrasound arrays.

This module implements the frequency-domain pressure field calculation
using Fraunhofer (far-field) and Fresnel (paraxial) approximations.


The main function is `pfield()` which computes RMS pressure at specified
points. For time-domain signals, use `simus()` instead.
"""
```

## Type Hints + Docstrings

With jaxtyping, types are in signature. Docstring describes semantics:

```python
@jaxtyped(typechecker=typechecker)
def compute_delays(
    element_positions: Float[ArrayAPIObj, "n_elements"],
    focus_x: float,
    focus_z: float,
    c: float = 1540.0,
) -> Float[ArrayAPIObj, "n_elements"]:
    """Compute transmit delays for focused wave.

    Args:
        element_positions: X-coordinates of element centers in meters.
        focus_x: X-coordinate of focal point in meters.
        focus_z: Z-coordinate (depth) of focal point in meters.
        c: Speed of sound in m/s.

    Returns:
        Transmit delays in seconds. Elements farther from focus have
        smaller delays to synchronize wavefront arrival.
    """
```

## Doctest Validation

Run doctests:

```bash
uv run pytest --doctest-modules src/fast_simus/
```

## Common Mistakes

❌ **Don't repeat type hints in Args:**

```python
Args:
    x (np.ndarray): ...  # Wrong - type is in signature
```

✅ **Do focus on description:**

```python
Args:
    x: X-coordinates of scatterers in meters.
```

❌ **Don't use NumPy-style section headers:**

```python
Parameters
----------  # Wrong style
```

✅ **Do use Google-style:**

```python
Args:
    ...
```
