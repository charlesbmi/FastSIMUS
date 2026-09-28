# Finite 3D simulation

FastSIMUS models homogeneous linear acoustics, using oriented rectangular elements and independent transmit events. All
lengths are meters, delays are seconds, and frequencies are Hz. Output amplitudes are arbitrary units.

## Describe an aperture

```python
import numpy as np
from fast_simus import Transducer, matrix_aperture, pfield, simus

aperture = matrix_aperture(
    shape=(4, 3), pitch=(0.0003, 0.0003), size=(0.0002, 0.0002),
    xp=np, dtype=np.float64,
)
probe = Transducer(aperture, model="3d", freq_center=2e6)
points = np.array([[0.001, 0.002, 0.02]])
delays = np.zeros(12)
pressure = pfield(points, delays, probe)
result = simus(points, np.ones(1), delays, probe)
```

Matrix channels are x-fastest: `iy * nx + ix`. The default normal is +z, width axis +x, height axis +y.
`RectangularAperture(centers, width_axes, height_axes, sizes)` describes sparse, unequal, tilted or noncoplanar elements
through the same solver. Axes must be orthonormal; `transform_aperture` requires a proper rotation and translation.
Visibility uses each element's local normal, including after transforms.

`focus_delays` supports a focus or diverging virtual source. `plane_wave_delays` takes a unit propagation direction. NaN
delays disable transmission only; receive channels remain active. Other delays must be finite and nonnegative.
Apodization controls relative element strength. Buffers are borrowed: do not mutate geometry while a plan uses it.

## Physics and normalization

Existing `TransducerParams` calls retain the 2D strip model and `1/sqrt(r)` spreading. An explicit
`Transducer(model="3d")` uses finite rectangles with `1/r` spreading and sinc directivity along both local axes. An
in-plane 3D slice therefore differs physically from a 2D calculation. Patch weights sum to one per electrical element;
element area does not automatically multiply its source strength. Direction cosines use actual displacement. Propagation
uses a minimum distance of half the center wavelength; this regularization does not establish near-surface accuracy.

The same complex element response is used on transmit and receive, without conjugating the receive response. Field
spectra are raw frequency samples. Pulsed RMS is `sqrt(df * sum(abs(spectrum)**2))`; CW uses one center-frequency sample
with integration weight one. Receive spectra use the PyMUST3 unscaled inverse-DFT convention. Transient pressure uses
inverse-integral scaling `n_time * df`. RF and transient pressure consequently have different amplitude conventions.

`element_splitting=(nu,nv)` controls rectangular subdivision. Automatic subdivision uses the shortest retained-model
wavelength; finite lenses impose an additional height-phase bound. Subdivision convergence must be checked for demanding
near-field or lens cases. Center-frequency and full-frequency directivity are distinct approximations.

## Plans, timing and workspace

`pfield_precompute` returns `FieldPlan`; `simus_precompute` returns `EchoPlan` for new descriptions. Legacy tuple plans
retain their layouts. New plans bind the description object, medium, array precision/device, shapes, spectral settings
and support bounds. Call `plan.validate_inputs(points, delays)` outside JIT before using changed runtime inputs.
Low-level compute checks static compatibility; compiled callers close over a plan and must satisfy its validated bounds.

RF requires a finite positive pulse and `fs >= 4*fc`. `EchoPlan.sample_times` uses the effective rate `n_fft * df`,
which may slightly differ from the requested rate. Time zero is the individual transmit trigger. CW supports field RMS
and spectrum, but not RF or transient conversion.

`ExecutionOptions(workspace_bytes=...)` bounds estimated live numerical intermediates, excluding inputs, dense outputs,
compiler memory and allocator overhead. New plans expose `estimated_workspace_bytes` and retain their execution option.
The portable calculation tiles points, elements and patches, uses complete coherent transmit sums and a second receive
pass, and applies RF thresholding once per completed event. JAX uses device loops; MLX evaluates tile state to release
lazy graphs. NumPy and CuPy execute the same portable formulas. Native Metal/CUDA kernels remain restricted to supported
legacy 2D calls. An explicit unsupported native request raises; automatic dispatch selects a supported path.

Legacy calls with a budget use portable point blocks and retain a full aperture per point. A budget too small for that
minimum tile is rejected. Pass the same execution option explicitly to legacy precompute and compute.

## Elevation lenses

`ElevationLens(focal_lengths)` is a thin quadratic phase model along each element's local height coordinate. Infinite
focus is valid. The delay is `tau0 - v**2/(2*c*F)`, where `tau0=max(height**2/(8*c*F))` is one common aperture-wide
offset. `plan.lens_reference_delay` exposes this offset. It appears once in emitted fields and twice in echoes. Rigid
transforms leave focal lengths unchanged. This approximation is distinct from PyMUST's Gaussian elevation model.

`transducer_from_params(params, xp=..., dtype=...)` explicitly converts finite-height linear or convex probes,
preserving their existing origin and constructing their optional elevation lens. Infinite height is rejected.

## Sequences and spatial blocks

`TransmitSequence(delays, apodization=None)` stores `(event,element)` arrays. `sequence_precompute` plans against the
largest active delay over every event. `iter_simus_sequence` yields `SequenceEvent(index, result)` without allocating
all event outputs. `simus_sequence` stacks RF as `(event,time,element)` and spectra as `(event,frequency,element)`.
Events are independent; indices do not imply PRF or absolute acquisition time. Both interfaces expose effective timing
and lens delay.

`iter_pfield_spectrum` and `iter_wavefield` accept the original points, delays and common field plan. Each `FieldBlock`
contains flat `[start,stop)` indices and `(point,frequency)` or `(point,time)` values. `wavefield_times(plan)` provides
the shared time axis before any blocks are collected. Dense field outputs preserve every original spatial axis.

From the repository root:

```sh
uv run python examples/matrix_field.py --smoke
uv run python examples/volumetric_rf.py --smoke
```

These examples simulate an off-axis transient slice and a seeded static 3D cloud with multiple transmissions.
Beamforming, heterogeneous media, multiple scattering, moving scenes and native 3D GPU kernels are outside this model.

See the [validation record](3d-validation.md) for reference tests, backend coverage and measured scaling.
