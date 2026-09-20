# FastSIMUS scattering explorer

## General context

The scattering explorer is an interactive marimo example for understanding how an ultrasound transmit field propagates,
interacts with point scatterers, and returns to a physical receive array. It extends the transmit-only wavefield explorer
without replacing it and provides a visual bridge between FastSIMUS field calculations and channel-domain SIMUS output.

The example is intended for education, model inspection, and simulator development. It is not a calibrated acoustic
measurement tool or a B-mode reconstruction pipeline. Pressure and RF amplitudes are displayed relative to separate
references because the current model does not include the source drive, electroacoustic sensitivity, or dimensioned
scattering calibration required to claim absolute pressure units.

The implementation deliberately separates reusable simulation and data-preparation code from notebook composition:

- `src/fast_simus/scattering.py` owns the public point-scattering API and result objects.
- `src/fast_simus/_scattering_math.py` owns portable Array API propagation and contraction primitives.
- `examples/_scattering_explorer.py` owns scene, sampling, coordinate, editing, and estimation helpers.
- `examples/_scattering_viewer.py` owns the Python-side anywidget data contract and normalization.
- `examples/_scattering_viewer.js` and `examples/_scattering_viewer.css` own interactive rendering and layout.
- `examples/scattering_explorer.py` composes those pieces as reactive marimo cells.

This separation is the preferred direction for future work: notebook cells should coordinate controls and results, while
logic that is independently testable or useful to another example should live in a focused module.

## Behavioral contract

### Physics

- The model is scalar, two-dimensional, homogeneous-medium, first-order scattering.
- The incident path is array element to observation point. Each scattered path is array element to point scatterer to
  observation point.
- A scatterer source is its incident spectrum multiplied by its reflection coefficient. Scatterer-to-scatterer multiple
  scattering is not modeled.
- Propagation uses FastSIMUS's attenuating cylindrical Green's function, including phase delay, attenuation, and
  `1 / sqrt(r)` spreading.
- Coincident and near-coincident distances use the existing half-wavelength floor.
- Observation-grid samples are ideal omnidirectional points. SIMUS receive channels remain finite physical probe
  elements with probe response and directivity.
- Incident and scattered spectra share a frequency grid and time record. `total` is computed as their sum rather than
  stored as a third simulation.
- Public `reflection_coefficients` remain signed. Negative values represent a phase-inverted point response and are
  supported by the numerical API even though the notebook presents a simpler nonnegative model.
- Numerical code preserves the caller's Array API backend, does not mutate inputs, and does not convert arrays to NumPy
  inside the simulation path.

### Sampling and coordinates

- Physical coordinates are the source of truth. Display-pixel coordinates are conversion-boundary data only.
- Observation positions preserve arbitrary spatial shapes and place frequency or time last.
- Grid spacing defaults to one third of the propagation wavelength. The user may specify spacing in wavelengths or
  millimetres.
- Grid dimensions are `ceil(span / requested_spacing) + 1`. Both ROI endpoints are included and effective spacing never
  exceeds the requested spacing.
- Wavelength uses the actual propagation sound speed and transmit center frequency. Steering does not affect wavelength
  or recreate the grid-spacing control.
- Lateral and axial field axes use the same screen-pixel scale per millimetre. The physical plot is centered and
  letterboxed when its aspect ratio differs from the available canvas.
- The axial ROI may begin at zero so array contact and near-field behavior can be inspected.

### Scenes and editing

- **Single reflector:** one point at 30 mm depth with relative amplitude `0.005`; scene-specific lateral and depth
  controls move it through the field using the normal debounced simulation path.
- **PICMUS point targets:** exactly 20 unique points. Axial points lie at `x = 0`, `z = 10...45 mm` in 5 mm increments;
  lateral rows lie at `z = 20 mm` and `z = 40 mm`, `x = -15...15 mm` in 5 mm increments. Intersections are deduplicated.
- **Speckle lesion:** deterministic seeded Rayleigh amplitudes with mean `0.001`; targets inside the 7 mm-radius lesion
  centered near 35 mm depth are multiplied by `0.2`.
- **Custom:** a canonical table of physical `x_mm`, `z_mm`, and `rc` values drives simulation. Drawdata contains staged
  additions only. Committing a drawing appends converted points and clears the staging canvas.
- Notebook reflectivities must be nonnegative. Invalid table edits retain the last valid canonical table and display an
  inline validation message.
- Draw classes default to `0.001`, `0.002`, `0.005`, and `0.02`. Precise addition, deletion, and coefficient editing take
  place in the table.

### Transmit and receive

- Probe presets provide geometry, bandwidth, and nominal center frequency. Selecting a probe resets center frequency to
  the preset nominal value without changing unrelated controls.
- Supported transmit modes are focused, plane wave, and diverging wave. Only controls relevant to the active mode are
  shown.
- The assumed focusing sound speed affects delay calculation; the actual medium sound speed affects propagation.
- Tukey apodization is continuous from `alpha = 0` (uniform) to `alpha = 1` (Hann), with a live aperture preview. The
  same weights are applied to field, scattering, and SIMUS receive calculations.
- Pulse length defaults to two wavelengths.
- SIMUS RF is computed at four samples per center-frequency cycle and displayed against its own receive-channel peak.

### Display and interaction

- The primary view is a large physical-aspect field canvas with a shallow receive-time panel underneath.
- A cursor over the receive panel selects an absolute wavefield time. Clicking or dragging seeks the field.
- Wavefield frames use a 2x zero-padded inverse FFT in the notebook, producing eight phase-faithful samples per nominal
  center-frequency cycle. The cursor displays only those computed samples; the browser does not interpolate field data.
- The selected instantaneous component may be incident, scattered, or total.
- Instantaneous field components share the peak incident-pressure reference. Receive RF uses its own peak reference.
- Magnitude is mapped to relative dB while red/blue retains waveform polarity. Exact zero and values below the selected
  range are visually neutral.
- The incident RMS field is an optional cyan background with a separate relative-dB range.
- Scatterers use a light-to-dark grayscale proportional to relative reflectivity, avoiding competition with pressure
  polarity colors.
- Probe elements, focus geometry, scatterers, field data, and axes share physical coordinates.
- The displayed RF record ends at the last selectable wavefield time. Its channel-versus-time aspect is based on physical
  probe aperture and propagation distance `c * delta_t`, with practical height bounds for responsive layouts.
- Configuration groups are visible when the desktop sidebar first opens. The whole sidebar remains hideable through
  marimo's standard sidebar toggle.
- Backend, grid, workload, and memory estimates are diagnostic information and appear below the main plots.
- Display-only changes update the viewer without rerunning physics. Physics controls trigger downstream simulation and
  cached calls reuse identical configurations.

## General design ideas

1. **Keep physics reusable.** Public scattering functions and portable numerical primitives must not depend on marimo,
   browser state, or example-specific scene definitions.
2. **Keep the notebook declarative.** Cells should assemble controls, derive immutable values, call simulation functions,
   and display results. Stateful synchronization is reserved for the canonical custom-scene editor.
3. **Use explicit boundaries.** Backend-to-NumPy conversion occurs only when preparing browser buffers. Physical-to-pixel
   conversion occurs only in editing or rendering helpers.
4. **Share behavior, not incidental layout.** Coordinate conversion, grid derivation, apodization, scene generation,
   normalization, and workload estimation are reusable. Sidebar ordering and card styling may remain notebook-specific.
5. **Separate physics from presentation.** Relative-dB clipping, component selection, RMS visibility, and cursor movement
   must not invalidate cached spectra or RF.
6. **Prefer bounded intermediates.** Frequency work and observation-scatterer contractions are chunked so temporary arrays
   do not scale as the full Cartesian product.
7. **Make limitations visible.** Large requested problems are estimated and warned about, but the notebook does not
   silently cap, coarsen, or require confirmation.
8. **Test contracts rather than cell structure.** Tests should target physical invariants, reusable helpers, widget data
   contracts, synchronization sequences, and backend preservation rather than exact cell ordering or private names.

## Feature set

### Public FastSIMUS capability

- `scattering_pfield_spectrum(...) -> ScatteringSpectrumResult`
- `scattering_wavefield(...) -> ScatteringWavefieldResult`
- Incident, scattered, and derived total components
- Shared frequency metadata and time axis
- Arbitrary observation-grid shapes
- Empty-scatterer and zero-reflectivity behavior
- Signed coefficients and linear superposition
- Attenuation and distance regularization
- Chunked Array API execution across supported backends

### Notebook capability

- Four scene workflows: single reflector, PICMUS targets, speckle lesion, and custom editing
- Four FastSIMUS probe presets
- Focused, plane-wave, and diverging transmissions
- Center frequency, pulse length, steering, focusing, Tukey apodization, sound-speed, and attenuation controls
- Wavelength- or millimetre-based endpoint-inclusive sampling
- Physical-aspect time-frame visualization
- Interactive lateral/depth controls for the single-reflector scene
- Incident RMS background
- Signed receive-channel RF heatmap with a physically informed aspect and draggable time cursor
- Probe, scatterer, focus, axes, dB-reference, and reflectivity annotations
- Additive drawdata workflow plus canonical editable table
- Workload and field-movie memory diagnostics
- Deterministic script-mode configuration for headless checks

## Non-goals

- Multiple scattering, nonlinear propagation, heterogeneous media, and three-dimensional fields
- Calibrated pascal-valued pressure or receive voltage
- Treating arbitrary observation points as physical receive elements
- Envelope detection, beamforming, scan conversion, or B-mode image formation
- Numerical compatibility with MUST MKMOVIE or delay-and-sum.com
- Replacing the simpler transmit-only wavefield explorer

## Review and modularization guidance

Reviewers should preserve the contracts above while looking for smaller shared components. High-value review areas are:

- whether field-spectrum and SIMUS paths share the smallest useful Array API contraction without weakening finite-aperture
  SIMUS behavior;
- whether notebook calculations that are pure and testable have leaked into reactive cells;
- whether viewer traits are minimal and display-only updates avoid Python simulation work;
- whether custom editing has one canonical source of truth and avoids stale callback snapshots;
- whether browser buffers and large intermediate arrays have clear memory bounds;
- whether backend-specific synchronization is limited to demonstrated accelerated-kernel requirements;
- whether documentation and tests describe public behavior rather than transient implementation details.

Any refactor should retain PyMUST parity, portable-versus-accelerated SIMUS parity, Array API backend preservation, input
immutability, deterministic scenes, marimo static checking, headless execution, and a rendered wide/narrow layout check.
