# FastSIMUS Product Requirements Document

## 1. Overview

**Product**: FastSIMUS - Fast Simulator for Medical Ultrasound **Version**: 0.1.0 (Initial Release) **License**:
LGPL-2.1 (matching MUST)

### 1.1 Problem Statement

SIMUS (part of MUST - MATLAB UltraSound Toolbox) is the open-source alternative to Field II for academic ultrasound
simulation. Current limitations:

- MATLAB-only: Limits accessibility and integration with ML pipelines
- PyMUST: Direct port, maintains MATLAB idioms, no GPU acceleration
- Performance: parfor MATLAB or Single-threaded NumPy, ~1-5 seconds per simulation

### 1.2 Solution

FastSIMUS provides:

- Clean cross-array Python API following Array API Standard
- 50-100x speedup via JAX/CuPy JIT+GPU acceleration
- Numerical equivalence to MATLAB SIMUS
- Interactive examples with minimal installation

### 1.3 Target Users

1. Ultrasound researchers: Simulation studies, building ULM flow simulators
1. ML practitioners: Training data generation, physics-informed networks
1. Students: Learning ultrasound physics with fast feedback
1. Industry: Rapid prototyping, sequence design optimization

## 2. Features

### 2.1 Core Simulation (P0 - MVP)

| Feature       | Description                | Validation              |
| ------------- | -------------------------- | ----------------------- |
| simus         | 2D RF signal simulation    | Match MATLAB rtol=1e-4  |
| pfield        | Pressure field computation | Match MATLAB rtol=1e-4  |
| Linear arrays | Rectilinear transducers    | P4-2v, L11-5v presets   |
| Convex arrays | Curved transducers         | C5-2v preset            |
| Focused waves | Transmit focusing          | txdelay validation      |
| Plane waves   | Ultrafast imaging          | Multi-angle compounding |
| genscat       | Ultrafast imaging          | Multi-angle compounding |

### 2.2 Advanced Simulation (P1)

| Feature            | Description         | Validation             |
| ------------------ | ------------------- | ---------------------- |
| Diverging waves    | Echocardiography    | Match MUST DW example  |
| Elevation focusing | 3D beam patterns    | MGBM validation        |
| Attenuation        | Frequency-dependent | Example: 0.5 dB/cm/MHz |
| simus3             | Matrix array (3D)   | Match MUST 3D examples |

### 2.3 Signal Processing (P2)

| Feature           | Description                             | Validation                           |
| ----------------- | --------------------------------------- | ------------------------------------ |
| getpulse          | Transmit pulse generation               | Spectrum validation                  |
| txdelay           | Transmit delay computation              | Pattern validation                   |
| txdelay3          | 3-D Transmit delay computation          | Pattern validation                   |
| vbeam integration | cross-platform delay-and-sum beamformer | generate images, optional dependency |

### 2.4 Performance Features (P0)

| Feature       | Description              | Target             |
| ------------- | ------------------------ | ------------------ |
| NumPy backend | CPU baseline             | Reference          |
| JAX backend   | CPU/GPU/TPU acceleration | 50-100x speedup    |
| CuPy backend  | CUDA acceleration        | 50-100x speedup    |
| jax.jit       | JIT compilation          | Eliminate overhead |
| cupy.fuse     | CuPy kernel fusing       | Eliminate overhead |

### 2.5 Secondary performance improvements (P2)

| Feature                 | Description           |
| ----------------------- | --------------------- |
| jax.vmap or jax.lax.map | Vectorized simulation |
| jax.grad                | Auto differentiation  |

### 2.6 Examples & Tutorials (P0-P1)

| Example                          | Description                                           | Inspiration                              | Priority |
| -------------------------------- | ----------------------------------------------------- | ---------------------------------------- | -------- |
| **Pressure field visualization** | Interactive pfield with phased array, show focal zone | MUST pfield examples, SIMUS Part I Fig 9 | P0       |
| **Heart phantom simulation**     | 3-chamber echo view with ~40k scatterers              | SIMUS Part I Fig 12                      | P0       |
| **Rotating disk phantom**        | Synthetic flow phantom for Doppler validation         | PyMUST rotatingDisk examples             | P1       |
| **Point spread function**        | Single scatterer, measure FWHM                        | Validation reference                     | P0       |
| **Marimo WASM notebook**         | Browser-based interactive demo (no install)           | Modern scientific Python                 | P1       |

## 3. Technical Requirements

### 3.1 Array API Compliance

- ALL numerical operations via Array API Standard
- Test compatibility with array-api-strict
- Support NumPy, JAX, CuPy backends
- Preserve input array type in outputs

### 3.2 Type Safety & Runtime Validation

Use `jaxtyping` + `beartype` for:

- Self-documenting array dimensions
- Runtime type/shape checking (catches bugs early)
- Works with all Array API backends

See: `.claude/skills/array-api-typing.md`

### 3.3 Documentation Style

Use **Google-style docstrings** for all public functions:

- Args/Returns/Raises sections
- Include physical units (m, s, Hz)

See: `.claude/skills/googledoc-docstrings.md`

### 3.4 Numerical Precision

| Metric            | Requirement          |
| ----------------- | -------------------- |
| MATLAB agreement  | rtol=1e-4, atol=1e-8 |
| Backend agreement | rtol=1e-4            |
| Float precision   | float32 default      |

### 3.5 Performance Targets

These targets may need to be benchmarked on PyMUST first as a baseline.

| Configuration   | Target Time | Speedup |
| --------------- | ----------- | ------- |
| NumPy CPU       | 1.5s        | 1x      |
| JAX CPU         | 100-150ms   | 10-15x  |
| JAX GPU (A100)  | 15-30ms     | 50-100x |
| CuPy GPU (A100) | 15-30ms     | 50-100x |

Benchmark: 100x100 grid, 128 elements, 555 frequencies 10k scatterers

## 4. Validation Plan

### 4.1 Unit Tests (CI - Every PR)

**Framework:** pytest

**Targets:**

- Array API compliance: 100% public API (test with `array-api-strict`)
- Physical bounds: Delays >= 0, frequencies > 0
- Runtime: < 60s total

See: `.claude/skills/lint-test-poe.md`

### 4.2 Backend Validation (CI - Main Branch)

Backends: NumPy, JAX (CPU), array-api-strict

- Cross-backend equivalence: rtol=1e-5

### 4.3 Validation against MUST/PyMUST (CI - Main branch for long-running or pytest-hypothesis, CI - every PR for faster parameter sets of each test)

Reference: PyMUST, or MUST toolbox output files

| Example              | Metric       |
| -------------------- | ------------ |
| Focused phased array | RF rtol=1e-4 |
| Plane wave linear    | RF rtol=1e-4 |
| Diverging wave echo  | RF rtol=1e-4 |
| B-mode image         | SSIM > 0.95  |

May need small atol for near-zero values (e.g. phase-shift)

### 4.4 Performance Benchmarks (Weekly)

Tool: pytest-benchmark

- pfield frequency loop: NumPy, JAX CPU, JAX GPU (if available), CuPy (if available)
- simus full simulation: 1k, 10k, 100k points
- Batch simulation: 1, 10, 100 configs

## 5. MUST Examples Mapping

### Phase 1 (MVP)

| MUST Example       | Test File              | Priority |
| ------------------ | ---------------------- | -------- |
| Start demo         | test_quickstart.py     | P0       |
| Focus phased array | test_focused_phased.py | P0       |
| Focus linear array | test_focused_linear.py | P0       |
| Plane wave         | test_planewave.py      | P0       |

### Phase 2

| MUST Example         | Test File              | Priority |
| -------------------- | ---------------------- | -------- |
| DWI (diverging)      | test_diverging_wave.py | P1       |
| image reconstruction | test_das_beamform.py   | P1       |
| IQ signals           | test_iq_demod.py       | P1       |

### Phase 3

| MUST Example    | Test File               | Priority |
| --------------- | ----------------------- | -------- |
| Color Doppler   | test_color_doppler.py   | P2       |
| Vector Doppler  | test_vector_doppler.py  | P2       |
| Matrix array 3D | test_matrix_array_3d.py | P2       |

### Phase 4

| MUST Example                                                                                                                                                                                               | File | Priority |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | -------- |
| No-install marimo-WASM example like delay-and-sum.com                                                                                                                                                      | TBD  | P2       |
| ML training data on speckle: generate independent realizations of a scattering profile, like https://ieee-dataport.org/open-access/cnn-based-image-reconstruction-method-ultrafast-ultrasound-imaging-data | TBD  | P2       |

## 6. API Design

### 6.1 Main API

```python
import fast_simus as fs

# Define transducer
params = fs.transducers.P4_2v()

# Compute transmit delays
delays = fs.txdelay(focus_x=0.0, focus_z=0.03, params=params)

# Define scatterers
x = xp.array([0.0, 0.01, -0.01])
z = xp.array([0.02, 0.04, 0.06])
rc = xp.array([1.0, 1.0, 1.0])

# Simulate RF signals
rf = fs.simus(x, z, rc, delays, params)
```

### 6.2 TransducerParams (dataclass)

Required fields:

- fc: Center frequency (Hz)
- pitch: Element pitch (m)
- n_elements: Number of elements
- width or kerf: Element/kerf width (m)

Optional fields:

- height: Element height (m), default=inf
- focus: Elevation focus (m), default=inf
- radius: Curvature radius (m), default=inf
- bandwidth: Fractional bandwidth (%), default=75
- baffle: "soft", "rigid", or float, default="soft"
- c: Speed of sound (m/s), default=1540
- attenuation: dB/cm/MHz, default=0
- fs: Sampling frequency (Hz), default=4\*fc

## 7. Milestones

### M1: Foundation (Week 1-2)

- Project structure and CI
- Array API typing Protocols, tests
- TransducerParams dataclass
- Standard presets (P4-2v, L11-5v, C5-2v)
- txdelay implementation

### M2: Core Simulation (Week 3-4)

- pfield implementation (pure Array API, test with NumPy)
- simus implementation (pure Array API, test with NumPy)
- array-api-strict tests

### M3: JAX Acceleration (Week 5-6)

- JAX pfield frequency loop
- jax.jit compilation
- jax.lax.scan accumulation
- Performance benchmarks
- 50x speedup validation

### M4: Signal Processing (Week 7-8)

- getpulse
- rf2iq
- Full pipeline examples

### M5: Release (Week 9-10)

- API documentation
- Tutorial notebooks
- MUST examples
- PyPI release 0.1.0

## 8. Out of Scope (v0.1)

- Nonlinear propagation
- Inhomogeneous speed of sound
- Real-time visualization
- DICOM/NIFTI I/O
- Multi-GPU distribution

## 9. References

1. Garcia D. SIMUS: an open-source simulator for medical ultrasound imaging. Part I: theory & examples. CMPB,
   2022;218:106726. https://doi.org/10.1016/j.cmpb.2022.106726

1. Cigier A, Varray F, Garcia D. SIMUS Part II: comparison with four simulators. CMPB, 2022;220:106774.
   https://doi.org/10.1016/j.cmpb.2022.106774

1. Array API Standard 2024.12: https://data-apis.org/array-api/2024.12/

1. MUST Toolbox: https://www.biomecardio.com/MUST/

1. ultrasound-metrics (typing reference): https://github.com/Forest-Neurotech/ultrasound-metrics
