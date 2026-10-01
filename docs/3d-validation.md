# 3D implementation validation

Verification recorded on 2026-09-28. These measurements cover the portable implementation; they are not native 3D GPU
kernel benchmarks.

## Correctness and compatibility

- Local suite: 343 passed, 42 skipped. NumPy, JAX and MLX execute locally. The backend-specific skips include
  unavailable local CUDA.
- Final CUDA checks: 11 passed, 11 deselected, including field/RF workspace parity and existing native CUDA tests.
- Independent complex128 quadrature checks complex field and receive spectra, local visibility, both directivity modes,
  baffles, attenuation, noncoplanar geometry, channel permutations and lens phase on both propagation legs.
- Raw pinned PyMUST3 field and RF fixtures pass the `1e-4` peak-relative gate without independent amplitude
  normalization. Provenance and matching discretization are recorded in `tests/data/3d/`.
- Three workspace budgets and uneven point/element/patch tails agree. JAX compiled field and echo calls, including
  lenses, pass. Array API strict checks the field path.
- Legacy 2D outputs match 16 saved baseline arrays within the existing tolerance. Four warmed local benchmark cases
  changed by -0.9%, -2.5%, -2.3% and -3.3%; none exceeded the 10% regression gate.
- Ruff, type checking, formatting, spelling, pre-commit and strict documentation build pass. Dependency checking passes
  with the original project dependency declarations. The user's temporarily commented CuPy declarations make the
  ordinary lint command report two missing-CuPy dependency errors; those local edits were preserved.

## CUDA scaling

Hardware: NVIDIA GeForce GTX 1060 with Max-Q Design on `dell`, CuPy 14.0.1. Float32, 2 MHz, 0.3 mm matrix pitch, 0.2 mm
square elements, no lens, zero electronic delays. Seed 2026, uniform points in x/y ±5 mm and z 10–30 mm. One patch per
element. Each compute measurement synchronizes the CUDA stream. Planning is eager and measured separately.

The following field sweep uses one CW frequency and a 4 MiB workspace budget:

| Points  | Elements | Planning (s) | First compute (s) | Warm compute (s) | Estimated workspace (bytes) | Complex spectrum estimate (bytes) | Allocator reserved (bytes) |
| ------- | -------- | ------------ | ----------------- | ---------------- | --------------------------- | --------------------------------- | -------------------------- |
| 1,000   | 64       | 0.317        | 2.194             | 0.0415           | 4,182,016                   | 8,000                             | 567,808                    |
| 10,000  | 64       | 3.039        | 0.413             | 0.4133           | 4,182,016                   | 80,000                            | 928,256                    |
| 100,000 | 64       | 30.064       | 4.083             | 4.1043           | 4,182,016                   | 800,000                           | 4,528,640                  |
| 1,000   | 256      | 1.275        | 0.160             | 0.1607           | 4,194,304                   | 8,000                             | 4,528,640                  |
| 1,000   | 1,024    | 5.009        | 0.636             | 0.6357           | 4,194,304                   | 8,000                             | 4,528,640                  |

Reserved allocator memory includes cached blocks from preceding cases. Process peak RSS was 500,440–500,696 KiB and
includes Python, CUDA and library overhead. Input/output storage and runtime overhead are excluded from the workspace
estimate. The output column reports the plan's complex spectrum estimate; the measured CW RMS output is half that size.
The bounded workspace estimate is conservative for both supported precisions.

The 16×16 / 10K-scatterer RF run used 278 frequencies and a 16 MiB workspace budget. Planning took 12.13 s; first
compute 209.06 s and warm compute 215.73 s. Input storage was 161,024 bytes; declared output storage 1,280,000 bytes;
allocator reserved memory 8,514,560 bytes; process peak RSS 512,548 KiB. This measurement used unit reflectivity.

The 32×32 / 100K-scatterer RF case is an opt-in benchmark and has not been run. The reproducible pytest benchmark is
`tests/benchmarks/bench_3d.py`; it includes both 16×16 / 10K and 32×32 / 100K scenes, seeded reflectivity, dimensions,
frequency/subdivision counts and explicit input/output accounting. Large portable RF runs have substantial Python/GPU
launch overhead; native 3D acceleration remains future work.

## Ownership review

Geometry and local element frames belong to `aperture.py`; lens delay support belongs to `lens.py`. The private
`_propagation.py` owns phase/attenuation exponentials used by both physical models. `_transfer.py` and `_transfer_3d.py`
retain model-specific spreading, masks and normalization. `_contractions.py` owns ordinary, nonconjugated TX/RX
reductions. `_frequency.py` owns canonical grids and pulse support; `_spectral_output.py` owns RF placement and
thresholding. `_blocking.py` owns traversal and backend synchronization. Field, RF, sequence and spatial-block consumers
reuse these calculations. Native 2D kernels retain specialized formulas and capability gates.

Public additions are geometry/transducer/lens descriptions, general delay laws, execution options, field/echo plans,
sequence records and functions, spatial-block iterators and `wavefield_times`. Existing tuple layouts and 2D signatures
remain compatible. Sparse and transformed layouts use the same finite-rectangle solver. No plugin registry or public
solver callback protocol was added.

Planning uses a conservative center-distance plus half-diagonal path bound in bounded point blocks. This avoids
retaining all patch-to-point distances. Frequency tiles contain one frequency; points, elements and patches use static
bounded tiles. This trades conservative time support and planning overhead for compact reusable plans.

## Main integration (2026-10-01)

The integrated branch includes main through `30975d7`, including backend dispatch, scattering, canonical-frequency
precision, and CUDA shared-memory safeguards. SIMUS and sequence calls use main's `backend=` policy: explicit portable
names validate the input namespace, automatic requests permit portable fallback, and explicit native requests reject
finite 3D apertures or workspace budgets. Portable strip pressure and RF share transfer setup; both scattering models
reuse the ordinary receive contraction.

The final local suite passes with 431 tests and 64 skips. Post-integration checks cover namespace validation, 2D
workspace parity, finite 3D backend selection, oversampled wavefield timing, and native precision regressions. The
public-index lockfile validates without changing package versions. The full lint gate, strict documentation build, and
notebook validation pass in a clean environment. Live Marimo checks confirm linked RF pan and zoom, equal travel-time
aspect, and synchronized cursors during scrubbing.

Dell SSH timed out during this integration session. Earlier CUDA results above predate this merge; CUDA regression
checks and the shallow-default GPU timing must be repeated when the machine is reachable.
