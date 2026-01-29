# Example Scientific Python CLAUDE.md

This is an example project-level CLAUDE.md file for Python scientific computing projects. Place this in your project root.

## Project Overview

FastSIMUS is an Array API-compliant ultrasound simulation library with NumPy/JAX/CuPy backends for 50-100x GPU acceleration.

## Rules

### 1. Array API Compliance

- ALL numerical operations via Array API Standard
- Use `array_api_compat` and `xp` namespace for backend portability
- Test with array-api-strict
- Preserve input array type in outputs
- See `python-scientific-patterns` skill for detailed patterns

### 2. Type Safety

- jaxtyping for shape documentation
- beartype for runtime validation
- Google-style docstrings with physical units
- Type hints on all public functions

### 3. Code Organization

- src/ layout with clear module structure
- Organize by algorithm/domain
- Many small files over few large files
- High cohesion, low coupling
- 200-400 lines typical, 800 max per file
- No emojis in code, comments, or documentation

### 4. Testing

- TDD: Write tests first
- 80% minimum coverage (95%+ for core algorithms)
- Numerical validation against PyMUST (rtol=1e-4)
- Edge cases: empty, NaN, Inf, large arrays
- Backend compatibility tests (NumPy, JAX, CuPy)
- See `python-scientific-testing` skill for test patterns

## File Structure

```
src/
├── fast_simus/
│   ├── __init__.py
│   ├── py.typed              # PEP 561 marker
│   ├── transducers.py        # Array geometry
│   ├── txdelay.py            # Delay computation
│   ├── pfield.py             # Pressure field
│   ├── simus.py              # RF simulation
│   └── utils/
│       ├── constants.py      # Physical constants
│       └── _array_api.py     # Array API helpers and types
tests/
├── test_transducers.py
├── test_pfield.py
├── conftest.py               # pytest fixtures
└── data/
    └── reference_pfield.zarr # Reference data (Zarr format)
```

## Available Commands

- `/python-test` - Run pytest with TDD workflow
- `/python-review` - Review code quality
- `/python-build` - Fix type/lint errors

## Key Constraints

### Array API Usage

- Use `array_api_compat.array_namespace()` to get `xp` namespace
- Avoid NumPy-specific functions (e.g., `np.asmatrix`, `np.asscalar`)
- Use `xp` namespace for all array operations
- Preserve backend type through function calls

### Data Storage

- Use Zarr format (`.zarr`) for reference data and test fixtures
- Zarr provides better compression and chunked access for large arrays
- See `python-scientific-testing` skill for Zarr loading patterns

### Immutability

- Avoid mutating input arrays (breaks JAX compatibility)
- Create new arrays for all operations
- See `python-scientific-patterns` skill for immutable patterns

## Environment Variables

```bash
# Optional
JAX_ENABLE_X64=true           # 64-bit precision in JAX
CUDA_VISIBLE_DEVICES=0        # GPU selection
DATA_DIR=./data               # Data directory
MAX_ARRAY_MB=1000             # Memory limit
```

## Git Workflow

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- All tests must pass before merge
- Coverage must not decrease
- Type checking must pass (ty check)
- Linting must pass (ruff check)

## Reference Skills

For detailed patterns and examples, see:
- `python-scientific-patterns` - Array API patterns, function signatures, dataclasses
- `python-scientific-testing` - Test patterns, fixtures, Zarr loading
- `coding-standards` - General Python coding standards
