# FastSIMUS

Array API-compliant ultrasound simulation library with NumPy/JAX/CuPy backends for 50-100x GPU acceleration.

## Code Organization

- src/ layout, organize by algorithm/domain
- Many small files over few large files
- 200-400 lines typical, 800 max per file
- No emojis in code, comments, or documentation
- `uv` package manager

## Key Constraints

- ALL numerical operations via Array API (`xp` namespace) -- see `array-api-typing` skill
- Never mutate input arrays (breaks JAX)
- Validate against PyMUST reference implementation (rtol=1e-4)
- TDD: write tests first -- see `python-unit-testing` skill

## File Structure

```
src/fast_simus/
  __init__.py
  transducer_params.py   # Transducer parameter model
  transducer_presets.py   # Preset transducer configs (P4-2v, L11-5v, etc.)
  medium_params.py        # Medium/tissue parameters
  tx_delay.py             # Transmit delay computation
  utils/
    __init__.py
    _array_api.py         # Array API helpers and types
    geometry.py           # Geometry utilities
tests/
  conftest.py
  test_transducers.py
  test_tx_delay.py
  utils/
    test_array_api.py
```

## Workflow

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- All tests must pass: `poe test`
- Linting and type-checking: `poe lint`
