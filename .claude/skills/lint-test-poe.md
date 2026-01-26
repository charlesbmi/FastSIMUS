# Lint & Test with Poe the Poet

## Overview

FastSIMUS uses [Poe the Poet](https://poethepoet.natn.io/) as a task runner with `uv` for dependency management. All commands are defined in `pyproject.toml`.

## Quick Reference

```bash
# Install dependencies
uv sync

# Run affected tests only (fast)
uv run poe test

# Run all tests with coverage
uv run poe test-all

# Lint and format
uv run poe lint

# Run benchmarks
uv run poe benchmark
```

## Linting (`poe lint`)

Runs the full lint sequence:

```bash
uv run poe lint
```

This executes (in order):
1. `ruff format src tests` - Format code
2. `ruff check --fix src tests` - Lint with auto-fix
3. `ty check` - Type checking
4. `mdformat README.md docs --wrap 120` - Format markdown
5. `codespell src tests docs README.md` - Spell check
6. `deptry src` - Check for unused/missing dependencies

### Ruff Configuration

From `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 120
fix = true

[tool.ruff.lint]
select = [
    "B",     # flake8-bugbear
    "E",     # pycodestyle errors
    "F",     # pyflakes
    "I",     # isort
    "RUF",   # ruff-specific
    "S",     # flake8-bandit (security)
    "SIM",   # flake8-simplify
    "UP",    # pyupgrade
    "W",     # pycodestyle warnings
]
```

### Type Checking with ty

`ty` is a fast Rust-based type checker:

```bash
uv run ty check          # Check types
uv run ty check --watch  # Watch mode
```

## Testing (`poe test`)

### Fast Tests (Affected Only)

```bash
uv run poe test
```

Uses `pytest-testmon` to only run tests affected by recent changes:
- Tracks file dependencies
- Caches results in `.testmondata`
- Runs in parallel with `-n auto`

### All Tests with Coverage

```bash
uv run poe test-all
```

Runs full test suite with coverage reporting:
- Parallel execution (`-n auto`)
- Coverage XML output for Codecov

### Running Specific Tests

```bash
# Single file
uv run pytest tests/fast_simus/core/test_pfield.py

# Single test
uv run pytest tests/fast_simus/core/test_pfield.py::test_shape -v

# By marker
uv run pytest -m "not slow"      # Skip slow tests
uv run pytest -m slow            # Only slow tests

# By keyword
uv run pytest -k "delays"        # Tests matching "delays"
```

### Test Markers

Define in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "matlab: requires MATLAB reference files",
    "gpu: requires GPU",
]
```

Usage:

```python
import pytest

@pytest.mark.slow
def test_full_simulation():
    """Long-running validation test."""
    ...

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_backend_equivalence(backend):
    """Test across backends."""
    ...
```

## Benchmarks (`poe benchmark`)

```bash
uv run poe benchmark
```

Uses `pytest-benchmark`:
- Auto-saves results to `.benchmarks/`
- Generates histogram comparisons

Example benchmark test:

```python
def test_pfield_performance(benchmark):
    """Benchmark pfield frequency loop."""
    result = benchmark(pfield, x, z, delays, params)
    assert result.shape[0] > 0
```

## Pre-commit Hooks

Install hooks (one-time):

```bash
uv run prek install
```

Run manually:

```bash
uv run prek run --all-files
```

## CI Integration

Tests run automatically on:
- Push to `main`
- Pull requests touching `src/`, `tests/`, `pyproject.toml`

Matrix: Python 3.11-3.14 on Linux, macOS, Windows

## Troubleshooting

### Testmon cache issues

```bash
rm .testmondata*
uv run poe test
```

### Type errors with jaxtyping

Ensure beartype is imported correctly:

```python
from beartype import beartype as typechecker  # Must be named 'typechecker'
from jaxtyping import jaxtyped

@jaxtyped(typechecker=typechecker)  # This exact pattern
def my_func(...):
    ...
```

### Ruff conflicts with formatter

Some lint rules conflict with the formatter. These are pre-disabled in `pyproject.toml` under `[tool.ruff.lint.ignore]`.
