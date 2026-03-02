# Backend-Specific Tests

Tests in this folder verify features that go beyond the Array API abstraction and
are specific to a particular backend.

The main test suite (`tests/`) uses `array_api_strict` and is backend-agnostic.
Tests here exercise capabilities that are only meaningful or available for a given
backend, such as JAX JIT compilation, CuPy GPU execution, or NumPy-specific behaviour.

Each file is named after the backend it targets (`test_jax.py`, `test_cupy.py`, …).
Tests are skipped automatically when the relevant backend is not installed.
