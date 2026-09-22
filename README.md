# FastSIMUS

[![PyPI](https://img.shields.io/pypi/v/fastsimus)](https://pypi.org/project/fastsimus/)
[![Check](https://github.com/charlesbmi/FastSIMUS/actions/workflows/check.yml/badge.svg)](https://github.com/charlesbmi/FastSIMUS/actions/workflows/check.yml)
[![Test](https://github.com/charlesbmi/FastSIMUS/actions/workflows/test.yml/badge.svg)](https://github.com/charlesbmi/FastSIMUS/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/charlesbmi/FastSIMUS)](https://github.com/charlesbmi/FastSIMUS/blob/main/LICENSE)

Fast simulator for medical ultrasound, based on SIMUS/MUST

## Backend selection

Choose an available array namespace once, then pass its arrays to the public numerical functions. SIMUS infers the
implementation from its inputs: MLX arrays use the custom Metal kernel, CuPy arrays use the custom CUDA kernel, and JAX
or NumPy arrays use the portable implementation.

```python
import fast_simus as fs

backend = fs.get_backend()
xp = backend.xp

result = fs.simus(xp.asarray(scatterers), xp.asarray(reflection_coefficients), xp.asarray(delays), params)
```

Install the optional runtime for the target platform with `fastsimus[metal]`, `fastsimus[cuda12]`, or
`fastsimus[cuda13]`. Pass `backend="metal"` or `backend="cuda"` to require a custom kernel, or `backend="mlx"` or
`backend="cupy"` to force the portable same-library implementation.

Backend contexts choose where arrays are created; the `simus(..., backend=...)` name controls execution policy. For
example, `get_backend("mlx")` selects MLX arrays, while `simus(..., backend="mlx")` forces portable MLX.

## Development

This project uses [Flox](https://flox.dev/download) for reproducible system dependencies (Python, uv, CUDA runtime on
Linux), [uv](https://docs.astral.sh/uv/) for Python package management, and [Poe the Poet](https://poethepoet.natn.io/)
for running tasks.

### Setup

Install [Flox](https://flox.dev/download), then from the repository root:

```bash
flox activate
```

This creates the uv virtual environment and runs `poe install`.

For uv-only development without Flox, install [uv](https://docs.astral.sh/uv/getting-started/installation/). On macOS
and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows, see the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

For development, you can optionally install the task runner:

```bash
uv tool install poethepoet
```

This puts `poe` in your path; otherwise, run `uv run poe` instead of `poe`.

### Pre-commit Hooks

This project uses [prek](https://prek.j178.dev/) (a faster Rust-based alternative to pre-commit):

```bash
# Install hooks
uv run prek install

# Run manually
uv run prek run --all-files
```

### Available Commands

Run inside `flox activate` (or prefix with `flox activate --`):

- `poe test` - Run tests affected by recent changes
- `poe lint` - Format and lint code
- `poe docs` - Build and serve documentation

Without Flox: `uv run poe test`, `uv run poe lint`, etc.
