# FastSIMUS

[![Release](https://img.shields.io/github/v/release/charlesbmi/FastSIMUS)](https://img.shields.io/github/v/release/charlesbmi/FastSIMUS)
[![Build status](https://img.shields.io/github/actions/workflow/status/charlesbmi/FastSIMUS/ci.yml?branch=main)](https://github.com/charlesbmi/FastSIMUS/actions/workflows/ci.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/charlesbmi/FastSIMUS)](https://img.shields.io/github/license/charlesbmi/FastSIMUS)

Fast simulator for medical ultrasound, based on SIMUS/MUST

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

### CUDA Development on Linux

The Flox environment installs CUDA 12.2 runtime/NVRTC libraries and Nsight Compute. The NVIDIA driver still comes from
the host.

```bash
flox activate -c 'nvidia-smi'
flox activate -c 'python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"'
flox activate -c 'command -v ncu && ncu --version'
```

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
