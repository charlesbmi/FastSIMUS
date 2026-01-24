# FastSIMUS

[![Release](https://img.shields.io/github/v/release/charlesbmi/FastSIMUS)](https://img.shields.io/github/v/release/charlesbmi/FastSIMUS)
[![Build status](https://img.shields.io/github/actions/workflow/status/charlesbmi/FastSIMUS/ci.yml?branch=main)](https://github.com/charlesbmi/FastSIMUS/actions/workflows/ci.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/charlesbmi/FastSIMUS)](https://img.shields.io/github/license/charlesbmi/FastSIMUS)

Fast simulator for medical ultrasound, based on SIMUS/MUST

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and
[Poe the Poet](https://poethepoet.natn.io/) for running tasks, instead of a Makefile.

### Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). On macOS and Linux, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows, see the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

For development, you can optionally install the task runner:

```bash
uv tool install poethepoet
```

This puts `poe` in your path; otherwise, you will need to run `uv run poe` instead of `poe`.

### Pre-commit Hooks

This project uses [prek](https://prek.j178.dev/) (a faster Rust-based alternative to pre-commit):

```bash
# Install hooks
uv run prek install

# Run manually
uv run prek run --all-files
```

### Available Commands

To see all available tasks: `poe` or `uv run poe`

- `poe test` - Run tests affected by recent changes
- `poe lint` - Format and lint code
- `poe docs` - Build and serve documentation
