# Wavefield explorer

Interactive time slices along the pressure field, alongside RMS pressure.

Run from this directory:

```bash
# Linux + NVIDIA
uv run --group plot --group cuda12 marimo edit --no-token wavefield_explorer.py

# macOS
uv run --group plot marimo edit --no-token wavefield_explorer.py

# Sandboxed alternative
uv run marimo edit --sandbox wavefield_explorer.py

# Headless smoke test
uv run --script wavefield_explorer.py
```
