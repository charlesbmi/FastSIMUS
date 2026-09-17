# Wavefield explorer

Interactive time slices alongside the RMS pressure field. Pressure is shown in
arbitrary units; relative amplitudes across space and time are meaningful.

```bash
# Linux + NVIDIA
uv run --group plot --group cuda12 marimo edit --no-token examples/wavefield_explorer.py

# macOS (MLX is included in the plot extra on Darwin)
uv run --group plot marimo edit --no-token examples/wavefield_explorer.py

# Sandboxed alternative
uv run marimo edit --sandbox examples/wavefield_explorer.py

# Headless smoke test
uv run --group plot --script examples/wavefield_explorer.py
```

The explorer selects CuPy, MLX, or NumPy automatically. Interactive mode uses a
256 x 256 grid; script mode uses 48 x 48 for a quick smoke test.
