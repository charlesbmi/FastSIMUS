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

## 3D incident and scattered pressure

From the repository root:

```bash
uv run --group plot3d marimo edit examples/wavefield_3d_explorer.py
```

Choose the phantom, transmit law and observation resolution, then press **Simulate**. The rotatable scene and linked
orthoslices show incident, scattered or total pressure. Playback controls reuse the in-notebook simulation. See the
[scattered-wavefield guide](../docs/scattered-wavefields.md) for the physical model and public API.
