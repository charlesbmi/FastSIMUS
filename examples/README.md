# Examples

## Wavefield explorer

A two-panel marimo app that puts the propagating pressure field next to the RMS
beam pattern FastSIMUS's `pfield` returns. Time slices use anyplotlib so the
slider only sends the current frame; RMS stays on matplotlib. The story: a
beam pattern is a long-exposure photograph.

Pressure is shown in arbitrary units: MUST documents SIMUS RF as arbitrary
unit, and FastSIMUS has no medium density, so the values are not Pascals.
Relative amplitudes across the field are still physically meaningful.

```bash
# Preferred: project extras, no sandbox
uv run --group plot --group cuda12 marimo edit --no-token examples/wavefield_explorer.py

# Sandboxed alternative (resolves fast_simus from this repo)
uv run marimo edit --sandbox examples/wavefield_explorer.py

# Headless smoke test
uv run --group plot --script examples/wavefield_explorer.py
```

Interactive mode uses a **256 x 256** grid (script/smoke tests stay at 48 px).

Bipolar colormap registration and anyplotlib colorbar ticks live in
`examples/wavefield_plot.py`. If another example needs the same colorplot,
that file is the one to promote next to `fast_simus.utils.display`.
