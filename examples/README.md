# Interactive field explorers

`wavefield_explorer.py` shows transmit propagation. `scattering_explorer.py` adds multiple point scatterers,
total/scattered fields, and receive-channel RF. Its endpoint-inclusive field grid defaults to one third of the
center wavelength; spacing can instead be entered in millimetres. The center frequency and continuous Tukey
apodization are adjustable, with the same aperture weights used for transmit, scattering, and receive simulation.

Instantaneous pressure and receive RF are shown in relative dB. Red and blue preserve the polarity (phase) of the
band-limited waveform, while exact zero and samples below the display range are neutral. Field components share the
peak incident-pressure reference; receive RF has its own peak reference because finite probe response and ideal field
samples do not share an amplitude calibration. The RMS overlay has a separate dB reference.

Notebook reflectivities are nonnegative relative point-scatterer amplitudes, following the Rayleigh-amplitude
convention used by MUST GENSCAT. They are simulation weights, not calibrated material-interface coefficients. The
public FastSIMUS scattering API remains signed for phase-inverting interfaces and advanced simulations.

Run from this directory:

```bash
# Linux + NVIDIA
uv run --group plot --group cuda12 marimo edit --no-token wavefield_explorer.py

# macOS
uv run --group plot marimo edit --no-token wavefield_explorer.py

# Propagation and scattering
uv run --group plot marimo edit --no-token scattering_explorer.py

# Sandboxed alternative
uv run marimo edit --sandbox wavefield_explorer.py

# Headless smoke test
uv run --script wavefield_explorer.py
uv run --script scattering_explorer.py
```
