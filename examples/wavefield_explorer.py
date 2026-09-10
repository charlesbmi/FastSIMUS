# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anyplotlib>=0.7",
#     "fastsimus",
#     "marimo",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
# ]
#
# [tool.uv.sources]
# fastsimus = { path = "../", editable = true }
# ///
"""Time-slice and RMS pressure from FastSIMUS.

uv run --group plot --group cuda12 marimo edit --no-token examples/wavefield_explorer.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Wavefield explorer")


@app.cell(hide_code=True)
def _(backend_note, mo):
    mo.md(f"""
    # Wavefield explorer

    The first plot is one time slice of the pressure field (white at zero).
    **Display in dB** is on by default: amplitude is mapped on a signed log
    scale so weak wavefronts stay visible, and the colorbar is ± dynamic range.
    Turn it off for linear pressure (arbitrary units). Drag **time** to scrub.
    The second plot is RMS pressure from `pfield`.

    {backend_note}
    """)

    return


@app.cell(hide_code=True)
def _(SCENARIOS, mo):
    scenario = mo.ui.dropdown(
        options=list(SCENARIOS),
        value="1. Focus is one instant",
        label="",
    )
    mo.hstack([mo.md("**Scenario**"), scenario], justify="start", gap=1.5)
    return (scenario,)


@app.cell(hide_code=True)
def _(APODIZATIONS, PRESETS, SCENARIOS, mo, scenario):
    preset = SCENARIOS[scenario.value]

    probe_ui = mo.ui.dropdown(list(PRESETS), value=preset["probe"], label="probe")
    transmit_ui = mo.ui.dropdown(["Focused", "Plane wave"], value=preset["transmit"], label="transmit")
    apod_ui = mo.ui.dropdown(list(APODIZATIONS), value=preset["apod"], label="apodization")
    depth_ui = mo.ui.slider(10, 60, 2, value=preset["depth_mm"], label="focus (mm)", show_value=True)
    steer_ui = mo.ui.slider(-30, 30, 1, value=preset["steer_deg"], label="steer (deg)", show_value=True)
    range_ui = mo.ui.slider(20, 60, 5, value=40, label="dynamic range (dB)", show_value=True)
    db_ui = mo.ui.checkbox(label="display in dB", value=True)

    mo.vstack(
        [
            mo.hstack([probe_ui, transmit_ui, apod_ui], justify="start", gap=1.5),
            mo.hstack([depth_ui, steer_ui, range_ui, db_ui], justify="start", gap=1.5),
        ],
        gap=0.4,
    )

    return apod_ui, db_ui, depth_ui, probe_ui, range_ui, steer_ui, transmit_ui


@app.cell(hide_code=True)
def _(mo):
    time_ui = mo.ui.slider(0, 100, 1, value=45, label="time", show_value=False, full_width=True)
    time_ui
    return (time_ui,)


@app.cell(hide_code=True)
def _(apod_ui, depth_ui, probe_ui, simulate, steer_ui, transmit_ui):
    sim = simulate(probe_ui.value, transmit_ui.value, depth_ui.value, steer_ui.value, apod_ui.value)
    return (sim,)


@app.cell(hide_code=True)
def _(apl_img, db_ui, np, range_ui, signed_db, sim):
    # Signed dB vs the movie peak (default), like delay-and-sum.com.
    # Linear mode keeps the same global peak; white is still zero.
    _dr = float(range_ui.value)
    _peak = float(sim["peak"])
    if db_ui.value:
        pressure = signed_db(sim["frames"], peak=_peak, dynamic_range=_dr)
        clim = (-_dr, _dr)
        apl_img.set_colorbar_label("dB")
    else:
        _floor = _peak * (10.0 ** (-_dr / 20.0))
        pressure = np.where(np.abs(sim["frames"]) < _floor, 0.0, sim["frames"])
        clim = (-_peak, _peak)
        apl_img.set_colorbar_label("a.u.")
    _nz, _nx = pressure.shape[:2]
    _x0, _x1, _z1, _z0 = sim["extent"]
    apl_img.set_extent(np.linspace(_x0, _x1, _nx), np.linspace(_z0, _z1, _nz), units="mm")
    apl_img.set_clim(*clim)

    return clim, pressure


@app.cell(hide_code=True)
def _(apl_widget):
    apl_widget
    return


@app.cell(hide_code=True)
def _(apl_img, clim, pressure, sim, time_ui):
    _index = int(round(time_ui.value / 100.0 * (sim["frames"].shape[-1] - 1)))
    apl_img.set_data(pressure[..., _index], clim=clim)
    apl_img.set_title(f"pressure at t = {sim['times'][_index] * 1e6:.1f} us")

    return


@app.cell(hide_code=True)
def _(range_ui, rms_figure, sim):
    rms_figure(sim, float(range_ui.value))
    return


@app.cell(hide_code=True)
def _(SCENARIOS, mo, scenario, sim):
    mo.md(f"""
    {SCENARIOS[scenario.value]["caption"]}

    {sim["probe_note"]}. Movie spans {sim["times"][0] * 1e6:.1f}-{sim["times"][-1] * 1e6:.1f} us
    ({sim["frames"].shape[-1]} frames).
    """)
    return


@app.cell(hide_code=True)
def _():
    import functools
    import sys
    from pathlib import Path

    import anyplotlib as apl
    import numpy as np
    from matplotlib.figure import Figure

    import fast_simus as fs
    from fast_simus.transducer_presets import C5_2v, L11_5v, L12_3v, P4_2v
    from fast_simus.utils.display import signed_db

    _examples = Path(__file__).resolve().parent
    if str(_examples) not in sys.path:
        sys.path.insert(0, str(_examples))
    import wavefield_plot

    wavefield_plot.register_das_bipolar()
    wavefield_plot.enable_colorbar_ticks()

    return C5_2v, Figure, L11_5v, L12_3v, P4_2v, apl, fs, functools, np, signed_db, wavefield_plot


@app.cell(hide_code=True)
def _(C5_2v, L11_5v, L12_3v, P4_2v, mo, np):
    def detect_backend():
        try:
            import cupy
        except ImportError:
            return np, np.asarray, False, "NumPy (CuPy not installed)"
        try:
            n_devices = int(cupy.cuda.runtime.getDeviceCount())
        except Exception:
            return np, np.asarray, False, "NumPy (CUDA runtime unavailable)"
        if n_devices < 1:
            return np, np.asarray, False, "NumPy (no CUDA device)"
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return cupy, (lambda a: a.get()), True, f"CuPy on {name}"

    is_script_mode = mo.app_meta().mode == "script"
    xp, to_numpy, on_gpu, backend_note = detect_backend()
    if is_script_mode:
        n_pixels = 48
        frequency_step = 1.0
    else:
        n_pixels = 256
        frequency_step = 0.5

    PRESETS = {
        "P4-2v (phased, 0.53 lam)": P4_2v,
        "L12-3v (linear, 0.98 lam)": L12_3v,
        "C5-2v (convex, 1.18 lam)": C5_2v,
        "L11-5v (linear, 1.48 lam)": L11_5v,
    }
    APODIZATIONS = {"none": np.ones, "Hann": np.hanning, "Hamming": np.hamming}
    return (
        APODIZATIONS,
        PRESETS,
        backend_note,
        frequency_step,
        n_pixels,
        to_numpy,
        xp,
    )


@app.cell(hide_code=True)
def _():
    SCENARIOS = {
        "1. Focus is one instant": dict(
            probe="P4-2v (phased, 0.53 lam)",
            transmit="Focused",
            apod="none",
            depth_mm=30,
            steer_deg=0,
            caption=(
                "RMS looks like a narrow beam at 30 mm. In time, a curved wavefront "
                "converges there for about one pulse length, then opens again. The "
                "hourglass is early and late frames stacked; nothing stays focused."
            ),
        ),
        "2. Side lobes from the aperture edges": dict(
            probe="P4-2v (phased, 0.53 lam)",
            transmit="Focused",
            apod="none",
            depth_mm=24,
            steer_deg=0,
            caption=(
                "The faint arms next to the main beam are circular waves launched at "
                "the aperture ends. Switch apodization to Hann: the edge weights taper, "
                "those waves weaken, and the waist gets a bit wider."
            ),
        ),
        "3. Grating lobes are a second wavefront": dict(
            probe="L11-5v (linear, 1.48 lam)",
            transmit="Focused",
            apod="none",
            depth_mm=30,
            steer_deg=20,
            caption=(
                "L11-5v elements sit 1.48 wavelengths apart, so a +20 deg steer also "
                "launches a wavefront the other way. That is the extra arm in RMS. "
                "P4-2v (0.53 wavelength pitch) does not produce it."
            ),
        ),
        "Free exploration": dict(
            probe="P4-2v (phased, 0.53 lam)",
            transmit="Plane wave",
            apod="none",
            depth_mm=30,
            steer_deg=0,
            caption=(
                "A steered plane wave stays flat; its edge waves stay circular. A focus "
                "deeper than the aperture is long stops tightening. C5-2v diverges from "
                "a virtual apex behind the array."
            ),
        ),
    }
    return (SCENARIOS,)


@app.cell(hide_code=True)
def _(
    APODIZATIONS,
    PRESETS,
    frequency_step,
    fs,
    functools,
    n_pixels,
    np,
    to_numpy,
    xp,
):
    @functools.lru_cache(maxsize=32)
    def simulate(probe, transmit, depth_mm, steer_deg, apod):
        """Run one transmit and return arrays both panels read."""
        params = PRESETS[probe]()
        medium = fs.MediumParams()
        c = medium.speed_of_sound
        elements, _, apex = fs.element_positions(params.n_elements, params.pitch, params.radius, xp)

        aperture = params.pitch * (params.n_elements - 1)
        wavelength = c / params.freq_center
        extent = min(2.0 * aperture, 250.0 * wavelength)
        x_axis = np.linspace(-extent / 2, extent / 2, n_pixels, dtype=np.float32)
        z_axis = np.linspace(1e-4, extent, n_pixels, dtype=np.float32)
        grid = xp.asarray(np.stack(np.meshgrid(x_axis, z_axis), axis=-1))

        tilt = float(np.deg2rad(steer_deg))
        focus = None
        if transmit == "Focused":
            focus = (depth_mm * 1e-3 * np.sin(tilt), depth_mm * 1e-3 * np.cos(tilt))
            delays = fs.focused(
                elements,
                xp.asarray(np.asarray(focus, dtype=np.float32)),
                speed_of_sound=c,
                radius=params.radius,
                apex_offset=apex,
            )
        else:
            delays = fs.plane_wave(elements, tilt, speed_of_sound=c, radius=params.radius, apex_offset=apex)

        weights = APODIZATIONS[apod](params.n_elements).astype(np.float32)
        spectrum, info = fs.pfield_spectrum(
            grid,
            delays,
            params,
            medium,
            tx_apodization=xp.asarray(weights),
            frequency_step=frequency_step,
        )
        rms = to_numpy(fs.rms_from_spectrum(spectrum, info))
        movie = fs.spectrum_to_wavefield(spectrum, info)
        frames, times = to_numpy(movie.frames), to_numpy(movie.times)

        envelope = np.max(np.abs(frames), axis=(0, 1))
        live = np.flatnonzero(envelope > 0.01 * (envelope.max() + 1e-12))
        last = int(live[-1]) + 1 if live.size else frames.shape[-1]
        frames, times = frames[..., :last], times[:last]
        frames = frames.astype(np.float32)
        peak = float(np.max(np.abs(frames)))

        return dict(
            frames=frames,
            peak=peak,
            times=times,
            rms_db=20 * np.log10(rms / rms.max() + 1e-12),
            extent=(x_axis[0] * 1e3, x_axis[-1] * 1e3, z_axis[-1] * 1e3, z_axis[0] * 1e3),
            elements=to_numpy(elements) * 1e3,
            focus=None if focus is None else (focus[0] * 1e3, focus[1] * 1e3),
            probe_note=(
                f"{params.n_elements} elements, "
                f"{params.pitch / wavelength:.2f} wavelength pitch, "
                f"{params.freq_center / 1e6:.1f} MHz, "
                f"{n_pixels}x{n_pixels} grid"
            ),
        )

    return (simulate,)


@app.cell(hide_code=True)
def _(Figure, n_pixels):
    PROBE_COLOR, FOCUS_COLOR = "#1D5199", "#FF195E"

    def decorate(ax, sim, title):
        ax.scatter(sim["elements"][:, 0], sim["elements"][:, 1], s=3, c=PROBE_COLOR, zorder=3)
        if sim["focus"] is not None:
            ax.plot(*sim["focus"], "x", color=FOCUS_COLOR, ms=9, mew=2, zorder=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("z (mm)", fontsize=8)
        ax.tick_params(labelsize=7)

    def _panel():
        fig = Figure(figsize=(4.6, 4.4), layout="tight", dpi=90)
        return fig, fig.add_subplot(1, 1, 1)

    def rms_figure(sim, dynamic_range):
        fig, ax = _panel()
        ax.imshow(
            sim["rms_db"],
            cmap="hot",
            vmin=-dynamic_range,
            vmax=0,
            extent=sim["extent"],
            aspect="equal",
            interpolation="nearest",
        )
        decorate(ax, sim, f"RMS pressure, all time ({dynamic_range:.0f} dB)")
        fig.suptitle(f"{n_pixels}x{n_pixels}", fontsize=8)
        return fig

    return (rms_figure,)


@app.cell(hide_code=True)
def viz_widgets(apl, mo, n_pixels, np, wavefield_plot):
    apl_fig, apl_ax = apl.subplots(1, 1, figsize=(480, 400))
    apl_img = apl_ax.imshow(
        np.zeros((n_pixels, n_pixels), np.float32),
        cmap=wavefield_plot.DAS_BIPOLAR,
        vmin=-40,
        vmax=40,
        origin="upper",
    )
    apl_img.set_xlabel("x (mm)")
    apl_img.set_ylabel("z (mm)")
    apl_img.set_colorbar_visible(True)
    apl_img.set_colorbar_label("dB")
    apl_widget = mo.ui.anywidget(apl_fig)

    return apl_img, apl_widget


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
