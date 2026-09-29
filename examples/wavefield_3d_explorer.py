"""Run from the repository root: uv run --group plot3d marimo edit examples/wavefield_3d_explorer.py."""

import marimo

__generated_with = "0.24.1"
app = marimo.App(width="full")


@app.cell
def _():
    import threading

    import marimo as mo
    from _wavefield3d import simulate
    from _wavefield3d_view import ViewerSlot

    return ViewerSlot, mo, simulate, threading


@app.cell
def _(mo):
    mo.md("""
    # 3D wavefield explorer
    Watch incident and single-scattered pressure on three physical orthoslices.
    Every phantom scatterer participates; the observation planes stay fixed during playback.
    Change the scene and press **Simulate**. Display controls reuse the completed result.
    """)
    return


@app.cell
def _(mo):
    controls = (
        mo.md("""
    **Scene** {scene}  {count}  {backend}

    **Transmission** {transmit}  {steer}  {focus}

    **Observation planes** {spacing_factor}
    """)
        .batch(
            **{
                "scene": mo.ui.dropdown(["None", "Point", "Phantom"], value="Phantom", label="Scatterers"),
                "count": mo.ui.number(2, 100000, value=10000, step=1, label="Phantom scatterers"),
                "backend": mo.ui.dropdown(["Auto", "NumPy", "JAX", "MLX", "CuPy"], value="Auto", label="Backend"),
                "transmit": mo.ui.dropdown(
                    ["Plane wave", "Focused", "Diverging"], value="Plane wave", label="Transmit"
                ),
                "steer": mo.ui.slider(-25, 25, value=0, step=1, label="Steering (degrees)"),
                "focus": mo.ui.slider(8, 30, value=18, step=1, label="Focus / virtual-source depth (mm)"),
                "spacing_factor": mo.ui.dropdown(
                    {"Acoustic sampling": 1, "Coarse preview (4x spacing)": 4},
                    value="Coarse preview (4x spacing)",
                    label="Observation grid",
                ),
            }
        )
        .form(submit_button_label="Simulate", show_clear_button=False)
    )
    controls
    return (controls,)


@app.cell
def _(mo, threading):
    get_result, set_result = mo.state(None)
    get_progress, set_progress = mo.state("Choose a scene, then press Simulate.")
    simulation_lock = threading.Lock()
    return get_progress, get_result, set_progress, set_result, simulation_lock


@app.cell
def _(controls, mo, set_progress, set_result, simulate, simulation_lock, threading):
    config = controls.value
    if mo.app_meta().mode == "script":
        config = dict(
            scene="Point", count=2, backend="NumPy", transmit="Plane wave", steer=0, focus=18, smoke=True, side=2
        )
    cancel_event = threading.Event()
    if config is not None:
        set_result(None)
        set_progress("Planning simulation...")

        def _run():
            current = mo.current_thread()

            def cancelled():
                return cancel_event.is_set() or current.should_exit

            def progress(component, done, total):
                if not current.should_exit:
                    if component == "receive RF":
                        set_progress("Computing received RF..." if done == 0 else "Received RF complete.")
                    else:
                        set_progress(f"{component.capitalize()}: {done:,} / {total:,} observation points")

            # Serializes replacement runs; invalidated threads stop at a block boundary.
            with simulation_lock:
                try:
                    result = simulate(config, progress, cancelled)
                    if not cancelled():
                        set_result(result)
                        set_progress("Simulation complete.")
                except InterruptedError:
                    if not current.should_exit:
                        set_progress("Simulation cancelled. Change settings and press Simulate to restart.")
                except Exception as error:
                    if not current.should_exit:
                        set_progress(f"Simulation failed: {error}")

        _worker = mo.Thread(target=_run, daemon=True)
        _worker.start()
        if mo.app_meta().mode == "script":
            _worker.join()
    return (cancel_event,)


@app.cell
def _(cancel_event, mo):
    mo.ui.button(label="Cancel simulation", on_click=lambda _: cancel_event.set())
    return


@app.cell
def _(get_progress, mo):
    mo.md(get_progress())
    return


@app.cell
def _(get_result, mo):
    simulation = get_result()
    mo.stop(simulation is None)
    mo.md(
        f"Computed **{len(simulation.slices.points):,} observation points** and "
        f"**{len(simulation.scatterers):,} scatterers** in **{simulation.seconds:.1f} s**. "
        f"Estimated numerical workspace: {simulation.workspace_bytes / 2**20:.1f} MiB."
    )
    return (simulation,)


@app.cell
def _(ViewerSlot):
    viewer_slot = ViewerSlot()
    return (viewer_slot,)


@app.cell
def _(mo, simulation, viewer_slot):
    viewer = viewer_slot.replace(simulation)
    mo.ui.anywidget(viewer.widget)
    return (viewer,)


@app.cell
def _(mo, simulation):
    time_control = mo.ui.slider(0, len(simulation.times) - 1, value=0, label="Time sample", show_value=True)
    playing = mo.ui.switch(label="Play")
    component = mo.ui.dropdown(["incident", "scattered", "total"], value="incident", label="Pressure")
    magnitude = mo.ui.switch(label="Magnitude")
    gain = mo.ui.slider(1, 20, value=5, label="Display gain", show_value=True)
    geometry = mo.ui.switch(value=True, label="Show probe / scatterers")
    mo.hstack([playing, time_control, component, magnitude, gain, geometry], wrap=True)
    return component, gain, geometry, magnitude, playing, time_control


@app.cell
def _(mo, simulation):
    _nx, _ny = simulation.matrix_shape
    _centers = simulation.elements.reshape(_ny, _nx, 3)
    rf_row = mo.ui.dropdown(
        {f"{i}: y = {_centers[i, 0, 1] * 1000:.2f} mm": i for i in range(_ny)},
        value=f"{(_ny - 1) // 2}: y = {_centers[(_ny - 1) // 2, 0, 1] * 1000:.2f} mm",
        label="RF Y row (X-directed)",
    )
    rf_column = mo.ui.dropdown(
        {f"{i}: x = {_centers[0, i, 0] * 1000:.2f} mm": i for i in range(_nx)},
        value=f"{(_nx - 1) // 2}: x = {_centers[0, (_nx - 1) // 2, 0] * 1000:.2f} mm",
        label="RF X column (Y-directed)",
    )
    mo.hstack([rf_row, rf_column], wrap=True)
    return rf_column, rf_row


@app.cell
def _(rf_column, rf_row, viewer):
    viewer.received.select(rf_row.value, rf_column.value)
    return


@app.cell
def _(component, gain, geometry, magnitude, mo, playing, time_control, viewer):
    viewer.playing = playing.value
    viewer.configure(component.value, magnitude.value, gain.value, geometry.value)
    if not playing.value:
        viewer.update(time_control.value)
    mo.md(
        f"Showing {viewer.displayed_scatterers:,} of {len(viewer.simulation.scatterers):,} scatterer markers. "
        "Each component uses its own fixed peak scale over the full movie; gain is display-only. "
        "Both RF cross-sections share a fixed signed amplitude scale and follow playback. "
        "Lateral ticks are millimeters; vertical ticks are microseconds. "
        "Equal screen distances represent equal travel times (position divided by sound speed). "
        "Pan and zoom are linked between the RF panels."
    )
    return


if __name__ == "__main__":
    app.run()
