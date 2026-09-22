# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anyplotlib>=0.8",
#     "anywidget>=0.11",
#     "drawdata>=0.5",
#     "fastsimus",
#     "marimo>=0.24.0",
#     "cupy-cuda13x[ctk]>=13.0; sys_platform == 'linux' and platform_machine == 'x86_64'",
#     "mlx>=0.31; sys_platform == 'darwin' and platform_machine == 'arm64'",
#     "numpy>=2.0",
# ]
#
# [tool.uv.sources]
# fastsimus = { path = "../", editable = true }
# ///
"""Interactive transmit, scattering, and receive-field explorer."""

# ruff: noqa: B018

import marimo

__generated_with = "0.24.1"
app = marimo.App(width="full", app_title="FastSIMUS scattering explorer")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from _scattering_explorer import (
        apodization_svg,
        append_drawn_points,
        custom_rows_are_dirty,
        drawing_axis_ticks,
        estimate_simulation_workload,
        grid_from_spacing,
        normalize_custom_rows,
        picmus_point_targets,
        snapshot_custom_rows,
        spacing_in_mm,
        speckle_lesion,
        status_text,
        tukey_apodization,
        wavelength_mm,
    )
    from _scattering_simulation import (
        DEFAULT_TIME_OVERSAMPLING,
        PROBE_PRESETS,
        SimulationConfig,
        cached_simulation,
    )
    from _scattering_viewer import (
        ScatteringFigure,
        crop_receive_to_field_window,
        prepare_viewer_data,
        reflectivity_legend,
    )
    from drawdata import ScatterWidget

    import fast_simus as fs


@app.cell(hide_code=True)
def _():
    is_script_mode = mo.app_meta().mode == "script"
    backend = fs.get_backend()
    xp = backend.xp
    backend_note = f"Numerical backend: **{backend.label}**"
    PRESETS = PROBE_PRESETS
    SCENES = ["Single reflector", "PICMUS point targets", "Speckle lesion", "Custom"]
    return PRESETS, SCENES, backend_note, is_script_mode, xp


@app.cell(hide_code=True)
def _(SCENES):
    scene_ui = mo.ui.dropdown(SCENES, value="Single reflector", label="scene", full_width=True)
    probe_ui = mo.ui.dropdown(
        ["P4-2v phased", "L12-3v linear", "C5-2v convex", "L11-5v linear"],
        value="P4-2v phased",
        label="probe",
        full_width=True,
    )
    return probe_ui, scene_ui


@app.cell(hide_code=True)
def _():
    single_x_ui = mo.ui.slider(
        -40.0,
        40.0,
        0.5,
        value=0.0,
        label="target lateral position (mm)",
        include_input=True,
        debounce=True,
        full_width=True,
    )
    single_z_ui = mo.ui.slider(
        0.0,
        100.0,
        0.5,
        value=30.0,
        label="target depth (mm)",
        include_input=True,
        debounce=True,
        full_width=True,
    )
    return single_x_ui, single_z_ui


@app.cell(hide_code=True)
def _():
    transmit_ui = mo.ui.dropdown(
        ["Focused", "Plane wave", "Diverging"], value="Focused", label="transmit", full_width=True
    )
    return (transmit_ui,)


@app.cell(hide_code=True)
def _(PRESETS, probe_ui):
    nominal_frequency_mhz = PRESETS[probe_ui.value]().freq_center / 1e6
    center_frequency_ui = mo.ui.slider(
        1.0,
        15.0,
        0.01,
        value=nominal_frequency_mhz,
        label="center frequency (MHz)",
        include_input=True,
        debounce=True,
        full_width=True,
    )
    return (center_frequency_ui,)


@app.cell(hide_code=True)
def _():
    focus_depth_ui = mo.ui.slider(
        10,
        60,
        1,
        value=30,
        label="focus / source depth (mm)",
        include_input=True,
        debounce=True,
        full_width=True,
    )
    steer_ui = mo.ui.slider(
        -30,
        30,
        1,
        value=0,
        label="steering (deg)",
        include_input=True,
        debounce=True,
        full_width=True,
    )
    width_ui = mo.ui.slider(
        20,
        120,
        5,
        value=70,
        label="diverging width (deg)",
        include_input=True,
        debounce=True,
        full_width=True,
    )
    pulse_ui = mo.ui.slider(
        0.5,
        4.0,
        0.5,
        value=2.0,
        label="pulse length (wavelengths)",
        include_input=True,
        debounce=True,
        full_width=True,
    )
    tukey_roll_ui = mo.ui.slider(
        0.0,
        1.0,
        0.01,
        value=0.0,
        label="Tukey \N{GREEK SMALL LETTER ALPHA}",
        include_input=True,
        debounce=True,
        full_width=True,
    )
    return focus_depth_ui, pulse_ui, steer_ui, tukey_roll_ui, width_ui


@app.cell(hide_code=True)
def _():
    propagation_c_ui = mo.ui.number(
        1400, 1700, 5, value=1540, label="actual speed of sound (m/s)", debounce=True, full_width=True
    )
    focusing_c_ui = mo.ui.number(
        1400, 1700, 5, value=1540, label="assumed focusing speed (m/s)", debounce=True, full_width=True
    )
    attenuation_ui = mo.ui.slider(
        0.0,
        1.0,
        0.05,
        value=0.5,
        label="attenuation (dB/cm/MHz)",
        include_input=True,
        debounce=True,
        full_width=True,
    )
    x_roi_ui = mo.ui.range_slider(
        -40, 40, 1, value=(-20, 20), label="lateral ROI (mm)", show_value=True, debounce=True, full_width=True
    )
    z_roi_ui = mo.ui.range_slider(
        0, 100, 1, value=(0, 55), label="axial ROI (mm)", show_value=True, debounce=True, full_width=True
    )
    grid_unit_ui = mo.ui.dropdown(
        ["Wavelengths (λ)", "Millimetres"], value="Wavelengths (λ)", label="grid spacing unit", full_width=True
    )
    speckle_count_ui = mo.ui.number(0, None, 100, value=800, label="speckle targets", debounce=True, full_width=True)
    return attenuation_ui, focusing_c_ui, grid_unit_ui, propagation_c_ui, speckle_count_ui, x_roi_ui, z_roi_ui


@app.cell(hide_code=True)
def _(center_frequency_ui, grid_unit_ui, propagation_c_ui):
    center_wavelength_mm = wavelength_mm(float(propagation_c_ui.value), float(center_frequency_ui.value))
    if grid_unit_ui.value == "Wavelengths (λ)":
        grid_spacing_ui = mo.ui.slider(
            1.0 / 60.0,
            2.0,
            1.0 / 60.0,
            value=1.0 / 3.0,
            label="grid spacing (λ)",
            include_input=True,
            debounce=True,
            full_width=True,
        )
    else:
        default_spacing_mm = round(center_wavelength_mm / 3.0, 2)
        grid_spacing_ui = mo.ui.slider(
            0.01,
            5.0,
            0.01,
            value=max(0.01, min(5.0, default_spacing_mm)),
            label="grid spacing (mm)",
            include_input=True,
            debounce=True,
            full_width=True,
        )
    return center_wavelength_mm, grid_spacing_ui


@app.cell(hide_code=True)
def _(center_wavelength_mm, grid_spacing_ui, grid_unit_ui):
    requested_spacing_mm = spacing_in_mm(
        float(grid_spacing_ui.value),
        center_wavelength_mm,
        wavelength_units=grid_unit_ui.value == "Wavelengths (λ)",
    )
    return (requested_spacing_mm,)


@app.cell(hide_code=True)
def _():
    component_ui = mo.ui.dropdown(
        ["Incident", "Scattered", "Total"], value="Total", label="field component", full_width=True
    )
    rms_visible_ui = mo.ui.checkbox(value=True, label="incident RMS background")
    waveform_range_ui = mo.ui.slider(
        20, 100, 5, value=60, label="waveform dynamic range (dB)", include_input=True, full_width=True
    )
    rms_range_ui = mo.ui.slider(
        10, 40, 5, value=20, label="RMS dynamic range (dB)", include_input=True, full_width=True
    )
    return component_ui, rms_range_ui, rms_visible_ui, waveform_range_ui


@app.cell(hide_code=True)
def _(PRESETS, probe_ui, tukey_roll_ui):
    preview_weights = tukey_apodization(PRESETS[probe_ui.value]().n_elements, float(tukey_roll_ui.value))
    apodization_preview = mo.Html(apodization_svg(preview_weights))
    return apodization_preview, preview_weights


@app.cell(hide_code=True)
def _():
    initial_custom_rows = [
        {"x_mm": -5.0, "z_mm": 25.0, "rc": 0.005},
        {"x_mm": 5.0, "z_mm": 35.0, "rc": 0.005},
    ]
    get_custom_draft_rows, set_custom_draft_rows = mo.state(snapshot_custom_rows(initial_custom_rows))
    get_custom_applied_rows, set_custom_applied_rows = mo.state(snapshot_custom_rows(initial_custom_rows))
    get_custom_error, set_custom_error = mo.state(None)
    return (
        get_custom_applied_rows,
        get_custom_draft_rows,
        get_custom_error,
        set_custom_applied_rows,
        set_custom_draft_rows,
        set_custom_error,
    )


@app.cell(hide_code=True)
def _(
    apodization_preview,
    attenuation_ui,
    center_frequency_ui,
    component_ui,
    focus_depth_ui,
    focusing_c_ui,
    grid_spacing_ui,
    grid_unit_ui,
    probe_ui,
    propagation_c_ui,
    pulse_ui,
    rms_range_ui,
    rms_visible_ui,
    scene_ui,
    single_x_ui,
    single_z_ui,
    speckle_count_ui,
    steer_ui,
    transmit_ui,
    tukey_roll_ui,
    width_ui,
    waveform_range_ui,
    x_roi_ui,
    z_roi_ui,
):
    if transmit_ui.value == "Focused":
        mode_controls = [focus_depth_ui, steer_ui]
    elif transmit_ui.value == "Plane wave":
        mode_controls = [steer_ui]
    else:
        mode_controls = [steer_ui, width_ui]
    transmit_controls = mo.vstack(
        [
            transmit_ui,
            center_frequency_ui,
            focusing_c_ui,
            *mode_controls,
            pulse_ui,
            mo.md("**apodization**"),
            tukey_roll_ui,
            apodization_preview,
        ],
        gap=0.45,
    )
    medium_controls = mo.vstack([propagation_c_ui, attenuation_ui], gap=0.45)
    sampling_items = [x_roi_ui, z_roi_ui, grid_unit_ui, grid_spacing_ui]
    if scene_ui.value == "Speckle lesion":
        sampling_items.append(speckle_count_ui)
    sampling_controls = mo.vstack(sampling_items, gap=0.45)
    display_controls = mo.vstack(
        [
            component_ui,
            waveform_range_ui,
            rms_visible_ui,
            rms_range_ui,
        ],
        gap=0.45,
    )
    scene_controls = [scene_ui, probe_ui]
    if scene_ui.value == "Single reflector":
        scene_controls.extend([mo.md("### Target"), single_x_ui, single_z_ui])
    sidebar = mo.sidebar(
        mo.vstack(
            [
                mo.md("## Explorer controls"),
                *scene_controls,
                mo.md("### Transmit"),
                transmit_controls,
                mo.md("### Medium"),
                medium_controls,
                mo.md("### Sampling"),
                sampling_controls,
                mo.md("### Display"),
                display_controls,
            ],
            gap=0.65,
        ),
        width="310px",
    )
    sidebar
    return


@app.cell(hide_code=True)
def _(set_custom_draft_rows, set_custom_error, x_roi_ui, z_roi_ui):
    x_min, x_max = x_roi_ui.value
    z_min, z_max = z_roi_ui.value
    custom_extent = (float(x_min), float(x_max), float(z_min), float(z_max))
    class_a_ui = mo.ui.number(0.0, None, 0.001, value=0.001, label="class A reflectivity (default)")
    class_b_ui = mo.ui.number(0.0, None, 0.001, value=0.002, label="class B reflectivity")
    class_c_ui = mo.ui.number(0.0, None, 0.001, value=0.005, label="class C reflectivity")
    class_d_ui = mo.ui.number(0.0, None, 0.001, value=0.02, label="class D reflectivity")
    draw_widget = mo.ui.anywidget(ScatterWidget(data=[], width=640, height=480, brushsize=14, n_classes=4))

    x_ticks, z_ticks = drawing_axis_ticks(custom_extent)
    y_tick_markup = "".join(
        f'<text x="62" y="{min(max(4 + index * 120, 10), 474)}" text-anchor="end">{value:g}</text>'
        f'<line x1="65" y1="{index * 120}" x2="72" y2="{index * 120}" />'
        for index, value in enumerate(z_ticks)
    )
    y_axis = mo.Html(
        f"""
        <svg width="72" height="526" viewBox="0 0 72 526" role="img"
             aria-label="Depth z in millimetres, from {z_ticks[-1]:g} to {z_ticks[0]:g}">
          <g transform="translate(0 42)" fill="currentColor" stroke="currentColor" font-size="12">
            <line x1="71" y1="0" x2="71" y2="480" />
            {y_tick_markup}
            <text transform="rotate(-90)" x="-240" y="13" text-anchor="middle" stroke="none">
              depth z (mm)
            </text>
          </g>
        </svg>
        """
    )
    x_tick_markup = "".join(
        f'<line x1="{index * 160}" y1="0" x2="{index * 160}" y2="7" />'
        f'<text x="{max(4, min(index * 160, 636))}" y="20" '
        f'text-anchor="{"start" if index == 0 else "end" if index == 4 else "middle"}" '
        f'stroke="none">{value:g}</text>'
        for index, value in enumerate(x_ticks)
    )
    x_axis = mo.Html(
        f"""
        <svg width="640" height="44" viewBox="0 0 640 44" role="img"
             aria-label="Lateral x in millimetres, from {x_ticks[0]:g} to {x_ticks[-1]:g}">
          <g fill="currentColor" stroke="currentColor" font-size="12">
            <line x1="0" y1="1" x2="640" y2="1" />
            {x_tick_markup}
            <text x="320" y="40" text-anchor="middle" stroke="none">lateral x (mm)</text>
          </g>
        </svg>
        """
    )
    drawing_pad = mo.hstack(
        [y_axis, mo.vstack([draw_widget, x_axis], align="start", gap=0.0)],
        justify="start",
        align="start",
        gap=0.0,
    )

    def add_drawing(_value):
        class_values = (
            float(class_a_ui.value if class_a_ui.value is not None else 0.001),
            float(class_b_ui.value if class_b_ui.value is not None else 0.002),
            float(class_c_ui.value if class_c_ui.value is not None else 0.005),
            float(class_d_ui.value if class_d_ui.value is not None else 0.02),
        )
        drawing_data = draw_widget.value.get("data", [])
        if not drawing_data:
            return
        if any(value < 0.0 for value in class_values):
            set_custom_error("Custom reflectivity must be nonnegative")
            return
        set_custom_draft_rows(
            lambda rows: append_drawn_points(
                rows,
                drawing_data,
                custom_extent,
                class_values,
                width=640,
                height=480,
            )
        )
        set_custom_error(None)
        draw_widget.data = []

    add_drawing_ui = mo.ui.button(label="Add drawn points to draft", kind="success", on_click=add_drawing)
    return add_drawing_ui, class_a_ui, class_b_ui, class_c_ui, class_d_ui, draw_widget, drawing_pad


@app.cell(hide_code=True)
def _(get_custom_applied_rows, get_custom_draft_rows, get_custom_error):
    custom_draft_rows = snapshot_custom_rows(get_custom_draft_rows())
    custom_applied_rows = snapshot_custom_rows(get_custom_applied_rows())
    custom_is_dirty = custom_rows_are_dirty(custom_draft_rows, custom_applied_rows)
    custom_error = get_custom_error()
    table_data = {
        "x_mm": [row["x_mm"] for row in custom_draft_rows],
        "z_mm": [row["z_mm"] for row in custom_draft_rows],
        "rc": [row["rc"] for row in custom_draft_rows],
    }
    return custom_applied_rows, custom_draft_rows, custom_error, custom_is_dirty, table_data


@app.cell(hide_code=True)
def _(set_custom_draft_rows, set_custom_error, table_data):
    def replace_table(value):
        try:
            normalized = normalize_custom_rows(value)
        except ValueError as error:
            set_custom_error(str(error))
            return
        set_custom_draft_rows(normalized)
        set_custom_error(None)

    table_editor = mo.ui.data_editor(
        table_data,
        label="Draft scatterers",
        editable_columns=["x_mm", "z_mm", "rc"],
        on_change=replace_table,
    )
    return (table_editor,)


@app.cell(hide_code=True)
def _(custom_draft_rows, set_custom_applied_rows, set_custom_error):
    def apply_custom_scene(_value):
        try:
            applied = snapshot_custom_rows(custom_draft_rows)
        except ValueError as error:
            set_custom_error(str(error))
            return
        set_custom_applied_rows(applied)
        set_custom_error(None)

    run_custom_ui = mo.ui.button(label="Run custom simulation", kind="success", on_click=apply_custom_scene)
    return (run_custom_ui,)


@app.cell(hide_code=True)
def _(
    add_drawing_ui,
    class_a_ui,
    class_b_ui,
    class_c_ui,
    class_d_ui,
    custom_error,
    custom_applied_rows,
    custom_draft_rows,
    custom_is_dirty,
    drawing_pad,
    scene_ui,
    table_editor,
    run_custom_ui,
):
    draw_panel = mo.vstack(
        [
            mo.md(
                "Draw staged additions, selecting reflectivity class **A-D** in the canvas. "
                "Class **A** is selected by default. "
                "Undo and Reset affect only the staged drawing."
            ),
            mo.hstack([class_a_ui, class_b_ui, class_c_ui, class_d_ui], widths="equal", gap=0.5),
            drawing_pad,
            mo.hstack(
                [add_drawing_ui, mo.md(f"**{len(custom_draft_rows)} staged points** in the draft table")],
                justify="start",
                gap=1.0,
            ),
        ],
        gap=0.55,
    )
    custom_editor = mo.ui.tabs(
        {
            "Draw additions": draw_panel,
            "Edit draft table": mo.vstack(
                [
                    mo.md("Edit, add, or delete draft physical coordinates and nonnegative relative amplitudes."),
                    mo.callout(custom_error, kind="danger") if custom_error else mo.md(""),
                    table_editor,
                ]
            ),
        }
    )
    custom_status = (
        mo.callout("Changes not yet simulated.", kind="warn")
        if custom_is_dirty
        else mo.callout("Draft matches the currently simulated Custom scene.", kind="success")
    )
    custom_output = (
        mo.vstack(
            [
                custom_editor,
                mo.hstack(
                    [
                        run_custom_ui,
                        mo.md(
                            f"**{len(custom_draft_rows)} staged** · **{len(custom_applied_rows)} currently simulated**"
                        ),
                    ],
                    justify="start",
                    gap=1.0,
                ),
                custom_status,
            ],
            gap=0.55,
        )
        if scene_ui.value == "Custom"
        else None
    )
    return (custom_output,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # FastSIMUS scattering explorer
    """)
    return


@app.cell(hide_code=True)
def _(custom_output):
    custom_output
    return


@app.cell(hide_code=True)
def _(get_custom_applied_rows, is_script_mode, scene_ui, single_x_ui, single_z_ui, speckle_count_ui):
    custom_rows = snapshot_custom_rows(get_custom_applied_rows())
    if scene_ui.value == "Single reflector":
        scatterers_mm = np.asarray([[float(single_x_ui.value), float(single_z_ui.value)]])
        reflection_coefficients = np.asarray([0.005])
    elif scene_ui.value == "PICMUS point targets":
        scatterers_mm, reflection_coefficients = picmus_point_targets()
    elif scene_ui.value == "Speckle lesion":
        speckle_count = 80 if is_script_mode else int(speckle_count_ui.value)
        scatterers_mm, reflection_coefficients = speckle_lesion(speckle_count, seed=2026)
    else:
        scatterers_mm = np.asarray([[row["x_mm"], row["z_mm"]] for row in custom_rows], dtype=float).reshape((-1, 2))
        reflection_coefficients = np.asarray([row["rc"] for row in custom_rows], dtype=float)
    return reflection_coefficients, scatterers_mm


@app.cell(hide_code=True)
def _(
    center_frequency_ui,
    is_script_mode,
    propagation_c_ui,
    pulse_ui,
    reflection_coefficients,
    requested_spacing_mm,
    scatterers_mm,
    x_roi_ui,
    z_roi_ui,
):
    x_limits_mm = (float(x_roi_ui.value[0]), float(x_roi_ui.value[1]))
    z_limits_mm = (float(z_roi_ui.value[0]), float(z_roi_ui.value[1]))
    requested_grid = grid_from_spacing(x_limits_mm, z_limits_mm, requested_spacing_mm)
    nx = 40 if is_script_mode else requested_grid.nx
    nz = 48 if is_script_mode else requested_grid.nz
    time_oversampling = DEFAULT_TIME_OVERSAMPLING
    estimate = estimate_simulation_workload(
        grid_shape=(nx, nz),
        x_limits_mm=x_limits_mm,
        z_limits_mm=z_limits_mm,
        scatterers_mm=scatterers_mm,
        n_scatterers=reflection_coefficients.size,
        propagation_speed=float(propagation_c_ui.value),
        center_frequency_mhz=float(center_frequency_ui.value),
        pulse_wavelengths=float(pulse_ui.value),
        time_oversampling=time_oversampling,
    )
    grid_note = (
        f"Requested spacing {requested_spacing_mm:.3f} mm; effective "
        f"{estimate.effective_dx_mm:.3f} \N{MULTIPLICATION SIGN} {estimate.effective_dz_mm:.3f} mm. "
        f"Estimated {time_oversampling}\N{MULTIPLICATION SIGN} time-sampled field movies: "
        f"{estimate.field_movie_bytes / 2**30:.2f} GiB."
    )
    workload_note = (
        mo.callout(
            f"{grid_note} This configuration evaluates about {estimate.spatial_pairs / 1e6:.1f} million "
            "observation-scatterer pairs per frequency. Computation will continue and may be slow or memory-heavy.",
            kind="warn",
        )
        if estimate.spatial_pairs > 25_000_000 or estimate.field_movie_bytes > 2**30
        else mo.md(
            f"Estimated spatial work: {estimate.spatial_pairs / 1e6:.2f} million pairs per frequency. {grid_note}"
        )
    )
    return estimate, nx, nz, workload_note


@app.cell(hide_code=True)
def _(
    attenuation_ui,
    center_frequency_ui,
    focus_depth_ui,
    focusing_c_ui,
    is_script_mode,
    nx,
    nz,
    probe_ui,
    propagation_c_ui,
    pulse_ui,
    preview_weights,
    reflection_coefficients,
    scatterers_mm,
    steer_ui,
    transmit_ui,
    width_ui,
    x_roi_ui,
    xp,
    z_roi_ui,
):
    config = SimulationConfig(
        probe_name=probe_ui.value,
        transmit_name=transmit_ui.value,
        center_frequency_mhz=float(center_frequency_ui.value),
        apodization=tuple(float(value) for value in preview_weights),
        focus_depth_mm=float(focus_depth_ui.value),
        steering_deg=float(steer_ui.value),
        diverging_width_deg=float(width_ui.value),
        pulse_wavelengths=float(pulse_ui.value),
        propagation_speed=float(propagation_c_ui.value),
        focusing_speed=float(focusing_c_ui.value),
        attenuation=float(attenuation_ui.value),
        x_limits_mm=(float(x_roi_ui.value[0]), float(x_roi_ui.value[1])),
        z_limits_mm=(float(z_roi_ui.value[0]), float(z_roi_ui.value[1])),
        grid_shape=(nx, nz),
        scatterers_mm=tuple((float(row[0]), float(row[1])) for row in scatterers_mm),
        reflection_coefficients=tuple(float(value) for value in reflection_coefficients),
        frequency_step=2.0 if is_script_mode else 0.5,
        time_oversampling=DEFAULT_TIME_OVERSAMPLING,
    )
    with mo.status.spinner(title="Running full-fidelity simulation..."):
        sim = cached_simulation(config, xp)
    return (sim,)


@app.cell(hide_code=True)
def _(sim):
    receive, receive_times = crop_receive_to_field_window(sim.receive, sim.receive_times, sim.times)
    viewer_data = prepare_viewer_data(sim.incident, sim.scattered, sim.incident_rms, receive)
    initial_time = min(sim.times.size - 1, int(0.45 * sim.times.size))
    viewer = ScatteringFigure.from_data(
        viewer_data,
        field_times=sim.times,
        receive_times=receive_times,
        extent=sim.extent_mm,
        elements=sim.elements_mm,
        scatterers=sim.scatterers_mm,
        coefficients=sim.reflection_coefficients,
        propagation_speed=sim.propagation_speed,
        focus=sim.focus_mm,
        time_index=initial_time,
    )
    viewer_ui = mo.ui.anywidget(viewer.figure)
    reflectivity_key = mo.Html(reflectivity_legend(sim.reflection_coefficients))
    return reflectivity_key, viewer, viewer_ui


@app.cell(hide_code=True)
def _(
    component_ui,
    rms_range_ui,
    rms_visible_ui,
    viewer,
    viewer_ui,
    waveform_range_ui,
    reflectivity_key,
):
    viewer.update_display(
        component=component_ui.value,
        waveform_dynamic_range=float(waveform_range_ui.value),
        rms_visible=bool(rms_visible_ui.value),
        rms_dynamic_range=float(rms_range_ui.value),
    )
    mo.vstack([viewer_ui, reflectivity_key], gap=0.2)
    return


@app.cell(hide_code=True)
def _(backend_note, center_wavelength_mm, estimate, reflection_coefficients, sim, workload_note):
    summary = status_text(reflection_coefficients.size, sim.probe.n_elements, sim.incident.shape[:2])
    spacing_summary = (
        f"λ = **{center_wavelength_mm:.3f} mm** · effective grid spacing "
        f"**{estimate.effective_dx_mm:.3f} \N{MULTIPLICATION SIGN} {estimate.effective_dz_mm:.3f} mm**"
    )
    mo.vstack(
        [
            mo.md("### Simulation diagnostics"),
            mo.md(backend_note),
            mo.md(summary),
            mo.md(spacing_summary),
            workload_note,
        ],
        gap=0.15,
    )
    return


if __name__ == "__main__":
    app.run()
