# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anywidget>=0.11",
#     "drawdata>=0.5",
#     "fastsimus",
#     "marimo>=0.24.0",
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
    import functools
    from math import inf

    import marimo as mo
    import numpy as np
    from _scattering_explorer import (
        apodization_svg,
        append_drawn_points,
        estimate_field_movie_bytes,
        estimate_spatial_pairs,
        grid_from_spacing,
        normalize_custom_rows,
        picmus_point_targets,
        probe_with_center_frequency,
        spacing_in_mm,
        speckle_lesion,
        status_text,
        tukey_apodization,
        wavelength_mm,
    )
    from _scattering_viewer import ScatteringViewer, crop_receive_to_field_window, prepare_viewer_data
    from drawdata import ScatterWidget

    import fast_simus as fs
    from fast_simus.transducer_presets import C5_2v, L11_5v, L12_3v, P4_2v
    from fast_simus.utils import as_numpy

    TIME_OVERSAMPLING = 2


@app.cell(hide_code=True)
def _():
    is_script_mode = mo.app_meta().mode == "script"
    xp = fs.default_namespace()
    backend_note = f"Numerical backend: **{xp.__name__}**"
    PRESETS = {
        "P4-2v phased": P4_2v,
        "L12-3v linear": L12_3v,
        "C5-2v convex": C5_2v,
        "L11-5v linear": L11_5v,
    }
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
    phase_range_ui = mo.ui.slider(
        20, 100, 5, value=60, label="waveform dynamic range (dB)", include_input=True, full_width=True
    )
    rms_range_ui = mo.ui.slider(
        10, 40, 5, value=20, label="RMS dynamic range (dB)", include_input=True, full_width=True
    )
    return component_ui, phase_range_ui, rms_range_ui, rms_visible_ui


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
    get_custom_rows, set_custom_rows = mo.state(initial_custom_rows)
    get_custom_error, set_custom_error = mo.state(None)
    return get_custom_error, get_custom_rows, set_custom_error, set_custom_rows


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
    phase_range_ui,
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
            phase_range_ui,
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
def _(set_custom_error, set_custom_rows, x_roi_ui, z_roi_ui):
    x_min, x_max = x_roi_ui.value
    z_min, z_max = z_roi_ui.value
    custom_extent = (float(x_min), float(x_max), float(z_min), float(z_max))
    class_a_ui = mo.ui.number(0.0, None, 0.001, value=0.001, label="class a")
    class_b_ui = mo.ui.number(0.0, None, 0.001, value=0.002, label="class b")
    class_c_ui = mo.ui.number(0.0, None, 0.001, value=0.005, label="class c (default)")
    class_d_ui = mo.ui.number(0.0, None, 0.001, value=0.02, label="class d")
    draw_widget = mo.ui.anywidget(ScatterWidget(data=[], width=640, height=480, brushsize=14, n_classes=4))

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
        set_custom_rows(
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

    add_drawing_ui = mo.ui.button(label="Add drawn points", kind="success", on_click=add_drawing)
    return add_drawing_ui, class_a_ui, class_b_ui, class_c_ui, class_d_ui, draw_widget


@app.cell(hide_code=True)
def _(get_custom_error, get_custom_rows):
    custom_rows_snapshot = normalize_custom_rows(get_custom_rows())
    custom_error = get_custom_error()
    table_data = {
        "x_mm": [row["x_mm"] for row in custom_rows_snapshot],
        "z_mm": [row["z_mm"] for row in custom_rows_snapshot],
        "rc": [row["rc"] for row in custom_rows_snapshot],
    }
    return custom_error, custom_rows_snapshot, table_data


@app.cell(hide_code=True)
def _(set_custom_error, set_custom_rows, table_data):
    def replace_table(value):
        try:
            normalized = normalize_custom_rows(value)
        except ValueError as error:
            set_custom_error(str(error))
            return
        set_custom_rows(normalized)
        set_custom_error(None)

    table_editor = mo.ui.data_editor(
        table_data,
        label="Final scatterers",
        editable_columns=["x_mm", "z_mm", "rc"],
        on_change=replace_table,
    )
    return (table_editor,)


@app.cell(hide_code=True)
def _(
    add_drawing_ui,
    class_a_ui,
    class_b_ui,
    class_c_ui,
    class_d_ui,
    custom_error,
    custom_rows_snapshot,
    draw_widget,
    scene_ui,
    table_editor,
):
    draw_panel = mo.vstack(
        [
            mo.md(
                "Draw staged additions, selecting class **a-d** in the canvas. "
                "Undo and Reset affect only the staged drawing."
            ),
            mo.hstack([class_a_ui, class_b_ui, class_c_ui, class_d_ui], widths="equal", gap=0.5),
            draw_widget,
            mo.hstack(
                [add_drawing_ui, mo.md(f"**{len(custom_rows_snapshot)} points** currently in the final table")],
                justify="start",
                gap=1.0,
            ),
        ],
        gap=0.55,
    )
    custom_editor = mo.ui.tabs(
        {
            "Draw additions": draw_panel,
            "Edit final table": mo.vstack(
                [
                    mo.md("Edit, add, or delete final physical coordinates and nonnegative relative amplitudes."),
                    mo.callout(custom_error, kind="danger") if custom_error else mo.md(""),
                    table_editor,
                ]
            ),
        }
    )
    custom_output = custom_editor if scene_ui.value == "Custom" else None
    return (custom_output,)


@app.cell(hide_code=True)
def _(get_custom_rows, is_script_mode, scene_ui, single_x_ui, single_z_ui, speckle_count_ui):
    custom_rows = normalize_custom_rows(get_custom_rows())
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
    effective_dx_mm = (x_limits_mm[1] - x_limits_mm[0]) / (nx - 1)
    effective_dz_mm = (z_limits_mm[1] - z_limits_mm[0]) / (nz - 1)
    estimated_pairs = estimate_spatial_pairs((nz, nx), reflection_coefficients.size)
    lateral_values = [
        abs(x_limits_mm[0]),
        abs(x_limits_mm[1]),
    ]
    axial_values = [
        abs(z_limits_mm[0]),
        abs(z_limits_mm[1]),
    ]
    if scatterers_mm.size:
        lateral_values.extend(np.abs(scatterers_mm[:, 0]).tolist())
        axial_values.extend(np.abs(scatterers_mm[:, 1]).tolist())
    maximum_range_mm = float(np.hypot(max(lateral_values), max(axial_values)))
    estimated_record_seconds = 2.0 * maximum_range_mm * 1e-3 / float(propagation_c_ui.value) + float(pulse_ui.value) / (
        float(center_frequency_ui.value) * 1e6
    )
    estimated_times = max(1, int(np.ceil(4.0 * float(center_frequency_ui.value) * 1e6 * estimated_record_seconds)))
    estimated_movie_bytes = estimate_field_movie_bytes(
        (nz, nx), estimated_times, temporal_oversampling=TIME_OVERSAMPLING
    )
    grid_note = (
        f"Requested spacing **{requested_spacing_mm:.3f} mm**; effective "
        f"**{effective_dx_mm:.3f} \N{MULTIPLICATION SIGN} {effective_dz_mm:.3f} mm**. "
        f"Estimated {TIME_OVERSAMPLING}\N{MULTIPLICATION SIGN} time-sampled field movies: "
        f"**{estimated_movie_bytes / 2**30:.2f} GiB**."
    )
    workload_note = (
        mo.callout(
            f"{grid_note} This configuration evaluates about **{estimated_pairs / 1e6:.1f} million** "
            "observation-scatterer pairs per frequency. Computation will continue and may be slow or memory-heavy.",
            kind="warn",
        )
        if estimated_pairs > 25_000_000 or estimated_movie_bytes > 2**30
        else mo.md(f"Estimated spatial work: **{estimated_pairs / 1e6:.2f} million pairs per frequency**. {grid_note}")
    )
    return effective_dx_mm, effective_dz_mm, nx, nz, workload_note


@app.cell(hide_code=True)
def _(PRESETS, xp):
    @functools.lru_cache(maxsize=8)
    def simulate(
        probe_name,
        transmit_name,
        center_frequency_mhz,
        apodization_key,
        focus_depth_mm,
        steer_deg,
        width_deg,
        pulse_wavelengths,
        propagation_speed,
        focusing_speed,
        attenuation,
        x_limits,
        z_limits,
        grid_shape,
        scatterer_key,
        coefficient_key,
        script_mode,
    ):
        params = probe_with_center_frequency(PRESETS[probe_name](), center_frequency_mhz)
        medium = fs.MediumParams(speed_of_sound=propagation_speed, attenuation=attenuation)
        elements, _theta, apex = fs.element_positions(params.n_elements, params.pitch, params.radius, xp)
        steer_rad = np.deg2rad(steer_deg)
        if transmit_name == "Focused":
            focus = xp.asarray([focus_depth_mm * 1e-3 * np.tan(steer_rad), focus_depth_mm * 1e-3])
            delays = fs.focused(elements, focus, speed_of_sound=focusing_speed, radius=params.radius, apex_offset=apex)
        elif transmit_name == "Plane wave":
            focus = None
            delays = fs.plane_wave(
                elements, steer_rad, speed_of_sound=focusing_speed, radius=params.radius, apex_offset=apex
            )
        elif params.radius == inf:
            focus = None
            delays = fs.diverging_wave(
                elements,
                steer_rad,
                np.deg2rad(width_deg),
                aperture_length=(params.n_elements - 1) * params.pitch,
                speed_of_sound=focusing_speed,
            )
        else:
            virtual_depth = -focus_depth_mm * 1e-3
            focus = xp.asarray([abs(virtual_depth) * np.tan(steer_rad), virtual_depth])
            delays = fs.focused(elements, focus, speed_of_sound=focusing_speed, radius=params.radius, apex_offset=apex)

        apodization = xp.asarray(apodization_key)
        x_axis = xp.linspace(x_limits[0] * 1e-3, x_limits[1] * 1e-3, grid_shape[0])
        z_axis = xp.linspace(z_limits[0] * 1e-3, z_limits[1] * 1e-3, grid_shape[1])
        x_grid, z_grid = xp.meshgrid(x_axis, z_axis)
        positions = xp.stack([x_grid, z_grid], axis=-1)
        scatterers = xp.asarray(scatterer_key)
        coefficients = xp.asarray(coefficient_key)
        frequency_step = 2.0 if script_mode else 0.5
        spectrum = fs.scattering_pfield_spectrum(
            positions,
            scatterers,
            coefficients,
            delays,
            params,
            medium,
            tx_apodization=apodization,
            tx_n_wavelengths=pulse_wavelengths,
            frequency_step=frequency_step,
        )
        incident_movie = fs.spectrum_to_wavefield(spectrum.incident, spectrum.info, time_oversampling=TIME_OVERSAMPLING)
        scattered_movie = fs.spectrum_to_wavefield(
            spectrum.scattered, spectrum.info, time_oversampling=TIME_OVERSAMPLING
        )
        incident_rms = fs.rms_from_spectrum(spectrum.incident, spectrum.info)

        sampling_frequency = 4.0 * params.freq_center
        if coefficients.shape[0] == 0:
            rf = xp.zeros((1, params.n_elements))
            rf_times = xp.zeros(1)
        else:
            rf_result = fs.simus(
                scatterers,
                coefficients,
                delays,
                params,
                medium,
                fs=sampling_frequency,
                tx_apodization=apodization,
                tx_n_wavelengths=pulse_wavelengths,
                frequency_step=frequency_step,
            )
            rf = rf_result.rf
            rf_times = xp.arange(rf.shape[0], dtype=rf.dtype) / sampling_frequency

        return {
            "incident": as_numpy(incident_movie.frames),
            "scattered": as_numpy(scattered_movie.frames),
            "incident_rms": as_numpy(incident_rms),
            "times": as_numpy(incident_movie.times),
            "rf": as_numpy(rf),
            "rf_times": as_numpy(rf_times),
            "elements_mm": as_numpy(elements) * 1e3,
            "scatterers_mm": np.asarray(scatterer_key) * 1e3,
            "rc": np.asarray(coefficient_key),
            "extent": (x_limits[0], x_limits[1], z_limits[0], z_limits[1]),
            "focus_mm": None if focus is None else as_numpy(focus) * 1e3,
            "probe": params,
            "propagation_speed": propagation_speed,
        }

    return (simulate,)


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
    simulate,
    steer_ui,
    transmit_ui,
    width_ui,
    x_roi_ui,
    z_roi_ui,
):
    scatterer_key = tuple(tuple(float(value) * 1e-3 for value in row) for row in scatterers_mm)
    coefficient_key = tuple(float(value) for value in reflection_coefficients)
    sim = simulate(
        probe_ui.value,
        transmit_ui.value,
        float(center_frequency_ui.value),
        tuple(float(value) for value in preview_weights),
        float(focus_depth_ui.value),
        float(steer_ui.value),
        float(width_ui.value),
        float(pulse_ui.value),
        float(propagation_c_ui.value),
        float(focusing_c_ui.value),
        float(attenuation_ui.value),
        tuple(map(float, x_roi_ui.value)),
        tuple(map(float, z_roi_ui.value)),
        (nx, nz),
        scatterer_key,
        coefficient_key,
        is_script_mode,
    )
    return (sim,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # FastSIMUS scattering explorer
    """)
    return


@app.cell(hide_code=True)
def _(sim):
    receive, receive_times = crop_receive_to_field_window(sim["rf"], sim["rf_times"], sim["times"])
    viewer_data = prepare_viewer_data(sim["incident"], sim["scattered"], sim["incident_rms"], receive)
    initial_time = min(sim["times"].size - 1, int(0.45 * sim["times"].size))
    viewer = ScatteringViewer.from_data(
        viewer_data,
        field_times=sim["times"],
        receive_times=receive_times,
        extent=sim["extent"],
        elements=sim["elements_mm"],
        scatterers=sim["scatterers_mm"],
        coefficients=sim["rc"],
        propagation_speed=sim["propagation_speed"],
        focus=sim["focus_mm"],
        time_index=initial_time,
    )
    viewer_ui = mo.ui.anywidget(viewer)
    return viewer, viewer_ui


@app.cell(hide_code=True)
def _(
    component_ui,
    phase_range_ui,
    rms_range_ui,
    rms_visible_ui,
    viewer,
    viewer_ui,
):
    viewer.component = component_ui.value
    viewer.rms_visible = bool(rms_visible_ui.value)
    viewer.phase_dynamic_range = float(phase_range_ui.value)
    viewer.rms_dynamic_range = float(rms_range_ui.value)
    viewer_ui
    return


@app.cell(hide_code=True)
def _(custom_output):
    custom_output
    return


@app.cell(hide_code=True)
def _(
    backend_note, center_wavelength_mm, effective_dx_mm, effective_dz_mm, reflection_coefficients, sim, workload_note
):
    summary = status_text(reflection_coefficients.size, sim["probe"].n_elements, sim["incident"].shape[:2])
    spacing_summary = (
        f"λ = **{center_wavelength_mm:.3f} mm** · effective grid spacing "
        f"**{effective_dx_mm:.3f} \N{MULTIPLICATION SIGN} {effective_dz_mm:.3f} mm**"
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
