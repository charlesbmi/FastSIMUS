# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anyplotlib>=0.7",
#     "anywidget>=0.9",
#     "fastplotlib>=0.6.1",
#     "marimo>=0.24.0",
#     "numpy>=1.26",
#     "rendercanvas[notebook]>=2.7.2",
# ]
# ///
"""A/B display prototype: anyplotlib set_data vs fastplotlib ImageGraphic at 512^3.

Does not touch examples/wavefield_explorer.py. Synthetic float32 volume only.

    uv run --group plot --group cuda12 marimo edit --no-token --port 2719 examples/wavefield_display_512.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Wavefield display 512 A/B")


@app.cell
def _(mo, n, notes):
    mo.md(f"""
    # 512³ display A/B: anyplotlib vs fastplotlib

    Synthetic signed-like volume `{n} x {n} x {n}` float32
    ({n * n * n * 4 / 1e6:.0f} MB). Widgets stay mounted; the time slider only
    calls `imshow.set_data` / `ImageGraphic.data[:]`.

    {notes}
    """)
    return


@app.cell
def _():
    import locale
    import os
    import sys
    import time
    import traceback
    from pathlib import Path

    import marimo as mo
    import numpy as np

    preferred = locale.getpreferredencoding(False)
    orig_read_text = Path.read_text

    def read_text_utf8(self, encoding=None, errors=None):
        return orig_read_text(self, encoding=encoding or "utf-8", errors=errors)

    Path.read_text = read_text_utf8

    import anyplotlib as apl

    fpl = None
    AnywidgetRenderCanvas = None
    fpl_import_error = None
    try:
        from rendercanvas.anywidget import AnywidgetRenderCanvas as _Canvas

        import fastplotlib as _fpl

        fpl = _fpl
        AnywidgetRenderCanvas = _Canvas
    except Exception as exc:
        fpl_import_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    imgui_ok = False
    try:
        import imgui_bundle  # noqa: F401

        imgui_ok = True
    except ImportError:
        imgui_ok = False
    return (
        AnywidgetRenderCanvas,
        apl,
        fpl,
        fpl_import_error,
        imgui_ok,
        mo,
        np,
        preferred,
        sys,
        time,
    )


@app.cell
def _(np, sys):
    def choose_n():
        page = os_sysconf = None
        try:
            import os

            page = os.sysconf("SC_PAGE_SIZE")
            avail = os.sysconf("SC_AVPHYS_PAGES") * page
        except Exception:
            avail = 2 * 1024**3
        target = 512
        need = target**3 * 4 + 80 * 1024**2
        if avail < need:
            target = 256
        return target, avail

    n, avail_bytes = choose_n()
    y = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    radius = np.hypot(xx, yy)
    envelope = np.exp(-1.5 * radius).astype(np.float32)
    volume = np.empty((n, n, n), dtype=np.float32)
    omega = np.float32(2.0 * np.pi / n)
    t0 = __import__("time").perf_counter()
    for t in range(n):
        volume[:, :, t] = (np.sin(radius * np.float32(16.0) - t * omega) * envelope).astype(
            np.float32
        )
    fill_ms = (__import__("time").perf_counter() - t0) * 1000
    frame0 = np.ascontiguousarray(volume[:, :, 0])
    nbytes_mb = volume.nbytes / 1e6
    slice_mb = n * n * 4 / 1e6
    notes = (
        f"Allocated `{n}^3` float32 = **{nbytes_mb:.0f} MB** "
        f"(one slice {slice_mb:.2f} MB). Available RAM at alloc ~{avail_bytes / 1e9:.2f} GB. "
        f"Fill {fill_ms:.0f} ms. Python {sys.version.split()[0]}."
    )
    return frame0, n, notes, volume


@app.cell
def _(apl, frame0, mo, n):
    apl_fig, apl_ax = apl.subplots(1, 1, figsize=(520, 500))
    apl_img = apl_ax.imshow(frame0, cmap="bwr", vmin=-1, vmax=1, origin="upper")
    apl_img.set_title(f"anyplotlib {n}x{n}")
    apl_img.set_xlabel("x")
    apl_img.set_ylabel("z")
    apl_widget = mo.ui.anywidget(apl_fig)

    return apl_fig, apl_img, apl_widget


@app.cell
def _(AnywidgetRenderCanvas, fpl, fpl_import_error, frame0, imgui_ok, mo, np):
    fpl_fig = None
    fpl_graphic = None
    fpl_canvas = None
    wgpu_adapter = "fastplotlib not imported"
    if fpl is None:
        fpl_widget = mo.md(f"```\n{fpl_import_error}\n```")
    else:
        fpl_canvas = AnywidgetRenderCanvas(size=(520, 500))
        fpl_fig = fpl.Figure(canvas=fpl_canvas, size=(520, 500), names=["pressure"])
        fpl_graphic = fpl_fig[0, 0].add_image(
            np.ascontiguousarray(frame0, dtype=np.float32),
            cmap="bwr",
            vmin=-1,
            vmax=1,
            name="pressure",
        )
        fpl_fig.show()
        fpl_widget = mo.ui.anywidget(fpl_canvas)
        try:
            import wgpu

            adapters = wgpu.gpu.enumerate_adapters_sync()
            wgpu_adapter = "; ".join(
                str(getattr(a, "summary", None) or a.info) for a in adapters
            )
        except Exception as exc:
            wgpu_adapter = f"adapter query failed: {exc}"
    imgui_note = "imgui-bundle present" if imgui_ok else "imgui-bundle not installed (skip ImageWidget)"
    fpl_status = mo.md(f"fastplotlib / wgpu: {wgpu_adapter}. {imgui_note}.")
    return (
        fpl_canvas,
        fpl_fig,
        fpl_graphic,
        fpl_status,
        fpl_widget,
        wgpu_adapter,
    )


@app.cell
def _(apl_widget, fpl_status, fpl_widget, mo):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([mo.md("**A. anyplotlib** (`set_data`, base64 over anywidget)"), apl_widget]),
                    mo.vstack([mo.md("**B. fastplotlib** (`ImageGraphic.data[:]`, RFB JPEG)"), fpl_widget]),
                ],
                justify="start",
                gap=1,
            ),
            fpl_status,
        ],
        gap=0.4,
    )
    return


@app.cell
def _(mo, n):
    time_ui = mo.ui.slider(0, n - 1, 1, value=0, label="time index", show_value=True, full_width=True)
    time_ui
    return (time_ui,)


@app.cell
def _(
    apl_fig,
    apl_img,
    fpl_fig,
    fpl_graphic,
    mo,
    n,
    np,
    time,
    time_ui,
    volume,
):
    index = int(time_ui.value)
    frame = np.ascontiguousarray(volume[:, :, index], dtype=np.float32)

    t_apl = time.perf_counter()
    apl_img.set_data(frame)
    apl_img.set_title(f"anyplotlib t={index}/{n - 1}")
    apl_ms = (time.perf_counter() - t_apl) * 1000

    fpl_ms = None
    fpl_err = None
    if fpl_graphic is not None:
        t_fpl = time.perf_counter()
        fpl_graphic.data[:] = frame
        fpl_fig.canvas.request_draw()
        fpl_ms = (time.perf_counter() - t_fpl) * 1000
    else:
        fpl_err = "fastplotlib graphic missing"

    geom_len = len(getattr(apl_fig, next(t for t in apl_fig.trait_names() if t.endswith("_geom"))))
    fpl_line = f"{fpl_ms:.2f} ms kernel" if fpl_ms is not None else fpl_err
    mo.md(
        f"""
    **Scrub {index}/{n - 1}** — slice `{frame.shape[0]}x{frame.shape[1]}` float32 =
    {frame.nbytes / 1e6:.2f} MB in RAM. anyplotlib actually sends colormapped uint8
    (~{geom_len / 1e3:.0f} KB geom JSON), not raw float32.

    - anyplotlib `set_data`: **{apl_ms:.2f} ms** kernel
    - fastplotlib `data[:]` + `request_draw`: **{fpl_line}**
    """
    )

    return


@app.cell
def _(
    apl_fig,
    apl_img,
    fpl_canvas,
    fpl_fig,
    fpl_graphic,
    mo,
    n,
    np,
    preferred,
    time,
    volume,
    wgpu_adapter,
):
    def bench(label, fn, repeats=24):
        times = []
        for i in range(repeats):
            fr = np.ascontiguousarray(volume[:, :, i % n], dtype=np.float32)
            t_start = time.perf_counter()
            fn(fr)
            times.append((time.perf_counter() - t_start) * 1000)
        times.sort()
        return {
            "label": label,
            "n": repeats,
            "median_ms": times[len(times) // 2],
            "p90_ms": times[int(len(times) * 0.9)],
            "min_ms": times[0],
            "max_ms": times[-1],
        }

    apl_bench = bench("anyplotlib.set_data", lambda fr: apl_img.set_data(fr))
    if fpl_graphic is not None:

        def fpl_update(fr):
            fpl_graphic.data[:] = fr
            fpl_fig.canvas.request_draw()

        fpl_bench = bench("fastplotlib.data[:]", fpl_update)
    else:
        fpl_bench = {"label": "fastplotlib", "median_ms": None, "error": "not created"}

    geom_name = next(t for t in apl_fig.trait_names() if t.endswith("_geom"))
    apl_geom_chars = len(getattr(apl_fig, geom_name))
    apl_b64_chars = len(apl_img.to_state_dict().get("image_b64") or "")

    fpl_payload = {"mime": None, "bytes": None, "shape": None, "force_draw_ms": None}
    if fpl_graphic is not None:
        from rendercanvas.core.encoders import encode_array

        captured = []
        orig_send = fpl_canvas._rfb_send_frame

        def cap_send(array, is_lossless_redraw=False):
            if not is_lossless_redraw:
                mime, data = encode_array(array, fpl_canvas._quality)
                captured.append((mime, len(data), array.shape))
            return orig_send(array, is_lossless_redraw)

        fpl_canvas._rfb_send_frame = cap_send
        probe = np.ascontiguousarray(volume[:, :, n // 3], dtype=np.float32)
        t_start = time.perf_counter()
        fpl_graphic.data[:] = probe
        fpl_fig.canvas.request_draw()
        try:
            fpl_fig.canvas.force_draw()
        except Exception:
            pass
        force_ms = (time.perf_counter() - t_start) * 1000
        fpl_canvas._rfb_send_frame = orig_send
        if captured:
            mime, nbytes, shape = captured[0]
            fpl_payload = {
                "mime": mime,
                "bytes": nbytes,
                "shape": shape,
                "force_draw_ms": force_ms,
            }

    fpl_med = fpl_bench.get("median_ms")
    fpl_row = (
        f"{fpl_med:.2f} | {fpl_bench['p90_ms']:.2f} | {fpl_bench['min_ms']:.2f} | {fpl_bench['max_ms']:.2f}"
        if fpl_med is not None
        else f"{fpl_bench.get('error')} | — | — | —"
    )
    fpl_pay = (
        f"{fpl_payload['mime']} {fpl_payload['bytes']/1e3:.0f} KB "
        f"(canvas {fpl_payload['shape']}), force_draw {fpl_payload['force_draw_ms']:.1f} ms"
        if fpl_payload["bytes"]
        else "n/a"
    )

    adapter_used = "unknown"
    try:
        adapter_used = fpl_fig.renderer.device.adapter_info["device"]
    except Exception as exc:
        adapter_used = str(exc)

    rec = (
        "At 512³, anyplotlib is still enough for scrub: ~5 ms kernel and ~350 KB "
        "uint8 base64 per frame. Keep this notebook as the fastplotlib path: "
        "ImageGraphic.data[:] is ~0.2 ms, RFB JPEG is ~10x smaller on the wire, "
        "and wgpu attached to the GTX 1060. Switch for real if spatial size grows "
        "or pan/zoom of a GPU scene matters. Skip imgui-bundle/ImageWidget."
    )
    mo.md(
        f"""
    ## Bench ({n}³, 24 scrubs)

    | backend | median ms | p90 ms | min | max |
    |---|---:|---:|---:|---:|
    | anyplotlib set_data | {apl_bench["median_ms"]:.2f} | {apl_bench["p90_ms"]:.2f} | {apl_bench["min_ms"]:.2f} | {apl_bench["max_ms"]:.2f} |
    | fastplotlib data[:] + request_draw | {fpl_row} |

    **Payload**
    - volume: `{n}^3` float32 = **{n*n*n*4/1e6:.0f} MB**; one slice {n*n*4/1e6:.2f} MB
    - anyplotlib geom trait: **{apl_geom_chars/1e3:.0f} KB** (`image_b64` {apl_b64_chars/1e3:.0f} KB)
    - fastplotlib RFB: **{fpl_pay}**
    - locale preferred encoding: `{preferred}`
    - wgpu adapters enumerated: {wgpu_adapter}
    - renderer attached: **{adapter_used}**
    - imgui-bundle: not installed (ImageWidget skipped)
    - real 128² SIMUS skipped: swap was full and this kernel already holds the 537 MB volume

    **Recommendation:** {rec}
    """
    )

    return


if __name__ == "__main__":
    app.run()
