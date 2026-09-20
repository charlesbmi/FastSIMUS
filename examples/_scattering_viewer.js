function floatArray(value) {
  if (!value) return new Float32Array();
  if (value instanceof DataView) {
    return new Float32Array(value.buffer, value.byteOffset, value.byteLength / 4);
  }
  if (value instanceof Uint8Array) {
    return new Float32Array(value.buffer, value.byteOffset, value.byteLength / 4);
  }
  if (value.buffer) {
    const offset = value.byteOffset || 0;
    const length = value.byteLength || value.buffer.byteLength;
    return new Float32Array(value.buffer, offset, length / 4);
  }
  return new Float32Array();
}

function blend(base, color, alpha) {
  return [
    base[0] * (1 - alpha) + color[0] * alpha,
    base[1] * (1 - alpha) + color[1] * alpha,
    base[2] * (1 - alpha) + color[2] * alpha,
  ];
}

function phaseColor(value) {
  return value < 0 ? [37, 99, 235] : [220, 38, 38];
}

function dbVisibility(value, dynamicRange) {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return 0;
  const level = 20 * Math.log10(magnitude);
  return Math.max(0, Math.min(1, (level + dynamicRange) / dynamicRange));
}

function setupCanvas(canvas, height) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, canvas.clientWidth);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function physicalPlot(width, height, margins, xDomain, zDomain) {
  const [minimumLeft, minimumTop, minimumRight, minimumBottom] = margins;
  const availableWidth = Math.max(1, width - minimumLeft - minimumRight);
  const availableHeight = Math.max(1, height - minimumTop - minimumBottom);
  const xSpan = Math.max(Number.EPSILON, Math.abs(xDomain[1] - xDomain[0]));
  const zSpan = Math.max(Number.EPSILON, Math.abs(zDomain[1] - zDomain[0]));
  const pixelsPerMillimetre = Math.min(availableWidth / xSpan, availableHeight / zSpan);
  const plotWidth = xSpan * pixelsPerMillimetre;
  const plotHeight = zSpan * pixelsPerMillimetre;
  return {
    left: minimumLeft + (availableWidth - plotWidth) / 2,
    top: minimumTop + (availableHeight - plotHeight) / 2,
    plotWidth,
    plotHeight,
  };
}

function render({ model, el }) {
  const controller = new AbortController();
  const { signal } = controller;
  el.classList.add("fs-scattering-viewer");
  el.innerHTML = `
    <section class="fs-viewer-card">
      <header class="fs-viewer-header">
        <div><strong class="fs-component"></strong><span class="fs-time"></span></div>
      </header>
      <canvas class="fs-field" aria-label="Ultrasound propagation and scattering field"></canvas>
      <div class="fs-caption-row">
        <span class="fs-field-caption"></span>
        <span class="fs-reflectivity"><span class="fs-reflectivity-bar"></span><span class="fs-reflectivity-label"></span></span>
      </div>
      <canvas class="fs-receive" aria-label="Signed SIMUS receive channels"></canvas>
      <div class="fs-caption-row"><span class="fs-receive-caption"></span></div>
    </section>`;

  const fieldCanvas = el.querySelector(".fs-field");
  const receiveCanvas = el.querySelector(".fs-receive");
  const componentLabel = el.querySelector(".fs-component");
  const timeLabel = el.querySelector(".fs-time");
  const fieldCaption = el.querySelector(".fs-field-caption");
  const receiveCaption = el.querySelector(".fs-receive-caption");
  const reflectivityLabel = el.querySelector(".fs-reflectivity-label");

  function drawAxes(context, plot, xDomain, zDomain) {
    const { left, top, plotWidth, plotHeight } = plot;
    context.strokeStyle = "rgba(100, 116, 139, 0.45)";
    context.fillStyle = "#64748b";
    context.lineWidth = 1;
    context.font = "11px ui-sans-serif, system-ui";
    context.textAlign = "center";
    context.textBaseline = "top";
    for (let index = 0; index <= 4; index += 1) {
      const fraction = index / 4;
      const x = left + fraction * plotWidth;
      context.beginPath();
      context.moveTo(x, top);
      context.lineTo(x, top + plotHeight);
      context.stroke();
      const value = xDomain[0] + fraction * (xDomain[1] - xDomain[0]);
      context.fillText(`${value.toFixed(0)} mm`, x, top + plotHeight + 5);
    }
    context.textAlign = "right";
    context.textBaseline = "middle";
    for (let index = 0; index <= 5; index += 1) {
      const fraction = index / 5;
      const y = top + fraction * plotHeight;
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(left + plotWidth, y);
      context.stroke();
      const value = zDomain[0] + fraction * (zDomain[1] - zDomain[0]);
      context.fillText(`${value.toFixed(0)} mm`, left - 7, y);
    }
  }

  function drawField() {
    const { context, width, height } = setupCanvas(fieldCanvas, Math.min(720, Math.max(430, window.innerHeight * 0.67)));
    const [nz, nx, nt] = model.get("field_shape");
    if (!nz || !nx || !nt) return;
    const incident = floatArray(model.get("incident"));
    const scattered = floatArray(model.get("scattered"));
    const rms = floatArray(model.get("incident_rms"));
    const component = model.get("component");
    const phaseRange = Math.max(1, model.get("phase_dynamic_range"));
    const opacity = model.get("rms_visible") ? 0.5 : 0;
    const range = Math.max(1, model.get("rms_dynamic_range"));
    const times = model.get("field_times");
    const timeIndex = Math.max(0, Math.min(nt - 1, model.get("time_index")));
    const [xmin, xmax, zmin, zmax] = model.get("extent");
    const elements = model.get("elements") || [];
    const elementDepths = elements.map((point) => point[1]);
    const viewZMin = Math.min(0, zmin, ...(elementDepths.length ? elementDepths : [zmin]));
    const viewZMax = zmax;
    const xDomain = [xmin, xmax];
    const zDomain = [viewZMin, viewZMax];
    const plot = physicalPlot(width, height, [54, 12, 16, 34], xDomain, zDomain);
    drawAxes(context, plot, xDomain, zDomain);
    const xToCanvas = (x) => plot.left + ((x - xmin) / (xmax - xmin)) * plot.plotWidth;
    const zToCanvas = (z) => plot.top + ((z - viewZMin) / (viewZMax - viewZMin)) * plot.plotHeight;

    const image = new ImageData(nx, nz);
    for (let z = 0; z < nz; z += 1) {
      for (let x = 0; x < nx; x += 1) {
        const spatial = z * nx + x;
        const offset = spatial * nt + timeIndex;
        let value = incident[offset];
        if (component === "Scattered") value = scattered[offset];
        if (component === "Total") value += scattered[offset];
        const rawValue = value;
        value = Math.max(-1, Math.min(1, rawValue));
        const rmsDb = 20 * Math.log10(Math.max(rms[spatial], 1e-12));
        const rmsAmount = Math.max(0, Math.min(1, (rmsDb + range) / range));
        let color = blend([255, 255, 255], [94, 218, 232], opacity * rmsAmount);
        color = blend(color, phaseColor(value), dbVisibility(rawValue, phaseRange));
        const pixel = spatial * 4;
        image.data[pixel] = color[0];
        image.data[pixel + 1] = color[1];
        image.data[pixel + 2] = color[2];
        image.data[pixel + 3] = 255;
      }
    }
    const raster = document.createElement("canvas");
    raster.width = nx;
    raster.height = nz;
    raster.getContext("2d").putImageData(image, 0, 0);
    context.imageSmoothingEnabled = true;
    context.drawImage(raster, plot.left, zToCanvas(zmin), plot.plotWidth, zToCanvas(zmax) - zToCanvas(zmin));

    const focus = model.get("focus");
    if (focus && focus.length === 2) {
      context.strokeStyle = "rgba(219, 39, 119, 0.65)";
      context.setLineDash([5, 5]);
      context.beginPath();
      context.moveTo(xToCanvas(focus[0]), plot.top);
      context.lineTo(xToCanvas(focus[0]), zToCanvas(focus[1]));
      context.stroke();
      context.setLineDash([]);
    }
    context.fillStyle = "#1d4ed8";
    for (const point of elements) {
      context.beginPath();
      context.arc(xToCanvas(point[0]), zToCanvas(point[1]), 2.2, 0, 2 * Math.PI);
      context.fill();
    }
    const scatterers = model.get("scatterers") || [];
    const coefficients = model.get("coefficients") || [];
    const reflectivityReference = coefficients.reduce((peak, value) => Math.max(peak, Math.abs(value)), 0);
    scatterers.forEach((point, index) => {
      const strength = reflectivityReference === 0 ? 0 : Math.abs(coefficients[index]) / reflectivityReference;
      const shade = Math.round(245 - 220 * Math.sqrt(strength));
      context.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
      context.strokeStyle = "white";
      context.lineWidth = 1;
      context.beginPath();
      context.arc(xToCanvas(point[0]), zToCanvas(point[1]), 3.5, 0, 2 * Math.PI);
      context.fill();
      context.stroke();
    });
    componentLabel.textContent = `${component} field`;
    timeLabel.textContent = ` · ${(times[timeIndex] * 1e6).toFixed(1)} µs`;
    fieldCaption.textContent = `${phaseRange.toFixed(0)} dB range · relative to peak incident pressure`;
    reflectivityLabel.textContent = `relative reflectivity 0–${reflectivityReference.toPrecision(2)}`;
  }

  function drawReceive() {
    const [nTimes, nChannels] = model.get("receive_shape");
    const receive = floatArray(model.get("receive"));
    if (!nTimes || !nChannels) return;
    const margins = [54, 8, 16, 26];
    const receiveTimes = model.get("receive_times");
    const start = receiveTimes[0] || 0;
    const end = receiveTimes[receiveTimes.length - 1] || start;
    const elements = model.get("elements") || [];
    let apertureSpan = 0;
    for (let index = 1; index < elements.length; index += 1) {
      apertureSpan += Math.hypot(
        elements[index][0] - elements[index - 1][0],
        elements[index][1] - elements[index - 1][1],
      );
    }
    const propagationSpan = (end - start) * Math.max(0, model.get("propagation_speed")) * 1e3;
    const availableWidth = Math.max(320, receiveCanvas.clientWidth) - margins[0] - margins[2];
    const physicalHeight = propagationSpan > 0 ? availableWidth * apertureSpan / propagationSpan : 98;
    const plotHeight = Math.max(72, Math.min(280, physicalHeight));
    const canvasHeight = margins[1] + plotHeight + margins[3];
    const { context, width } = setupCanvas(receiveCanvas, canvasHeight);
    const phaseRange = Math.max(1, model.get("phase_dynamic_range"));
    const image = new ImageData(nTimes, nChannels);
    for (let time = 0; time < nTimes; time += 1) {
      for (let channel = 0; channel < nChannels; channel += 1) {
        const source = time * nChannels + channel;
        const rawValue = receive[source];
        const value = Math.max(-1, Math.min(1, rawValue));
        const color = blend([255, 255, 255], phaseColor(value), dbVisibility(rawValue, phaseRange));
        const pixel = (channel * nTimes + time) * 4;
        image.data[pixel] = color[0];
        image.data[pixel + 1] = color[1];
        image.data[pixel + 2] = color[2];
        image.data[pixel + 3] = 255;
      }
    }
    const raster = document.createElement("canvas");
    raster.width = nTimes;
    raster.height = nChannels;
    raster.getContext("2d").putImageData(image, 0, 0);
    const plotWidth = width - margins[0] - margins[2];
    context.drawImage(raster, margins[0], margins[1], plotWidth, plotHeight);
    context.fillStyle = "#64748b";
    context.font = "11px ui-sans-serif, system-ui";
    context.textAlign = "center";
    context.textBaseline = "top";
    for (let index = 0; index <= 5; index += 1) {
      const fraction = index / 5;
      const x = margins[0] + fraction * plotWidth;
      context.fillText(`${((start + fraction * (end - start)) * 1e6).toFixed(0)} µs`, x, margins[1] + plotHeight + 5);
    }
    context.save();
    context.translate(14, margins[1] + plotHeight / 2);
    context.rotate(-Math.PI / 2);
    context.fillText("receive element", 0, 0);
    context.restore();

    const fieldTimes = model.get("field_times");
    const timeIndex = Math.max(0, Math.min(fieldTimes.length - 1, model.get("time_index")));
    const selected = fieldTimes[timeIndex];
    const fraction = end === start ? 0 : (selected - start) / (end - start);
    const cursorX = margins[0] + Math.max(0, Math.min(1, fraction)) * plotWidth;
    context.strokeStyle = "#db2777";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(cursorX, margins[1]);
    context.lineTo(cursorX, margins[1] + plotHeight);
    context.stroke();
    receiveCaption.textContent = `${phaseRange.toFixed(0)} dB range · relative to peak receive RF`;
  }

  function drawAll() {
    drawField();
    drawReceive();
  }

  function seek(event) {
    const bounds = receiveCanvas.getBoundingClientRect();
    const left = 54;
    const right = 16;
    const fraction = Math.max(0, Math.min(1, (event.clientX - bounds.left - left) / (bounds.width - left - right)));
    const receiveTimes = model.get("receive_times");
    const start = receiveTimes[0] || 0;
    const end = receiveTimes[receiveTimes.length - 1] || start;
    const selected = start + fraction * (end - start);
    const fieldTimes = model.get("field_times");
    let nearest = 0;
    let distance = Infinity;
    fieldTimes.forEach((time, index) => {
      const candidate = Math.abs(time - selected);
      if (candidate < distance) {
        distance = candidate;
        nearest = index;
      }
    });
    model.set("time_index", nearest);
    model.save_changes();
    drawAll();
  }

  let dragging = false;
  receiveCanvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    receiveCanvas.setPointerCapture(event.pointerId);
    seek(event);
  }, { signal });
  receiveCanvas.addEventListener("pointermove", (event) => {
    if (dragging) seek(event);
  }, { signal });
  receiveCanvas.addEventListener("pointerup", () => { dragging = false; }, { signal });

  const traits = [
    "incident", "scattered", "incident_rms", "receive", "field_shape", "receive_shape",
    "field_times", "receive_times", "extent", "elements", "scatterers", "coefficients",
    "focus", "component", "rms_visible", "rms_dynamic_range", "phase_dynamic_range", "propagation_speed", "time_index",
  ];
  traits.forEach((name) => model.on(`change:${name}`, drawAll));
  const observer = new ResizeObserver(drawAll);
  observer.observe(el);
  drawAll();
  return () => {
    observer.disconnect();
    controller.abort();
  };
}

export default { render };
