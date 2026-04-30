# Handoff: PyMUST (NumPy SIMUS) scaling benchmark (1e3–1e7) on a larger host

This document is for continuing the scaling study on a **16-core (or larger) Linux x86_64** instance with enough RAM.
The goal is to extend the **PyMUST** baseline curve, displayed in the paper as **SIMUS (NumPy)**, to `10^6` and
optionally `10^7` scatterers, then merge results with the existing **Proposed (CUDA)** JSON and regenerate the figure
with publication-friendly labels.

## What is already done on the dev box

- **CUDA (Proposed)** sweep `10^3 … 10^7` completed and autosaved under `.benchmarks/Linux-CPython-3.12-64bit/`. Example
  file from the last run: `0007_9127dc8e28527687d9bfed7dde17ed636a474af4_20260430_185951_uncommited-changes.json`
  (device label: `Proposed (CUDA)`).
- **PyMUST baseline** partial sweep on a **4 vCPU** slice of **AMD EPYC 7713** with **~16 GB RAM** completed only
  through `10^5` (larger sizes were not attempted to avoid multi-hour runs and OOM risk). Example:
  `0008_9127dc8e28527687d9bfed7dde17ed636a474af4_20260430_190438_uncommited-changes.json` (benchmark backend: `pymust`;
  device label: `SIMUS (NumPy)`).
- Plot script now supports **`--reference-backend`**, **`--reference-label`**, **`--series-column machine`**, and
  **`--title`** so the legend can show `SIMUS (NumPy)` vs `Proposed (CUDA)` without renaming benchmark internals.

Copy the two JSON paths above to the larger machine if you are not pushing the whole `.benchmarks/` tree.

## Prerequisites on the larger instance

```bash
cd /path/to/FastSIMUS
git fetch origin
git checkout feat/cuda-cupy-backend   # or your branch containing the plot changes
uv sync --group bench
```

Confirm CPU and memory baseline:

```bash
lscpu | sed -n '1,25p'
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
```

Optional: set a stable hostname label for the machine (shown in JSON `machine_info.node` if you do not set
`FASTSIMUS_DEVICE_LABEL`).

## Why PyMUST progress and memory matter

This handoff is **not** asking for FastSIMUS's NumPy backend (`tests/benchmarks/bench_simus_scaling.py -k numpy`). For
the paper comparison, the baseline is **PyMUST** (`tests/benchmarks/bench_pymust_scaling.py`), displayed as
`SIMUS (NumPy)`.

PyMUST's `simus()` delegates the expensive spectrum calculation to `pymust.pfield()`. In the 2-D P4-2v benchmark with
`ElementSplitting=1`, the dominant arrays still scale like **`(n_scat, n_elem, n_sub)`** per complex field. With
`n_elem = 64` and `n_sub = 1`, each complex `(n_scat, 64, 1)` tensor is about:

```text
bytes ≈ n_scat * 64 * sizeof(complex)
```

Use **16 bytes** if an intermediate resolves to `complex128`, or **8 bytes** when it stays `complex64`. The sweep holds
several such arrays plus intermediates, so treat `10^7` scatterers as **tens of GB** unless you have verified dtypes and
peak RSS on a smaller `n_scat` first.

The other failure mode is simply runtime: PyMUST advances through a frequency-sample loop in `pymust.pfield()`. If a
larger case exceeds the extrapolated time window, it is acceptable to stop once progress logging shows that the loop is
still moving. In that case, record the observed rate instead of waiting for a completed autosave JSON.

## Monitoring memory while a benchmark runs

### 1. Watch system memory in another terminal

```bash
watch -n 2 'grep -E "^(MemAvailable|MemFree|SwapFree):" /proc/meminfo'
```

Or:

```bash
watch -n 2 free -h
```

Stop the benchmark if `MemAvailable` approaches zero and swap starts climbing unexpectedly.

### 2. Peak RSS for the pytest process (GNU time)

```bash
/usr/bin/time -v uv run pytest ... 2>&1 | tee pymust-bench.log
# At end of output, look for:
#   Maximum resident set size (kbytes): ...
```

### 3. Live RSS for a known PID

In one terminal start the job, note the PID from `ps` or `pgrep -af pytest`, then:

```bash
top -p <PID>
# or
pidstat -r -p <PID> 2
```

### 4. cgroup limit (cloud VMs)

If the instance uses cgroups v2:

```bash
cat /sys/fs/cgroup/$(cat /proc/self/cgroup | tail -1 | cut -d: -f3)/memory.max 2>/dev/null || true
```

If this prints a finite cap, that is your hard ceiling regardless of `MemTotal`.

## Optional PyMUST progress instrumentation

If a large PyMUST run passes its extrapolated time window without completing, add temporary progress logging to the
installed PyMUST source before rerunning. This is only for diagnosing whether the job is stalled or merely slow; do not
commit changes under `.venv/`.

Locate the installed sources:

```bash
uv run python - <<'PY'
import inspect
import pymust

print(inspect.getsourcefile(pymust.simus))
print(inspect.getsourcefile(pymust.pfield))
PY
```

In `pymust/pfield.py`, instrument the frequency loop around `for k in range(nSampling):`. A minimal local patch is:

```python
import os
import time

# just before the loop
progress_every = int(os.environ.get("PYMUST_PROGRESS_EVERY", "10"))
progress_t0 = time.monotonic()

for k in range(nSampling):
    if progress_every and (k == 0 or (k + 1) % progress_every == 0 or k + 1 == nSampling):
        elapsed = time.monotonic() - progress_t0
        rate = (k + 1) / elapsed if elapsed else 0.0
        remaining = (nSampling - k - 1) / rate if rate else float("inf")
        print(
            f"[pymust.pfield] frequency {k + 1}/{nSampling} "
            f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m",
            flush=True,
        )
```

Then rerun a single large point with a coarse interval:

```bash
PYMUST_PROGRESS_EVERY=5 /usr/bin/time -v uv run pytest tests/benchmarks/bench_pymust_scaling.py \
  --benchmark-only --benchmark-autosave -p no:xdist -m scaling \
  -k pymust \
  --n-scat=1000000 2>&1 | tee pymust-1e6-progress.log
```

If the log advances steadily but the ETA is too long, stop the run and keep the progress log as evidence that PyMUST is
slow rather than stalled.

## Recommended run order (avoid wasting hours)

1. **Smoke:** `--n-scat=1000` only, confirm autosave JSON appears under `.benchmarks/Linux-CPython-3.12-64bit/`.
1. **Ladder:** `1000,10000,100000` — confirm throughput and wall time look sane.
1. **`10^6`:** run once with `/usr/bin/time -v`, `watch` on `MemAvailable`, and optional PyMUST progress logging.
1. **`10^7`:** only if step 3 peak RSS plus headroom fits your policy and the progress ETA is reasonable. It is fine to
   stop early after collecting enough progress data to show that PyMUST is still advancing.

## PyMUST benchmark command (full sweep)

Use a machine label that includes **CPU model** so the plot legend stays honest when you merge JSON from multiple hosts:

```bash
export FASTSIMUS_DEVICE_LABEL="SIMUS (NumPy) $(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2 | xargs) ($(nproc) cores)"

uv run pytest tests/benchmarks/bench_pymust_scaling.py \
  --benchmark-only --benchmark-autosave -p no:xdist -m scaling \
  -k pymust \
  --n-scat=1000,10000,100000,1000000,10000000
```

If `10^7` is too heavy, split:

```bash
uv run pytest tests/benchmarks/bench_pymust_scaling.py ... -k pymust --n-scat=1000000
uv run pytest tests/benchmarks/bench_pymust_scaling.py ... -k pymust --n-scat=10000000
```

Each autosave creates a new JSON; keep the ones you need for plotting.

## CUDA reference (optional re-run on same machine)

If you want CUDA and PyMUST baseline rows from the **same commit** on the large box:

```bash
export FASTSIMUS_DEVICE_LABEL="Proposed (CUDA) $(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2 | xargs)"

uv run pytest tests/benchmarks/bench_simus_scaling.py \
  --benchmark-only --benchmark-autosave -p no:xdist -m scaling \
  -k cupy \
  --n-scat=1000,10000,100000,1000000,10000000
```

## Merge JSON and regenerate the figure

From a checkout that has **both** JSON files (copy paths as needed):

```bash
uv run python scripts/plot_benchmark_scaling.py \
  path/to/cuda.json \
  path/to/pymust.json \
  -o docs/figures/2026-04-cuda-scaling.png \
  --reference-backend pymust \
  --reference-label "SIMUS (NumPy)" \
  --series-column machine \
  --title "SIMUS scaling: SIMUS (NumPy) vs Proposed"
```

Speedup panel is **vs PyMUST**, which is displayed as `SIMUS (NumPy)` for the paper.

## After results exist

- Commit updated `docs/figures/2026-04-cuda-scaling.png` and any new `.benchmarks/**/*.json` you intend to keep in-repo
  (or attach JSON to the PR if policy prefers not committing large artifacts).
- Run `uv run --group lint mdformat README.md docs --wrap 120` if you touch docs.

## Plotting unit tests

```bash
uv run pytest tests/test_plot_benchmark_scaling.py -q -p no:testmon
```
