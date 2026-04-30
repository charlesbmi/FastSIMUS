# Handoff: NumPy SIMUS scaling benchmark (1e3–1e7) on a larger host

This document is for continuing the scaling study on a **16-core (or larger) Linux x86_64** instance with enough RAM.
The goal is to extend the **SIMUS (NumPy)** curve to `10^6` and optionally `10^7` scatterers, then merge results with
the existing **Proposed (CUDA)** JSON and regenerate the figure with publication-friendly labels.

## What is already done on the dev box

- **CUDA (Proposed)** sweep `10^3 … 10^7` completed and autosaved under `.benchmarks/Linux-CPython-3.12-64bit/`. Example
  file from the last run: `0007_9127dc8e28527687d9bfed7dde17ed636a474af4_20260430_185951_uncommited-changes.json`
  (device label: `Proposed (CUDA)`).
- **NumPy (SIMUS)** partial sweep on a **4 vCPU** slice of **AMD EPYC 7713** with **~16 GB RAM** completed only through
  `10^5` (larger sizes were not attempted to avoid multi-hour runs and OOM risk). Example:
  `0008_9127dc8e28527687d9bfed7dde17ed636a474af4_20260430_190438_uncommited-changes.json` (device label:
  `SIMUS (NumPy)`).
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

## Why memory matters for NumPy SIMUS

For `SimusStrategy.PYTHON`, `simus_compute()` calls `_prepare_simus_sweep()`, which materializes large arrays whose
dominant terms scale like **`(n_scat, n_elem, n_sub)`** per complex field (e.g. `phase_init`, `phase_step`), plus
geometry and similar terms. For **P4-2v** with `element_splitting=1` (as in the scaling bench), `n_elem = 64`,
`n_sub = 1`, so each complex `(n_scat, 64, 1)` tensor is about:

```text
bytes ≈ n_scat * 64 * sizeof(complex)
```

Use **16 bytes** if the stack resolves to `complex128` (worst case on many NumPy builds), or **8 bytes** if everything
stays `complex64`. The sweep holds **several** such arrays plus intermediates, so treat `10^7` scatterers as **tens of
GB** unless you have verified dtypes and peak RSS on a smaller `n_scat` first.

**Rule of thumb:** if `MemAvailable` (from `/proc/meminfo`) is not comfortably above your rough tensor budget (say
**2–3x** the estimated peak array footprint), do not run `10^7` on NumPy in one shot.

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
/usr/bin/time -v uv run pytest ... 2>&1 | tee numpy-bench.log
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

## Recommended run order (avoid wasting hours)

1. **Smoke:** `--n-scat=1000` only, confirm autosave JSON appears under `.benchmarks/Linux-CPython-3.12-64bit/`.
1. **Ladder:** `1000,10000,100000` — confirm throughput and wall time look sane.
1. **`10^6`:** run once with `/usr/bin/time -v` and `watch` on `MemAvailable`.
1. **`10^7`:** only if step 3 peak RSS plus headroom fits your policy; expect long runtime even on 16 cores because the
   Python frequency loop is still **`O(n_freq)`** per scatterer batch semantics.

## NumPy benchmark command (full sweep)

Use a machine label that includes **CPU model** so the plot legend stays honest when you merge JSON from multiple hosts:

```bash
export FASTSIMUS_DEVICE_LABEL="SIMUS (NumPy) $(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2 | xargs) ($(nproc) cores)"

uv run pytest tests/benchmarks/bench_simus_scaling.py \
  --benchmark-only --benchmark-autosave -p no:xdist -m scaling \
  -k numpy \
  --n-scat=1000,10000,100000,1000000,10000000
```

If `10^7` is too heavy, split:

```bash
uv run pytest tests/benchmarks/bench_simus_scaling.py ... -k numpy --n-scat=1000000
uv run pytest tests/benchmarks/bench_simus_scaling.py ... -k numpy --n-scat=10000000
```

Each autosave creates a new JSON; keep the ones you need for plotting.

## CUDA reference (optional re-run on same machine)

If you want CUDA and NumPy from the **same commit** on the large box:

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
  path/to/numpy.json \
  -o docs/figures/2026-04-cuda-scaling.png \
  --reference-backend numpy \
  --reference-label "SIMUS (NumPy)" \
  --series-column machine \
  --title "SIMUS scaling: SIMUS (NumPy) vs Proposed"
```

Speedup panel is **vs NumPy** (`SIMUS (NumPy)`), not PyMUST.

## After results exist

- Commit updated `docs/figures/2026-04-cuda-scaling.png` and any new `.benchmarks/**/*.json` you intend to keep in-repo
  (or attach JSON to the PR if policy prefers not committing large artifacts).
- Run `uv run --group lint mdformat README.md docs --wrap 120` if you touch docs.

## Plotting unit tests

```bash
uv run pytest tests/test_plot_benchmark_scaling.py -q -p no:testmon
```
