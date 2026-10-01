"""Scene construction and in-notebook three-plane simulation, independent of plotting."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

import fast_simus as fs
from fast_simus.utils._array_api import as_numpy


@dataclass(frozen=True)
class Orthoslices:
    """Unique physical observation points and maps back to XY, XZ, YZ planes."""

    points: np.ndarray
    indices: tuple
    axes: tuple
    center: tuple


def orthoslices(x, y, z):
    """Deduplicate intersections; arrays retain row-major image order."""
    center = (x[len(x) // 2], y[len(y) // 2], z[len(z) // 2])
    xy = np.stack(np.meshgrid(x, y, [center[2]], indexing="xy"), axis=-1).reshape(len(y), len(x), 3)
    xz = np.stack(np.meshgrid(x, [center[1]], z, indexing="ij"), axis=-1).reshape(len(x), len(z), 3).transpose(1, 0, 2)
    yz = np.stack(np.meshgrid([center[0]], y, z, indexing="ij"), axis=-1).reshape(len(y), len(z), 3).transpose(1, 0, 2)
    points, inverse = np.unique(np.concatenate([v.reshape(-1, 3) for v in (xy, xz, yz)]), axis=0, return_inverse=True)
    a, b = xy.shape[0] * xy.shape[1], xz.shape[0] * xz.shape[1]
    indices = (
        inverse[:a].reshape(xy.shape[:2]),
        inverse[a : a + b].reshape(xz.shape[:2]),
        inverse[a + b :].reshape(yz.shape[:2]),
    )
    return Orthoslices(points.astype(np.float32), indices, (x, y, z), center)


def phantom(kind, count, seed=2026, depth_mm=20):
    """Signed weak-scattering cloud with a 1.5 mm anechoic sphere and bright targets."""
    if kind == "None":
        return np.empty((0, 3), np.float32), np.empty(0, np.float32)
    if kind == "Point":
        return np.array([[0, 0, 0.007 if depth_mm == 20 else 0.017]], np.float32), np.array([2e-4], np.float32)
    shallow = depth_mm == 20
    lower, upper = (0.005, 0.018) if shallow else (0.009, 0.025)
    cavity_depth = 0.012 if shallow else 0.018
    targets = (0.007, 0.017) if shallow else (0.014, 0.023)
    rng = np.random.default_rng(seed)
    chunks = []
    remaining = max(0, count - 2)
    while remaining:
        candidates = rng.uniform([-0.004, -0.004, lower], [0.004, 0.004, upper], (remaining + 64, 3))
        candidates = candidates[np.linalg.norm(candidates - [0.001, 0, cavity_depth], axis=-1) > 0.0015][:remaining]
        chunks.append(candidates)
        remaining -= len(candidates)
    background = np.concatenate(chunks) if chunks else np.empty((0, 3))
    points = np.concatenate([background, [[-0.002, 0, targets[0]], [0.002, 0, targets[1]]]]).astype(np.float32)
    rc = rng.normal(0, 2e-6, len(points)).astype(np.float32)
    rc[-2:] = 2e-4
    return points, rc


@dataclass(frozen=True)
class Simulation:
    """Latest completed notebook result, with time last and coordinates in meters."""

    slices: Orthoslices
    incident: np.ndarray
    scattered: np.ndarray
    times: np.ndarray
    elements: np.ndarray
    scatterers: np.ndarray
    seconds: float
    workspace_bytes: int
    backend: str
    rf: np.ndarray
    rf_times: np.ndarray
    matrix_shape: tuple[int, int]
    sound_speed: float


def observation_planes(config, sound_speed, frequency):
    """Central slices sampled from retained bandwidth, with explicit preview coarsening."""
    # Physical Nyquist spacing for the retained grid up to 2*fc; preview is explicit.
    spacing = sound_speed / (4 * frequency) * config.get("spacing_factor", 1)
    nx = max(3, int(np.ceil(0.008 / spacing)) + 1)
    depth_mm = config.get("depth_mm", 20)
    z_min, z_max = (0.002 if depth_mm == 20 else 0.004), depth_mm / 1000
    nz = max(3, int(np.ceil((z_max - z_min) / spacing)) + 1)
    nx += (nx + 1) % 2
    nz += (nz + 1) % 2
    if config.get("smoke"):
        nx, nz = 5, 7
    return orthoslices(np.linspace(-0.004, 0.004, nx), np.linspace(-0.004, 0.004, nx), np.linspace(z_min, z_max, nz))


def simulate(config, progress=None, cancelled=None):
    """Simulate all scatterers on just three planes; callbacks run between blocks."""
    start = perf_counter()
    xp = fs.get_backend(config["backend"].lower()).xp
    fc = 2e6
    medium = fs.MediumParams()
    depth_mm = config.get("depth_mm", 20)
    slices = observation_planes(config, medium.speed_of_sound, fc)
    scatterers, rc = phantom(config["scene"], config["count"], depth_mm=depth_mm)
    side = config.get("side", 16)
    aperture = fs.matrix_aperture(
        shape=(side, side), pitch=(0.0003, 0.0003), size=(0.0002, 0.0002), xp=xp, dtype=xp.float32
    )
    probe = fs.Transducer(aperture, "3d", fc)
    angle = np.deg2rad(config["steer"])
    if config["transmit"] == "Plane wave":
        delays = fs.plane_wave_delays(aperture.centers, xp.asarray([np.sin(angle), 0, np.cos(angle)], dtype=xp.float32))
    else:
        depth = config["focus"] / 1000
        target = xp.asarray(
            [depth * np.tan(angle), 0, -depth if config["transmit"] == "Diverging" else depth], dtype=xp.float32
        )
        delays = fs.focus_delays(aperture.centers, target, diverging=config["transmit"] == "Diverging")
    observers = xp.asarray(slices.points, dtype=xp.float32)
    sources = xp.asarray(scatterers, dtype=xp.float32)
    strengths = xp.asarray(rc, dtype=xp.float32)
    plan = fs.scattered_field_precompute(
        observers,
        sources,
        strengths,
        delays,
        probe,
        medium,
        execution=fs.ExecutionOptions(128 * 1024 * 1024),
        frequency_step=1.0,
    )
    times = as_numpy(fs.wavefield_times(plan))
    outputs = []
    for component in ("incident", "scattered"):
        if cancelled is not None and cancelled():
            raise InterruptedError("Simulation cancelled")
        if progress:
            progress(component, 0, len(slices.points))
        frames = np.empty((len(slices.points), len(times)), np.float32)
        iterator = fs.iter_scattered_pfield_spectrum(
            observers, sources, strengths, delays, plan, probe, medium, component=component, cancelled=cancelled
        )
        for block in iterator:
            if cancelled is not None and cancelled():
                raise InterruptedError("Simulation cancelled between spatial blocks")
            frames[block.start : block.stop] = as_numpy(fs.spectrum_to_wavefield(block.values, plan).frames)
            if progress:
                progress(component, block.stop, len(slices.points))
        outputs.append(frames)
    if cancelled is not None and cancelled():
        raise InterruptedError("Simulation cancelled before receive RF")
    if progress:
        progress("receive RF", 0, 1)
    workspace_bytes = plan.estimated_workspace_bytes
    if len(scatterers):
        rf_plan = fs.simus_precompute(
            sources,
            strengths,
            delays,
            probe,
            medium,
            frequency_step=1.0,
            execution=fs.ExecutionOptions(128 * 1024 * 1024),
        )
        rf = as_numpy(fs.simus_compute(sources, strengths, delays, rf_plan, probe, medium).rf)
        rf_times = as_numpy(rf_plan.sample_times)
        workspace_bytes = max(workspace_bytes, rf_plan.estimated_workspace_bytes)
    else:
        rf = np.zeros((len(times), probe.n_elements), dtype=np.float32)
        rf_times = times
    if cancelled is not None and cancelled():
        raise InterruptedError("Simulation cancelled during receive RF")
    if progress:
        progress("receive RF", 1, 1)
    return Simulation(
        slices=slices,
        incident=outputs[0],
        scattered=outputs[1],
        times=times,
        elements=as_numpy(aperture.centers),
        scatterers=scatterers,
        seconds=perf_counter() - start,
        workspace_bytes=workspace_bytes,
        backend=config["backend"],
        rf=rf,
        rf_times=rf_times,
        matrix_shape=(side, side),
        sound_speed=medium.speed_of_sound,
    )
