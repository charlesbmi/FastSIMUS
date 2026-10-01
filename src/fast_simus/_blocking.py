"""Static tile sizing and backend loop adapters for bounded numerical execution."""

from dataclasses import dataclass
from math import ceil
from types import ModuleType
from typing import cast

from array_api_compat import is_jax_namespace

from fast_simus.utils._array_api import is_mlx_namespace


@dataclass(frozen=True)
class Tiles:
    """Conservative live-buffer estimate with one frequency per tile."""

    points: int
    patches: int
    elements: int
    max_patches: int
    workspace_bytes: int


def choose_tiles(n_points, counts, options):
    """Budget displacement, angles, complex phase and reduction temporaries."""
    maximum = max(nu * nv for nu, nv in counts)
    fixed = len(counts) * 64  # subdivision metadata and cleaned excitation arrays
    available = options.workspace_bytes - fixed
    if available < 1024:
        raise ValueError("Workspace is too small for aperture metadata and one tile")
    elements = min(len(counts), 16, max(1, available // 4096))
    patches = min(maximum, 32, max(1, available // (1024 * elements)))
    points = min(n_points, 1024, max(1, available // (elements * (512 * patches + 512))))
    return Tiles(points, patches, elements, maximum, fixed + points * elements * (512 * patches + 512))


def run_loop(count, body, state, xp):
    """Use device control flow for JAX and release lazy MLX tile graphs eagerly."""
    if is_jax_namespace(cast(ModuleType, xp)):
        import jax  # noqa: PLC0415

        return jax.lax.fori_loop(0, count, body, state)
    for i in range(count):
        state = body(i, state)
        if is_mlx_namespace(xp):
            xp.eval(state)
    return state


def point_block(points, index, size, xp):
    """Gather a fixed-size finite block and its validity mask."""
    indices = index * size + xp.arange(size)
    valid = indices < points.shape[0]
    return xp.take(points, xp.minimum(indices, xp.asarray(points.shape[0] - 1)), axis=0), valid


def block_count(length, size):
    """Number of padded blocks."""
    return ceil(length / size)


def legacy_point_count(execution, n_elements, n_sub):
    """Bound a legacy strip tile while retaining its complete coherent aperture."""
    per_point = 512 * n_elements * n_sub
    if execution.workspace_bytes < per_point:
        raise ValueError(f"Legacy strip execution needs at least {per_point} workspace bytes for one point")
    return max(1, execution.workspace_bytes // per_point)


def element_block(index, size, count, xp):
    """Safe channel indices and validity for one static element tile."""
    indices = index * size + xp.arange(size)
    valid = indices < count
    return xp.minimum(indices, xp.asarray(count - 1)), valid


def scatterer_block(points, coefficients, index, size, xp):
    """Pad a source tile with finite positions and zero scattering strength."""
    block, valid = point_block(points, index, size, xp)
    strengths, _ = point_block(coefficients, index, size, xp)
    return block, xp.where(valid, strengths, xp.zeros_like(strengths))
