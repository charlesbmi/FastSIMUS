"""Static tile sizing and backend loop adapters for bounded numerical execution."""

from dataclasses import dataclass
from math import ceil
from types import ModuleType
from typing import cast

from array_api_compat import is_jax_namespace

from fast_simus.utils._array_api import is_mlx_namespace


@dataclass(frozen=True)
class Tiles:
    """Conservative live-buffer estimate with frequency/element tiles of one."""

    points: int
    patches: int
    max_patches: int
    workspace_bytes: int


def choose_tiles(n_points, counts, options):
    """Budget displacement, angles, complex phase and reduction temporaries."""
    maximum = max(nu * nv for nu, nv in counts)
    patches = min(maximum, 32, max(1, options.workspace_bytes // 1024))
    points = min(n_points, 1024, max(1, options.workspace_bytes // (512 * patches + 512)))
    return Tiles(points, patches, maximum, points * (512 * patches + 512))


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
    return points[xp.minimum(indices, points.shape[0] - 1)], valid


def block_count(length, size):
    """Number of padded blocks."""
    return ceil(length / size)


def legacy_point_count(execution, n_elements, n_sub):
    """Bound a legacy strip tile while retaining its complete coherent aperture."""
    per_point = 512 * n_elements * n_sub
    if execution.workspace_bytes < per_point:
        raise ValueError(f"Legacy strip execution needs at least {per_point} workspace bytes for one point")
    return max(1, execution.workspace_bytes // per_point)
