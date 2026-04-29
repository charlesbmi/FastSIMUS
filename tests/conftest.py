"""Pytest configuration for FastSIMUS tests."""

import contextlib
import math
from pathlib import Path
from typing import Any, cast

import pytest

from fast_simus.pfield import PfieldStrategy
from fast_simus.simus import SimusStrategy
from fast_simus.utils._array_api import _ArrayNamespace


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register FastSIMUS test-specific command-line options."""
    parser.addoption(
        "--plot-picmus-phantom",
        action="store",
        default=None,
        metavar="PATH",
        help="Write optional reconstructed PICMUS phantom diagnostic figure to PATH.",
    )
    parser.addoption(
        "--picmus-phantom-angles",
        action="store",
        choices=("broadside", "full"),
        default="broadside",
        help="Choose broadside or full 75-angle sequence for optional PICMUS phantom plots.",
    )
    parser.addoption(
        "--picmus-phantom-grid-wavelengths",
        action="store",
        type=float,
        default=None,
        metavar="N_WAVELENGTHS",
        help=(
            "Use a diagnostic plot beamforming grid with this spacing in center-frequency wavelengths "
            "(for example, 0.25 for lambda/4)."
        ),
    )


@pytest.fixture
def picmus_phantom_plot(request: pytest.FixtureRequest) -> Path | None:
    """Return the optional PICMUS phantom diagnostic plot output path."""
    figure_path = request.config.getoption("--plot-picmus-phantom")
    if figure_path is None:
        return None
    return Path(figure_path)


@pytest.fixture
def picmus_phantom_angles(request: pytest.FixtureRequest) -> str:
    """Return the PICMUS phantom angle mode for optional diagnostic plots."""
    return cast("str", request.config.getoption("--picmus-phantom-angles"))


@pytest.fixture
def picmus_phantom_grid_wavelengths(request: pytest.FixtureRequest) -> float | None:
    """Return optional diagnostic plot grid spacing in center-frequency wavelengths."""
    grid_wavelengths = cast("float | None", request.config.getoption("--picmus-phantom-grid-wavelengths"))
    if grid_wavelengths is not None and (not math.isfinite(grid_wavelengths) or grid_wavelengths <= 0.0):
        raise pytest.UsageError("--picmus-phantom-grid-wavelengths must be finite and positive")
    return grid_wavelengths


# Try to import array backends


def _cupy_has_cuda_device(cupy_module: Any) -> bool:
    """Return whether CuPy can see at least one CUDA device."""
    try:
        return int(cupy_module.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


HAS_NUMPY = False
np = None
with contextlib.suppress(ImportError):
    import numpy as np

    HAS_NUMPY = True

HAS_JAX = False
jnp = None
with contextlib.suppress(ImportError):
    import jax.numpy as jnp

    HAS_JAX = True

HAS_MLX = False
mx = None
with contextlib.suppress(ImportError):
    import mlx.core as mx

    from fast_simus.backends.mlx import ensure_compat

    ensure_compat(mx)
    HAS_MLX = True

HAS_CUPY = False
cp = None
with contextlib.suppress(ImportError):
    import cupy as cp

    HAS_CUPY = _cupy_has_cuda_device(cp)


@pytest.fixture(
    params=[
        pytest.param(np, id="numpy", marks=pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")),
        pytest.param(
            jnp,
            id="jax",
            marks=[
                pytest.mark.skipif(not HAS_JAX, reason="JAX not available"),
            ],
        ),
        pytest.param(
            mx,
            id="mlx",
            marks=pytest.mark.skipif(not HAS_MLX, reason="MLX not available"),
        ),
        pytest.param(
            cp,
            id="cupy",
            marks=pytest.mark.skipif(not HAS_CUPY, reason="CuPy CUDA device not available"),
        ),
    ]
)
def xp(request) -> _ArrayNamespace:
    """Fixture providing different array API backends.

    Parametrizes tests to run with NumPy, JAX, MLX, and CuPy.
    Does not include array-api-strict, which can be used in place of a parametrized backend
    for Array API compliance testing.
    """
    return cast("_ArrayNamespace", request.param)


@pytest.fixture(
    params=[
        pytest.param(None, id="auto"),
        pytest.param(PfieldStrategy.VECTORIZED, id="vectorized"),
        pytest.param(PfieldStrategy.SCAN, id="scan"),
        pytest.param(PfieldStrategy.METAL, id="metal"),
    ],
)
def strategy(request) -> PfieldStrategy | None:
    """Fixture providing different pfield strategies."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(None, id="auto"),
        pytest.param(SimusStrategy.PYTHON, id="python"),
        pytest.param(SimusStrategy.SCAN, id="scan"),
        pytest.param(SimusStrategy.METAL, id="metal"),
        pytest.param(SimusStrategy.CUDA, id="cuda"),
    ]
)
def simus_strategy(request) -> SimusStrategy | None:
    """Fixture providing different simus strategies."""
    return request.param
