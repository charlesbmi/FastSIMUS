"""Tests for public backend discovery and selection."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import cast

import pytest

from fast_simus import Backend, BackendKind, get_backend
from fast_simus.utils._array_api import ArrayNamespace

selection = import_module("fast_simus.backends._selection")


def test_explicit_numpy_backend_is_usable() -> None:
    """An explicit NumPy context exposes an Array API namespace."""
    backend = get_backend("numpy")

    assert isinstance(backend, Backend)
    assert backend.kind is BackendKind.NUMPY
    assert backend.xp.asarray([1.0, 2.0]).shape == (2,)
    assert "NumPy" in backend.label


def test_auto_backend_resolves_to_concrete_backend() -> None:
    """Automatic selection returns a concrete, reusable context."""
    backend = get_backend()

    assert backend.kind is not BackendKind.AUTO
    assert backend.xp.asarray([1.0]).shape == (1,)
    assert backend.label


def test_unknown_backend_name_is_rejected() -> None:
    """Typos in explicit backend names fail at the public boundary."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("quantum")


def test_auto_backend_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA wins before Metal and JAX when a device is available."""
    cuda = cast(ArrayNamespace, SimpleNamespace(__name__="cupy"))
    monkeypatch.setattr(selection, "_cupy_namespace", lambda: cuda)
    monkeypatch.setattr(selection, "_mlx_namespace", lambda: pytest.fail("Metal loader should not run"))
    monkeypatch.setattr(selection, "_jax_namespace", lambda: pytest.fail("JAX loader should not run"))

    backend = get_backend()

    assert backend.kind is BackendKind.CUDA
    assert backend.xp is cuda


def test_auto_backend_falls_back_after_missing_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery proceeds to JAX when CUDA and Metal are unavailable."""
    jax = cast(ArrayNamespace, SimpleNamespace(__name__="jax.numpy"))
    monkeypatch.setattr(selection, "_cupy_namespace", lambda: None)
    monkeypatch.setattr(selection, "_mlx_namespace", lambda: None)
    monkeypatch.setattr(selection, "_jax_namespace", lambda: jax)

    backend = get_backend()

    assert backend.kind is BackendKind.JAX
    assert backend.xp is jax


def test_explicit_unavailable_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit acceleration never silently selects a different runtime."""
    monkeypatch.setattr(selection, "_cupy_namespace", lambda: None)

    with pytest.raises(RuntimeError, match="visible CUDA device"):
        get_backend(BackendKind.CUDA)
