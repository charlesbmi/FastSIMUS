"""Behavioral contracts for CUDA SIMUS resource eligibility."""

from __future__ import annotations

import pytest

from fast_simus.kernels import _cuda_capabilities as capabilities


@pytest.mark.parametrize("device_limit", [48 * 1024, 96 * 1024])
def test_cuda_shared_memory_accepts_workload_within_device_limit(monkeypatch, device_limit) -> None:
    """A workload fitting the current device remains eligible for CUDA."""
    monkeypatch.setattr(capabilities, "cuda_dynamic_shared_memory_limit", lambda: device_limit)

    assert capabilities.cuda_shared_memory_unsupported_reason(128, 2) is None


def test_cuda_shared_memory_reports_device_specific_limit(monkeypatch) -> None:
    """Eligibility reports both the required memory and the device limit."""
    required = (7 * 128 * 2 + 3 * 128) * 4  # One scatterer is the smallest executable tile.
    device_limit = required - 1
    monkeypatch.setattr(capabilities, "cuda_dynamic_shared_memory_limit", lambda: device_limit)

    reason = capabilities.cuda_shared_memory_unsupported_reason(128, 2)

    assert reason is not None
    assert str(required) in reason
    assert str(device_limit) in reason
