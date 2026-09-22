"""Behavioral contracts for CUDA SIMUS resource eligibility."""

from __future__ import annotations

from fast_simus.kernels import _cuda_capabilities as capabilities


def test_cuda_shared_memory_accepts_workload_within_device_limit(monkeypatch) -> None:
    """A workload fitting the current device remains eligible for CUDA."""
    required = capabilities.required_cuda_shared_memory(128, 2)
    monkeypatch.setattr(capabilities, "cuda_dynamic_shared_memory_limit", lambda: required)

    assert capabilities.cuda_shared_memory_unsupported_reason(128, 2) is None


def test_cuda_shared_memory_reports_device_specific_limit(monkeypatch) -> None:
    """Eligibility reports both the required memory and the device limit."""
    required = capabilities.required_cuda_shared_memory(128, 2)
    device_limit = required - 1
    monkeypatch.setattr(capabilities, "cuda_dynamic_shared_memory_limit", lambda: device_limit)

    reason = capabilities.cuda_shared_memory_unsupported_reason(128, 2)

    assert reason is not None
    assert str(required) in reason
    assert str(device_limit) in reason
