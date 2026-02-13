"""Unit tests for execution device selection helpers."""

from __future__ import annotations

from serve.device_selection import resolve_execution_device


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeMpsBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _FakeMpsBackend(mps_available)


class _FakeTorch:
    def __init__(self, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.backends = _FakeBackends(mps_available)

    def device(self, value: str) -> str:
        return value


def test_resolve_execution_device_prefers_cuda() -> None:
    """Resolver should prefer CUDA when both CUDA and MPS are available."""
    device = resolve_execution_device(_FakeTorch(cuda_available=True, mps_available=True))

    assert device == "cuda"


def test_resolve_execution_device_uses_mps_when_cuda_is_unavailable() -> None:
    """Resolver should fall back to MPS before CPU."""
    device = resolve_execution_device(_FakeTorch(cuda_available=False, mps_available=True))

    assert device == "mps"


def test_resolve_execution_device_uses_cpu_when_no_accelerator_is_available() -> None:
    """Resolver should return CPU when CUDA and MPS are unavailable."""
    device = resolve_execution_device(_FakeTorch(cuda_available=False, mps_available=False))

    assert device == "cpu"
