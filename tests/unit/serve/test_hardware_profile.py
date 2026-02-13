"""Unit tests for hardware capability detection."""

from __future__ import annotations

from serve.hardware_profile import detect_hardware_profile


class _FakeDeviceProps:
    def __init__(self, name: str, total_memory: int, major: int, minor: int) -> None:
        self.name = name
        self.total_memory = total_memory
        self.major = major
        self.minor = minor


class _FakeCuda:
    def __init__(self, available: bool, bf16: bool, device_name: str, memory_gb: int) -> None:
        self._available = available
        self._bf16 = bf16
        self._device_name = device_name
        self._total_memory = memory_gb * (1024**3)

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return 1

    def is_bf16_supported(self) -> bool:
        return self._bf16

    def get_device_properties(self, index: int) -> _FakeDeviceProps:
        _ = index
        return _FakeDeviceProps(self._device_name, self._total_memory, 9, 0)


class _FakeMpsBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _FakeMpsBackend(available=mps_available)


class _FakeTorch:
    def __init__(
        self,
        available: bool,
        bf16: bool,
        device_name: str,
        memory_gb: int,
        mps_available: bool = False,
    ) -> None:
        self.cuda = _FakeCuda(
            available=available, bf16=bf16, device_name=device_name, memory_gb=memory_gb
        )
        self.backends = _FakeBackends(mps_available=mps_available)


def test_detect_hardware_profile_returns_cpu_defaults_when_cuda_unavailable() -> None:
    """Detector should return CPU defaults when CUDA is unavailable."""
    profile = detect_hardware_profile(
        torch_module=_FakeTorch(available=False, bf16=False, device_name="none", memory_gb=0)
    )

    assert profile.accelerator == "cpu" and profile.recommended_precision_mode == "fp32"


def test_detect_hardware_profile_detects_a100_and_bf16() -> None:
    """Detector should infer A100 profile and bf16 precision recommendation."""
    profile = detect_hardware_profile(
        torch_module=_FakeTorch(available=True, bf16=True, device_name="NVIDIA A100", memory_gb=80)
    )

    assert (
        profile.accelerator == "cuda"
        and profile.suggested_profile == "a100"
        and profile.recommended_precision_mode == "bf16"
        and profile.recommended_batch_size == 64
    )


def test_detect_hardware_profile_uses_mps_when_cuda_is_unavailable() -> None:
    """Detector should expose MPS profile on Apple silicon setups."""
    profile = detect_hardware_profile(
        torch_module=_FakeTorch(
            available=False,
            bf16=False,
            device_name="none",
            memory_gb=0,
            mps_available=True,
        )
    )

    assert (
        profile.accelerator == "mps"
        and profile.gpu_count == 1
        and profile.recommended_precision_mode == "fp32"
        and profile.suggested_profile == "apple_mps"
    )
