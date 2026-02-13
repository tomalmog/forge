"""Unit tests for mixed-precision runtime helpers."""

from __future__ import annotations

from serve.training_precision import build_training_precision_runtime


class _FakeGradScaler:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled


class _FakeAmp:
    GradScaler = _FakeGradScaler


class _FakeCuda:
    def __init__(self, bf16_supported: bool) -> None:
        self._bf16_supported = bf16_supported
        self.amp = _FakeAmp()

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported


class _FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self, bf16_supported: bool) -> None:
        self.cuda = _FakeCuda(bf16_supported=bf16_supported)


class _FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


def test_build_precision_runtime_auto_on_cpu_uses_fp32() -> None:
    """Auto precision should resolve to fp32 on non-CUDA devices."""
    runtime = build_training_precision_runtime(
        torch_module=_FakeTorch(bf16_supported=True),
        requested_mode="auto",
        device=_FakeDevice("cpu"),
    )

    assert (
        runtime.resolved_mode == "fp32"
        and not runtime.autocast_enabled
        and runtime.scaler is None
    )


def test_build_precision_runtime_auto_on_cuda_prefers_bf16() -> None:
    """Auto precision should prefer bf16 when CUDA bf16 support exists."""
    runtime = build_training_precision_runtime(
        torch_module=_FakeTorch(bf16_supported=True),
        requested_mode="auto",
        device=_FakeDevice("cuda"),
    )

    assert (
        runtime.resolved_mode == "bf16"
        and runtime.autocast_enabled
        and runtime.autocast_dtype == "bfloat16"
        and runtime.scaler is None
    )


def test_build_precision_runtime_bf16_falls_back_to_fp16_when_unsupported() -> None:
    """bf16 request should fall back to fp16 when hardware lacks bf16 support."""
    runtime = build_training_precision_runtime(
        torch_module=_FakeTorch(bf16_supported=False),
        requested_mode="bf16",
        device=_FakeDevice("cuda"),
    )

    assert (
        runtime.resolved_mode == "fp16"
        and runtime.autocast_enabled
        and runtime.autocast_dtype == "float16"
        and runtime.scaler is not None
        and runtime.scaler.enabled
    )
