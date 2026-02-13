"""Unit tests for model-format detection helpers."""

from __future__ import annotations

from serve.model_format import detect_model_format


def test_detect_model_format_returns_pytorch_for_pt() -> None:
    """Detector should classify .pt files as pytorch artifacts."""
    assert detect_model_format("./outputs/model.pt") == "pytorch"


def test_detect_model_format_returns_onnx_for_onnx() -> None:
    """Detector should classify .onnx files as ONNX artifacts."""
    assert detect_model_format("./exports/model.onnx") == "onnx"


def test_detect_model_format_returns_unknown_for_other_suffix() -> None:
    """Detector should mark unsupported file extensions as unknown."""
    assert detect_model_format("./outputs/model.bin") == "unknown"
