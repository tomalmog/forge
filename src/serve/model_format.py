"""Model artifact format helpers.

This module centralizes file-format detection for model artifacts used by
training and inference paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

ModelFormat = Literal["pytorch", "onnx", "unknown"]


def detect_model_format(model_path: str) -> ModelFormat:
    """Detect model format from artifact file extension."""
    suffix = Path(model_path).expanduser().resolve().suffix.lower()
    if suffix == ".pt":
        return "pytorch"
    if suffix == ".onnx":
        return "onnx"
    return "unknown"
