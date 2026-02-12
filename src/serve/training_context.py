"""Runtime context passed to custom training loops.

This module defines the context object that user-defined loops receive.
It allows custom loops to reuse Forge-prepared datasets and model state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import TrainingOptions
from serve.tokenization import SequenceBatch, VocabularyTokenizer


@dataclass
class TrainingRuntimeContext:
    """Runtime context object for custom training implementations."""

    torch_module: Any
    model: Any
    optimizer: Any
    loss_function: Any
    train_batches: list[SequenceBatch]
    validation_batches: list[SequenceBatch]
    tokenizer: VocabularyTokenizer
    options: TrainingOptions
    output_dir: Path
    device: Any
