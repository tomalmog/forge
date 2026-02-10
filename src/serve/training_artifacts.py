"""Training artifacts persistence and graph generation.

This module writes training histories and produces standard curves.
It captures epoch loss metrics for train and validation phases.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from core.constants import (
    DEFAULT_TRAIN_HISTORY_FILE_NAME,
    DEFAULT_TRAIN_PLOT_FILE_NAME,
    DEFAULT_TRAINED_MODEL_FILE_NAME,
)
from core.errors import ForgeDependencyError
from core.types import EpochMetric


def ensure_training_output_dir(output_dir: str) -> Path:
    """Create and return resolved training output directory.

    Args:
        output_dir: Configured output directory.

    Returns:
        Resolved output path.
    """
    resolved_path = Path(output_dir).expanduser().resolve()
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def save_training_history(output_dir: Path, metrics: list[EpochMetric]) -> Path:
    """Persist training history JSON file.

    Args:
        output_dir: Training output directory.
        metrics: Epoch metrics.

    Returns:
        History JSON file path.
    """
    history_path = output_dir / DEFAULT_TRAIN_HISTORY_FILE_NAME
    payload = {"epochs": [asdict(metric) for metric in metrics]}
    history_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return history_path


def save_model_weights(output_dir: Path, torch_module: Any, model: Any) -> Path:
    """Persist model state dict to output directory.

    Args:
        output_dir: Training output directory.
        torch_module: Imported torch module.
        model: Trained torch model.

    Returns:
        Model file path.
    """
    model_path = output_dir / DEFAULT_TRAINED_MODEL_FILE_NAME
    torch_module.save(model.state_dict(), str(model_path))
    return model_path


def save_training_plot(output_dir: Path, metrics: list[EpochMetric]) -> Path | None:
    """Save loss curves as PNG when matplotlib is available.

    Args:
        output_dir: Training output directory.
        metrics: Epoch metrics.

    Returns:
        Plot file path when generated; otherwise None.

    Raises:
        ForgeDependencyError: If matplotlib is missing.
    """
    if not metrics:
        return None
    try:
        import matplotlib.pyplot as plot
    except ImportError as error:
        raise ForgeDependencyError(
            "Training plot generation requires matplotlib. "
            "Install matplotlib to produce loss graphs."
        ) from error
    epoch_indexes = [metric.epoch for metric in metrics]
    train_losses = [metric.train_loss for metric in metrics]
    validation_losses = [metric.validation_loss for metric in metrics]
    figure = plot.figure(figsize=(8, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(epoch_indexes, train_losses, label="train_loss")
    axis.plot(epoch_indexes, validation_losses, label="validation_loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training Curves")
    axis.grid(alpha=0.3)
    axis.legend()
    plot_path = output_dir / DEFAULT_TRAIN_PLOT_FILE_NAME
    figure.tight_layout()
    figure.savefig(plot_path)
    plot.close(figure)
    return plot_path
