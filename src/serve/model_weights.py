"""Model weight loading utilities for training runs.

This module applies optional initial model weights for fine-tuning.
It keeps checkpoint parsing and error reporting separate from loop logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.errors import ForgeServeError


def load_initial_weights(
    torch_module: Any,
    model: Any,
    initial_weights_path: str | None,
    device: Any,
) -> None:
    """Load optional initial model weights into the training model.

    Args:
        torch_module: Imported torch module.
        model: Model instance receiving state dict.
        initial_weights_path: Optional checkpoint path.
        device: Resolved training device.

    Raises:
        ForgeServeError: If weights cannot be read or applied.
    """
    if initial_weights_path is None:
        return
    resolved_path = Path(initial_weights_path).expanduser().resolve()
    if not resolved_path.exists():
        raise ForgeServeError(
            f"Initial weights not found at {resolved_path}. "
            "Provide a valid --initial-weights-path or omit it to train from scratch."
        )
    try:
        checkpoint_payload = torch_module.load(str(resolved_path), map_location=device)
    except (OSError, RuntimeError) as error:
        raise ForgeServeError(
            f"Failed to load initial weights from {resolved_path}: {error}. "
            "Verify the checkpoint file is readable and valid."
        ) from error
    model_state = _extract_model_state(checkpoint_payload, resolved_path)
    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise ForgeServeError(
            f"Failed to apply initial weights from {resolved_path}: {error}. "
            "Use matching architecture settings or train from scratch."
        ) from error


def _extract_model_state(checkpoint_payload: object, weights_path: Path) -> Mapping[str, object]:
    """Extract model state dict from raw checkpoint payload."""
    if isinstance(checkpoint_payload, Mapping):
        nested_payload = checkpoint_payload.get("model_state_dict")
        if isinstance(nested_payload, Mapping):
            return nested_payload
        if all(isinstance(key, str) for key in checkpoint_payload):
            return checkpoint_payload
    raise ForgeServeError(
        f"Invalid checkpoint format at {weights_path}: expected a state_dict mapping "
        "or a mapping containing model_state_dict."
    )
