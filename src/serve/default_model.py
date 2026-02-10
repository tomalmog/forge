"""Default PyTorch model architecture for language modeling.

This module defines a compact causal language model used when
no custom architecture is supplied by the user.
"""

from __future__ import annotations

from typing import Any, cast

from core.constants import DEFAULT_TRAIN_DROPOUT, DEFAULT_TRAIN_EMBED_DIM


def build_default_model(
    torch_module: Any,
    vocab_size: int,
    hidden_dim: int,
    num_layers: int,
) -> Any:
    """Build the default causal language model.

    Args:
        torch_module: Imported torch module.
        vocab_size: Vocabulary size.
        hidden_dim: Recurrent hidden dimension.
        num_layers: Recurrent layer count.

    Returns:
        torch.nn.Module default model.
    """

    module_base = cast(type, torch_module.nn.Module)

    class DefaultCausalModel(module_base):  # type: ignore[misc,valid-type]
        """Embedding + GRU causal language model."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch_module.nn.Embedding(vocab_size, DEFAULT_TRAIN_EMBED_DIM)
            self.recurrent = torch_module.nn.GRU(
                input_size=DEFAULT_TRAIN_EMBED_DIM,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=DEFAULT_TRAIN_DROPOUT if num_layers > 1 else 0.0,
            )
            self.output = torch_module.nn.Linear(hidden_dim, vocab_size)

        def forward(self, inputs: Any) -> Any:
            embedded = self.embedding(inputs)
            recurrent_output, _ = self.recurrent(embedded)
            logits = self.output(recurrent_output)
            return logits

    return DefaultCausalModel()
