"""Unit tests for default model positional embedding behavior."""

from __future__ import annotations

import pytest

from core.types import TrainingOptions
from serve.default_model import build_default_model


def test_build_default_model_supports_sinusoidal_positions(tmp_path) -> None:
    """Sinusoidal mode should run forward pass without learned position table."""
    torch = pytest.importorskip("torch")
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        hidden_dim=32,
        attention_heads=4,
        mlp_hidden_dim=64,
        mlp_layers=2,
        num_layers=1,
        max_token_length=64,
        position_embedding_type="sinusoidal",
    )
    model = build_default_model(torch, vocab_size=40, options=options)
    sample_inputs = torch.randint(0, 40, (2, 8))

    logits = model(sample_inputs)

    assert (
        hasattr(model, "sinusoidal_position_encoding")
        and not hasattr(model, "position_embedding")
        and tuple(logits.shape) == (2, 8, 40)
    )


def test_build_default_model_learned_positions_uses_embedding(tmp_path) -> None:
    """Learned mode should construct trainable positional embedding table."""
    torch = pytest.importorskip("torch")
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        hidden_dim=32,
        attention_heads=4,
        mlp_hidden_dim=64,
        mlp_layers=2,
        num_layers=1,
        max_token_length=64,
        position_embedding_type="learned",
    )
    model = build_default_model(torch, vocab_size=40, options=options)

    assert hasattr(model, "position_embedding") and model.position_embedding.num_embeddings == 64
