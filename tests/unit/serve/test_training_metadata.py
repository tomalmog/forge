"""Unit tests for training metadata persistence helpers."""

from __future__ import annotations

from pathlib import Path

from core.types import TrainingOptions
from serve.tokenization import VocabularyTokenizer
from serve.training_metadata import (
    load_tokenizer,
    load_training_config,
    save_tokenizer_vocabulary,
    save_training_config,
)


def test_save_and_load_training_config_round_trip(tmp_path: Path) -> None:
    """Saved training config should be readable beside model weights."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        max_token_length=384,
        position_embedding_type="sinusoidal",
    )
    save_training_config(tmp_path, options)
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")

    payload = load_training_config(str(model_path))

    assert payload == {
        "dataset_name": "demo",
        "output_dir": str(tmp_path),
        "version_id": None,
        "architecture_path": None,
        "custom_loop_path": None,
        "epochs": 3,
        "learning_rate": 0.001,
        "batch_size": 16,
        "max_token_length": 384,
        "validation_split": 0.1,
        "hidden_dim": 256,
        "num_layers": 2,
        "attention_heads": 8,
        "mlp_hidden_dim": 1024,
        "mlp_layers": 2,
        "dropout": 0.1,
        "position_embedding_type": "sinusoidal",
        "vocabulary_size": None,
        "initial_weights_path": None,
    }


def test_save_and_load_tokenizer_round_trip(tmp_path: Path) -> None:
    """Saved tokenizer vocabulary should be reused for chat loading."""
    tokenizer = VocabularyTokenizer(vocabulary={"<pad>": 0, "<unk>": 1, "alpha": 2, "beta": 3})
    save_tokenizer_vocabulary(tmp_path, tokenizer)
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")

    loaded = load_tokenizer(str(model_path))

    assert loaded is not None and loaded.vocabulary == tokenizer.vocabulary


def test_load_training_config_returns_none_when_missing(tmp_path: Path) -> None:
    """Missing training config should return None without error."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")

    payload = load_training_config(str(model_path))

    assert payload is None
