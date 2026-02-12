"""Unit tests for chat option resolution helpers."""

from __future__ import annotations

from pathlib import Path

from core.chat_types import ChatOptions
from core.types import DataRecord, RecordMetadata, TrainingOptions
from serve.chat_option_resolver import resolve_chat_tokenizer, resolve_chat_training_options
from serve.tokenization import VocabularyTokenizer
from serve.training_metadata import save_tokenizer_vocabulary, save_training_config


def test_resolve_chat_training_options_prefers_persisted_config(tmp_path: Path) -> None:
    """Persisted training config should drive chat model options."""
    save_training_config(
        tmp_path,
        TrainingOptions(
            dataset_name="demo",
            output_dir=str(tmp_path),
            hidden_dim=320,
            position_embedding_type="sinusoidal",
            max_token_length=300,
        ),
    )
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    options = ChatOptions(dataset_name="demo", model_path=str(model_path), prompt="hello")

    resolved = resolve_chat_training_options(options, model_state={})

    assert (
        resolved.hidden_dim == 320
        and resolved.position_embedding_type == "sinusoidal"
        and resolved.max_token_length == 300
    )


def test_resolve_chat_tokenizer_prefers_persisted_vocabulary(tmp_path: Path) -> None:
    """Persisted tokenizer should be reused instead of refitting records."""
    save_tokenizer_vocabulary(
        tmp_path,
        VocabularyTokenizer(vocabulary={"<pad>": 0, "<unk>": 1, "persisted": 2}),
    )
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    options = ChatOptions(dataset_name="demo", model_path=str(model_path), prompt="hello")
    records = [
        DataRecord(
            record_id="r-1",
            text="fresh tokens only",
            metadata=RecordMetadata(
                source_uri="data.txt",
                language="en",
                quality_score=0.5,
                perplexity=1.0,
            ),
        )
    ]
    training_options = TrainingOptions(dataset_name="demo", output_dir=str(tmp_path))

    tokenizer = resolve_chat_tokenizer(records, options, training_options)

    assert tokenizer.vocabulary == {"<pad>": 0, "<unk>": 1, "persisted": 2}
