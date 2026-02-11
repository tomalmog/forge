"""Unit tests for training setup helpers."""

from __future__ import annotations

import pytest

from core.errors import ForgeServeError
from core.types import DataRecord, RecordMetadata, TrainingOptions
from serve.training_setup import fit_training_tokenizer, validate_training_options


def _build_records() -> list[DataRecord]:
    metadata = RecordMetadata(
        source_uri="input.txt",
        language="en",
        quality_score=0.8,
        perplexity=2.1,
    )
    return [DataRecord(record_id="rid", text="alpha beta gamma", metadata=metadata)]


def test_validate_training_options_rejects_non_divisible_attention_shape(tmp_path) -> None:
    """Validation should fail when hidden dimension is not divisible by heads."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        hidden_dim=10,
        attention_heads=3,
    )

    with pytest.raises(ForgeServeError):
        validate_training_options(options)

    assert True


def test_fit_training_tokenizer_applies_vocabulary_limit(tmp_path) -> None:
    """Tokenizer fitting should use configured vocabulary size cap."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        vocabulary_size=3,
    )
    tokenizer = fit_training_tokenizer(_build_records(), options)

    assert len(tokenizer.vocabulary) == 3
