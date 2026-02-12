"""Training setup helpers for tokenizer and option validation.

This module centralizes pre-loop setup checks for training runs.
It keeps validation and tokenizer fitting isolated from loop execution.
"""

from __future__ import annotations

from core.errors import ForgeServeError
from core.types import DataRecord, TrainingOptions
from serve.tokenization import VocabularyTokenizer


def fit_training_tokenizer(
    records: list[DataRecord],
    options: TrainingOptions,
) -> VocabularyTokenizer:
    """Build tokenizer vocabulary from record texts."""
    tokenizer = VocabularyTokenizer.create()
    tokenizer.fit(
        (record.text for record in records),
        max_vocabulary_size=options.vocabulary_size,
    )
    return tokenizer


def validate_training_options(options: TrainingOptions) -> None:
    """Validate training options before loop execution."""
    if options.epochs < 1:
        raise ForgeServeError(f"Invalid epochs value {options.epochs}: expected value >= 1.")
    if options.batch_size < 1:
        raise ForgeServeError(f"Invalid batch_size {options.batch_size}: expected value >= 1.")
    if options.max_token_length < 4:
        raise ForgeServeError(
            f"Invalid max_token_length {options.max_token_length}: expected value >= 4."
        )
    if options.learning_rate <= 0:
        raise ForgeServeError(
            f"Invalid learning_rate {options.learning_rate}: expected positive value."
        )
    if not 0 <= options.validation_split < 1:
        raise ForgeServeError("Invalid validation_split: expected value in [0, 1).")
    if options.hidden_dim < 1:
        raise ForgeServeError(f"Invalid hidden_dim {options.hidden_dim}: expected value >= 1.")
    if options.num_layers < 1:
        raise ForgeServeError(f"Invalid num_layers {options.num_layers}: expected value >= 1.")
    if options.attention_heads < 1:
        raise ForgeServeError(
            f"Invalid attention_heads {options.attention_heads}: expected value >= 1."
        )
    if options.hidden_dim % options.attention_heads != 0:
        raise ForgeServeError(
            f"Invalid configuration: hidden_dim {options.hidden_dim} must be divisible by "
            f"attention_heads {options.attention_heads}."
        )
    if options.mlp_hidden_dim < 1:
        raise ForgeServeError(
            f"Invalid mlp_hidden_dim {options.mlp_hidden_dim}: expected value >= 1."
        )
    if options.mlp_layers < 1:
        raise ForgeServeError(f"Invalid mlp_layers {options.mlp_layers}: expected value >= 1.")
    if not 0 <= options.dropout < 1:
        raise ForgeServeError(f"Invalid dropout {options.dropout}: expected value in [0, 1).")
    if options.position_embedding_type not in {"learned", "sinusoidal"}:
        raise ForgeServeError(
            "Invalid position_embedding_type "
            f"{options.position_embedding_type!r}: expected 'learned' or 'sinusoidal'."
        )
    if options.vocabulary_size is not None and options.vocabulary_size < 2:
        raise ForgeServeError(
            f"Invalid vocabulary_size {options.vocabulary_size}: expected value >= 2."
        )
