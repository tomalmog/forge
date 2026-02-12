"""Persistence helpers for training configuration and tokenizer artifacts.

This module stores and reloads metadata that keeps training and chat
runtime settings consistent across separate commands.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from core.constants import (
    DEFAULT_TOKENIZER_VOCAB_FILE_NAME,
    DEFAULT_TRAINING_CONFIG_FILE_NAME,
)
from core.errors import ForgeServeError
from core.types import TrainingOptions
from serve.tokenization import VocabularyTokenizer


def save_training_config(output_dir: Path, options: TrainingOptions) -> Path:
    """Persist training options used to build the model architecture."""
    config_path = output_dir / DEFAULT_TRAINING_CONFIG_FILE_NAME
    config_path.write_text(json.dumps(asdict(options), indent=2) + "\n", encoding="utf-8")
    return config_path


def save_tokenizer_vocabulary(output_dir: Path, tokenizer: VocabularyTokenizer) -> Path:
    """Persist fitted tokenizer vocabulary used during training."""
    vocabulary_path = output_dir / DEFAULT_TOKENIZER_VOCAB_FILE_NAME
    payload = {
        token: token_id
        for token, token_id in sorted(tokenizer.vocabulary.items(), key=_sort_vocabulary_item)
    }
    vocabulary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return vocabulary_path


def load_training_config(model_path: str) -> dict[str, object] | None:
    """Load persisted training config located beside model weights."""
    config_path = _artifact_dir(model_path) / DEFAULT_TRAINING_CONFIG_FILE_NAME
    if not config_path.exists():
        return None
    payload = _read_json_payload(config_path)
    if not isinstance(payload, dict):
        raise ForgeServeError(
            f"Invalid training config format at {config_path}: expected JSON object."
        )
    return cast(dict[str, object], payload)


def load_tokenizer(model_path: str) -> VocabularyTokenizer | None:
    """Load persisted training tokenizer located beside model weights."""
    vocabulary_path = _artifact_dir(model_path) / DEFAULT_TOKENIZER_VOCAB_FILE_NAME
    if not vocabulary_path.exists():
        return None
    payload = _read_json_payload(vocabulary_path)
    if not isinstance(payload, dict):
        raise ForgeServeError(
            f"Invalid tokenizer vocabulary format at {vocabulary_path}: expected JSON object."
        )
    vocabulary: dict[str, int] = {}
    for raw_token, raw_token_id in payload.items():
        if not isinstance(raw_token, str):
            raise ForgeServeError(
                f"Invalid tokenizer vocabulary at {vocabulary_path}: token keys must be strings."
            )
        if not isinstance(raw_token_id, int):
            raise ForgeServeError(
                f"Invalid tokenizer vocabulary at {vocabulary_path}: token id for "
                f"{raw_token!r} must be an integer."
            )
        vocabulary[raw_token] = raw_token_id
    return VocabularyTokenizer(vocabulary=vocabulary)


def _artifact_dir(model_path: str) -> Path:
    """Resolve output directory containing model and metadata artifacts."""
    return Path(model_path).expanduser().resolve().parent


def _read_json_payload(payload_path: Path) -> object:
    """Read JSON payload from disk with traceable errors."""
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ForgeServeError(
            f"Failed to parse JSON artifact at {payload_path}: {error.msg}. "
            "Re-run training to regenerate valid artifacts."
        ) from error
    except OSError as error:
        raise ForgeServeError(
            f"Failed to read artifact at {payload_path}: {error}. "
            "Verify file permissions and retry."
        ) from error


def _sort_vocabulary_item(item: tuple[str, int]) -> tuple[int, str]:
    """Sort vocabulary output by token id for deterministic storage."""
    token, token_id = item
    return token_id, token
