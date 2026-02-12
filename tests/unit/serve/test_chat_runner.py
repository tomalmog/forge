"""Unit tests for chat runner dependency behavior."""

from __future__ import annotations

import builtins

import pytest

from core.chat_types import ChatOptions
from core.errors import ForgeDependencyError
from core.types import DataRecord, RecordMetadata
from serve.chat_runner import run_chat


def _build_records() -> list[DataRecord]:
    metadata = RecordMetadata(
        source_uri="a.txt",
        language="en",
        quality_score=0.9,
        perplexity=1.5,
    )
    return [DataRecord(record_id="id-1", text="alpha beta gamma", metadata=metadata)]


def test_run_chat_raises_without_torch(monkeypatch) -> None:
    """Chat should fail clearly when torch dependency is missing."""
    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)
    options = ChatOptions(
        dataset_name="demo",
        model_path="./outputs/train/demo/model.pt",
        prompt="hello",
    )

    with pytest.raises(ForgeDependencyError):
        run_chat(_build_records(), options)

    assert True
