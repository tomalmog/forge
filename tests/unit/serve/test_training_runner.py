"""Unit tests for training runner dependency behavior."""

from __future__ import annotations

import builtins

import pytest

from core.errors import ForgeDependencyError
from core.types import DataRecord, RecordMetadata, TrainingOptions
from serve.training_runner import run_training


def _build_records() -> list[DataRecord]:
    metadata = RecordMetadata(
        source_uri="a.txt",
        language="en",
        quality_score=0.9,
        perplexity=1.5,
    )
    return [DataRecord(record_id="id-1", text="alpha beta gamma", metadata=metadata)]


def test_run_training_raises_without_torch(monkeypatch, tmp_path) -> None:
    """Training should fail clearly when torch dependency is missing."""
    original_import = builtins.__import__

    def _patched_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _patched_import)
    options = TrainingOptions(dataset_name="demo", output_dir=str(tmp_path))

    with pytest.raises(ForgeDependencyError):
        run_training(_build_records(), options, random_seed=1)

    assert True
