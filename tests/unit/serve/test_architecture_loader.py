"""Unit tests for architecture loader behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import serve.architecture_loader as architecture_loader
from core.errors import ForgeServeError
from core.types import TrainingOptions
from serve.architecture_loader import load_training_model


class _FakeTorch:
    class nn:
        class Module:
            pass


def test_load_training_model_raises_for_missing_architecture_file(tmp_path: Path) -> None:
    """Missing architecture file should raise serve error."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        architecture_path=str(tmp_path / "missing.py"),
    )

    with pytest.raises(ForgeServeError):
        load_training_model(_FakeTorch(), options, vocab_size=10)

    assert True


def test_load_training_model_reads_json_default_architecture(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """JSON default architecture config should load model instance."""

    def _fake_build_default_model(torch_module, vocab_size, hidden_dim, num_layers):
        return _FakeTorch.nn.Module()

    monkeypatch.setattr(architecture_loader, "build_default_model", _fake_build_default_model)
    architecture_path = tmp_path / "architecture.json"
    architecture_path.write_text(
        json.dumps({"architecture": "default", "hidden_dim": 8, "num_layers": 1}),
        encoding="utf-8",
    )
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        architecture_path=str(architecture_path),
    )

    model = load_training_model(_FakeTorch(), options, vocab_size=10)

    assert isinstance(model, _FakeTorch.nn.Module)
