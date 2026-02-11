"""Unit tests for train CLI command wiring."""

from __future__ import annotations

from cli.main import main
from core.types import TrainingRunResult
from store.dataset_sdk import ForgeClient


def test_cli_train_prints_training_artifact_paths(monkeypatch, tmp_path, capsys) -> None:
    """Train command should print returned training artifact paths."""
    captured: dict[str, object] = {}

    def _fake_train(self, options):
        captured["attention_heads"] = options.attention_heads
        captured["mlp_layers"] = options.mlp_layers
        captured["vocabulary_size"] = options.vocabulary_size
        captured["initial_weights_path"] = options.initial_weights_path
        return TrainingRunResult(
            model_path=str(tmp_path / "model.pt"),
            history_path=str(tmp_path / "history.json"),
            plot_path=str(tmp_path / "curves.png"),
            epochs_completed=2,
        )

    monkeypatch.setattr(ForgeClient, "train", _fake_train)
    args = [
        "train",
        "--dataset",
        "demo",
        "--output-dir",
        str(tmp_path),
        "--attention-heads",
        "4",
        "--mlp-layers",
        "3",
        "--vocabulary-size",
        "1200",
        "--initial-weights-path",
        str(tmp_path / "init.pt"),
    ]

    exit_code = main(args)
    output = capsys.readouterr().out.strip().splitlines()

    assert (
        exit_code == 0
        and output[0].startswith("model_path=")
        and captured
        == {
            "attention_heads": 4,
            "mlp_layers": 3,
            "vocabulary_size": 1200,
            "initial_weights_path": str(tmp_path / "init.pt"),
        }
    )
