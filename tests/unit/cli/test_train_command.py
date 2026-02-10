"""Unit tests for train CLI command wiring."""

from __future__ import annotations

from core.types import TrainingRunResult
from cli.main import main
from store.dataset_sdk import ForgeClient


def test_cli_train_prints_training_artifact_paths(monkeypatch, tmp_path, capsys) -> None:
    """Train command should print returned training artifact paths."""

    def _fake_train(self, options):
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
    ]

    exit_code = main(args)
    output = capsys.readouterr().out.strip().splitlines()

    assert exit_code == 0 and output[0].startswith("model_path=")
