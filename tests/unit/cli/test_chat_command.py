"""Unit tests for chat CLI command wiring."""

from __future__ import annotations

from cli.main import main
from core.chat_types import ChatResult
from store.dataset_sdk import ForgeClient


def test_cli_chat_prints_response_text(monkeypatch, capsys) -> None:
    """Chat command should print model response text."""
    captured: dict[str, object] = {}

    def _fake_chat(self, options):
        captured["position_embedding_type"] = options.position_embedding_type
        _ = options
        return ChatResult(response_text="hello from model")

    monkeypatch.setattr(ForgeClient, "chat", _fake_chat)
    args = [
        "chat",
        "--dataset",
        "demo",
        "--model-path",
        "./outputs/train/demo/model.pt",
        "--prompt",
        "hello",
        "--position-embedding-type",
        "sinusoidal",
    ]

    exit_code = main(args)
    output = capsys.readouterr().out.strip()

    assert (
        exit_code == 0
        and output == "hello from model"
        and captured == {"position_embedding_type": "sinusoidal"}
    )
