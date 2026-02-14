"""Unit tests for dataset SDK chat integration."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.chat_types import ChatOptions
from core.config import ForgeConfig
from core.errors import ForgeServeError
from core.types import IngestOptions
from store.dataset_sdk import ForgeClient


def test_dataset_chat_raises_for_mismatched_dataset_name(tmp_path) -> None:
    """Dataset handle should reject chat options for different dataset name."""
    config = replace(ForgeConfig.from_env(), data_root=tmp_path)
    client = ForgeClient(config)
    source_path = "tests/fixtures/raw_valid/local_a.txt"
    client.ingest(IngestOptions(dataset_name="demo", source_uri=source_path))
    dataset = client.dataset("demo")

    with pytest.raises(ForgeServeError):
        dataset.chat(
            ChatOptions(
                model_path="./outputs/train/demo/model.pt",
                prompt="hello",
                dataset_name="other",
            )
        )

    assert True
