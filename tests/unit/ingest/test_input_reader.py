"""Unit tests for input reader module."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import ForgeConfig
from core.errors import ForgeIngestError
from ingest.input_reader import read_source_records
from tests.fixture_paths import fixture_path


def test_read_source_records_reads_directory_files() -> None:
    """Reader should collect supported files from a directory."""
    config = ForgeConfig.from_env()
    records = read_source_records(str(fixture_path("raw_valid")), config)

    assert len(records) >= 4


def test_read_source_records_raises_for_missing_path(tmp_path: Path) -> None:
    """Reader should fail when source path is missing."""
    config = ForgeConfig.from_env()
    missing_path = tmp_path / "does-not-exist"

    with pytest.raises(ForgeIngestError):
        read_source_records(str(missing_path), config)

    assert missing_path.exists() is False


def test_read_source_records_raises_for_invalid_jsonl() -> None:
    """Reader should fail for malformed or invalid JSONL payloads."""
    config = ForgeConfig.from_env()

    with pytest.raises(ForgeIngestError):
        read_source_records(str(fixture_path("raw/bad_records.jsonl")), config)

    assert fixture_path("raw/bad_records.jsonl").exists()
