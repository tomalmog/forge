"""Unit tests for snapshot store persistence."""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.config import ForgeConfig
from core.errors import ForgeStoreError
from core.types import DataRecord, MetadataFilter, RecordMetadata, SnapshotWriteRequest
from store.snapshot_store import SnapshotStore


def _sample_record() -> DataRecord:
    metadata = RecordMetadata(
        source_uri="tests/fixtures/raw/local_a.txt",
        language="en",
        quality_score=0.9,
        perplexity=3.2,
    )
    return DataRecord(record_id="id-1", text="sample text", metadata=metadata)


def test_create_snapshot_persists_manifest(tmp_path) -> None:
    """Store should create an immutable version with metadata."""
    config = replace(ForgeConfig.from_env(), data_root=tmp_path)
    store = SnapshotStore(config)
    request = SnapshotWriteRequest(
        dataset_name="demo",
        records=(_sample_record(),),
        recipe_steps=("step",),
    )

    manifest = store.create_snapshot(request)

    assert manifest.dataset_name == "demo"


def test_load_records_returns_written_payload(tmp_path) -> None:
    """Store should return records written into a snapshot."""
    config = replace(ForgeConfig.from_env(), data_root=tmp_path)
    store = SnapshotStore(config)
    request = SnapshotWriteRequest(
        dataset_name="demo",
        records=(_sample_record(),),
        recipe_steps=("step",),
    )
    store.create_snapshot(request)

    _, records = store.load_records("demo")

    assert records[0].text == "sample text"


def test_filter_records_creates_child_version(tmp_path) -> None:
    """Filter operation should create a derived snapshot version."""
    config = replace(ForgeConfig.from_env(), data_root=tmp_path)
    store = SnapshotStore(config)
    request = SnapshotWriteRequest(
        dataset_name="demo",
        records=(_sample_record(),),
        recipe_steps=("step",),
    )
    parent_manifest = store.create_snapshot(request)

    child_manifest = store.filter_records("demo", MetadataFilter(language="en"))

    assert child_manifest.parent_version == parent_manifest.version_id


def test_load_records_raises_for_unknown_dataset(tmp_path) -> None:
    """Loading should fail when dataset catalog is missing."""
    config = replace(ForgeConfig.from_env(), data_root=tmp_path)
    store = SnapshotStore(config)

    with pytest.raises(ForgeStoreError):
        store.load_records("missing")

    assert (tmp_path / "datasets" / "missing").exists()
