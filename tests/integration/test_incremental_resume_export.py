"""Integration tests for incremental, resume, and training export workflows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.config import ForgeConfig
from core.errors import ForgeStoreError
from core.types import IngestOptions
from ingest.pipeline import ingest_dataset
from store.dataset_sdk import ForgeClient
from store.snapshot_store import SnapshotStore


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_incremental_ingest_creates_child_version(tmp_path: Path) -> None:
    """Incremental run should create a child version from latest snapshot."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_text(source_dir / "a.txt", "alpha")
    _write_text(source_dir / "b.txt", "beta")
    config = replace(ForgeConfig.from_env(), data_root=tmp_path / "forge")
    client = ForgeClient(config)

    first_version = client.ingest(
        IngestOptions(dataset_name="demo", source_uri=str(source_dir), incremental=False)
    )
    _write_text(source_dir / "b.txt", "beta updated")
    second_version = client.ingest(
        IngestOptions(dataset_name="demo", source_uri=str(source_dir), incremental=True)
    )
    second_manifest = client.dataset("demo").list_versions()[-1]

    assert second_manifest.parent_version == first_version and second_version != first_version


def test_resume_uses_checkpoint_without_source_files(tmp_path: Path, monkeypatch) -> None:
    """Resume should finish from checkpoint after a snapshot-write failure."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "a.txt"
    _write_text(source_file, "alpha")
    config = replace(ForgeConfig.from_env(), data_root=tmp_path / "forge")
    options = IngestOptions(dataset_name="demo", source_uri=str(source_dir), incremental=False)

    original_create_snapshot = SnapshotStore.create_snapshot

    def _failing_create_snapshot(self, request):
        raise ForgeStoreError("forced failure")

    monkeypatch.setattr(SnapshotStore, "create_snapshot", _failing_create_snapshot)
    with pytest.raises(ForgeStoreError):
        ingest_dataset(options, config)
    source_file.unlink()
    monkeypatch.setattr(SnapshotStore, "create_snapshot", original_create_snapshot)

    version_id = ingest_dataset(replace(options, resume=True), config)

    assert version_id.startswith("demo-")


def test_export_training_creates_manifest(tmp_path: Path) -> None:
    """Training export should create manifest file on disk."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_text(source_dir / "a.txt", "alpha training sample")
    config = replace(ForgeConfig.from_env(), data_root=tmp_path / "forge")
    client = ForgeClient(config)
    client.ingest(IngestOptions(dataset_name="demo", source_uri=str(source_dir)))

    manifest_path = client.dataset("demo").export_training(output_dir=str(tmp_path / "exports"))

    assert Path(manifest_path).exists()
