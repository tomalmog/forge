"""Unit tests for incremental ingest helper logic."""

from __future__ import annotations

from core.types import DataRecord, RecordMetadata, SourceTextRecord
from ingest.incremental_ingest import merge_incremental_records, select_incremental_records
from transforms.exact_deduplication import build_record_id


def _record(source_uri: str, text: str) -> DataRecord:
    metadata = RecordMetadata(
        source_uri=source_uri,
        language="en",
        quality_score=0.9,
        perplexity=1.5,
    )
    return DataRecord(record_id=build_record_id(text), text=text, metadata=metadata)


def test_select_incremental_records_keeps_unchanged_rows() -> None:
    """Selection should retain unchanged records by source and content."""
    source = [
        SourceTextRecord(source_uri="a.txt", text="alpha"),
        SourceTextRecord(source_uri="b.txt", text="beta-new"),
    ]
    existing = [_record("a.txt", "alpha"), _record("b.txt", "beta-old")]

    selection = select_incremental_records(source, existing)

    assert len(selection.unchanged_records) == 1


def test_merge_incremental_records_deduplicates_record_ids() -> None:
    """Merge should keep one row when record ids collide."""
    unchanged = [_record("a.txt", "shared")]
    updated = [_record("b.txt", "shared")]

    merged = merge_incremental_records(unchanged, updated)

    assert len(merged) == 1
