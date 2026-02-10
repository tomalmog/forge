"""Incremental ingest selection and merge helpers.

This module detects changed source records against a prior snapshot.
It produces an updated record set while preserving unchanged records.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.types import DataRecord, SourceTextRecord
from transforms.exact_deduplication import build_record_id


@dataclass(frozen=True)
class IncrementalSelection:
    """Selection output for incremental ingest."""

    records_to_process: list[SourceTextRecord]
    unchanged_records: list[DataRecord]


def select_incremental_records(
    source_records: list[SourceTextRecord],
    existing_records: list[DataRecord] | None,
) -> IncrementalSelection:
    """Select changed/new records and retain unchanged records.

    Args:
        source_records: Source records from the current ingest run.
        existing_records: Latest dataset records when available.

    Returns:
        Incremental selection containing work and unchanged records.
    """
    if not existing_records:
        return IncrementalSelection(records_to_process=source_records, unchanged_records=[])
    existing_by_source = {record.metadata.source_uri: record for record in existing_records}
    records_to_process: list[SourceTextRecord] = []
    unchanged_records: list[DataRecord] = []
    for source_record in source_records:
        existing_record = existing_by_source.get(source_record.source_uri)
        source_record_id = build_record_id(source_record.text)
        if existing_record and existing_record.record_id == source_record_id:
            unchanged_records.append(existing_record)
            continue
        records_to_process.append(source_record)
    return IncrementalSelection(
        records_to_process=records_to_process,
        unchanged_records=unchanged_records,
    )


def merge_incremental_records(
    unchanged_records: list[DataRecord],
    updated_records: list[DataRecord],
) -> list[DataRecord]:
    """Merge unchanged and updated records with exact dedup by id.

    Args:
        unchanged_records: Records carried over from parent version.
        updated_records: Freshly transformed records.

    Returns:
        Deduplicated merged records.
    """
    merged_records = unchanged_records + updated_records
    deduplicated_records: list[DataRecord] = []
    seen_record_ids: set[str] = set()
    for record in merged_records:
        if record.record_id in seen_record_ids:
            continue
        seen_record_ids.add(record.record_id)
        deduplicated_records.append(record)
    return deduplicated_records
