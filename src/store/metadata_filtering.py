"""Record metadata filtering helpers.

This module applies metadata constraints used for derived snapshots.
It keeps filtering logic reusable across store and export flows.
"""

from __future__ import annotations

from core.types import DataRecord, MetadataFilter


def filter_records(
    records: list[DataRecord],
    filter_spec: MetadataFilter,
) -> list[DataRecord]:
    """Filter records using metadata constraints.

    Args:
        records: Input records to filter.
        filter_spec: Filter constraints.

    Returns:
        Filtered records list.
    """
    filtered: list[DataRecord] = []
    for record in records:
        metadata = record.metadata
        if filter_spec.language and metadata.language != filter_spec.language:
            continue
        if filter_spec.min_quality_score is not None:
            if metadata.quality_score < filter_spec.min_quality_score:
                continue
        if filter_spec.source_prefix:
            if not metadata.source_uri.startswith(filter_spec.source_prefix):
                continue
        filtered.append(record)
    return filtered
