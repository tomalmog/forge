"""Ingest orchestration for phase-one pipelines.

This module coordinates source loading, transforms, and snapshot writes.
It defines the single ingest workflow used by SDK and CLI.
"""

from __future__ import annotations

from core.config import ForgeConfig
from core.logging_config import get_logger
from core.types import (
    DataRecord,
    IngestOptions,
    RecordMetadata,
    SnapshotWriteRequest,
    SourceTextRecord,
    VersionExportRequest,
)
from ingest.input_reader import read_source_records
from store.snapshot_store import SnapshotStore
from transforms.exact_deduplication import build_record_id, remove_exact_duplicates
from transforms.language_detection import detect_languages
from transforms.perplexity_quality import score_texts_with_perplexity

_LOGGER = get_logger(__name__)


def ingest_dataset(options: IngestOptions, config: ForgeConfig) -> str:
    """Run the phase-one ingest pipeline and persist a snapshot.

    Args:
        options: Ingest request options.
        config: Runtime configuration.

    Returns:
        Created snapshot version id.

    Raises:
        ForgeIngestError: If source read or transforms fail.
        ForgeStoreError: If snapshot persistence fails.
    """
    source_records = read_source_records(options.source_uri, config)
    deduplicated_records = remove_exact_duplicates(source_records)
    enriched_records = _build_enriched_records(deduplicated_records)
    store = SnapshotStore(config)
    write_request = SnapshotWriteRequest(
        dataset_name=options.dataset_name,
        records=tuple(enriched_records),
        recipe_steps=(
            "exact_deduplication",
            "language_detection",
            "perplexity_quality_scoring",
        ),
    )
    manifest = store.create_snapshot(write_request)
    if options.output_uri:
        export_request = VersionExportRequest(
            dataset_name=options.dataset_name,
            version_id=manifest.version_id,
            output_uri=options.output_uri,
        )
        store.export_version_to_s3(export_request)
    _log_ingest_completion(
        options, len(source_records), len(enriched_records), manifest.version_id
    )
    return manifest.version_id


def _build_enriched_records(source_records: list[SourceTextRecord]) -> list[DataRecord]:
    """Build fully-scored records from deduplicated input.

    Args:
        source_records: Deduplicated source records.

    Returns:
        Records ready for immutable snapshot persistence.
    """
    texts = [record.text for record in source_records]
    languages = detect_languages(texts)
    quality_pairs = score_texts_with_perplexity(texts)
    enriched_records: list[DataRecord] = []
    for source_record, language, quality_pair in zip(
        source_records, languages, quality_pairs
    ):
        perplexity, quality_score = quality_pair
        metadata = RecordMetadata(
            source_uri=source_record.source_uri,
            language=language,
            quality_score=quality_score,
            perplexity=perplexity,
        )
        enriched_records.append(
            DataRecord(
                record_id=build_record_id(source_record.text),
                text=source_record.text,
                metadata=metadata,
            )
        )
    return enriched_records


def _log_ingest_completion(
    options: IngestOptions,
    input_count: int,
    output_count: int,
    version_id: str,
) -> None:
    """Log pipeline completion with contextual metadata.

    Args:
        options: Ingest options.
        input_count: Number of source records.
        output_count: Number of records after dedup.
        version_id: Snapshot version id.
    """
    _LOGGER.info(
        "ingest_completed",
        dataset_name=options.dataset_name,
        source_uri=options.source_uri,
        input_count=input_count,
        output_count=output_count,
        version_id=version_id,
        output_uri=options.output_uri,
    )
