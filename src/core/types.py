"""Shared typed models.

This module defines immutable data models used by ingest, store,
SDK, and serving layers to keep interfaces explicit and stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True)
class RecordMetadata:
    """Metadata attached to each training record.

    Attributes:
        source_uri: Origin path or URI of the record.
        language: Detected language code.
        quality_score: Normalized quality score in [0, 1].
        perplexity: Perplexity value used to derive quality score.
        extra_fields: User-extensible metadata dictionary.
    """

    source_uri: str
    language: str
    quality_score: float
    perplexity: float
    extra_fields: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DataRecord:
    """Canonical text training record.

    Attributes:
        record_id: Stable content hash identifier.
        text: Raw text payload.
        metadata: Typed metadata fields.
    """

    record_id: str
    text: str
    metadata: RecordMetadata


@dataclass(frozen=True)
class SnapshotManifest:
    """Immutable snapshot metadata for versioning.

    Attributes:
        dataset_name: Logical dataset identifier.
        version_id: Immutable snapshot id.
        created_at: UTC creation timestamp.
        parent_version: Previous version id when derived.
        recipe_steps: Ordered transforms used to create snapshot.
        record_count: Number of records in snapshot.
    """

    dataset_name: str
    version_id: str
    created_at: datetime
    parent_version: str | None
    recipe_steps: tuple[str, ...]
    record_count: int


@dataclass(frozen=True)
class IngestOptions:
    """Ingest command options.

    Attributes:
        dataset_name: Dataset name to create/update.
        source_uri: Input file path, directory, or S3 URI.
        output_uri: Optional output object-store URI for snapshot export.
    """

    dataset_name: str
    source_uri: str
    output_uri: str | None = None


@dataclass(frozen=True)
class SourceTextRecord:
    """Raw text input record before transforms.

    Attributes:
        source_uri: Source path or URI where text was loaded.
        text: Raw text content extracted from source.
    """

    source_uri: str
    text: str


@dataclass(frozen=True)
class MetadataFilter:
    """Metadata filter constraints for snapshot slicing.

    Attributes:
        language: Optional exact language match.
        min_quality_score: Optional lower quality threshold.
        source_prefix: Optional source URI prefix match.
    """

    language: str | None = None
    min_quality_score: float | None = None
    source_prefix: str | None = None


@dataclass(frozen=True)
class SnapshotWriteRequest:
    """Request payload for snapshot persistence.

    Attributes:
        dataset_name: Logical dataset identifier.
        records: Final transformed records to persist.
        recipe_steps: Ordered list of transform names.
        parent_version: Optional parent version id.
    """

    dataset_name: str
    records: tuple[DataRecord, ...]
    recipe_steps: tuple[str, ...]
    parent_version: str | None = None


@dataclass(frozen=True)
class VersionExportRequest:
    """Request payload for exporting a version.

    Attributes:
        dataset_name: Dataset identifier.
        version_id: Version to export.
        output_uri: Destination URI (currently s3:// only).
    """

    dataset_name: str
    version_id: str
    output_uri: str


@dataclass(frozen=True)
class DataLoaderOptions:
    """PyTorch serving options.

    Attributes:
        batch_size: Number of tokenized records per batch.
        shuffle: Whether to shuffle records before yielding.
        shuffle_buffer_size: Buffer size for shuffle algorithm.
        max_token_length: Truncation length for tokenized records.
    """

    batch_size: int
    shuffle: bool
    shuffle_buffer_size: int
    max_token_length: int
