"""Shared typed models.

This module defines immutable data models used by ingest, store,
SDK, and serving layers to keep interfaces explicit and stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_POSITION_EMBEDDING_TYPE,
    DEFAULT_QUALITY_MODEL,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS,
    DEFAULT_TRAIN_DROPOUT,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_MAX_CHECKPOINT_FILES,
    DEFAULT_TRAIN_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_LAYERS,
    DEFAULT_TRAIN_NUM_LAYERS,
    DEFAULT_TRAIN_OPTIMIZER_TYPE,
    DEFAULT_TRAIN_PRECISION_MODE,
    DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS,
    DEFAULT_TRAIN_SCHEDULER_ETA_MIN,
    DEFAULT_TRAIN_SCHEDULER_GAMMA,
    DEFAULT_TRAIN_SCHEDULER_STEP_SIZE,
    DEFAULT_TRAIN_SCHEDULER_T_MAX_EPOCHS,
    DEFAULT_TRAIN_SCHEDULER_TYPE,
    DEFAULT_TRAIN_SGD_MOMENTUM,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
    DEFAULT_TRAIN_WEIGHT_DECAY,
)

PositionEmbeddingType = Literal["learned", "sinusoidal"]
PrecisionMode = Literal["auto", "fp32", "fp16", "bf16"]
OptimizerType = Literal["adam", "adamw", "sgd"]
SchedulerType = Literal["none", "step", "cosine"]


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
        resume: Resume from the latest matching ingest checkpoint.
        incremental: Update an existing dataset from changed/new source records.
        quality_model: Quality scoring model identifier.
    """

    dataset_name: str
    source_uri: str
    output_uri: str | None = None
    resume: bool = False
    incremental: bool = False
    quality_model: str = DEFAULT_QUALITY_MODEL


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
class TrainingExportRequest:
    """Request payload for training shard export.

    Attributes:
        dataset_name: Dataset identifier.
        version_id: Optional version id; latest if omitted.
        output_dir: Local output directory for shard files.
        shard_size: Number of records per shard file.
        include_metadata: Whether to include metadata in each JSONL row.
    """

    dataset_name: str
    output_dir: str
    version_id: str | None = None
    shard_size: int = 1000
    include_metadata: bool = False


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


@dataclass(frozen=True)
class TrainingOptions:
    """Training command options used by CLI, SDK, and run-spec workflows."""

    dataset_name: str
    output_dir: str
    version_id: str | None = None
    architecture_path: str | None = None
    custom_loop_path: str | None = None
    hooks_path: str | None = None
    epochs: int = DEFAULT_TRAIN_EPOCHS
    learning_rate: float = DEFAULT_TRAIN_LEARNING_RATE
    precision_mode: PrecisionMode = DEFAULT_TRAIN_PRECISION_MODE
    optimizer_type: OptimizerType = DEFAULT_TRAIN_OPTIMIZER_TYPE
    weight_decay: float = DEFAULT_TRAIN_WEIGHT_DECAY
    sgd_momentum: float = DEFAULT_TRAIN_SGD_MOMENTUM
    scheduler_type: SchedulerType = DEFAULT_TRAIN_SCHEDULER_TYPE
    scheduler_step_size: int = DEFAULT_TRAIN_SCHEDULER_STEP_SIZE
    scheduler_gamma: float = DEFAULT_TRAIN_SCHEDULER_GAMMA
    scheduler_t_max_epochs: int | None = DEFAULT_TRAIN_SCHEDULER_T_MAX_EPOCHS
    scheduler_eta_min: float = DEFAULT_TRAIN_SCHEDULER_ETA_MIN
    batch_size: int = DEFAULT_BATCH_SIZE
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH
    validation_split: float = DEFAULT_TRAIN_VALIDATION_SPLIT
    hidden_dim: int = DEFAULT_TRAIN_HIDDEN_DIM
    num_layers: int = DEFAULT_TRAIN_NUM_LAYERS
    attention_heads: int = DEFAULT_TRAIN_ATTENTION_HEADS
    mlp_hidden_dim: int = DEFAULT_TRAIN_MLP_HIDDEN_DIM
    mlp_layers: int = DEFAULT_TRAIN_MLP_LAYERS
    dropout: float = DEFAULT_TRAIN_DROPOUT
    position_embedding_type: PositionEmbeddingType = DEFAULT_POSITION_EMBEDDING_TYPE
    vocabulary_size: int | None = None
    initial_weights_path: str | None = None
    checkpoint_every_epochs: int = DEFAULT_TRAIN_CHECKPOINT_EVERY_EPOCHS
    save_best_checkpoint: bool = True
    max_checkpoint_files: int | None = DEFAULT_TRAIN_MAX_CHECKPOINT_FILES
    resume_checkpoint_path: str | None = None
    progress_log_interval_steps: int = DEFAULT_TRAIN_PROGRESS_LOG_INTERVAL_STEPS


@dataclass(frozen=True)
class EpochMetric:
    """One epoch metric row.

    Attributes:
        epoch: One-based epoch index.
        train_loss: Average training loss.
        validation_loss: Average validation loss.
    """

    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class BatchLossMetric:
    """One training batch metric row.

    Attributes:
        epoch: One-based epoch index.
        batch_index: One-based batch index inside the epoch.
        global_step: One-based global optimizer step index.
        train_loss: Batch training loss.
    """

    epoch: int
    batch_index: int
    global_step: int
    train_loss: float


@dataclass(frozen=True)
class TrainingRunResult:
    """Training command output artifact paths and summary metadata."""

    model_path: str
    history_path: str
    plot_path: str | None
    epochs_completed: int
    checkpoint_dir: str | None = None
    best_checkpoint_path: str | None = None
    resumed_from_checkpoint: str | None = None
    run_id: str | None = None
    artifact_contract_path: str | None = None
