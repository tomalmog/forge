"""Core constants used across Forge modules.

This module centralizes non-domain-specific constants.
Keeping values here avoids magic literals in business logic.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_ROOT = Path(".forge")
DATASETS_DIR_NAME = "datasets"
VERSIONS_DIR_NAME = "versions"
CATALOG_FILE_NAME = "catalog.json"
MANIFEST_FILE_NAME = "manifest.json"
RECORDS_FILE_NAME = "records.jsonl"
LANCE_DIR_NAME = "data.lance"
DEFAULT_LANGUAGE_CODE = "unknown"
ENGLISH_LANGUAGE_CODE = "en"
DEFAULT_QUALITY_SCORE = 0.0
DEFAULT_QUALITY_FLOOR = 0.0
DEFAULT_SHUFFLE_BUFFER_SIZE = 1024
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_TOKEN_LENGTH = 512
HASH_ALGORITHM = "sha256"
SUPPORTED_TEXT_EXTENSIONS = (".txt", ".md", ".text", ".jsonl")
INGEST_CHECKPOINT_DIR_NAME = "ingest_checkpoint"
CHECKPOINT_STATE_FILE_NAME = "state.json"
CHECKPOINT_SOURCE_RECORDS_FILE_NAME = "source_records.jsonl"
CHECKPOINT_WORK_RECORDS_FILE_NAME = "work_records.jsonl"
CHECKPOINT_UNCHANGED_RECORDS_FILE_NAME = "unchanged_records.jsonl"
CHECKPOINT_DEDUP_RECORDS_FILE_NAME = "dedup_records.jsonl"
CHECKPOINT_ENRICHED_RECORDS_FILE_NAME = "enriched_records.jsonl"
DEFAULT_QUALITY_MODEL = "perplexity"
SUPPORTED_QUALITY_MODELS = ("hybrid", "perplexity")
DEFAULT_EXPORT_SHARD_SIZE = 1000
DEFAULT_TRAIN_EPOCHS = 3
DEFAULT_TRAIN_LEARNING_RATE = 1e-3
DEFAULT_TRAIN_VALIDATION_SPLIT = 0.1
DEFAULT_TRAIN_EMBED_DIM = 128
DEFAULT_TRAIN_HIDDEN_DIM = 256
DEFAULT_TRAIN_NUM_LAYERS = 2
DEFAULT_TRAIN_DROPOUT = 0.1
DEFAULT_TRAINED_MODEL_FILE_NAME = "model.pt"
DEFAULT_TRAIN_HISTORY_FILE_NAME = "history.json"
DEFAULT_TRAIN_PLOT_FILE_NAME = "training_curves.png"
