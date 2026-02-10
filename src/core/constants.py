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
