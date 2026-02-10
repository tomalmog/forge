"""Exact text deduplication transform.

This module removes duplicate records using normalized text hashes.
It is the first transform stage in the phase-one ingest pipeline.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from core.constants import HASH_ALGORITHM
from core.types import SourceTextRecord


def remove_exact_duplicates(records: Iterable[SourceTextRecord]) -> list[SourceTextRecord]:
    """Remove duplicate records by normalized content hash.

    Args:
        records: Source records to evaluate.

    Returns:
        Ordered records with duplicates removed.
    """
    unique_records: list[SourceTextRecord] = []
    seen_hashes: set[str] = set()
    for record in records:
        normalized_text = _normalize_text(record.text)
        text_hash = _hash_text(normalized_text)
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)
        unique_records.append(record)
    return unique_records


def build_record_id(text: str) -> str:
    """Build a stable record id from normalized text.

    Args:
        text: Raw text content.

    Returns:
        Stable hash id suitable for immutable snapshots.
    """
    return _hash_text(_normalize_text(text))


def _normalize_text(text: str) -> str:
    """Normalize text for stable dedup hashing.

    Args:
        text: Raw text content.

    Returns:
        Lowercased text with normalized whitespace.
    """
    normalized = " ".join(text.lower().split())
    return normalized.strip()


def _hash_text(text: str) -> str:
    """Hash a string using configured digest algorithm.

    Args:
        text: Input text.

    Returns:
        Hex digest string.
    """
    hasher = hashlib.new(HASH_ALGORITHM)
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()
