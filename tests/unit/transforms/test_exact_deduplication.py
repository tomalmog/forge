"""Unit tests for exact deduplication transform."""

from __future__ import annotations

from core.types import SourceTextRecord
from transforms.exact_deduplication import build_record_id, remove_exact_duplicates


def test_remove_exact_duplicates_drops_matching_text() -> None:
    """Exact dedup should collapse equivalent text rows."""
    records = [
        SourceTextRecord(source_uri="a", text="Hello World"),
        SourceTextRecord(source_uri="b", text=" hello   world "),
    ]

    deduped = remove_exact_duplicates(records)

    assert len(deduped) == 1


def test_build_record_id_is_stable_for_equivalent_text() -> None:
    """Record id should normalize whitespace and case before hashing."""
    record_id_one = build_record_id("Stable ID")
    record_id_two = build_record_id(" stable   id ")

    assert record_id_one == record_id_two
