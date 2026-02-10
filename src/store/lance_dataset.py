"""Lance dataset persistence helpers.

This module writes snapshot records to Apache Lance when available.
It also maintains JSONL mirror files for lightweight compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.constants import LANCE_DIR_NAME, RECORDS_FILE_NAME
from core.errors import ForgeStoreError
from core.types import DataRecord, RecordMetadata


def write_version_payload(version_dir: Path, records: list[DataRecord]) -> bool:
    """Persist snapshot records and attempt Lance conversion.

    Args:
        version_dir: Snapshot version directory.
        records: Records to persist.

    Returns:
        ``True`` when Lance export succeeded, else ``False``.

    Raises:
        ForgeStoreError: If JSONL persistence fails.
    """
    _write_jsonl_records(version_dir, records)
    return _try_write_lance_dataset(version_dir, records)


def read_version_payload(version_dir: Path) -> list[DataRecord]:
    """Load snapshot records from JSONL mirror file.

    Args:
        version_dir: Snapshot version directory.

    Returns:
        Parsed records in persisted order.

    Raises:
        ForgeStoreError: If records file is missing or invalid.
    """
    records_path = version_dir / RECORDS_FILE_NAME
    if not records_path.exists():
        raise ForgeStoreError(
            f"Failed to load snapshot at {version_dir}: missing {RECORDS_FILE_NAME}."
        )
    parsed_records: list[DataRecord] = []
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = _parse_json_line(version_dir, line, line_number)
        parsed_records.append(_record_from_payload(payload))
    return parsed_records


def _write_jsonl_records(version_dir: Path, records: list[DataRecord]) -> None:
    """Write records to JSONL mirror file.

    Args:
        version_dir: Snapshot version directory.
        records: Snapshot records.

    Raises:
        ForgeStoreError: If write fails.
    """
    records_path = version_dir / RECORDS_FILE_NAME
    lines = [json.dumps(_payload_from_record(record), sort_keys=True) for record in records]
    try:
        records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as error:
        raise ForgeStoreError(
            f"Failed to persist snapshot payload at {records_path}: {error}. "
            "Check write permissions and available disk space."
        ) from error


def _try_write_lance_dataset(version_dir: Path, records: list[DataRecord]) -> bool:
    """Attempt to write records to Apache Lance.

    Args:
        version_dir: Snapshot version directory.
        records: Snapshot records.

    Returns:
        Whether Lance export succeeded.
    """
    try:
        import lance
        import pyarrow as pa
    except ImportError:
        return False

    table = pa.table(
        {
            "record_id": [record.record_id for record in records],
            "text": [record.text for record in records],
            "source_uri": [record.metadata.source_uri for record in records],
            "language": [record.metadata.language for record in records],
            "quality_score": [record.metadata.quality_score for record in records],
            "perplexity": [record.metadata.perplexity for record in records],
            "extra_fields": [json.dumps(record.metadata.extra_fields) for record in records],
        }
    )
    lance_uri = str(version_dir / LANCE_DIR_NAME)
    try:
        lance.write_dataset(table, lance_uri, mode="overwrite")
    except Exception as error:
        raise ForgeStoreError(
            f"Failed to write Lance dataset at {lance_uri}: {error}. "
            "Validate lance/pyarrow compatibility and retry ingest."
        ) from error
    return True


def _payload_from_record(record: DataRecord) -> dict[str, object]:
    """Serialize a DataRecord to JSON-safe dictionary.

    Args:
        record: Data record instance.

    Returns:
        Dictionary payload.
    """
    metadata = record.metadata
    return {
        "record_id": record.record_id,
        "text": record.text,
        "metadata": {
            "source_uri": metadata.source_uri,
            "language": metadata.language,
            "quality_score": metadata.quality_score,
            "perplexity": metadata.perplexity,
            "extra_fields": dict(metadata.extra_fields),
        },
    }


def _parse_json_line(version_dir: Path, line: str, line_number: int) -> dict[str, Any]:
    """Parse one records JSONL line.

    Args:
        version_dir: Parent version directory for context.
        line: Raw JSONL line.
        line_number: One-based line number.

    Returns:
        Parsed dictionary payload.

    Raises:
        ForgeStoreError: If JSON is invalid.
    """
    try:
        payload = json.loads(line)
        if isinstance(payload, dict):
            return payload
        raise ForgeStoreError(
            f"Failed to parse snapshot payload at {version_dir}/{RECORDS_FILE_NAME}:{line_number}: "
            "expected a JSON object per line. Recreate the dataset snapshot."
        )
    except json.JSONDecodeError as error:
        raise ForgeStoreError(
            f"Failed to parse snapshot payload at {version_dir}/{RECORDS_FILE_NAME}:{line_number}: "
            f"{error.msg}. Recreate the dataset snapshot."
        ) from error


def _record_from_payload(payload: dict[str, Any]) -> DataRecord:
    """Deserialize JSON payload into DataRecord.

    Args:
        payload: Serialized record payload.

    Returns:
        Typed record.
    """
    metadata_payload = payload["metadata"]
    metadata_dict = metadata_payload if isinstance(metadata_payload, dict) else {}
    metadata = RecordMetadata(
        source_uri=str(metadata_dict.get("source_uri", "")),
        language=str(metadata_dict.get("language", "unknown")),
        quality_score=float(metadata_dict.get("quality_score", 0.0)),
        perplexity=float(metadata_dict.get("perplexity", 0.0)),
        extra_fields={
            key: str(value) for key, value in dict(metadata_dict.get("extra_fields", {})).items()
        },
    )
    return DataRecord(
        record_id=str(payload.get("record_id", "")),
        text=str(payload.get("text", "")),
        metadata=metadata,
    )
