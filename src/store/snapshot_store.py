"""Snapshot store and metadata catalog.

This module persists immutable dataset versions with lineage metadata.
It provides load, list, filter, and export operations for the SDK.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from core.config import ForgeConfig
from core.constants import (
    CATALOG_FILE_NAME,
    DATASETS_DIR_NAME,
    MANIFEST_FILE_NAME,
    VERSIONS_DIR_NAME,
)
from core.errors import ForgeDependencyError, ForgeStoreError
from core.logging_config import get_logger
from core.s3_uri import parse_s3_uri
from core.types import (
    DataRecord,
    MetadataFilter,
    SnapshotManifest,
    SnapshotWriteRequest,
    VersionExportRequest,
)
from store.lance_dataset import read_version_payload, write_version_payload

_LOGGER = get_logger(__name__)


class SnapshotStore:
    """Immutable snapshot store implementation.

    This class owns dataset directories, version manifests,
    and metadata catalog updates for phase-one operations.
    """

    def __init__(self, config: ForgeConfig) -> None:
        """Initialize snapshot store from config.

        Args:
            config: Runtime configuration.
        """
        self._config = config
        self._datasets_root = config.data_root / DATASETS_DIR_NAME
        self._datasets_root.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, request: SnapshotWriteRequest) -> SnapshotManifest:
        """Create a new immutable dataset snapshot.

        Args:
            request: Snapshot write request payload.

        Returns:
            Persisted snapshot manifest.

        Raises:
            ForgeStoreError: If persistence fails.
        """
        dataset_root = self._dataset_root(request.dataset_name)
        version_id = _build_version_id(request.dataset_name, request.records)
        version_dir = dataset_root / VERSIONS_DIR_NAME / version_id
        version_dir.mkdir(parents=True, exist_ok=False)
        lance_written = write_version_payload(version_dir, list(request.records))
        manifest = SnapshotManifest(
            dataset_name=request.dataset_name,
            version_id=version_id,
            created_at=datetime.now(timezone.utc),
            parent_version=request.parent_version,
            recipe_steps=request.recipe_steps,
            record_count=len(request.records),
        )
        _write_manifest_file(version_dir, manifest, lance_written)
        _update_catalog(dataset_root / CATALOG_FILE_NAME, manifest)
        _LOGGER.info(
            "snapshot_created",
            dataset_name=request.dataset_name,
            version_id=version_id,
            record_count=manifest.record_count,
            parent_version=manifest.parent_version,
            lance_written=lance_written,
        )
        return manifest

    def list_versions(self, dataset_name: str) -> list[SnapshotManifest]:
        """List manifests for a dataset sorted by creation time.

        Args:
            dataset_name: Dataset identifier.

        Returns:
            Ordered manifest list.

        Raises:
            ForgeStoreError: If dataset catalog does not exist.
        """
        catalog_path = self._dataset_root(dataset_name) / CATALOG_FILE_NAME
        catalog = _read_catalog_file(catalog_path)
        version_payloads = cast(list[dict[str, Any]], catalog["versions"])
        versions = [_manifest_from_dict(item) for item in version_payloads]
        return sorted(versions, key=lambda item: item.created_at)

    def load_records(
        self,
        dataset_name: str,
        version_id: str | None = None,
    ) -> tuple[SnapshotManifest, list[DataRecord]]:
        """Load records for a dataset snapshot.

        Args:
            dataset_name: Dataset identifier.
            version_id: Optional snapshot version; latest when omitted.

        Returns:
            Pair of manifest and loaded records.

        Raises:
            ForgeStoreError: If dataset/version is missing.
        """
        manifest = self._resolve_manifest(dataset_name, version_id)
        version_dir = self._version_dir(dataset_name, manifest.version_id)
        records = read_version_payload(version_dir)
        return manifest, records

    def filter_records(
        self,
        dataset_name: str,
        filter_spec: MetadataFilter,
    ) -> SnapshotManifest:
        """Create a filtered immutable snapshot from latest version.

        Args:
            dataset_name: Dataset to filter.
            filter_spec: Metadata filter criteria.

        Returns:
            New filtered snapshot manifest.
        """
        parent_manifest, records = self.load_records(dataset_name)
        filtered_records = _filter_records(records, filter_spec)
        recipe_steps = parent_manifest.recipe_steps + ("metadata_filter",)
        request = SnapshotWriteRequest(
            dataset_name=dataset_name,
            records=tuple(filtered_records),
            recipe_steps=recipe_steps,
            parent_version=parent_manifest.version_id,
        )
        return self.create_snapshot(request)

    def export_version_to_s3(self, request: VersionExportRequest) -> None:
        """Export a version directory to S3.

        Args:
            request: Export request payload.

        Raises:
            ForgeStoreError: If export fails.
        """
        location = parse_s3_uri(request.output_uri, domain="store")
        version_dir = self._version_dir(request.dataset_name, request.version_id)
        s3_client = _create_s3_client(self._config)
        _upload_directory(s3_client, version_dir, location.bucket, location.prefix)
        _LOGGER.info(
            "snapshot_exported",
            dataset_name=request.dataset_name,
            version_id=request.version_id,
            output_uri=request.output_uri,
        )

    def _dataset_root(self, dataset_name: str) -> Path:
        """Return dataset root path and ensure base directories.

        Args:
            dataset_name: Dataset identifier.

        Returns:
            Dataset root path.
        """
        dataset_root = self._datasets_root / dataset_name
        (dataset_root / VERSIONS_DIR_NAME).mkdir(parents=True, exist_ok=True)
        return dataset_root

    def _resolve_manifest(self, dataset_name: str, version_id: str | None) -> SnapshotManifest:
        """Resolve a target manifest.

        Args:
            dataset_name: Dataset identifier.
            version_id: Optional version id.

        Returns:
            Resolved snapshot manifest.

        Raises:
            ForgeStoreError: If catalog or target version is missing.
        """
        manifests = self.list_versions(dataset_name)
        if not manifests:
            raise ForgeStoreError(
                f"No versions exist for dataset '{dataset_name}'. "
                "Ingest data before reading snapshots."
            )
        if version_id is None:
            return manifests[-1]
        for manifest in manifests:
            if manifest.version_id == version_id:
                return manifest
        raise ForgeStoreError(
            f"Version '{version_id}' not found for dataset '{dataset_name}'. "
            "Use list_versions to discover valid version ids."
        )

    def _version_dir(self, dataset_name: str, version_id: str) -> Path:
        """Return snapshot version directory.

        Args:
            dataset_name: Dataset identifier.
            version_id: Snapshot version id.

        Returns:
            Version directory path.

        Raises:
            ForgeStoreError: If version directory is missing.
        """
        version_dir = self._dataset_root(dataset_name) / VERSIONS_DIR_NAME / version_id
        if not version_dir.exists():
            raise ForgeStoreError(
                f"Missing snapshot directory for {dataset_name}:{version_id} at {version_dir}. "
                "Recreate the snapshot before loading or exporting."
            )
        return version_dir


def _build_version_id(dataset_name: str, records: tuple[DataRecord, ...]) -> str:
    """Build deterministic version id from dataset and timestamp.

    Args:
        dataset_name: Dataset identifier.
        records: Snapshot records.

    Returns:
        Version id string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest_seed = "|".join(record.record_id for record in records)
    digest = hashlib.sha256(digest_seed.encode("utf-8")).hexdigest()[:10]
    return f"{dataset_name}-{timestamp}-{digest}"


def _write_manifest_file(
    version_dir: Path, manifest: SnapshotManifest, lance_written: bool
) -> None:
    """Write per-version manifest file.

    Args:
        version_dir: Snapshot version directory.
        manifest: Manifest payload.
        lance_written: Whether Lance dataset was created.
    """
    manifest_dict = asdict(manifest)
    manifest_dict["created_at"] = manifest.created_at.isoformat()
    manifest_dict["lance_written"] = lance_written
    manifest_path = version_dir / MANIFEST_FILE_NAME
    manifest_path.write_text(json.dumps(manifest_dict, indent=2) + "\n", encoding="utf-8")


def _update_catalog(catalog_path: Path, manifest: SnapshotManifest) -> None:
    """Append manifest to catalog file.

    Args:
        catalog_path: Catalog JSON path.
        manifest: Manifest to append.
    """
    if catalog_path.exists():
        catalog = _read_catalog_file(catalog_path)
    else:
        catalog = {"latest_version": None, "versions": []}
    manifest_dict = asdict(manifest)
    manifest_dict["created_at"] = manifest.created_at.isoformat()
    versions = cast(list[dict[str, Any]], catalog["versions"])
    versions.append(manifest_dict)
    catalog["latest_version"] = manifest.version_id
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


def _read_catalog_file(catalog_path: Path) -> dict[str, Any]:
    """Read dataset catalog file.

    Args:
        catalog_path: Catalog JSON path.

    Returns:
        Parsed catalog payload.

    Raises:
        ForgeStoreError: If catalog is missing or invalid.
    """
    if not catalog_path.exists():
        raise ForgeStoreError(
            f"Dataset catalog not found at {catalog_path}. Ingest data before requesting versions."
        )
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        raise ForgeStoreError(
            f"Failed to parse dataset catalog at {catalog_path}: "
            "expected JSON object at top level. Recreate the catalog."
        )
    except json.JSONDecodeError as error:
        raise ForgeStoreError(
            f"Failed to parse dataset catalog at {catalog_path}: {error.msg}. "
            "Recreate the dataset catalog from source snapshots."
        ) from error


def _manifest_from_dict(payload: dict[str, Any]) -> SnapshotManifest:
    """Deserialize manifest payload from dictionary.

    Args:
        payload: Manifest dictionary.

    Returns:
        Typed snapshot manifest.
    """
    return SnapshotManifest(
        dataset_name=str(payload["dataset_name"]),
        version_id=str(payload["version_id"]),
        created_at=datetime.fromisoformat(str(payload["created_at"])),
        parent_version=str(payload["parent_version"]) if payload["parent_version"] else None,
        recipe_steps=tuple(str(step) for step in payload["recipe_steps"]),
        record_count=int(payload["record_count"]),
    )


def _filter_records(
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


def _create_s3_client(config: ForgeConfig) -> Any:
    """Create boto3 S3 client for exports.

    Args:
        config: Runtime config with optional session settings.

    Returns:
        Boto3 S3 client.

    Raises:
        ForgeDependencyError: If boto3 is missing.
    """
    try:
        import boto3
    except ImportError as error:
        raise ForgeDependencyError(
            "S3 export requires boto3, but it is not installed. "
            "Install boto3 to export versions to s3:// destinations."
        ) from error
    session_kwargs: dict[str, str] = {}
    if config.s3_profile:
        session_kwargs["profile_name"] = config.s3_profile
    if config.s3_region:
        session_kwargs["region_name"] = config.s3_region
    session = boto3.session.Session(**session_kwargs)
    return session.client("s3")


def _upload_directory(s3_client: Any, version_dir: Path, bucket: str, prefix: str) -> None:
    """Upload all version files to S3.

    Args:
        s3_client: Boto3 S3 client.
        version_dir: Local version directory.
        bucket: Destination bucket.
        prefix: Destination key prefix.

    Raises:
        ForgeStoreError: If upload fails.
    """
    for local_file in sorted(version_dir.rglob("*")):
        if not local_file.is_file():
            continue
        relative_path = local_file.relative_to(version_dir)
        object_key = f"{prefix.rstrip('/')}/{relative_path.as_posix()}"
        try:
            s3_client.upload_file(str(local_file), bucket, object_key)
        except Exception as error:
            raise ForgeStoreError(
                f"Failed to export snapshot file {local_file} to s3://{bucket}/{object_key}: {error}. "
                "Check AWS credentials and retry export."
            ) from error
