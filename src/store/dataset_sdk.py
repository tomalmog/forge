"""Python SDK for dataset operations.

This module exposes high-level APIs for ingest, loading, filtering,
and version inspection backed by the snapshot store.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.config import ForgeConfig
from core.types import (
    DataRecord,
    IngestOptions,
    MetadataFilter,
    SnapshotManifest,
    VersionExportRequest,
)
from ingest.pipeline import ingest_dataset
from store.snapshot_store import SnapshotStore


class ForgeClient:
    """Primary SDK entry point for phase-one workflows."""

    def __init__(self, config: ForgeConfig | None = None) -> None:
        """Create SDK client.

        Args:
            config: Optional runtime configuration.
        """
        self._config = config or ForgeConfig.from_env()
        self._store = SnapshotStore(self._config)

    def ingest(self, options: IngestOptions) -> str:
        """Ingest a source URI into a versioned dataset.

        Args:
            options: Ingest options.

        Returns:
            Created version id.

        Raises:
            ForgeIngestError: If ingest pipeline fails.
            ForgeStoreError: If snapshot persistence fails.
        """
        return ingest_dataset(options, self._config)

    def dataset(self, dataset_name: str) -> "Dataset":
        """Get dataset handle by name.

        Args:
            dataset_name: Dataset identifier.

        Returns:
            Dataset handle.
        """
        return Dataset(dataset_name, self._store)

    def with_data_root(self, data_root: str) -> "ForgeClient":
        """Clone the client with a different local data root.

        Args:
            data_root: New data root path.

        Returns:
            New SDK client instance.
        """
        resolved_root = Path(data_root).expanduser().resolve()
        updated_config = replace(self._config, data_root=resolved_root)
        return ForgeClient(updated_config)


class Dataset:
    """SDK dataset handle for versioned records."""

    def __init__(self, dataset_name: str, store: SnapshotStore) -> None:
        """Create dataset handle.

        Args:
            dataset_name: Dataset identifier.
            store: Snapshot store backend.
        """
        self._dataset_name = dataset_name
        self._store = store

    @property
    def name(self) -> str:
        """Return dataset identifier."""
        return self._dataset_name

    def list_versions(self) -> list[SnapshotManifest]:
        """List all dataset versions.

        Returns:
            Ordered list of snapshot manifests.
        """
        return self._store.list_versions(self._dataset_name)

    def load_records(
        self,
        version_id: str | None = None,
    ) -> tuple[SnapshotManifest, list[DataRecord]]:
        """Load records for latest or target version.

        Args:
            version_id: Optional specific snapshot id.

        Returns:
            Pair of manifest and data records.
        """
        return self._store.load_records(self._dataset_name, version_id)

    def filter(self, filter_spec: MetadataFilter) -> str:
        """Create a filtered snapshot from the latest version.

        Args:
            filter_spec: Metadata constraints.

        Returns:
            Newly created snapshot version id.
        """
        manifest = self._store.filter_records(self._dataset_name, filter_spec)
        return manifest.version_id

    def export(self, version_id: str, output_uri: str) -> None:
        """Export a snapshot version to an S3 destination.

        Args:
            version_id: Snapshot version id.
            output_uri: Destination URI.
        """
        request = VersionExportRequest(
            dataset_name=self._dataset_name,
            version_id=version_id,
            output_uri=output_uri,
        )
        self._store.export_version_to_s3(request)
