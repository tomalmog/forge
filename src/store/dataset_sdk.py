"""Python SDK for dataset operations.

This module exposes high-level APIs for ingest, loading, filtering,
and version inspection backed by the snapshot store.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.chat_types import ChatOptions, ChatResult
from core.config import ForgeConfig
from core.errors import ForgeServeError
from core.run_spec_execution import execute_run_spec_file
from core.types import (
    DataRecord,
    IngestOptions,
    MetadataFilter,
    SnapshotManifest,
    TrainingExportRequest,
    TrainingOptions,
    TrainingRunResult,
    VersionExportRequest,
)
from ingest.pipeline import ingest_dataset
from serve.chat_runner import run_chat
from serve.hardware_profile import detect_hardware_profile
from serve.training_run_registry import TrainingRunRegistry
from serve.training_run_types import TrainingRunRecord
from serve.training_runner import run_training
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

    def train(self, options: TrainingOptions) -> TrainingRunResult:
        """Train a model on a dataset version using PyTorch.

        Args:
            options: Training options.

        Returns:
            Training run artifact summary.
        """
        dataset = self.dataset(options.dataset_name)
        return dataset.train(options)

    def chat(self, options: ChatOptions) -> ChatResult:
        """Run chat inference against a trained model on a dataset.

        Args:
            options: Chat inference options.

        Returns:
            Generated response payload.
        """
        dataset = self.dataset(options.dataset_name)
        return dataset.chat(options)

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

    def run_spec(self, spec_file: str) -> tuple[str, ...]:
        """Execute a YAML run-spec through the shared execution engine.

        Args:
            spec_file: Path to YAML run-spec file.

        Returns:
            Ordered command output lines.
        """
        return execute_run_spec_file(self, spec_file)

    def hardware_profile(self) -> dict[str, object]:
        """Detect local hardware profile and recommended defaults.

        Returns:
            Hardware profile payload.
        """
        return detect_hardware_profile().to_dict()

    def list_training_runs(self) -> tuple[str, ...]:
        """List known training run IDs from lifecycle registry.

        Returns:
            Ordered tuple of run IDs.
        """
        return TrainingRunRegistry(self._config.data_root).list_runs()

    def get_training_run(self, run_id: str) -> TrainingRunRecord:
        """Load one training run lifecycle record by ID.

        Args:
            run_id: Training run identifier.

        Returns:
            Persisted lifecycle record.
        """
        return TrainingRunRegistry(self._config.data_root).load_run(run_id)

    def get_lineage_graph(self) -> dict[str, object]:
        """Load model/dataset lineage graph for this data root.

        Returns:
            Lineage graph payload.
        """
        return TrainingRunRegistry(self._config.data_root).load_lineage_graph()


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

    def export_training(
        self,
        output_dir: str,
        version_id: str | None = None,
        shard_size: int = 1000,
        include_metadata: bool = False,
    ) -> str:
        """Export snapshot into local sharded training files.

        Args:
            output_dir: Local output directory for exported shards.
            version_id: Optional version id, latest when omitted.
            shard_size: Number of records per shard file.
            include_metadata: Include metadata per exported row.

        Returns:
            Path to generated training manifest.
        """
        request = TrainingExportRequest(
            dataset_name=self._dataset_name,
            output_dir=output_dir,
            version_id=version_id,
            shard_size=shard_size,
            include_metadata=include_metadata,
        )
        manifest_path = self._store.export_training_data(request)
        return str(manifest_path)

    def train(self, options: TrainingOptions) -> TrainingRunResult:
        """Train a model on this dataset with default/custom loop.

        Args:
            options: Training options.

        Returns:
            Training run artifact summary.

        Raises:
            ForgeServeError: If dataset names mismatch.
        """
        if options.dataset_name != self._dataset_name:
            raise ForgeServeError(
                f"Training options dataset '{options.dataset_name}' does not match handle "
                f"'{self._dataset_name}'."
            )
        manifest, records = self.load_records(options.version_id)
        return run_training(
            records=records,
            options=options,
            random_seed=self._store.random_seed,
            data_root=self._store.data_root,
            dataset_version_id=manifest.version_id,
        )

    def chat(self, options: ChatOptions) -> ChatResult:
        """Run one chat completion on this dataset.

        Args:
            options: Chat inference options.

        Returns:
            Generated response payload.

        Raises:
            ForgeServeError: If dataset names mismatch.
        """
        if options.dataset_name != self._dataset_name:
            raise ForgeServeError(
                f"Chat options dataset '{options.dataset_name}' does not match handle "
                f"'{self._dataset_name}'."
            )
        _, records = self.load_records(options.version_id)
        return run_chat(records, options)
