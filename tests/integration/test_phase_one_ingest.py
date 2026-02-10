"""Integration tests for phase-one ingest workflow."""

from __future__ import annotations

from dataclasses import replace

from core.config import ForgeConfig
from core.types import IngestOptions, MetadataFilter
from store.dataset_sdk import ForgeClient


def test_phase_one_ingest_and_filter_flow(tmp_path) -> None:
    """End-to-end flow should ingest, version, and filter records."""
    config = replace(ForgeConfig.from_env(), data_root=tmp_path)
    client = ForgeClient(config)
    options = IngestOptions(
        dataset_name="integration-demo",
        source_uri="tests/fixtures/raw_valid",
    )

    version_id = client.ingest(options)
    filtered_version_id = client.dataset("integration-demo").filter(
        MetadataFilter(language="en", min_quality_score=0.0)
    )

    assert version_id != filtered_version_id
