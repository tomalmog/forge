"""Forge CLI entry points.
This module exposes phase-one commands for ingest and dataset operations.
It maps argparse commands onto SDK calls.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from cli.chat_command import add_chat_command, run_chat_command
from cli.hardware_profile_command import (
    add_hardware_profile_command,
    run_hardware_profile_command,
)
from cli.run_spec_command import add_run_spec_command, run_run_spec_command
from cli.train_command import add_train_command, run_train_command
from core.config import ForgeConfig
from core.constants import (
    DEFAULT_QUALITY_MODEL,
)
from core.types import IngestOptions, MetadataFilter
from store.dataset_sdk import ForgeClient
from transforms.quality_scoring import supported_quality_models


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(prog="forge", description="Forge phase-one CLI")
    parser.add_argument("--data-root", help="Override FORGE_DATA_ROOT for this command")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_ingest_command(subparsers)
    _add_versions_command(subparsers)
    _add_filter_command(subparsers)
    _add_export_training_command(subparsers)
    add_run_spec_command(subparsers)
    add_hardware_profile_command(subparsers)
    add_train_command(subparsers)
    add_chat_command(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Forge CLI.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    client = _build_client(args.data_root)
    if args.command == "ingest":
        return _run_ingest_command(client, args)
    if args.command == "versions":
        return _run_versions_command(client, args)
    if args.command == "filter":
        return _run_filter_command(client, args)
    if args.command == "export-training":
        return _run_export_training_command(client, args)
    if args.command == "train":
        return run_train_command(client, args)
    if args.command == "chat":
        return run_chat_command(client, args)
    if args.command == "run-spec":
        return run_run_spec_command(client, args)
    if args.command == "hardware-profile":
        return run_hardware_profile_command()
    parser.error(f"Unsupported command: {args.command}")
    return 2


def _build_client(data_root: str | None) -> ForgeClient:
    """Build SDK client with optional data-root override.

    Args:
        data_root: Optional override path.

    Returns:
        Configured SDK client.
    """
    config = ForgeConfig.from_env()
    if data_root:
        config = replace(config, data_root=Path(data_root).expanduser().resolve())
    return ForgeClient(config)


def _run_ingest_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle ingest command.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    options = IngestOptions(
        dataset_name=args.dataset,
        source_uri=args.source,
        output_uri=args.output_uri,
        resume=args.resume,
        incremental=args.incremental,
        quality_model=args.quality_model,
    )
    version_id = client.ingest(options)
    print(version_id)
    return 0


def _run_versions_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle versions command.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    dataset = client.dataset(args.dataset)
    versions = dataset.list_versions()
    for manifest in versions:
        print(
            f"{manifest.version_id}\t"
            f"{manifest.record_count}\t"
            f"{manifest.created_at.isoformat()}\t"
            f"{manifest.parent_version or '-'}"
        )
    return 0


def _run_filter_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle filter command.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    filter_spec = MetadataFilter(
        language=args.language,
        min_quality_score=args.min_quality,
        source_prefix=args.source_prefix,
    )
    dataset = client.dataset(args.dataset)
    version_id = dataset.filter(filter_spec)
    print(version_id)
    return 0


def _run_export_training_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle export-training command.

    Args:
        client: SDK client.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    dataset = client.dataset(args.dataset)
    manifest_path = dataset.export_training(
        output_dir=args.output_dir,
        version_id=args.version_id,
        shard_size=args.shard_size,
        include_metadata=args.include_metadata,
    )
    print(manifest_path)
    return 0


def _add_ingest_command(subparsers: Any) -> None:
    """Register ingest subcommand."""
    parser = subparsers.add_parser("ingest", help="Ingest a local path or S3 prefix")
    parser.add_argument("source", help="Source file, directory, or s3://bucket/prefix")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output-uri", help="Optional s3:// export destination")
    parser.add_argument("--resume", action="store_true", help="Resume from ingest checkpoint")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only transform new/changed records against latest version",
    )
    parser.add_argument(
        "--quality-model",
        default=DEFAULT_QUALITY_MODEL,
        choices=supported_quality_models(),
        help="Quality scoring model",
    )


def _add_versions_command(subparsers: Any) -> None:
    """Register versions subcommand."""
    parser = subparsers.add_parser("versions", help="List dataset versions")
    parser.add_argument("--dataset", required=True, help="Dataset name")


def _add_filter_command(subparsers: Any) -> None:
    """Register filter subcommand."""
    parser = subparsers.add_parser("filter", help="Create metadata-filtered snapshot")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--language", help="Language filter, e.g. en")
    parser.add_argument("--min-quality", type=float, help="Minimum quality score")
    parser.add_argument("--source-prefix", help="Source URI prefix filter")


def _add_export_training_command(subparsers: Any) -> None:
    """Register export-training subcommand."""
    parser = subparsers.add_parser(
        "export-training",
        help="Export a version into sharded local training files",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    parser.add_argument("--version-id", help="Optional specific version id")
    parser.add_argument("--shard-size", type=int, default=1000, help="Records per shard file")
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata fields in each shard row",
    )
