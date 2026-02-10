"""Shared fixture path helpers for tests."""

from __future__ import annotations

from pathlib import Path


def fixture_path(relative_path: str) -> Path:
    """Resolve a fixture path relative to tests/fixtures.

    Args:
        relative_path: Path under fixtures root.

    Returns:
        Absolute fixture path.
    """
    tests_root = Path(__file__).resolve().parent
    return tests_root / "fixtures" / relative_path
