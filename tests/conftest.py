"""Pytest configuration for repository test runs."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_sessionstart() -> None:
    """Add src directory to sys.path for test imports."""
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
