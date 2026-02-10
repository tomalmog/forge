"""Unit tests for core config parsing."""

from __future__ import annotations

import os

import pytest

from core.config import ForgeConfig
from core.errors import ForgeConfigError


def test_from_env_reads_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should resolve data root from environment."""
    monkeypatch.setenv("FORGE_DATA_ROOT", "./.tmp-forge")

    config = ForgeConfig.from_env()

    assert config.data_root.name == ".tmp-forge"


def test_from_env_raises_for_invalid_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config should fail for non-numeric random seed."""
    monkeypatch.setenv("FORGE_RANDOM_SEED", "not-a-number")

    with pytest.raises(ForgeConfigError):
        ForgeConfig.from_env()

    assert os.getenv("FORGE_RANDOM_SEED") == "not-a-number"
