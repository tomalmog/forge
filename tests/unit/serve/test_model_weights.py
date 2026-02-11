"""Unit tests for model weight loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import ForgeServeError
from serve.model_weights import load_initial_weights


class _FakeTorch:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def load(self, path: str, map_location: object) -> object:
        _ = (path, map_location)
        return self._payload


class _FakeModel:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self.loaded_state = state_dict


def test_load_initial_weights_raises_for_missing_path(tmp_path: Path) -> None:
    """Missing checkpoint path should raise a clear serve error."""
    with pytest.raises(ForgeServeError):
        load_initial_weights(
            torch_module=_FakeTorch(payload={}),
            model=_FakeModel(),
            initial_weights_path=str(tmp_path / "missing.pt"),
            device="cpu",
        )

    assert True


def test_load_initial_weights_accepts_raw_state_dict(tmp_path: Path) -> None:
    """Raw state_dict payload should be forwarded to model loader."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    model = _FakeModel()

    load_initial_weights(
        torch_module=_FakeTorch(payload={"linear.weight": object()}),
        model=model,
        initial_weights_path=str(model_path),
        device="cpu",
    )

    assert isinstance(model.loaded_state, dict)


def test_load_initial_weights_accepts_nested_model_state_dict(tmp_path: Path) -> None:
    """Checkpoint payload with model_state_dict should be unpacked."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    model = _FakeModel()

    load_initial_weights(
        torch_module=_FakeTorch(payload={"model_state_dict": {"linear.bias": object()}}),
        model=model,
        initial_weights_path=str(model_path),
        device="cpu",
    )

    assert isinstance(model.loaded_state, dict)


def test_load_initial_weights_rejects_invalid_payload(tmp_path: Path) -> None:
    """Invalid checkpoint payload should raise serve error."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")

    with pytest.raises(ForgeServeError):
        load_initial_weights(
            torch_module=_FakeTorch(payload=["not", "a", "mapping"]),
            model=_FakeModel(),
            initial_weights_path=str(model_path),
            device="cpu",
        )

    assert True
