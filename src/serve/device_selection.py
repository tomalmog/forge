"""Accelerator-aware torch device selection helpers.

This module centralizes runtime device selection for training and inference.
It keeps accelerator detection consistent across serve components.
"""

from __future__ import annotations

from typing import Any


def resolve_execution_device(torch_module: Any) -> Any:
    """Resolve the preferred torch device for execution."""
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and bool(cuda_module.is_available()):
        return torch_module.device("cuda")
    if is_mps_available(torch_module):
        return torch_module.device("mps")
    return torch_module.device("cpu")


def is_mps_available(torch_module: Any) -> bool:
    """Return True when torch reports MPS backend support and availability."""
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return False
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False
    probe = getattr(mps_backend, "is_available", None)
    if not callable(probe):
        return False
    return bool(probe())
