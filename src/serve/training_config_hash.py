"""Training config hashing for run identity and lineage tracking.

This module provides deterministic hashing for typed training options so
lifecycle records and lineage graphs can identify exact run configurations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from core.constants import HASH_ALGORITHM
from core.types import TrainingOptions


def compute_training_config_hash(options: TrainingOptions) -> str:
    """Compute a stable hash for one training options payload."""
    payload = asdict(options)
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    hash_builder = hashlib.new(HASH_ALGORITHM)
    hash_builder.update(normalized.encode("utf-8"))
    return hash_builder.hexdigest()
