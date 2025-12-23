"""Trainer state for checkpointing."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrainerState:
    """Complete training state for checkpointing and resumption.

    This dataclass captures all mutable training state. Immutable
    configuration (like model architecture) is not included.

    The key to proper checkpoint resume is saving ALL state:
    - Training progress (epoch, step)
    - Best metric tracking
    - Flags (should_stop)
    - RNG state for reproducibility
    """

    # Training progress
    epoch: int = 0
    global_step: int = 0

    # Best metrics tracking
    best_metric_value: float | None = None
    best_metric_name: str | None = None

    # Flags
    is_training: bool = False
    should_stop: bool = False

    # Accumulated metrics for current epoch
    epoch_metrics: dict[str, float] = field(default_factory=dict)

    # Random state for reproducibility (serialized)
    random_state: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Handle bytes specially - encode as base64
        if d["random_state"] is not None:
            d["random_state"] = base64.b64encode(d["random_state"]).decode("utf-8")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainerState:
        """Reconstruct from dictionary."""
        d = d.copy()  # Don't modify input
        if d.get("random_state") is not None:
            d["random_state"] = base64.b64decode(d["random_state"].encode("utf-8"))
        return cls(**d)
