"""Base class for learning rate schedulers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx


class LRScheduler(ABC):
    """Base class for learning rate schedulers.

    Wraps MLX's native callable lr_schedule pattern with additional
    features for checkpointing and monitoring.

    MLX optimizers accept learning_rate as a callable that takes the
    current step and returns the learning rate. Our schedulers
    implement this protocol while adding state management.

    Example:
        >>> scheduler = WarmupCosineScheduler(peak_lr=1e-3, warmup_steps=100, total_steps=1000)
        >>> optimizer = optim.AdamW(learning_rate=scheduler)
    """

    def __init__(self) -> None:
        self._step: int = 0
        self._last_lr: float = 0.0

    @abstractmethod
    def __call__(self, step: int) -> mx.array:
        """Return learning rate for the given step.

        This is called by the optimizer on each update.

        Args:
            step: Current optimizer step

        Returns:
            Learning rate as a scalar mx.array
        """
        pass

    @property
    def last_lr(self) -> float:
        """Returns the last computed learning rate."""
        return self._last_lr

    @property
    def current_step(self) -> int:
        """Returns the current step count."""
        return self._step

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state for checkpointing.

        This is critical for proper resume - the scheduler step must be
        saved and restored to continue training correctly.
        """
        return {"step": self._step, "last_lr": self._last_lr}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint.

        Args:
            state: Dictionary from state_dict()
        """
        self._step = state.get("step", 0)
        self._last_lr = state.get("last_lr", 0.0)


class ConstantLR(LRScheduler):
    """Constant learning rate (baseline scheduler).

    Example:
        >>> scheduler = ConstantLR(lr=1e-4)
        >>> optimizer = optim.Adam(learning_rate=scheduler)
    """

    def __init__(self, lr: float) -> None:
        super().__init__()
        self.lr = lr
        self._last_lr = lr

    def __call__(self, step: int) -> mx.array:
        self._step = step
        return mx.array(self.lr)
