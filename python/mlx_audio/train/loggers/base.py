"""Base logger class for mlx-train."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Logger(ABC):
    """Base class for all loggers in mlx-train.

    Loggers handle metric tracking and experiment management.
    Implementations should be stateless between runs.

    Example:
        >>> class MyLogger(Logger):
        ...     def log_metrics(self, metrics, step):
        ...         for k, v in metrics.items():
        ...             print(f"Step {step}: {k}={v}")
    """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics at the given step.

        Args:
            metrics: Dictionary of metric name to value
            step: Current training step
        """
        pass

    @abstractmethod
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of hyperparameter name to value
        """
        pass

    def finalize(self) -> None:
        """Called at the end of training. Override for cleanup."""
        pass
