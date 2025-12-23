"""TrainModule base class for mlx-train.

Users subclass TrainModule and implement compute_loss() and configure_optimizers()
to define their training logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

if TYPE_CHECKING:
    from mlx_audio.train.trainer import Trainer


@dataclass
class OptimizerConfig:
    """Configuration returned by configure_optimizers().

    Attributes:
        optimizer: The MLX optimizer instance. Can have a callable learning_rate
            for scheduling.
        lr_schedule_name: Optional name for logging purposes (e.g., "warmup_cosine").
    """

    optimizer: optim.Optimizer
    lr_schedule_name: str | None = None


class TrainModule(nn.Module):
    """Base class for all training modules in mlx-train.

    Users should subclass this and implement:
    - __init__: Define model architecture
    - __call__: Forward pass (inherited from nn.Module)
    - compute_loss: Compute loss given a batch
    - configure_optimizers: Return optimizer configuration

    Optional overrides:
    - validation_step: Custom validation logic
    - test_step: Custom test logic
    - on_train_start/end: Training lifecycle hooks
    - on_train_epoch_start/end: Epoch-level hooks
    - on_validation_start/end: Validation lifecycle hooks

    Example:
        >>> class MyModel(TrainModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(784, 10)
        ...
        ...     def __call__(self, x):
        ...         return self.linear(x)
        ...
        ...     def compute_loss(self, batch):
        ...         x, y = batch
        ...         logits = self(x)
        ...         loss = mx.mean(nn.losses.cross_entropy(logits, y))
        ...         return loss, {"accuracy": mx.mean(mx.argmax(logits, axis=-1) == y)}
        ...
        ...     def configure_optimizers(self):
        ...         return OptimizerConfig(optimizer=optim.Adam(learning_rate=1e-3))
    """

    def __init__(self) -> None:
        super().__init__()
        self._trainer: Trainer | None = None
        self._last_metrics: dict[str, mx.array] = {}

    # ==================== REQUIRED METHODS ====================

    def compute_loss(self, batch: Any) -> tuple[mx.array, dict[str, mx.array]]:
        """Compute the loss for a batch. This is the core training logic.

        IMPORTANT: This method is called inside the gradient computation.
        It should be a pure function of the model parameters and batch.

        Args:
            batch: A batch from the dataloader. Structure is user-defined.

        Returns:
            Tuple of:
            - loss: Scalar loss value (mx.array with shape ())
            - metrics: Dict of additional metrics to log (e.g., {"accuracy": acc})
                       All values should be scalar mx.array

        Example:
            >>> def compute_loss(self, batch):
            ...     x, y = batch
            ...     logits = self(x)
            ...     loss = mx.mean(nn.losses.cross_entropy(logits, y))
            ...     preds = mx.argmax(logits, axis=-1)
            ...     accuracy = mx.mean(preds == y)
            ...     return loss, {"accuracy": accuracy}
        """
        raise NotImplementedError("TrainModule.compute_loss must be implemented by subclass")

    def configure_optimizers(self) -> OptimizerConfig:
        """Configure the optimizer for training.

        MLX optimizers natively support callable learning rates for scheduling.
        Pass a schedule function directly to the optimizer's learning_rate parameter.

        Returns:
            OptimizerConfig with the optimizer instance

        Example (constant lr):
            >>> def configure_optimizers(self):
            ...     return OptimizerConfig(
            ...         optimizer=optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
            ...     )

        Example (with warmup + cosine schedule):
            >>> def configure_optimizers(self):
            ...     from mlx_audio.train.schedulers import WarmupCosineScheduler
            ...     schedule = WarmupCosineScheduler(
            ...         peak_lr=1e-3,
            ...         warmup_steps=500,
            ...         total_steps=self.trainer.max_steps or 10000
            ...     )
            ...     return OptimizerConfig(
            ...         optimizer=optim.AdamW(learning_rate=schedule),
            ...         lr_schedule_name="warmup_cosine"
            ...     )
        """
        raise NotImplementedError(
            "TrainModule.configure_optimizers must be implemented by subclass"
        )

    # ==================== OPTIONAL METHODS ====================

    def validation_step(self, batch: Any) -> dict[str, mx.array]:
        """Perform a validation step. Called during validation loop.

        By default, calls compute_loss and returns loss + metrics with "val_" prefix.
        Override for custom validation logic.

        Args:
            batch: A batch from the validation dataloader

        Returns:
            Dict of metrics to log (should include "val_loss")
        """
        loss, metrics = self.compute_loss(batch)
        return {"val_loss": loss, **{f"val_{k}": v for k, v in metrics.items()}}

    def test_step(self, batch: Any) -> dict[str, mx.array]:
        """Perform a test step. Called during test loop.

        By default, calls compute_loss and returns loss + metrics with "test_" prefix.
        Override for custom test logic.

        Args:
            batch: A batch from the test dataloader

        Returns:
            Dict of metrics to log (should include "test_loss")
        """
        loss, metrics = self.compute_loss(batch)
        return {"test_loss": loss, **{f"test_{k}": v for k, v in metrics.items()}}

    # ==================== LIFECYCLE HOOKS ====================

    def on_train_start(self) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self) -> None:
        """Called at the end of training."""
        pass

    def on_train_epoch_start(self, epoch: int) -> None:
        """Called at the beginning of each training epoch.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        pass

    def on_train_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each training epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Aggregated metrics from the epoch
        """
        pass

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        """Called at the end of validation.

        Args:
            metrics: Aggregated validation metrics
        """
        pass

    # ==================== TRAINER INTEGRATION ====================

    @property
    def trainer(self) -> Trainer:
        """Reference to the Trainer instance.

        Raises:
            RuntimeError: If module is not attached to a Trainer
        """
        if self._trainer is None:
            raise RuntimeError("TrainModule is not attached to a Trainer")
        return self._trainer

    @property
    def current_epoch(self) -> int:
        """Current training epoch (0-indexed)."""
        return self.trainer.current_epoch

    @property
    def global_step(self) -> int:
        """Total number of optimizer steps taken."""
        return self.trainer.global_step

    def log(self, name: str, value: float | mx.array, **kwargs: Any) -> None:
        """Log a metric during training.

        Args:
            name: Metric name
            value: Metric value (scalar float or mx.array)
            **kwargs: Additional logging options (reserved for future use)
        """
        self.trainer.log(name, value, **kwargs)

    def log_dict(self, metrics: dict[str, float | mx.array], **kwargs: Any) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value
            **kwargs: Additional logging options (reserved for future use)
        """
        for name, value in metrics.items():
            self.log(name, value, **kwargs)
