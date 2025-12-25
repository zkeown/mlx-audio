"""TrainModule base class for mlx-train.

Users subclass TrainModule and implement training_step() and configure_optimizers()
to define their training logic. For backward compatibility, compute_loss() is also
supported but deprecated.

This module aims for API parity with PyTorch Lightning's LightningModule.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

if TYPE_CHECKING:
    from mlx_audio.train.trainer import Trainer


def _has_param(method: Any, param_name: str) -> bool:
    """Check if a method accepts a specific parameter."""
    try:
        sig = inspect.signature(method)
        return param_name in sig.parameters
    except (ValueError, TypeError):
        return False


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
    - training_step: Compute loss given a batch (PyTorch Lightning compatible)
    - configure_optimizers: Return optimizer configuration

    Optional overrides:
    - validation_step: Custom validation logic
    - test_step: Custom test logic
    - predict_step: Custom prediction logic
    - on_train_start/end: Training lifecycle hooks
    - on_train_epoch_start/end: Epoch-level hooks
    - on_train_batch_start/end: Batch-level hooks
    - on_before_backward/on_after_backward: Optimization hooks
    - on_validation_start/end: Validation lifecycle hooks
    - on_test_start/end: Test lifecycle hooks

    Example:
        >>> class MyModel(TrainModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(784, 10)
        ...
        ...     def __call__(self, x):
        ...         return self.linear(x)
        ...
        ...     def training_step(self, batch, batch_idx):
        ...         x, y = batch
        ...         logits = self(x)
        ...         loss = mx.mean(nn.losses.cross_entropy(logits, y))
        ...         acc = mx.mean(mx.argmax(logits, axis=-1) == y)
        ...         return {"loss": loss, "accuracy": acc}
        ...
        ...     def configure_optimizers(self):
        ...         return OptimizerConfig(optimizer=optim.Adam(learning_rate=1e-3))
    """

    def __init__(self) -> None:
        super().__init__()
        self._trainer: Trainer | None = None
        self._last_metrics: dict[str, mx.array] = {}
        self._logged_metrics: dict[str, float] = {}

    # ==================== REQUIRED METHODS ====================

    def training_step(
        self, batch: Any, batch_idx: int
    ) -> mx.array | dict[str, mx.array]:
        """Perform a single training step. This is the core training logic.

        IMPORTANT: This method is called inside the gradient computation.
        It should be a pure function of the model parameters and batch.

        Args:
            batch: A batch from the dataloader. Structure is user-defined.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            Either:
            - A scalar loss tensor (mx.array with shape ())
            - A dict with 'loss' key and optional additional metrics

        Example:
            >>> def training_step(self, batch, batch_idx):
            ...     x, y = batch
            ...     logits = self(x)
            ...     loss = mx.mean(nn.losses.cross_entropy(logits, y))
            ...     preds = mx.argmax(logits, axis=-1)
            ...     accuracy = mx.mean(preds == y)
            ...     return {"loss": loss, "accuracy": accuracy}
        """
        # Default implementation: delegate to compute_loss for backward compat
        loss, metrics = self.compute_loss(batch)
        return {"loss": loss, **metrics}

    def compute_loss(self, batch: Any) -> tuple[mx.array, dict[str, mx.array]]:
        """[DEPRECATED] Compute the loss for a batch.

        This method is deprecated. Use training_step(batch, batch_idx) instead.
        Kept for backward compatibility with existing code.

        Args:
            batch: A batch from the dataloader. Structure is user-defined.

        Returns:
            Tuple of:
            - loss: Scalar loss value (mx.array with shape ())
            - metrics: Dict of additional metrics to log (e.g., {"accuracy": acc})
                       All values should be scalar mx.array
        """
        raise NotImplementedError(
            "TrainModule requires either training_step(batch, batch_idx) or "
            "compute_loss(batch) to be implemented. training_step is preferred."
        )

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

    def validation_step(self, batch: Any, batch_idx: int = 0) -> dict[str, mx.array]:
        """Perform a validation step. Called during validation loop.

        By default, calls training_step and returns metrics with "val_" prefix.
        Override for custom validation logic.

        Args:
            batch: A batch from the validation dataloader
            batch_idx: Index of the current batch within the validation epoch

        Returns:
            Dict of metrics to log (should include "val_loss")
        """
        result = self.training_step(batch, batch_idx)
        if isinstance(result, mx.array):
            return {"val_loss": result}
        return {f"val_{k}": v for k, v in result.items()}

    def test_step(self, batch: Any, batch_idx: int = 0) -> dict[str, mx.array]:
        """Perform a test step. Called during test loop.

        By default, calls training_step and returns metrics with "test_" prefix.
        Override for custom test logic.

        Args:
            batch: A batch from the test dataloader
            batch_idx: Index of the current batch within the test epoch

        Returns:
            Dict of metrics to log (should include "test_loss")
        """
        result = self.training_step(batch, batch_idx)
        if isinstance(result, mx.array):
            return {"test_loss": result}
        return {f"test_{k}": v for k, v in result.items()}

    def predict_step(self, batch: Any, batch_idx: int = 0) -> Any:
        """Perform a prediction step. Called during predict loop.

        By default, runs forward pass on the batch. Override for custom logic.

        Args:
            batch: A batch from the dataloader
            batch_idx: Index of the current batch

        Returns:
            Model predictions (structure depends on model)
        """
        # Default: assume batch is (input, target) or just input
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self(x)

    # ==================== LIFECYCLE HOOKS ====================

    def on_train_start(self) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self) -> None:
        """Called at the end of training."""
        pass

    def on_train_epoch_start(self) -> None:
        """Called at the beginning of each training epoch.

        Access epoch via self.current_epoch.
        """
        pass

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch.

        Access epoch via self.current_epoch, metrics via self.trainer.
        """
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Called at the start of each training batch.

        Args:
            batch: The current batch
            batch_idx: Index of the current batch
        """
        pass

    def on_train_batch_end(
        self, outputs: dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        """Called at the end of each training batch.

        Args:
            outputs: Dictionary containing 'loss' and any logged metrics
            batch: The current batch
            batch_idx: Index of the current batch
        """
        pass

    def on_before_backward(self, loss: mx.array) -> None:
        """Called after loss computation, before gradient computation.

        Note: In MLX, gradients are computed via nn.value_and_grad, so this
        hook is called after loss is computed but conceptually 'before backward'.

        Args:
            loss: The computed loss value
        """
        pass

    def on_after_backward(self) -> None:
        """Called after gradient computation."""
        pass

    def on_before_optimizer_step(
        self, optimizer: optim.Optimizer, optimizer_idx: int = 0
    ) -> None:
        """Called before the optimizer step.

        Args:
            optimizer: The optimizer being used
            optimizer_idx: Index of the optimizer (for multi-optimizer setups)
        """
        pass

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self) -> None:
        """Called at the end of validation.

        Access metrics via self.trainer.
        """
        pass

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        pass

    def on_test_end(self) -> None:
        """Called at the end of testing."""
        pass

    def on_predict_start(self) -> None:
        """Called at the beginning of prediction."""
        pass

    def on_predict_end(self) -> None:
        """Called at the end of prediction."""
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
