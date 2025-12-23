"""Learning rate monitor callback for mlx-train."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlx_audio.train.callbacks.base import Callback, CallbackContext, CallbackPriority

if TYPE_CHECKING:
    from mlx_audio.train.module import TrainModule
    from mlx_audio.train.trainer import Trainer


class LearningRateMonitor(Callback):
    """Logs the learning rate at each step or epoch.

    Automatically detects if the optimizer uses a schedule and logs
    the current learning rate.

    Example:
        >>> trainer = Trainer(callbacks=[LearningRateMonitor()])

        >>> # Log at epoch level instead of step level
        >>> trainer = Trainer(callbacks=[LearningRateMonitor(logging_interval="epoch")])
    """

    priority = CallbackPriority.HIGH  # Log early

    def __init__(self, logging_interval: str = "step") -> None:
        """Initialize the learning rate monitor.

        Args:
            logging_interval: "step" to log every step, "epoch" to log every epoch
        """
        if logging_interval not in ("step", "epoch"):
            raise ValueError(f"logging_interval must be 'step' or 'epoch', got {logging_interval}")
        self.logging_interval = logging_interval
        self._last_lr: float | None = None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        batch: Any,
        outputs: dict[str, Any],
        ctx: CallbackContext,
    ) -> None:
        if self.logging_interval == "step":
            self._log_lr(trainer, ctx)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        if self.logging_interval == "epoch":
            self._log_lr(trainer, ctx)

    def _log_lr(self, trainer: Trainer, ctx: CallbackContext) -> None:
        """Get and log the current learning rate."""
        optimizer = trainer._optimizer
        if optimizer is None:
            return

        lr = getattr(optimizer, "learning_rate", None)
        if lr is None:
            return

        if callable(lr):
            # It's a schedule - get the value for current step
            import mlx.core as mx

            lr_value = lr(ctx.global_step)
            if isinstance(lr_value, mx.array):
                lr_value = float(lr_value.item())
        else:
            lr_value = float(lr)

        self._last_lr = lr_value
        trainer.log("learning_rate", lr_value)

    @property
    def last_lr(self) -> float | None:
        """Returns the last logged learning rate."""
        return self._last_lr
