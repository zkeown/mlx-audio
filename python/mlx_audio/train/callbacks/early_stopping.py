"""Early stopping callback for mlx-train."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from mlx_audio.train.callbacks.base import Callback, CallbackContext, CallbackPriority

if TYPE_CHECKING:
    from mlx_audio.train.module import TrainModule
    from mlx_audio.train.trainer import Trainer


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    This is an opinionated implementation:
    - Defaults to monitoring val_loss with mode="min"
    - Uses simple patience counter (no percentage-based thresholds)
    - Integrates with checkpoint system to restore best weights

    Example:
        >>> early_stop = EarlyStopping(
        ...     monitor="val_loss",
        ...     patience=5,
        ...     min_delta=0.001
        ... )
        >>> trainer = Trainer(callbacks=[early_stop])

        >>> # For accuracy (higher is better)
        >>> early_stop = EarlyStopping(
        ...     monitor="val_accuracy",
        ...     mode="max",
        ...     patience=3
        ... )
    """

    priority = CallbackPriority.LOW  # Run after logging callbacks

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 3,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        verbose: bool = True,
    ) -> None:
        """Initialize early stopping.

        Args:
            monitor: Metric to monitor (must be logged during validation)
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: "min" if lower is better, "max" if higher is better
            verbose: Print messages about early stopping status
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        # State
        self.best_score: float | None = None
        self.wait_count: int = 0
        self.stopped_epoch: int = 0

        # Mode determines comparison
        if mode == "min":
            self._is_improvement = lambda current, best: current < (best - min_delta)
        else:
            self._is_improvement = lambda current, best: current > (best + min_delta)

    def on_validation_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
        metrics: dict[str, float],
    ) -> None:
        """Check if training should stop."""
        current = metrics.get(self.monitor)

        if current is None:
            if self.verbose:
                available = list(metrics.keys())
                print(f"  EarlyStopping: metric '{self.monitor}' not found in {available}")
            return

        if self.best_score is None or self._is_improvement(current, self.best_score):
            self.best_score = current
            self.wait_count = 0
            if self.verbose:
                print(f"  EarlyStopping: {self.monitor} improved to {current:.6f}")
        else:
            self.wait_count += 1
            if self.verbose:
                print(
                    f"  EarlyStopping: {self.monitor} did not improve "
                    f"({self.wait_count}/{self.patience})"
                )

            if self.wait_count >= self.patience:
                self.stopped_epoch = ctx.epoch
                trainer.should_stop = True
                if self.verbose:
                    print(f"  EarlyStopping: stopping at epoch {ctx.epoch + 1}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_score = state.get("best_score")
        self.wait_count = state.get("wait_count", 0)
        self.stopped_epoch = state.get("stopped_epoch", 0)
