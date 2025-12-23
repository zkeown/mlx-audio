"""Progress bar callback for mlx-train."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from mlx_audio.train.callbacks.base import Callback, CallbackContext, CallbackPriority

if TYPE_CHECKING:
    from mlx_audio.train.module import TrainModule
    from mlx_audio.train.trainer import Trainer


class ProgressBar(Callback):
    """Display training progress.

    Shows epoch, step, loss, metrics, and speed (samples/sec).
    Uses simple print-based output - no external dependencies.

    Example:
        >>> trainer = Trainer(callbacks=[ProgressBar()])

        Output:
        Epoch 1/10
          Step 100 | train_loss: 0.4523 | train_accuracy: 0.8750 | 1234.5 samples/sec
    """

    priority = CallbackPriority.LOW

    def __init__(
        self,
        refresh_rate: int = 10,
        show_eta: bool = True,
    ) -> None:
        """Initialize the progress bar.

        Args:
            refresh_rate: Update display every N batches
            show_eta: Whether to show estimated time remaining
        """
        self.refresh_rate = refresh_rate
        self.show_eta = show_eta

        self._epoch_start_time: float = 0
        self._fit_start_time: float = 0
        self._samples_processed: int = 0
        self._last_print_len: int = 0

    def on_fit_start(self, trainer: Trainer, module: TrainModule) -> None:
        self._fit_start_time = time.time()

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        self._epoch_start_time = time.time()
        self._samples_processed = 0

        epoch_str = f"Epoch {ctx.epoch + 1}"
        if ctx.max_epochs:
            epoch_str += f"/{ctx.max_epochs}"
        print(f"\n{epoch_str}")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        batch: Any,
        outputs: dict[str, Any],
        ctx: CallbackContext,
    ) -> None:
        batch_size = self._get_batch_size(batch)
        self._samples_processed += batch_size

        if ctx.batch_idx % self.refresh_rate == 0:
            self._print_progress(ctx, outputs)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        elapsed = time.time() - self._epoch_start_time
        # Clear line and print summary
        print(f"\r{' ' * self._last_print_len}", end="\r")
        print(f"  Epoch {ctx.epoch + 1} completed in {elapsed:.1f}s")

    def on_validation_start(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        print("  Validating...", end="\r")

    def on_validation_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
        metrics: dict[str, float],
    ) -> None:
        # Format validation metrics
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items()))
        print(f"  Validation: {metrics_str}")

    def on_fit_end(self, trainer: Trainer, module: TrainModule) -> None:
        total_time = time.time() - self._fit_start_time
        print(f"\nTraining completed in {self._format_time(total_time)}")

    def _print_progress(self, ctx: CallbackContext, outputs: dict[str, Any]) -> None:
        """Print progress line."""
        elapsed = time.time() - self._epoch_start_time
        samples_per_sec = self._samples_processed / elapsed if elapsed > 0 else 0

        # Build metrics string
        metrics_parts = []
        for k, v in sorted(ctx.metrics.items()):
            if isinstance(v, float):
                metrics_parts.append(f"{k}: {v:.4f}")

        metrics_str = " | ".join(metrics_parts)

        # Build progress string
        progress = f"  Step {ctx.global_step}"
        if metrics_str:
            progress += f" | {metrics_str}"
        progress += f" | {samples_per_sec:.1f} samples/sec"

        # Add ETA if enabled
        if self.show_eta and ctx.max_steps:
            remaining_steps = ctx.max_steps - ctx.global_step
            steps_per_sec = ctx.global_step / elapsed if elapsed > 0 else 0
            if steps_per_sec > 0:
                eta_sec = remaining_steps / steps_per_sec
                progress += f" | ETA: {self._format_time(eta_sec)}"

        # Print with carriage return for in-place update
        self._last_print_len = len(progress)
        print(f"\r{progress}", end="", flush=True)

    def _get_batch_size(self, batch: Any) -> int:
        """Extract batch size from batch data."""
        if isinstance(batch, tuple | list) and len(batch) > 0:
            first = batch[0]
            if hasattr(first, "shape"):
                return first.shape[0]
        elif hasattr(batch, "shape"):
            return batch.shape[0]
        return 1

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
