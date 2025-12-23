"""Model checkpoint callback for mlx-train."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mlx_audio.train.callbacks.base import Callback, CallbackContext, CallbackPriority

if TYPE_CHECKING:
    from mlx_audio.train.module import TrainModule
    from mlx_audio.train.trainer import Trainer


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Opinionated defaults:
    - Always saves as safetensors (MLX native format)
    - Tracks both "last" and "best" checkpoints
    - Includes full training state for resumption

    Directory structure:
        checkpoints/
            last/              # Most recent checkpoint
            best/              # Best according to monitored metric
            epoch_5/           # If save_top_k > 1

    Example:
        >>> ckpt = ModelCheckpoint(
        ...     dirpath="checkpoints",
        ...     monitor="val_loss",
        ...     save_top_k=3
        ... )
        >>> trainer = Trainer(callbacks=[ckpt])
    """

    priority = CallbackPriority.LOWEST  # Run last

    def __init__(
        self,
        dirpath: str = "checkpoints",
        filename: str = "epoch_{epoch}",
        monitor: str | None = "val_loss",
        mode: Literal["min", "max"] = "min",
        save_last: bool = True,
        save_top_k: int = 1,
        every_n_epochs: int = 1,
        save_on_exception: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize the checkpoint callback.

        Args:
            dirpath: Directory to save checkpoints
            filename: Filename template (supports {epoch}, {step} placeholders)
            monitor: Metric to monitor for best checkpoint. None to disable.
            mode: "min" if lower is better, "max" if higher is better
            save_last: Whether to maintain a "last" checkpoint
            save_top_k: Number of best checkpoints to keep
            every_n_epochs: Save every N epochs
            save_on_exception: Save emergency checkpoint on exception
            verbose: Print messages when saving checkpoints
        """
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs
        self.save_on_exception = save_on_exception
        self.verbose = verbose

        # State tracking
        self.best_score: float | None = None
        self.best_checkpoint_path: Path | None = None
        self._saved_checkpoints: list[tuple[float, Path]] = []

        # Mode determines comparison
        if mode == "min":
            self._is_better = lambda current, best: current < best
        else:
            self._is_better = lambda current, best: current > best

    def on_fit_start(self, trainer: Trainer, module: TrainModule) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_validation_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
        metrics: dict[str, float],
    ) -> None:
        """Save checkpoint after validation if conditions are met."""
        # Only save at epoch boundaries based on every_n_epochs
        # This is called after validation, which happens at epoch end
        if (ctx.epoch + 1) % self.every_n_epochs != 0:
            return

        # Get monitored metric
        current_score = metrics.get(self.monitor) if self.monitor else None

        # Determine if this is best
        is_best = False
        if current_score is not None:
            if self.best_score is None or self._is_better(current_score, self.best_score):
                self.best_score = current_score
                is_best = True

        # Build filename
        checkpoint_name = self.filename.format(epoch=ctx.epoch, step=ctx.global_step)

        # Save checkpoint
        trainer.save_checkpoint(self.dirpath / checkpoint_name, is_best=is_best)

        if is_best:
            self.best_checkpoint_path = self.dirpath / checkpoint_name
            if self.verbose:
                print(f"  Checkpoint: new best {self.monitor}={current_score:.6f}")

        # Manage top_k checkpoints
        if self.save_top_k > 0 and current_score is not None:
            self._manage_top_k(current_score, self.dirpath / checkpoint_name)

    def on_exception(
        self,
        trainer: Trainer,
        module: TrainModule,
        exception: Exception,
    ) -> None:
        """Save emergency checkpoint on exception."""
        if self.save_on_exception:
            emergency_name = "emergency_checkpoint"
            trainer.save_checkpoint(self.dirpath / emergency_name, is_best=False)
            print(f"  Emergency checkpoint saved to {self.dirpath / emergency_name}")

    def _manage_top_k(self, score: float, filepath: Path) -> None:
        """Keep only top_k checkpoints by score."""
        self._saved_checkpoints.append((score, filepath))

        # Sort by score (ascending for min, descending for max)
        self._saved_checkpoints.sort(
            key=lambda x: x[0],
            reverse=(self.mode == "max"),
        )

        # Remove excess checkpoints
        while len(self._saved_checkpoints) > self.save_top_k:
            _, old_path = self._saved_checkpoints.pop()
            # Don't delete best or last
            if old_path != self.best_checkpoint_path:
                self._delete_checkpoint(old_path)

    def _delete_checkpoint(self, path: Path) -> None:
        """Delete a checkpoint directory."""
        if path.exists() and path.is_dir():
            import shutil

            shutil.rmtree(path)

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "best_checkpoint_path": str(self.best_checkpoint_path)
            if self.best_checkpoint_path
            else None,
            "saved_checkpoints": [(s, str(p)) for s, p in self._saved_checkpoints],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_score = state.get("best_score")
        path = state.get("best_checkpoint_path")
        self.best_checkpoint_path = Path(path) if path else None
        self._saved_checkpoints = [(s, Path(p)) for s, p in state.get("saved_checkpoints", [])]
