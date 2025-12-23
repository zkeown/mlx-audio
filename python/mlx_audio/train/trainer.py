"""Trainer class for mlx-train.

Orchestrates the training loop with MLX-native patterns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_audio.train.callbacks.base import Callback, CallbackContext, CallbackRegistry
from mlx_audio.train.module import TrainModule

if TYPE_CHECKING:
    from mlx_audio.train.loggers.base import Logger


# Type alias for dataloaders - any iterable that yields batches
DataLoader = Iterator[Any]


class Trainer:
    """Orchestrates the training loop for MLX models.

    Handles:
    - Training loop with proper mx.eval() placement
    - Callback dispatch at lifecycle hooks
    - Gradient clipping
    - Checkpointing and resume
    - Logging integration

    Example:
        >>> model = MyModel()
        >>> trainer = Trainer(
        ...     max_epochs=10,
        ...     gradient_clip_val=1.0,
        ...     callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        ...     logger=WandbLogger(project="my-project")
        ... )
        >>> trainer.fit(model, train_loader, val_loader)
    """

    def __init__(
        self,
        *,
        max_epochs: int | None = None,
        max_steps: int | None = None,
        val_check_interval: int | float = 1.0,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
        default_root_dir: str = "./mlx_train_logs",
        enable_checkpointing: bool = True,
        compile: bool = True,
        seed: int | None = None,
        callbacks: list[Callback] | None = None,
        logger: Logger | list[Logger] | None = None,
        debug_lazy_eval: bool = False,
    ) -> None:
        """Initialize the Trainer.

        Args:
            max_epochs: Maximum number of training epochs. Either max_epochs or
                max_steps must be specified.
            max_steps: Maximum number of training steps. Takes precedence over
                max_epochs if both specified.
            val_check_interval: How often to run validation.
                - float: Fraction of epoch (e.g., 0.5 = twice per epoch)
                - int: Every N training steps
            gradient_clip_val: Maximum gradient norm/value. None disables clipping.
            gradient_clip_algorithm: "norm" for L2 norm clipping, "value" for
                element-wise clipping.
            default_root_dir: Default directory for checkpoints and logs.
            enable_checkpointing: Whether to save checkpoints automatically.
            compile: Whether to use mx.compile for the training step.
            seed: Random seed for reproducibility.
            callbacks: List of callbacks to use during training.
            logger: Logger or list of loggers for metric tracking.
            debug_lazy_eval: Enable debugging for lazy evaluation issues.
        """
        # Validation
        if max_epochs is None and max_steps is None:
            max_epochs = 1  # Default to 1 epoch

        # Configuration
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.val_check_interval = val_check_interval
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.default_root_dir = Path(default_root_dir)
        self.enable_checkpointing = enable_checkpointing
        self.compile = compile
        self.seed = seed
        self.debug_lazy_eval = debug_lazy_eval

        # State
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.should_stop: bool = False
        self._module: TrainModule | None = None
        self._optimizer: optim.Optimizer | None = None
        self._step_fn: Callable | None = None

        # Callbacks and loggers
        self._callbacks = CallbackRegistry(callbacks)
        if logger is None:
            self._loggers: list[Logger] = []
        elif isinstance(logger, list):
            self._loggers = logger
        else:
            self._loggers = [logger]

        # Metrics buffer for current step
        self._step_metrics: dict[str, float] = {}

        # Debug state
        self._eval_count: int = 0

    # ==================== PUBLIC API ====================

    def fit(
        self,
        module: TrainModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        """Train the model.

        Args:
            module: The TrainModule to train
            train_dataloader: Training data iterator (any iterable yielding batches)
            val_dataloader: Optional validation data iterator
            ckpt_path: Path to checkpoint for resuming training
        """
        self._setup(module, ckpt_path)

        try:
            self._callbacks.fire("on_fit_start", self, module)
            module.on_train_start()

            self._run_training_loop(train_dataloader, val_dataloader)

            module.on_train_end()
            self._callbacks.fire("on_fit_end", self, module)
        except Exception as e:
            self._callbacks.fire("on_exception", self, module, e)
            raise
        finally:
            self._teardown()
            for logger in self._loggers:
                logger.finalize()

    def validate(
        self,
        module: TrainModule,
        val_dataloader: DataLoader,
        ckpt_path: str | None = None,
    ) -> dict[str, float]:
        """Run validation and return metrics.

        Args:
            module: The TrainModule to validate
            val_dataloader: Validation data iterator
            ckpt_path: Optional checkpoint to load before validation

        Returns:
            Dictionary of validation metrics
        """
        self._setup(module, ckpt_path, training=False)
        try:
            return self._run_validation(val_dataloader)
        finally:
            self._teardown()

    def test(
        self,
        module: TrainModule,
        test_dataloader: DataLoader,
        ckpt_path: str | None = None,
    ) -> dict[str, float]:
        """Run testing and return metrics.

        Args:
            module: The TrainModule to test
            test_dataloader: Test data iterator
            ckpt_path: Optional checkpoint to load before testing

        Returns:
            Dictionary of test metrics
        """
        self._setup(module, ckpt_path, training=False)
        try:
            return self._run_test(test_dataloader)
        finally:
            self._teardown()

    def log(self, name: str, value: float | mx.array, **kwargs: Any) -> None:
        """Log a metric. Called by TrainModule.log().

        Args:
            name: Metric name
            value: Metric value (float or scalar mx.array)
            **kwargs: Reserved for future use
        """
        if isinstance(value, mx.array):
            value = float(value.item())
        self._step_metrics[name] = value

    def save_checkpoint(self, path: str | Path, is_best: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to save the checkpoint
            is_best: Whether this is the best checkpoint so far
        """
        from mlx_audio.train.checkpointing.manager import CheckpointManager

        path = Path(path)
        manager = CheckpointManager(path.parent)
        manager.save(
            checkpoint_name=path.stem,
            model=self._module,
            optimizer=self._optimizer,
            trainer_state=self._get_trainer_state(),
            callback_states=self._callbacks.get_state_dicts(),
            is_best=is_best,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to the checkpoint to load
        """
        from mlx_audio.train.checkpointing.manager import CheckpointManager

        path = Path(path)
        manager = CheckpointManager(path.parent)
        state, callback_states = manager.load(
            checkpoint_name=path.stem,
            model=self._module,
            optimizer=self._optimizer,
        )

        # Restore trainer state
        self.current_epoch = state.epoch
        self.global_step = state.global_step
        self.should_stop = state.should_stop

        # Restore RNG state if available
        if state.random_state is not None:
            mx.random.seed(int.from_bytes(state.random_state[:4], "little"))

        # Restore callback states
        self._callbacks.load_state_dicts(callback_states)

    # ==================== INTERNAL: SETUP/TEARDOWN ====================

    def _setup(
        self,
        module: TrainModule,
        ckpt_path: str | None,
        training: bool = True,
    ) -> None:
        """Initialize training state."""
        if self.seed is not None:
            mx.random.seed(self.seed)

        self._module = module
        module._trainer = self

        if training:
            # Configure optimizer
            opt_config = module.configure_optimizers()
            self._optimizer = opt_config.optimizer

            # Initialize optimizer state by doing a dummy step
            # This ensures optimizer.state is populated
            mx.eval(module.parameters())

            # Create step function
            self._step_fn = self._create_step_fn(module)

            # Load checkpoint if specified
            if ckpt_path:
                self.load_checkpoint(ckpt_path)

    def _teardown(self) -> None:
        """Clean up after training."""
        if self._module is not None:
            self._module._trainer = None
        self._module = None
        self._optimizer = None
        self._step_fn = None

    def _get_trainer_state(self) -> Any:
        """Get current trainer state for checkpointing."""
        from mlx_audio.train.checkpointing.state import TrainerState

        return TrainerState(
            epoch=self.current_epoch,
            global_step=self.global_step,
            should_stop=self.should_stop,
            random_state=self.seed.to_bytes(4, "little") if self.seed else None,
        )

    # ==================== INTERNAL: STEP FUNCTION ====================

    def _create_step_fn(self, module: TrainModule) -> Callable:
        """Create the training step function.

        This is the heart of MLX-native training:
        1. Create loss function compatible with nn.value_and_grad
        2. Wrap with gradient clipping if configured
        3. Optionally compile for performance
        """

        def loss_fn(model: TrainModule, batch: Any) -> mx.array:
            """Loss function for nn.value_and_grad."""
            loss, metrics = model.compute_loss(batch)
            # Store metrics in module for later retrieval
            model._last_metrics = metrics
            return loss

        # Create value_and_grad function
        loss_and_grad_fn = nn.value_and_grad(module, loss_fn)

        def step(batch: Any) -> tuple[mx.array, dict[str, Any]]:
            """Single training step."""
            # Forward + backward
            loss, grads = loss_and_grad_fn(module, batch)

            # Fire callback for gradient modification (e.g., clipping)
            grads = self._callbacks.fire(
                "on_after_backward",
                self,
                module,
                _chain_value=grads,
            )
            if grads is None:
                # No callbacks modified gradients, recompute
                _, grads = loss_and_grad_fn(module, batch)

            # Apply built-in gradient clipping if configured
            if self.gradient_clip_val is not None:
                grads = self._clip_gradients(grads)

            # Optimizer step
            self._callbacks.fire("on_before_optimizer_step", self, module)
            self._optimizer.update(module, grads)

            return loss, module._last_metrics

        # Optionally compile the step function
        if self.compile:
            # Note: We compile with state tracking for proper gradient flow
            step = mx.compile(step)

        return step

    def _clip_gradients(self, grads: dict[str, Any]) -> dict[str, Any]:
        """Apply gradient clipping."""
        from mlx.utils import tree_flatten, tree_unflatten

        flat_grads = tree_flatten(grads)

        if self.gradient_clip_algorithm == "norm":
            # Compute total norm
            total_norm_sq = mx.array(0.0)
            for _, g in flat_grads:
                if g is not None:
                    total_norm_sq = total_norm_sq + mx.sum(g * g)
            total_norm = mx.sqrt(total_norm_sq)

            # Compute clip coefficient
            clip_coef = self.gradient_clip_val / (total_norm + 1e-6)
            clip_coef = mx.minimum(clip_coef, mx.array(1.0))

            # Apply clipping
            clipped = [(k, g * clip_coef if g is not None else None) for k, g in flat_grads]
        else:
            # Value clipping
            clipped = [
                (
                    k,
                    mx.clip(g, -self.gradient_clip_val, self.gradient_clip_val)
                    if g is not None
                    else None,
                )
                for k, g in flat_grads
            ]

        return tree_unflatten(clipped)

    # ==================== INTERNAL: TRAINING LOOP ====================

    def _run_training_loop(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
    ) -> None:
        """Main training loop."""
        while not self.should_stop:
            # Check stopping conditions
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                break
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

            self._run_training_epoch(train_dataloader, val_dataloader)
            self.current_epoch += 1

    def _run_training_epoch(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
    ) -> None:
        """Run a single training epoch."""
        ctx = self._create_context(batch_idx=0)
        self._callbacks.fire("on_train_epoch_start", self, self._module, ctx)
        self._module.on_train_epoch_start(self.current_epoch)

        epoch_metrics: dict[str, float] = {}
        epoch_length = 0

        for batch_idx, batch in enumerate(train_dataloader):
            epoch_length += 1

            # Check step limit
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

            ctx = self._create_context(batch_idx=batch_idx)
            self._callbacks.fire("on_train_batch_start", self, self._module, batch, ctx)

            # Forward + backward + update
            loss, metrics = self._step_fn(batch)

            # CRITICAL: Force evaluation
            # This is where MLX's lazy evaluation meets reality
            self._force_eval(loss, metrics)

            # Collect metrics
            loss_value = float(loss.item())
            self._step_metrics["train_loss"] = loss_value

            # Add metrics from compute_loss
            for k, v in metrics.items():
                self._step_metrics[f"train_{k}"] = float(v.item())

            self.global_step += 1

            # Log to loggers
            self._log_metrics(self._step_metrics, step=self.global_step)

            # Fire batch end callback
            outputs = {"loss": loss_value, **{k: float(v.item()) for k, v in metrics.items()}}
            ctx = self._create_context(batch_idx=batch_idx)
            ctx.metrics.update(self._step_metrics)
            self._callbacks.fire("on_train_batch_end", self, self._module, batch, outputs, ctx)

            self._step_metrics.clear()

            # Check for validation
            if val_dataloader is not None and self._should_validate(batch_idx, epoch_length):
                val_metrics = self._run_validation(val_dataloader)
                epoch_metrics.update(val_metrics)

        # End of epoch
        ctx = self._create_context(batch_idx=epoch_length - 1)
        ctx.metrics.update(epoch_metrics)
        self._module.on_train_epoch_end(self.current_epoch, epoch_metrics)
        self._callbacks.fire("on_train_epoch_end", self, self._module, ctx)

    def _run_validation(self, val_dataloader: DataLoader) -> dict[str, float]:
        """Run validation loop."""
        ctx = self._create_context(batch_idx=0, is_validating=True)
        self._callbacks.fire("on_validation_start", self, self._module, ctx)
        self._module.on_validation_start()

        all_metrics: dict[str, list[float]] = {}

        for batch_idx, batch in enumerate(val_dataloader):
            # No gradients needed for validation
            metrics = self._module.validation_step(batch)

            # Evaluate and collect
            mx.eval(*[v for v in metrics.values() if isinstance(v, mx.array)])
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                val = float(v.item()) if isinstance(v, mx.array) else float(v)
                all_metrics[k].append(val)

        # Average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}

        self._log_metrics(avg_metrics, step=self.global_step)

        ctx = self._create_context(batch_idx=0, is_validating=True)
        ctx.metrics.update(avg_metrics)
        self._module.on_validation_end(avg_metrics)
        self._callbacks.fire("on_validation_end", self, self._module, ctx, avg_metrics)

        return avg_metrics

    def _run_test(self, test_dataloader: DataLoader) -> dict[str, float]:
        """Run test loop."""
        all_metrics: dict[str, list[float]] = {}

        for batch in test_dataloader:
            metrics = self._module.test_step(batch)

            mx.eval(*[v for v in metrics.values() if isinstance(v, mx.array)])
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                val = float(v.item()) if isinstance(v, mx.array) else float(v)
                all_metrics[k].append(val)

        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
        self._log_metrics(avg_metrics, step=self.global_step)
        return avg_metrics

    def _should_validate(self, batch_idx: int, epoch_length: int) -> bool:
        """Determine if validation should run."""
        if isinstance(self.val_check_interval, float):
            # Fraction of epoch - validate at end of epoch
            return batch_idx == epoch_length - 1
        else:
            # Every N steps
            return self.global_step > 0 and (self.global_step % self.val_check_interval) == 0

    def _force_eval(self, loss: mx.array, metrics: dict[str, mx.array]) -> None:
        """Force evaluation of lazy arrays."""
        arrays_to_eval = [loss, self._module.parameters(), self._optimizer.state]
        arrays_to_eval.extend(metrics.values())
        mx.eval(*arrays_to_eval)

        if self.debug_lazy_eval:
            self._eval_count += 1
            # TODO: Add more sophisticated lazy eval debugging

    def _create_context(
        self,
        batch_idx: int = 0,
        is_validating: bool = False,
    ) -> CallbackContext:
        """Create a callback context with current training state."""
        return CallbackContext(
            epoch=self.current_epoch,
            global_step=self.global_step,
            batch_idx=batch_idx,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            metrics=dict(self._step_metrics),
            is_training=not is_validating,
            is_validating=is_validating,
        )

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to all loggers."""
        for logger in self._loggers:
            logger.log_metrics(metrics, step=step)
