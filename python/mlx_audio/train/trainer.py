"""Trainer class for mlx-train.

Orchestrates the training loop with MLX-native patterns.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
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
        check_val_every_n_epoch: int = 1,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
        accumulate_grad_batches: int = 1,
        default_root_dir: str = "./mlx_train_logs",
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        compile: bool = True,
        seed: int | None = None,
        deterministic: bool = False,
        callbacks: list[Callback] | None = None,
        logger: Logger | list[Logger] | None = None,
        log_every_n_steps: int = 50,
        num_sanity_val_steps: int = 2,
        fast_dev_run: bool | int = False,
        limit_train_batches: int | float | None = None,
        limit_val_batches: int | float | None = None,
        limit_test_batches: int | float | None = None,
        limit_predict_batches: int | float | None = None,
        overfit_batches: int | float = 0,
        detect_anomaly: bool = False,
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
            check_val_every_n_epoch: Run validation every N epochs.
            gradient_clip_val: Maximum gradient norm/value. None disables clipping.
            gradient_clip_algorithm: "norm" for L2 norm clipping, "value" for
                element-wise clipping.
            accumulate_grad_batches: Accumulate gradients over N batches before
                updating. Effectively increases batch size without more memory.
            default_root_dir: Default directory for checkpoints and logs.
            enable_checkpointing: Whether to save checkpoints automatically.
            enable_progress_bar: Whether to show progress bar during training.
            enable_model_summary: Whether to print model summary at start.
            compile: Whether to use mx.compile for the training step.
            seed: Random seed for reproducibility.
            deterministic: If True, sets seed and ensures reproducible behavior.
            callbacks: List of callbacks to use during training.
            logger: Logger or list of loggers for metric tracking.
            log_every_n_steps: Log metrics every N training steps.
            num_sanity_val_steps: Number of validation batches to run before
                training starts to catch errors early. Set to 0 to disable.
            fast_dev_run: If True, runs 1 batch of train/val/test for debugging.
                If int, runs that many batches.
            limit_train_batches: Limit training to N batches (int) or fraction (float).
            limit_val_batches: Limit validation to N batches (int) or fraction (float).
            limit_test_batches: Limit testing to N batches (int) or fraction (float).
            limit_predict_batches: Limit prediction to N batches (int) or fraction.
            overfit_batches: If > 0, uses only this many batches for train/val/test
                and repeats them. Useful for debugging model capacity.
            detect_anomaly: If True, checks for NaN/Inf in gradients.
            debug_lazy_eval: Enable debugging for lazy evaluation issues.
        """
        # Handle fast_dev_run mode - overrides other settings
        if fast_dev_run:
            n_batches = 1 if fast_dev_run is True else int(fast_dev_run)
            limit_train_batches = n_batches
            limit_val_batches = n_batches
            limit_test_batches = n_batches
            limit_predict_batches = n_batches
            max_epochs = 1
            enable_checkpointing = False
            num_sanity_val_steps = 0

        # Validation
        if max_epochs is None and max_steps is None:
            max_epochs = 1  # Default to 1 epoch

        # Handle deterministic mode
        if deterministic and seed is None:
            seed = 42  # Default seed for deterministic mode

        # Configuration
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.val_check_interval = val_check_interval
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.default_root_dir = Path(default_root_dir)
        self.enable_checkpointing = enable_checkpointing
        self.enable_progress_bar = enable_progress_bar
        self.enable_model_summary = enable_model_summary
        self.compile = compile
        self.seed = seed
        self.deterministic = deterministic
        self.log_every_n_steps = log_every_n_steps
        self.num_sanity_val_steps = num_sanity_val_steps
        self.fast_dev_run = fast_dev_run
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.limit_predict_batches = limit_predict_batches
        self.overfit_batches = overfit_batches
        self.detect_anomaly = detect_anomaly
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

        # Metrics buffers
        self._step_metrics: dict[str, float] = {}
        self._callback_metrics: dict[str, float] = {}
        self._logged_metrics: dict[str, float] = {}

        # Overfit batches cache
        self._overfit_cache: list[Any] | None = None

        # Debug state
        self._eval_count: int = 0

    # ==================== PUBLIC PROPERTIES ====================

    @property
    def callback_metrics(self) -> dict[str, float]:
        """All metrics logged in callbacks and training."""
        return dict(self._callback_metrics)

    @property
    def logged_metrics(self) -> dict[str, float]:
        """Metrics logged in the most recent step."""
        return dict(self._logged_metrics)

    @property
    def progress_bar_metrics(self) -> dict[str, float]:
        """Metrics for progress bar display (excludes internal metrics)."""
        return {k: v for k, v in self._step_metrics.items() if not k.startswith("_")}

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
            # Print model summary if enabled
            if self.enable_model_summary:
                self._print_model_summary(module)

            # Run sanity validation before training
            if self.num_sanity_val_steps > 0 and val_dataloader is not None:
                self._run_sanity_validation(val_dataloader)

            self._callbacks.fire("on_fit_start", self, module)
            self._callbacks.fire("on_train_start", self, module)
            module.on_train_start()

            self._run_training_loop(train_dataloader, val_dataloader)

            module.on_train_end()
            self._callbacks.fire("on_train_end", self, module)
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
            self._callbacks.fire("on_test_start", self, module, self._create_context())
            module.on_test_start()

            metrics = self._run_test(test_dataloader)

            module.on_test_end()
            self._callbacks.fire(
                "on_test_end", self, module, self._create_context(), metrics
            )
            return metrics
        finally:
            self._teardown()

    def predict(
        self,
        module: TrainModule,
        dataloaders: DataLoader,
        ckpt_path: str | None = None,
    ) -> list[Any]:
        """Run prediction and return outputs.

        Args:
            module: The TrainModule to use for prediction
            dataloaders: Data iterator for prediction
            ckpt_path: Optional checkpoint to load before prediction

        Returns:
            List of prediction outputs from predict_step
        """
        self._setup(module, ckpt_path, training=False)
        try:
            self._callbacks.fire("on_predict_start", self, module, self._create_context())
            module.on_predict_start()

            predictions = self._run_predict(dataloaders)

            module.on_predict_end()
            self._callbacks.fire("on_predict_end", self, module, self._create_context())
            return predictions
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

        Note: When compiling, we use mx.compile with explicit state tracking
        via inputs/outputs parameters. Callbacks are kept outside the compiled
        function.
        """
        from functools import partial

        from mlx.utils import tree_flatten, tree_map, tree_unflatten

        # Capture optimizer reference for use in core_step
        optimizer = self._optimizer
        gradient_clip_val = self.gradient_clip_val
        accumulate_grad_batches = self.accumulate_grad_batches
        detect_anomaly = self.detect_anomaly

        # Storage for metrics from training_step
        # NOTE: We store (loss, metrics_dict) and extract inside compiled region
        # Use a list with single element to avoid dict overhead and allow clearing
        step_result_cache: list[Any] = [None]

        def training_step_fn(model: TrainModule, batch: Any, batch_idx: int) -> mx.array:
            """Loss function for nn.value_and_grad - caches result, returns loss."""
            result = model.training_step(batch, batch_idx)
            step_result_cache[0] = result
            if isinstance(result, mx.array):
                return result
            return result["loss"]

        # Create value_and_grad function
        loss_and_grad_fn = nn.value_and_grad(module, training_step_fn)

        # Gradient clipping helper (pure function)
        def clip_grads(grads: dict, clip_val: float) -> dict:
            """Apply gradient clipping - pure function."""
            flat_grads = tree_flatten(grads)
            total_norm = mx.sqrt(
                sum(mx.sum(g**2) for _, g in flat_grads if isinstance(g, mx.array))
            )
            scale = clip_val / (total_norm + 1e-6)
            scale = mx.minimum(scale, 1.0)

            clipped = [
                (k, g * scale if isinstance(g, mx.array) else g)
                for k, g in flat_grads
            ]
            return tree_unflatten(clipped)

        # Gradient accumulation state
        accumulated_grads: dict | None = None
        accumulation_step = 0

        def add_grads(g1: mx.array | None, g2: mx.array | None) -> mx.array | None:
            """Add two gradient arrays, handling None."""
            if g1 is None:
                return g2
            if g2 is None:
                return g1
            return g1 + g2

        # Core compute function - pure, no side effects
        def core_step(batch: Any, batch_idx: int) -> tuple[mx.array, dict, dict]:
            """Pure computation: forward + backward + clip + update.

            Returns (loss, grads, metrics_dict) where metrics values are mx.arrays.
            """
            nonlocal accumulated_grads, accumulation_step

            # Forward + backward
            loss, grads = loss_and_grad_fn(module, batch, batch_idx)

            # Extract metrics from cached result (set by training_step_fn)
            cached_result = step_result_cache[0]
            if cached_result is None or isinstance(cached_result, mx.array):
                metrics = {}
            else:
                metrics = {k: v for k, v in cached_result.items() if k != "loss"}
            # Clear cache to avoid memory leak
            step_result_cache[0] = None

            # Gradient anomaly detection
            if detect_anomaly:
                self._check_gradients(grads)

            # Gradient accumulation
            if accumulate_grad_batches > 1:
                # Scale loss for accumulation
                loss = loss / accumulate_grad_batches

                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = tree_map(add_grads, accumulated_grads, grads)

                accumulation_step += 1

                # Only update when we've accumulated enough
                if accumulation_step >= accumulate_grad_batches:
                    grads = accumulated_grads

                    # Apply gradient clipping if configured
                    if gradient_clip_val is not None:
                        grads = clip_grads(grads, gradient_clip_val)

                    # Optimizer step
                    optimizer.update(module, grads)

                    # Reset accumulation state
                    accumulated_grads = None
                    accumulation_step = 0
            else:
                # No accumulation - standard path
                if gradient_clip_val is not None:
                    grads = clip_grads(grads, gradient_clip_val)
                optimizer.update(module, grads)

            return loss, grads, metrics

        # Optionally compile with state tracking
        # Note: Compile is disabled when using gradient accumulation because
        # the nonlocal state tracking isn't compatible with mx.compile
        use_compile = self.compile and accumulate_grad_batches == 1
        if use_compile:
            # State includes model params, optimizer state, and random state
            # (random state needed for dropout, etc.)
            state = [module.state, optimizer.state, mx.random.state]
            compiled = partial(mx.compile, inputs=state, outputs=state)
            core_step = compiled(core_step)

        # Store state for _force_eval
        if use_compile:
            self._compile_state = [module.state, optimizer.state]
        else:
            self._compile_state = None

        def step(batch: Any, batch_idx: int) -> tuple[mx.array, dict[str, Any]]:
            """Full training step with callbacks."""
            # Pure computation (possibly compiled)
            # Returns (loss, grads, metrics) where metrics are from training_step
            loss, grads, metrics = core_step(batch, batch_idx)

            # Fire callbacks (outside compiled region)
            modified_grads = self._callbacks.fire(
                "on_after_backward",
                self,
                module,
                _chain_value=grads,
            )

            # If callbacks modified gradients, we need to re-apply update
            # This is rare and only used by specialized callbacks
            if modified_grads is not None and modified_grads is not grads:
                # Redo update with modified grads
                optimizer.update(module, modified_grads)

            self._callbacks.fire("on_before_optimizer_step", self, module, optimizer)

            return loss, metrics

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
        self._module.on_train_epoch_start()

        epoch_metrics: dict[str, float] = {}
        epoch_length = 0

        # Apply batch limiting and overfit mode
        dataloader = self._limit_batches(train_dataloader, self.limit_train_batches)

        for batch_idx, batch in enumerate(dataloader):
            epoch_length += 1

            # Check step limit
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

            ctx = self._create_context(batch_idx=batch_idx)
            self._callbacks.fire("on_train_batch_start", self, self._module, batch, ctx)

            # Fire module hook
            self._module.on_train_batch_start(batch, batch_idx)

            # Forward + backward + update
            loss, metrics = self._step_fn(batch, batch_idx)

            # CRITICAL: Force evaluation
            # This is where MLX's lazy evaluation meets reality
            self._force_eval(loss, metrics)

            # Collect metrics
            loss_value = float(loss.item())
            self._step_metrics["train_loss"] = loss_value

            # Add metrics from training_step
            for k, v in metrics.items():
                if isinstance(v, mx.array):
                    self._step_metrics[f"train_{k}"] = float(v.item())
                else:
                    self._step_metrics[f"train_{k}"] = float(v)

            self.global_step += 1

            # Update callback metrics
            self._callback_metrics.update(self._step_metrics)

            # Log to loggers (respecting log_every_n_steps)
            if self.global_step % self.log_every_n_steps == 0:
                self._log_metrics(self._step_metrics, step=self.global_step)
            self._logged_metrics = dict(self._step_metrics)

            # Fire batch end callbacks
            outputs = {"loss": loss_value}
            for k, v in metrics.items():
                if isinstance(v, mx.array):
                    outputs[k] = float(v.item())
                else:
                    outputs[k] = float(v)

            # Fire module hook
            self._module.on_train_batch_end(outputs, batch, batch_idx)

            ctx = self._create_context(batch_idx=batch_idx)
            ctx.metrics.update(self._step_metrics)
            self._callbacks.fire("on_train_batch_end", self, self._module, batch, outputs, ctx)

            self._step_metrics.clear()

            # Check for step-based validation (val_check_interval as int)
            if (
                val_dataloader is not None
                and isinstance(self.val_check_interval, int)
                and self._should_validate_step()
            ):
                val_metrics = self._run_validation(val_dataloader)
                epoch_metrics.update(val_metrics)

        # End of epoch - check for epoch-based validation (val_check_interval as float)
        if (
            val_dataloader is not None
            and isinstance(self.val_check_interval, float)
            and self._should_validate_epoch()
        ):
            val_metrics = self._run_validation(val_dataloader)
            epoch_metrics.update(val_metrics)

        ctx = self._create_context(batch_idx=max(0, epoch_length - 1))
        ctx.metrics.update(epoch_metrics)
        self._module.on_train_epoch_end()
        self._callbacks.fire("on_train_epoch_end", self, self._module, ctx)

    def _run_validation(self, val_dataloader: DataLoader) -> dict[str, float]:
        """Run validation loop."""
        ctx = self._create_context(batch_idx=0, is_validating=True)
        self._callbacks.fire("on_validation_start", self, self._module, ctx)
        self._module.on_validation_start()

        all_metrics: dict[str, list[float]] = {}

        # Apply batch limiting
        dataloader = self._limit_batches(val_dataloader, self.limit_val_batches)

        for batch_idx, batch in enumerate(dataloader):
            # Fire batch start callback
            ctx = self._create_context(batch_idx=batch_idx, is_validating=True)
            self._callbacks.fire(
                "on_validation_batch_start", self, self._module, batch, batch_idx, ctx
            )

            # No gradients needed for validation
            metrics = self._module.validation_step(batch, batch_idx)

            # Evaluate and collect
            mx.eval(*[v for v in metrics.values() if isinstance(v, mx.array)])
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                val = float(v.item()) if isinstance(v, mx.array) else float(v)
                all_metrics[k].append(val)

            # Fire batch end callback
            outputs = {k: float(v.item()) if isinstance(v, mx.array) else float(v)
                      for k, v in metrics.items()}
            self._callbacks.fire(
                "on_validation_batch_end", self, self._module, outputs, batch, batch_idx, ctx
            )

        # Average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}

        self._log_metrics(avg_metrics, step=self.global_step)
        self._callback_metrics.update(avg_metrics)

        ctx = self._create_context(batch_idx=0, is_validating=True)
        ctx.metrics.update(avg_metrics)
        self._module.on_validation_end()
        self._callbacks.fire("on_validation_end", self, self._module, ctx, avg_metrics)

        return avg_metrics

    def _run_test(self, test_dataloader: DataLoader) -> dict[str, float]:
        """Run test loop."""
        all_metrics: dict[str, list[float]] = {}

        # Apply batch limiting
        dataloader = self._limit_batches(test_dataloader, self.limit_test_batches)

        for batch_idx, batch in enumerate(dataloader):
            # Fire batch start callback
            ctx = self._create_context(batch_idx=batch_idx)
            self._callbacks.fire(
                "on_test_batch_start", self, self._module, batch, batch_idx, ctx
            )

            metrics = self._module.test_step(batch, batch_idx)

            mx.eval(*[v for v in metrics.values() if isinstance(v, mx.array)])
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                val = float(v.item()) if isinstance(v, mx.array) else float(v)
                all_metrics[k].append(val)

            # Fire batch end callback
            outputs = {k: float(v.item()) if isinstance(v, mx.array) else float(v)
                      for k, v in metrics.items()}
            self._callbacks.fire(
                "on_test_batch_end", self, self._module, outputs, batch, batch_idx, ctx
            )

        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
        self._log_metrics(avg_metrics, step=self.global_step)
        self._callback_metrics.update(avg_metrics)
        return avg_metrics

    def _run_predict(self, dataloaders: DataLoader) -> list[Any]:
        """Run prediction loop."""
        predictions = []

        # Apply batch limiting
        dataloader = self._limit_batches(dataloaders, self.limit_predict_batches)

        for batch_idx, batch in enumerate(dataloader):
            # Fire batch start callback
            ctx = self._create_context(batch_idx=batch_idx)
            self._callbacks.fire(
                "on_predict_batch_start", self, self._module, batch, batch_idx, ctx
            )

            pred = self._module.predict_step(batch, batch_idx)
            mx.eval(pred)
            predictions.append(pred)

            # Fire batch end callback
            self._callbacks.fire(
                "on_predict_batch_end", self, self._module, pred, batch, batch_idx, ctx
            )

        return predictions

    def _should_validate_epoch(self) -> bool:
        """Check if validation should run at end of epoch."""
        # Check epoch-based validation frequency
        return (self.current_epoch + 1) % self.check_val_every_n_epoch == 0

    def _should_validate_step(self) -> bool:
        """Check if validation should run at current step."""
        # Check epoch-based validation frequency first
        if (self.current_epoch + 1) % self.check_val_every_n_epoch != 0:
            return False
        # Every N steps
        interval = int(self.val_check_interval)
        return self.global_step > 0 and (self.global_step % interval) == 0

    def _limit_batches(
        self, dataloader: DataLoader, limit: int | float | None
    ) -> Generator[Any, None, None]:
        """Limit the number of batches from a dataloader."""
        if limit is None:
            yield from dataloader
            return

        if isinstance(limit, float):
            # For float, we'd need to know total length - just yield all for now
            # (Lightning handles this by inspecting dataloader length)
            yield from dataloader
            return

        # Integer limit
        for i, batch in enumerate(dataloader):
            if i >= limit:
                break
            yield batch

    def _run_sanity_validation(self, val_dataloader: DataLoader) -> None:
        """Run quick validation to catch errors early."""
        print(f"Running sanity validation ({self.num_sanity_val_steps} batches)...")

        self._callbacks.fire("on_sanity_check_start", self, self._module)

        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= self.num_sanity_val_steps:
                break
            metrics = self._module.validation_step(batch, batch_idx)
            mx.eval(*[v for v in metrics.values() if isinstance(v, mx.array)])

        self._callbacks.fire("on_sanity_check_end", self, self._module)
        print("Sanity validation passed!")

    def _print_model_summary(self, module: TrainModule) -> None:
        """Print model summary with parameter counts."""
        from mlx.utils import tree_flatten

        params = tree_flatten(module.parameters())
        total_params = sum(p.size for _, p in params if hasattr(p, "size"))

        print("\n" + "=" * 60)
        print(f"Model: {module.__class__.__name__}")
        print(f"  Total parameters: {total_params:,}")
        print("=" * 60 + "\n")

    def _check_gradients(self, grads: dict[str, Any]) -> None:
        """Check for NaN or Inf in gradients."""
        from mlx.utils import tree_flatten

        for key, grad in tree_flatten(grads):
            if grad is not None and isinstance(grad, mx.array):
                has_nan = mx.any(mx.isnan(grad))
                has_inf = mx.any(mx.isinf(grad))
                mx.eval(has_nan, has_inf)

                if has_nan.item() or has_inf.item():
                    raise RuntimeError(
                        f"Gradient anomaly detected in {key}: "
                        f"NaN={has_nan.item()}, Inf={has_inf.item()}"
                    )

    def _force_eval(self, loss: mx.array, metrics: dict[str, mx.array]) -> None:
        """Force evaluation of lazy arrays."""
        if self.compile and self._compile_state is not None:
            # When compiled, eval the state list
            mx.eval(self._compile_state)
        else:
            # Without compile, eval individual arrays
            arrays_to_eval = [
                loss,
                self._module.parameters(),
                self._optimizer.state,
            ]
            arrays_to_eval.extend(metrics.values())
            mx.eval(*arrays_to_eval)

        if self.debug_lazy_eval:
            self._eval_count += 1

        # Periodically clear memory cache to prevent leaks
        # Do this every 100 steps to avoid overhead
        if self.global_step % 100 == 0:
            mx.clear_cache()

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
