"""Callback system for mlx-train.

Provides a priority-ordered, composable callback system for extending training behavior.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlx.core as mx

    from mlx_audio.train.module import TrainModule
    from mlx_audio.train.trainer import Trainer


class CallbackPriority(IntEnum):
    """Priority levels for callback execution order.

    Lower values execute first. Built-in callbacks use specific priorities
    to ensure correct ordering (e.g., gradient clipping before optimizer step).
    """

    HIGHEST = 0  # System-critical (e.g., gradient clipping)
    HIGH = 25  # Monitoring/logging that needs early access
    NORMAL = 50  # User callbacks (default)
    LOW = 75  # Post-processing callbacks
    LOWEST = 100  # Cleanup callbacks (e.g., checkpoint saving)


@dataclass
class CallbackContext:
    """Context passed to callback hooks.

    Provides read-only access to training state. Callbacks should not
    modify this directly; use trainer methods for state changes.
    """

    epoch: int
    global_step: int
    batch_idx: int
    max_epochs: int | None
    max_steps: int | None

    # Current metrics (snapshot)
    metrics: dict[str, float] = field(default_factory=dict)

    # Flags
    is_training: bool = True
    is_validating: bool = False


class Callback(ABC):
    """Base class for all callbacks in mlx-train.

    Callbacks encapsulate non-essential training logic that can be
    composed and reused across projects. They should be:
    - Isolated: Not depend on other callbacks' behavior
    - Stateless or self-contained: Manage their own state
    - Non-destructive: Not modify model/optimizer directly (use trainer methods)

    Priority determines execution order within each hook. Lower values
    execute first. Use CallbackPriority enum for standard priorities.

    Example:
        >>> class MyCallback(Callback):
        ...     priority = CallbackPriority.NORMAL
        ...
        ...     def on_train_batch_end(self, trainer, module, batch, outputs, ctx):
        ...         if ctx.global_step % 100 == 0:
        ...             print(f"Step {ctx.global_step}: loss = {outputs['loss']:.4f}")
    """

    priority: int = CallbackPriority.NORMAL

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state in checkpoints.

        Override if multiple instances of the same callback type are used.
        Default uses class name.
        """
        return self.__class__.__name__

    # ==================== Setup/Teardown ====================

    def on_fit_start(self, trainer: Trainer, module: TrainModule) -> None:
        """Called at the very beginning of fit (training + validation)."""
        pass

    def on_fit_end(self, trainer: Trainer, module: TrainModule) -> None:
        """Called at the end of fit, even if training was interrupted."""
        pass

    # ==================== Training Lifecycle ====================

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        """Called at the start of each training epoch."""
        pass

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        """Called at the end of each training epoch, after validation."""
        pass

    def on_train_batch_start(
        self,
        trainer: Trainer,
        module: TrainModule,
        batch: Any,
        ctx: CallbackContext,
    ) -> None:
        """Called before processing each training batch."""
        pass

    def on_train_batch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        batch: Any,
        outputs: dict[str, Any],
        ctx: CallbackContext,
    ) -> None:
        """Called after processing each training batch.

        Args:
            outputs: Dictionary containing at minimum {"loss": float}.
                    May include additional metrics from training_step.
        """
        pass

    # ==================== Validation Lifecycle ====================

    def on_validation_start(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        """Called at the start of validation."""
        pass

    def on_validation_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
        metrics: dict[str, float],
    ) -> None:
        """Called at the end of validation with aggregated metrics."""
        pass

    # ==================== Optimization Hooks ====================

    def on_before_backward(
        self,
        trainer: Trainer,
        module: TrainModule,
        loss: mx.array,
    ) -> None:
        """Called after loss computation, before gradient calculation.

        Note: In MLX, gradients are computed via nn.value_and_grad, so this
        hook is called after loss is computed but conceptually "before backward".
        """
        pass

    def on_after_backward(
        self,
        trainer: Trainer,
        module: TrainModule,
        gradients: dict[str, Any],
    ) -> dict[str, Any]:
        """Called after gradients are computed, before optimizer step.

        This is the hook for gradient clipping/modification.

        Args:
            gradients: Dictionary of gradients from nn.value_and_grad

        Returns:
            Modified gradients dictionary. Return input gradients unchanged
            if no modification is needed.
        """
        return gradients

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        module: TrainModule,
    ) -> None:
        """Called immediately before optimizer.update()."""
        pass

    # ==================== Checkpoint Hooks ====================

    def state_dict(self) -> dict[str, Any]:
        """Return callback state to be saved in checkpoints.

        Override to persist callback-specific state (e.g., best metric
        value for early stopping).

        Returns:
            Dictionary of state to save. Must be JSON-serializable.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore callback state from checkpoint.

        Args:
            state: Dictionary returned by previous state_dict() call.
        """
        pass

    # ==================== Exception Handling ====================

    def on_exception(
        self,
        trainer: Trainer,
        module: TrainModule,
        exception: Exception,
    ) -> None:
        """Called when training encounters an exception.

        Useful for cleanup, saving emergency checkpoints, etc.
        Note: on_fit_end is still called after this.
        """
        pass


class CallbackRegistry:
    """Manages callback ordering and execution.

    Callbacks are sorted by priority (lower first) and executed in order.
    This class handles the complexity of hook dispatching.
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self._callbacks: list[Callback] = []
        if callbacks:
            for cb in callbacks:
                self.add(cb)

    def add(self, callback: Callback) -> None:
        """Add a callback and re-sort by priority."""
        self._callbacks.append(callback)
        self._callbacks.sort(key=lambda cb: cb.priority)

    def remove(self, callback: Callback) -> None:
        """Remove a specific callback instance."""
        self._callbacks.remove(callback)

    def remove_type(self, callback_type: type) -> None:
        """Remove all callbacks of a given type."""
        self._callbacks = [cb for cb in self._callbacks if not isinstance(cb, callback_type)]

    def __iter__(self):
        return iter(self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def fire(self, hook_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a hook on all callbacks in priority order.

        For hooks that return values (like on_after_backward), the return
        value is passed through each callback in sequence.

        Args:
            hook_name: Name of the hook method to call
            *args: Positional arguments to pass to the hook
            **kwargs: Keyword arguments to pass to the hook

        Returns:
            For on_after_backward: the final modified gradients
            For other hooks: None
        """
        result = kwargs.pop("_chain_value", None)

        for callback in self._callbacks:
            hook = getattr(callback, hook_name, None)
            if hook is not None:
                if hook_name == "on_after_backward" and result is not None:
                    # Chain gradient modifications
                    result = hook(*args, gradients=result, **kwargs)
                elif hook_name == "on_after_backward":
                    # First callback in chain
                    result = hook(*args, **kwargs)
                else:
                    hook(*args, **kwargs)

        return result

    def get_state_dicts(self) -> dict[str, dict[str, Any]]:
        """Collect state from all callbacks for checkpointing."""
        states = {}
        for cb in self._callbacks:
            state = cb.state_dict()
            if state:  # Only include non-empty states
                states[cb.state_key] = state
        return states

    def load_state_dicts(self, states: dict[str, dict[str, Any]]) -> None:
        """Restore callback states from checkpoint."""
        for cb in self._callbacks:
            if cb.state_key in states:
                cb.load_state_dict(states[cb.state_key])
