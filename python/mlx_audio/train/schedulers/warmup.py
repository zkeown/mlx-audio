"""Warmup and decay learning rate schedulers."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.optimizers as optim

from mlx_audio.train.schedulers.base import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup followed by cosine decay.

    This is the most common scheduler for transformer training.
    Uses MLX's native join_schedules under the hood.

    Args:
        peak_lr: Maximum learning rate after warmup
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate at end of decay (default: 0)

    Example:
        >>> scheduler = WarmupCosineScheduler(
        ...     peak_lr=1e-4,
        ...     warmup_steps=1000,
        ...     total_steps=100000,
        ...     min_lr=1e-6
        ... )
        >>> optimizer = optim.AdamW(learning_rate=scheduler)
    """

    def __init__(
        self,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        # Build the composite schedule using MLX primitives
        if warmup_steps > 0:
            warmup = optim.linear_schedule(
                init=0.0,
                end=peak_lr,
                steps=warmup_steps,
            )
            decay = optim.cosine_decay(
                init=peak_lr,
                decay_steps=max(1, total_steps - warmup_steps),
            )
            self._schedule = optim.join_schedules(
                schedules=[warmup, decay],
                boundaries=[warmup_steps],
            )
        else:
            self._schedule = optim.cosine_decay(
                init=peak_lr,
                decay_steps=total_steps,
            )

    def __call__(self, step: int) -> mx.array:
        self._step = step
        lr = self._schedule(step)

        # Apply min_lr floor
        lr = mx.maximum(lr, mx.array(self.min_lr))

        self._last_lr = float(lr.item())
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "last_lr": self._last_lr,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step = state.get("step", 0)
        self._last_lr = state.get("last_lr", 0.0)


class WarmupLinearScheduler(LRScheduler):
    """Linear warmup followed by linear decay to zero.

    Common for BERT-style training.

    Args:
        peak_lr: Maximum learning rate after warmup
        warmup_steps: Number of warmup steps
        total_steps: Total training steps

    Example:
        >>> scheduler = WarmupLinearScheduler(
        ...     peak_lr=5e-5,
        ...     warmup_steps=1000,
        ...     total_steps=50000
        ... )
    """

    def __init__(
        self,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        if warmup_steps > 0:
            warmup = optim.linear_schedule(
                init=0.0,
                end=peak_lr,
                steps=warmup_steps,
            )
            decay = optim.linear_schedule(
                init=peak_lr,
                end=0.0,
                steps=max(1, total_steps - warmup_steps),
            )
            self._schedule = optim.join_schedules(
                schedules=[warmup, decay],
                boundaries=[warmup_steps],
            )
        else:
            self._schedule = optim.linear_schedule(
                init=peak_lr,
                end=0.0,
                steps=total_steps,
            )

    def __call__(self, step: int) -> mx.array:
        self._step = step
        lr = self._schedule(step)
        self._last_lr = float(lr.item())
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "last_lr": self._last_lr,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


class StepLRScheduler(LRScheduler):
    """Step decay scheduler.

    Reduces learning rate by gamma every step_size steps.

    Args:
        initial_lr: Initial learning rate
        step_size: Number of steps between decays
        gamma: Multiplicative factor for decay (default: 0.1)

    Example:
        >>> scheduler = StepLRScheduler(initial_lr=0.1, step_size=30, gamma=0.1)
        >>> # LR will be 0.1 for steps 0-29, 0.01 for 30-59, etc.
    """

    def __init__(
        self,
        initial_lr: float,
        step_size: int,
        gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        self._last_lr = initial_lr

    def __call__(self, step: int) -> mx.array:
        self._step = step
        num_decays = step // self.step_size
        lr = self.initial_lr * (self.gamma**num_decays)
        self._last_lr = lr
        return mx.array(lr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "last_lr": self._last_lr,
            "initial_lr": self.initial_lr,
            "step_size": self.step_size,
            "gamma": self.gamma,
        }


class ExponentialLRScheduler(LRScheduler):
    """Exponential decay scheduler.

    Wraps MLX's native exponential_decay.

    Args:
        initial_lr: Initial learning rate
        decay_rate: Multiplicative decay factor per step

    Example:
        >>> scheduler = ExponentialLRScheduler(initial_lr=0.1, decay_rate=0.99)
        >>> # LR = 0.1 * (0.99 ^ step)
    """

    def __init__(
        self,
        initial_lr: float,
        decay_rate: float,
    ) -> None:
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self._last_lr = initial_lr

        self._schedule = optim.exponential_decay(
            init=initial_lr,
            decay_rate=decay_rate,
        )

    def __call__(self, step: int) -> mx.array:
        self._step = step
        lr = self._schedule(step)
        # MLX optim.exponential_decay returns float, not mx.array
        self._last_lr = float(lr) if isinstance(lr, int | float) else float(lr.item())
        return mx.array(lr) if isinstance(lr, int | float) else lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "last_lr": self._last_lr,
            "initial_lr": self.initial_lr,
            "decay_rate": self.decay_rate,
        }
