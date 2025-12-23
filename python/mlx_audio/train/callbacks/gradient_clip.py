"""Gradient clipping callback for mlx-train."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_audio.train.callbacks.base import Callback, CallbackPriority

if TYPE_CHECKING:
    from mlx_audio.train.module import TrainModule
    from mlx_audio.train.trainer import Trainer


def clip_grad_norm(
    gradients: dict[str, Any],
    max_norm: float,
    norm_type: float = 2.0,
) -> tuple[dict[str, Any], float]:
    """Clip gradients by global norm.

    Implements gradient clipping similar to torch.nn.utils.clip_grad_norm_.

    Args:
        gradients: Nested dictionary of gradient arrays
        max_norm: Maximum allowed norm
        norm_type: Type of norm (default: L2)

    Returns:
        Tuple of (clipped_gradients, total_norm)
    """
    flat_grads = tree_flatten(gradients)

    # Compute total norm
    if norm_type == float("inf"):
        norms = [mx.max(mx.abs(g)) for _, g in flat_grads if g is not None]
        total_norm = mx.max(mx.stack(norms)) if norms else mx.array(0.0)
    else:
        norms = [mx.sum(mx.abs(g) ** norm_type) for _, g in flat_grads if g is not None]
        total_norm = mx.sum(mx.stack(norms)) ** (1.0 / norm_type) if norms else mx.array(0.0)

    # Compute clip coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = mx.minimum(clip_coef, mx.array(1.0))

    # Apply clipping
    clipped = [(k, g * clip_coef if g is not None else None) for k, g in flat_grads]

    return tree_unflatten(clipped), float(total_norm.item())


def clip_grad_value(
    gradients: dict[str, Any],
    clip_value: float,
) -> dict[str, Any]:
    """Clip gradients element-wise by value.

    Args:
        gradients: Nested dictionary of gradient arrays
        clip_value: Maximum absolute value for any gradient element

    Returns:
        Clipped gradients dictionary
    """
    flat_grads = tree_flatten(gradients)
    clipped = [
        (k, mx.clip(g, -clip_value, clip_value) if g is not None else None) for k, g in flat_grads
    ]
    return tree_unflatten(clipped)


class GradientClipper(Callback):
    """Clip gradients during training.

    Supports two modes:
    - norm: Clip by global L2 norm (recommended for most cases)
    - value: Clip element-wise by absolute value

    This callback runs at HIGHEST priority to ensure gradients are
    clipped before any other callbacks process them.

    Example:
        >>> clipper = GradientClipper(max_norm=1.0)
        >>> trainer = Trainer(callbacks=[clipper])

        >>> # Or use value clipping
        >>> clipper = GradientClipper(clip_value=0.5)
    """

    priority = CallbackPriority.HIGHEST  # Must run before other callbacks

    def __init__(
        self,
        max_norm: float | None = None,
        clip_value: float | None = None,
        norm_type: float = 2.0,
        log_grad_norm: bool = True,
    ) -> None:
        """Initialize the gradient clipper.

        Args:
            max_norm: Maximum gradient norm (L2 by default). Mutually exclusive
                with clip_value.
            clip_value: Maximum absolute value for any gradient element. Mutually
                exclusive with max_norm.
            norm_type: Type of norm for norm clipping (default: 2.0 for L2).
            log_grad_norm: Whether to log the gradient norm before clipping.

        Raises:
            ValueError: If neither or both max_norm and clip_value are specified.
        """
        if max_norm is None and clip_value is None:
            raise ValueError("Must specify either max_norm or clip_value")
        if max_norm is not None and clip_value is not None:
            raise ValueError("Cannot specify both max_norm and clip_value")

        self.max_norm = max_norm
        self.clip_value = clip_value
        self.norm_type = norm_type
        self.log_grad_norm = log_grad_norm

        # Track gradient norms for logging
        self._last_grad_norm: float | None = None

    def on_after_backward(
        self,
        trainer: Trainer,
        module: TrainModule,
        gradients: dict[str, Any],
    ) -> dict[str, Any]:
        """Clip gradients after backward pass."""
        if self.max_norm is not None:
            clipped_grads, total_norm = clip_grad_norm(gradients, self.max_norm, self.norm_type)
            self._last_grad_norm = total_norm

            if self.log_grad_norm:
                trainer.log("grad_norm", total_norm)
                trainer.log("grad_norm_clipped", total_norm > self.max_norm)

            return clipped_grads
        else:
            return clip_grad_value(gradients, self.clip_value)

    @property
    def last_grad_norm(self) -> float | None:
        """Returns the last computed gradient norm (only for norm clipping)."""
        return self._last_grad_norm

    def state_dict(self) -> dict[str, Any]:
        """Return state for checkpointing."""
        return {"last_grad_norm": self._last_grad_norm}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self._last_grad_norm = state.get("last_grad_norm")
