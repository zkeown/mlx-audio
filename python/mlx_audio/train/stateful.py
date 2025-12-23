"""Stateful component mixin for checkpointing.

Provides a mixin class for components that need to save and
restore state during training checkpoints.
"""

from __future__ import annotations

from typing import Any, ClassVar


class StatefulMixin:
    """Mixin for components with serializable state.

    Classes using this mixin define which of their attributes should
    be included in checkpoints via the _state_fields class variable.

    Example:
        >>> class EarlyStoppingCallback(Callback, StatefulMixin):
        ...     _state_fields = ["best_score", "wait_count", "stopped_epoch"]
        ...
        ...     def __init__(self):
        ...         self.best_score = float("inf")
        ...         self.wait_count = 0
        ...         self.stopped_epoch = None
        ...
        >>> callback = EarlyStoppingCallback()
        >>> callback.best_score = 0.5
        >>> state = callback.state_dict()
        >>> state
        {'best_score': 0.5, 'wait_count': 0, 'stopped_epoch': None}
        >>> callback.load_state_dict({'best_score': 0.3})
        >>> callback.best_score
        0.3
    """

    # Subclasses define which fields to serialize
    _state_fields: ClassVar[list[str]] = []

    def state_dict(self) -> dict[str, Any]:
        """Return component state for checkpointing.

        Returns:
            Dictionary of state values for fields listed in _state_fields
        """
        return {
            k: getattr(self, k)
            for k in self._state_fields
            if hasattr(self, k)
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore component state from checkpoint.

        Only restores values for fields listed in _state_fields.
        Unknown keys in state are silently ignored.

        Args:
            state: Dictionary of state values from a previous state_dict()
        """
        for k, v in state.items():
            if k in self._state_fields and hasattr(self, k):
                setattr(self, k, v)


__all__ = ["StatefulMixin"]
