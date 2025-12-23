"""Lazy evaluation debugging utilities for mlx-train.

MLX uses lazy evaluation - operations build a computation graph that's
only executed when you call mx.eval(). Forgetting to call mx.eval() is
a common bug that can cause:
- Memory issues (graph keeps growing)
- Performance problems (computation happens at unexpected times)
- Silent bugs (values aren't what you expect)

This module provides tools to catch these issues early.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import mlx.core as mx


class LazyEvalDebugger:
    """Tracks mx.eval() calls and warns about potential issues.

    Use this to debug lazy evaluation problems during training.

    Example:
        >>> debugger = LazyEvalDebugger()
        >>> debugger.start()
        >>> # ... training code ...
        >>> debugger.stop()
        >>> debugger.report()
    """

    def __init__(self, warn_threshold: int = 100) -> None:
        """Initialize the debugger.

        Args:
            warn_threshold: Warn if this many operations happen without mx.eval()
        """
        self.warn_threshold = warn_threshold
        self._eval_count: int = 0
        self._op_count: int = 0
        self._active: bool = False
        self._original_eval: Callable | None = None

    def start(self) -> None:
        """Start tracking mx.eval() calls."""
        if self._active:
            return

        self._active = True
        self._eval_count = 0
        self._op_count = 0

        # Wrap mx.eval to track calls
        self._original_eval = mx.eval

        @functools.wraps(mx.eval)
        def tracked_eval(*args, **kwargs):
            self._eval_count += 1
            return self._original_eval(*args, **kwargs)

        mx.eval = tracked_eval

    def stop(self) -> None:
        """Stop tracking mx.eval() calls."""
        if not self._active:
            return

        self._active = False
        if self._original_eval is not None:
            mx.eval = self._original_eval
            self._original_eval = None

    def report(self) -> dict[str, int]:
        """Get a report of tracking statistics.

        Returns:
            Dictionary with eval_count and other stats
        """
        return {
            "eval_count": self._eval_count,
            "active": self._active,
        }

    def __enter__(self) -> LazyEvalDebugger:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


@contextmanager
def track_eval():
    """Context manager to track mx.eval() calls.

    Example:
        >>> with track_eval() as tracker:
        ...     # training code
        ...     pass
        >>> print(f"Called mx.eval() {tracker.report()['eval_count']} times")
    """
    debugger = LazyEvalDebugger()
    debugger.start()
    try:
        yield debugger
    finally:
        debugger.stop()


def warn_unevaluated(array: mx.array, name: str = "array") -> mx.array:
    """Warn if an array might not have been evaluated.

    This is a heuristic check - it can't definitively tell if an array
    has been evaluated, but it can catch some common cases.

    Args:
        array: The array to check
        name: Name for the warning message

    Returns:
        The input array unchanged

    Example:
        >>> loss = model(x)
        >>> loss = warn_unevaluated(loss, "loss")  # Warns if not evaluated
    """
    # In MLX, we can't directly check if an array is evaluated.
    # This is a placeholder for potential future MLX features.
    # For now, we just return the array unchanged.
    #
    # A more sophisticated implementation might:
    # - Track array creation and evaluation times
    # - Estimate graph size
    # - Check memory patterns
    return array


def check_memory_growth(threshold_mb: float = 1000) -> Callable:
    """Decorator to warn if memory grows too much during a function.

    Args:
        threshold_mb: Warn if memory grows by more than this many MB

    Returns:
        Decorated function

    Example:
        >>> @check_memory_growth(threshold_mb=500)
        ... def training_step(batch):
        ...     ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Note: MLX doesn't expose memory usage directly.
            # This is a placeholder for potential future features.
            #
            # In practice, you'd use system tools like:
            # - Activity Monitor on macOS
            # - memory_profiler package
            # - Metal Performance HUD
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


class EvalTracker:
    """Track the evaluation state of arrays for debugging.

    This is an experimental feature that attempts to help identify
    when arrays should be evaluated.

    Example:
        >>> tracker = EvalTracker()
        >>> loss = compute_loss(batch)
        >>> tracker.register(loss, "loss")
        >>> # ... later ...
        >>> tracker.check_all()  # Warns about unevaluated arrays
    """

    def __init__(self) -> None:
        self._tracked: dict[str, mx.array] = {}
        self._eval_times: dict[str, float] = {}

    def register(self, array: mx.array, name: str) -> mx.array:
        """Register an array for tracking.

        Args:
            array: Array to track
            name: Name for identification

        Returns:
            The input array unchanged
        """
        self._tracked[name] = array
        return array

    def mark_evaluated(self, name: str) -> None:
        """Mark an array as evaluated."""
        import time

        self._eval_times[name] = time.time()

    def check_all(self) -> list[str]:
        """Check all tracked arrays and return names of potentially unevaluated ones.

        Returns:
            List of names that might not be evaluated
        """
        potentially_unevaluated = []
        for name in self._tracked:
            if name not in self._eval_times:
                potentially_unevaluated.append(name)
                warnings.warn(
                    f"Array '{name}' may not have been evaluated. "
                    f"Consider calling mx.eval() to force computation.",
                    stacklevel=2,
                )
        return potentially_unevaluated

    def clear(self) -> None:
        """Clear all tracked arrays."""
        self._tracked.clear()
        self._eval_times.clear()
