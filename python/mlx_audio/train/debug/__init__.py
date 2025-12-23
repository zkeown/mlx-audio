"""Debugging utilities for mlx-train.

Helps catch common MLX lazy evaluation bugs.
"""

from mlx_audio.train.debug.lazy_eval import (
    LazyEvalDebugger,
    track_eval,
    warn_unevaluated,
)

__all__ = [
    "LazyEvalDebugger",
    "track_eval",
    "warn_unevaluated",
]
