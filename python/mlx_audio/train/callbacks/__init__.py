"""Callbacks for mlx-train."""

from mlx_audio.train.callbacks.base import (
    Callback,
    CallbackContext,
    CallbackPriority,
    CallbackRegistry,
)
from mlx_audio.train.callbacks.checkpoint import ModelCheckpoint
from mlx_audio.train.callbacks.early_stopping import EarlyStopping
from mlx_audio.train.callbacks.gradient_clip import GradientClipper, clip_grad_norm, clip_grad_value
from mlx_audio.train.callbacks.lr_monitor import LearningRateMonitor
from mlx_audio.train.callbacks.progress import ProgressBar, RichProgressBar

__all__ = [
    # Base
    "Callback",
    "CallbackContext",
    "CallbackPriority",
    "CallbackRegistry",
    # Built-in callbacks
    "ModelCheckpoint",
    "EarlyStopping",
    "GradientClipper",
    "LearningRateMonitor",
    "ProgressBar",
    "RichProgressBar",
    # Utilities
    "clip_grad_norm",
    "clip_grad_value",
]
