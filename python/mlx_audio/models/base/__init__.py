"""Base classes for mlx-audio models.

This module provides shared infrastructure for model configuration
and pretrained weight loading.
"""

from __future__ import annotations

from mlx_audio.models.base.config import ModelConfig
from mlx_audio.models.base.pretrained import PretrainedMixin

__all__ = [
    "ModelConfig",
    "PretrainedMixin",
]
