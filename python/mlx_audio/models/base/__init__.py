"""Base classes for mlx-audio models.

This module provides shared infrastructure for model configuration,
pretrained weight loading, protocol definitions, and weight conversion.
"""

from __future__ import annotations

from mlx_audio.models.base.config import ModelConfig
from mlx_audio.models.base.pretrained import PretrainedMixin
from mlx_audio.models.base.protocol import (
    AudioModel,
    EncoderModel,
    GenerativeModel,
    SeparationModel,
    StreamingModel,
    TranscriptionModel,
)
from mlx_audio.models.base.weight_converter import (
    IdentityConverter,
    WeightConverter,
)

__all__ = [
    # Config
    "ModelConfig",
    # Mixin
    "PretrainedMixin",
    # Protocols
    "AudioModel",
    "EncoderModel",
    "GenerativeModel",
    "StreamingModel",
    "SeparationModel",
    "TranscriptionModel",
    # Weight Conversion
    "WeightConverter",
    "IdentityConverter",
]
