"""Audio classification models."""

from mlx_audio.models.classifier.config import (
    ClassifierConfig,
    FreezeMode,
    MLPHeadConfig,
    TaskMode,
)
from mlx_audio.models.classifier.model import CLAPClassifier
from mlx_audio.models.classifier.layers import MLPHead

__all__ = [
    "CLAPClassifier",
    "ClassifierConfig",
    "FreezeMode",
    "MLPHead",
    "MLPHeadConfig",
    "TaskMode",
]
