"""Audio classification models."""

from mlx_audio.models.classifier.config import (
    ClassifierConfig,
    FreezeMode,
    MLPHeadConfig,
    TaskMode,
)
from mlx_audio.models.classifier.layers import MLPHead
from mlx_audio.models.classifier.model import CLAPClassifier

__all__ = [
    "CLAPClassifier",
    "ClassifierConfig",
    "FreezeMode",
    "MLPHead",
    "MLPHeadConfig",
    "TaskMode",
]
