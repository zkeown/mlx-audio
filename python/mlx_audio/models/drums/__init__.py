"""Drum transcription model for MLX.

This module provides a drum transcription model that detects drum onsets
and predicts velocities from mel-spectrogram input. The model architecture
consists of:
- CNN encoder for spectral feature extraction
- Transformer for temporal modeling
- Dual heads for onset detection and velocity prediction

Ported from PyTorch implementation at /Users/zakkeown/ml/drums.
"""

from mlx_audio.models.drums.config import DrumTranscriberConfig
from mlx_audio.models.drums.model import DrumTranscriber, create_model

__all__ = [
    "DrumTranscriber",
    "DrumTranscriberConfig",
    "create_model",
]
