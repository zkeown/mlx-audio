"""Drumux - Drum transcription model for MLX.

A 14-class drum transcription model that converts audio to MIDI/MusicXML
with cymbal granularity.
"""

from .data import (
    EGMDDataset,
    SpectrogramConfig,
    compute_class_weights,
    create_dataloader,
)
from .model import (
    NUM_CLASSES,
    DrumTranscriber,
    DrumTranscriberConfig,
    create_model,
)
from .train_module import DrumuxTrainModule

__all__ = [
    "DrumTranscriber",
    "DrumTranscriberConfig",
    "create_model",
    "DrumuxTrainModule",
    "EGMDDataset",
    "SpectrogramConfig",
    "create_dataloader",
    "compute_class_weights",
    "NUM_CLASSES",
]
