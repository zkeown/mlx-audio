"""Drumux - Drum transcription model for MLX.

A 14-class drum transcription model that converts audio to MIDI/MusicXML
with cymbal granularity.
"""

from .model import (
    DrumTranscriber,
    DrumTranscriberConfig,
    create_model,
    NUM_CLASSES,
)
from .train_module import DrumuxTrainModule
from .data import (
    EGMDDataset,
    SpectrogramConfig,
    create_dataloader,
    compute_class_weights,
)

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
