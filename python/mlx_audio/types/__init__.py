"""Type definitions for mlx-audio."""

from mlx_audio.types.audio import AudioData, load_audio, save_audio
from mlx_audio.types.results import (
    SeparationResult,
    TranscriptionResult,
    GenerationResult,
    EmbeddingResult,
)
from mlx_audio.types.vad import VADResult, SpeechSegment

__all__ = [
    "AudioData",
    "load_audio",
    "save_audio",
    "SeparationResult",
    "TranscriptionResult",
    "GenerationResult",
    "EmbeddingResult",
    "VADResult",
    "SpeechSegment",
]
