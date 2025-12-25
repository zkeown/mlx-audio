"""Type definitions for mlx-audio."""

from mlx_audio.types.audio import AudioData, load_audio, save_audio
from mlx_audio.types.results import (
    EmbeddingResult,
    GenerationResult,
    SeparationResult,
    TranscriptionResult,
)
from mlx_audio.types.vad import SpeechSegment, VADResult

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
