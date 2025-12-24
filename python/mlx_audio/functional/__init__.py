"""High-level functional API for mlx-audio.

This module provides one-liner functions for common audio tasks.
All functions can be imported directly from mlx_audio.functional.

Example:
    >>> from mlx_audio.functional import separate, transcribe, embed
    >>> stems = separate("song.mp3")
    >>> result = transcribe("speech.wav")
    >>> embeddings = embed(audio="sound.wav", text=["dog", "cat"])
"""

from mlx_audio.functional.classify import classify
from mlx_audio.functional.detect_speech import detect_speech
from mlx_audio.functional.diarize import diarize
from mlx_audio.functional.embed import embed, CLAPEmbeddingResult
from mlx_audio.functional.enhance import enhance
from mlx_audio.functional.generate import generate
from mlx_audio.functional.separate import separate
from mlx_audio.functional.speak import speak
from mlx_audio.functional.tag import tag
from mlx_audio.functional.transcribe import transcribe

__all__ = [
    # Functions
    "classify",
    "detect_speech",
    "diarize",
    "embed",
    "enhance",
    "generate",
    "separate",
    "speak",
    "tag",
    "transcribe",
    # Result type (temporary location until migrated)
    "CLAPEmbeddingResult",
]
