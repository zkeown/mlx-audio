"""High-level functional API for mlx-audio.

This module provides one-liner functions for common audio tasks.
"""

from mlx_audio.functional.separate import separate
from mlx_audio.functional.embed import embed, CLAPEmbeddingResult
from mlx_audio.functional.generate import generate
from mlx_audio.functional.detect_speech import detect_speech

__all__ = ["separate", "embed", "generate", "detect_speech", "CLAPEmbeddingResult"]
