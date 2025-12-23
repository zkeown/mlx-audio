"""Parler-TTS text-to-speech model for MLX."""

from mlx_audio.models.tts.config import ParlerTTSConfig
from mlx_audio.models.tts.model import ParlerTTS

__all__ = [
    "ParlerTTS",
    "ParlerTTSConfig",
]
