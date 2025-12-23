"""MusicGen text-to-music generation model."""

from mlx_audio.models.musicgen.config import MusicGenConfig
from mlx_audio.models.musicgen.model import MusicGen

__all__ = ["MusicGen", "MusicGenConfig"]
