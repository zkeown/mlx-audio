"""Path constants for caching and file operations.

These constants define default paths and cache parameters
used throughout mlx-audio.
"""

from __future__ import annotations

from pathlib import Path

# Cache directories
CACHE_DIR = Path.home() / ".cache" / "mlx_audio"
"""Root cache directory for mlx-audio."""

MODELS_CACHE_DIR = CACHE_DIR / "models"
"""Directory for cached model weights."""

# Model-specific cache directories
WHISPER_CACHE_DIR = MODELS_CACHE_DIR / "whisper"
"""Cache directory for Whisper models."""

DEMUCS_CACHE_DIR = MODELS_CACHE_DIR / "demucs"
"""Cache directory for Demucs models."""

ENCODEC_CACHE_DIR = MODELS_CACHE_DIR / "encodec"
"""Cache directory for EnCodec models."""

MUSICGEN_CACHE_DIR = MODELS_CACHE_DIR / "musicgen"
"""Cache directory for MusicGen models."""

CLAP_CACHE_DIR = MODELS_CACHE_DIR / "clap"
"""Cache directory for CLAP models."""

# Cache parameters
WINDOW_CACHE_MAXSIZE = 32
"""Maximum number of cached STFT windows."""

FILE_DOWNLOAD_CHUNK_SIZE = 8192
"""Chunk size in bytes for file downloads."""

__all__ = [
    "CACHE_DIR",
    "MODELS_CACHE_DIR",
    "WHISPER_CACHE_DIR",
    "DEMUCS_CACHE_DIR",
    "ENCODEC_CACHE_DIR",
    "MUSICGEN_CACHE_DIR",
    "CLAP_CACHE_DIR",
    "WINDOW_CACHE_MAXSIZE",
    "FILE_DOWNLOAD_CHUNK_SIZE",
]
