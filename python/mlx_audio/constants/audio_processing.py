"""Audio processing constants for STFT, buffers, and streaming.

These constants define parameters for spectral analysis and
audio buffering across mlx-audio.
"""

from __future__ import annotations

# STFT parameters by model
WHISPER_N_FFT = 400
"""Whisper FFT window size (25ms at 16kHz)."""

WHISPER_HOP_LENGTH = 160
"""Whisper STFT hop length (10ms at 16kHz)."""

DEMUCS_N_FFT = 4096
"""Demucs FFT window size for frequency domain processing."""

DEMUCS_HOP_LENGTH = 1024
"""Demucs STFT hop length."""

CLAP_N_FFT = 1024
"""CLAP FFT window size."""

CLAP_HOP_LENGTH = 480
"""CLAP STFT hop length."""

CLAP_WINDOW_LENGTH = 1024
"""CLAP spectrogram window length."""

# Default STFT parameters (used by primitives)
DEFAULT_N_FFT = 2048
"""Default FFT window size for general audio processing."""

DEFAULT_HOP_LENGTH = 512
"""Default STFT hop length for general audio processing."""

# Buffer and streaming parameters
DEFAULT_CHUNK_SIZE = 4096
"""Default chunk size for streaming processors."""

DEFAULT_BLOCKSIZE = 1024
"""Default block size for audio I/O devices."""

DEFAULT_CHANNELS = 2
"""Default number of audio channels (stereo)."""

# Whisper-specific audio parameters
WHISPER_CHUNK_LENGTH = 30
"""Whisper processes audio in 30-second chunks."""

WHISPER_N_AUDIO_CTX = 1500
"""Whisper encoder output frames (30s * 16kHz / 160 / 2)."""

WHISPER_N_TEXT_CTX = 448
"""Maximum tokens in Whisper decoder context."""

__all__ = [
    "WHISPER_N_FFT",
    "WHISPER_HOP_LENGTH",
    "DEMUCS_N_FFT",
    "DEMUCS_HOP_LENGTH",
    "CLAP_N_FFT",
    "CLAP_HOP_LENGTH",
    "CLAP_WINDOW_LENGTH",
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_BLOCKSIZE",
    "DEFAULT_CHANNELS",
    "WHISPER_CHUNK_LENGTH",
    "WHISPER_N_AUDIO_CTX",
    "WHISPER_N_TEXT_CTX",
]
