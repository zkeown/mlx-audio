"""Sample rate constants for audio processing.

These constants define the expected sample rates for various models
and streaming components in mlx-audio.
"""

from __future__ import annotations

# Model-specific sample rates
WHISPER_SAMPLE_RATE = 16000
"""Whisper expects 16kHz audio input."""

DEMUCS_SAMPLE_RATE = 44100
"""Demucs operates on 44.1kHz audio (CD quality)."""

MUSICGEN_SAMPLE_RATE = 32000
"""MusicGen generates audio at 32kHz."""

CLAP_SAMPLE_RATE = 48000
"""CLAP processes audio at 48kHz."""

# EnCodec supports multiple sample rates
ENCODEC_24KHZ_RATE = 24000
"""EnCodec 24kHz variant (speech-focused)."""

ENCODEC_32KHZ_RATE = 32000
"""EnCodec 32kHz variant (general purpose)."""

ENCODEC_48KHZ_RATE = 48000
"""EnCodec 48kHz variant (stereo music)."""

# Default for streaming components
DEFAULT_STREAMING_RATE = 44100
"""Default sample rate for streaming sources and sinks."""

__all__ = [
    "WHISPER_SAMPLE_RATE",
    "DEMUCS_SAMPLE_RATE",
    "MUSICGEN_SAMPLE_RATE",
    "CLAP_SAMPLE_RATE",
    "ENCODEC_24KHZ_RATE",
    "ENCODEC_32KHZ_RATE",
    "ENCODEC_48KHZ_RATE",
    "DEFAULT_STREAMING_RATE",
]
