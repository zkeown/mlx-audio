"""Shared utilities for functional API modules.

Provides common functions for audio input normalization and
model loading that are used across transcribe, separate, generate, and embed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mlx_audio.utils.audio_io import load_audio_file, resample_audio

if TYPE_CHECKING:
    from mlx_audio.types.audio import AudioData


def normalize_audio_input(
    audio: str | Path | np.ndarray | mx.array | AudioData,
    target_sample_rate: int | None = None,
    mono: bool = False,
) -> tuple[mx.array, int]:
    """Normalize various audio input formats to MLX array.

    Handles multiple input types:
    - File paths (str or Path) - loads the file
    - numpy arrays - converts directly
    - MLX arrays - uses directly
    - AudioData objects - extracts the array

    Args:
        audio: Audio input in various formats
        target_sample_rate: Optional target sample rate for resampling
        mono: Whether to convert to mono

    Returns:
        Tuple of (MLX array [channels, samples], sample rate)

    Example:
        >>> audio, sr = normalize_audio_input("audio.wav", target_sample_rate=16000)
        >>> audio, sr = normalize_audio_input(np.random.randn(2, 48000), 48000)
    """
    # Handle file paths
    if isinstance(audio, (str, Path)):
        np_audio, sr = load_audio_file(
            audio,
            target_sr=target_sample_rate,
            mono=mono,
        )
        return mx.array(np_audio), sr

    # Handle AudioData objects
    if hasattr(audio, "array") and hasattr(audio, "sample_rate"):
        # This is an AudioData object
        np_audio = np.array(audio.array)
        sr = audio.sample_rate

        if mono and np_audio.shape[0] > 1:
            np_audio = np_audio.mean(axis=0, keepdims=True)

        if target_sample_rate and sr != target_sample_rate:
            np_audio = resample_audio(np_audio, sr, target_sample_rate)
            sr = target_sample_rate

        return mx.array(np_audio), sr

    # Handle MLX arrays
    if isinstance(audio, mx.array):
        np_audio = np.array(audio)
    elif isinstance(audio, np.ndarray):
        np_audio = audio
    else:
        raise TypeError(
            f"Unsupported audio input type: {type(audio)}. "
            "Expected str, Path, np.ndarray, mx.array, or AudioData."
        )

    # Ensure [channels, samples] format
    if np_audio.ndim == 1:
        np_audio = np_audio[np.newaxis, :]

    if mono and np_audio.shape[0] > 1:
        np_audio = np_audio.mean(axis=0, keepdims=True)

    # We don't know the sample rate for raw arrays, use target or 0
    sr = target_sample_rate or 0

    return mx.array(np_audio), sr


def ensure_mono(audio: mx.array) -> mx.array:
    """Convert audio to mono if needed.

    Args:
        audio: Audio array with shape [channels, samples]

    Returns:
        Mono audio with shape [1, samples]
    """
    if audio.shape[0] > 1:
        return audio.mean(axis=0, keepdims=True)
    return audio


def ensure_stereo(audio: mx.array) -> mx.array:
    """Convert audio to stereo if needed.

    Args:
        audio: Audio array with shape [channels, samples]

    Returns:
        Stereo audio with shape [2, samples]
    """
    if audio.shape[0] == 1:
        return mx.concatenate([audio, audio], axis=0)
    return audio


__all__ = [
    "normalize_audio_input",
    "ensure_mono",
    "ensure_stereo",
]
