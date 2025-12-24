"""Centralized audio loading utilities for functional API.

This module provides unified audio loading that handles:
- File paths (str, Path)
- NumPy arrays
- MLX arrays

All functional modules should use these utilities instead of
implementing their own _load_audio() functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    import numpy as np

from mlx_audio.exceptions import AudioLoadError


def load_audio_input(
    audio: "str | Path | np.ndarray | mx.array",
    sample_rate: int | None = None,
    default_sample_rate: int = 44100,
    target_sample_rate: int | None = None,
    mono: bool = False,
) -> tuple["mx.array", int]:
    """Load audio from various input types.

    This is the unified audio loading function for all functional APIs.
    It handles file paths, numpy arrays, and MLX arrays uniformly.

    Args:
        audio: Audio input - can be:
            - str or Path: File path to load
            - np.ndarray: NumPy array [C, T] or [T]
            - mx.array: MLX array [C, T] or [T]
        sample_rate: Known sample rate (required for array inputs unless
            default_sample_rate is acceptable)
        default_sample_rate: Default sample rate when not inferable
            (e.g., 44100 for HTDemucs, 48000 for CLAP, 16000 for Whisper)
        target_sample_rate: If provided, resample audio to this rate.
            This eliminates the need for callers to resample after loading.
        mono: Convert to mono by averaging channels

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is an mx.array

    Raises:
        AudioLoadError: If audio cannot be loaded or is invalid

    Example:
        >>> # From file
        >>> audio, sr = load_audio_input("song.mp3")

        >>> # From numpy array with known sample rate
        >>> audio, sr = load_audio_input(np_array, sample_rate=44100)

        >>> # With model-specific default
        >>> audio, sr = load_audio_input(mx_array, default_sample_rate=48000)

        >>> # Auto-resample to model's expected rate
        >>> audio, sr = load_audio_input("song.mp3", target_sample_rate=16000)
    """
    import mlx.core as mx
    import numpy as np

    if isinstance(audio, (str, Path)):
        audio_array, sr = _load_from_file(audio, sample_rate, mono)
    elif isinstance(audio, np.ndarray):
        audio_array, sr = _load_from_numpy(
            audio, sample_rate, default_sample_rate, mono
        )
    elif isinstance(audio, mx.array):
        audio_array, sr = _load_from_mlx(
            audio, sample_rate, default_sample_rate, mono
        )
    else:
        raise AudioLoadError(
            f"Unsupported audio type: {type(audio).__name__}. "
            "Expected str, Path, np.ndarray, or mx.array."
        )

    # Resample if target_sample_rate specified and different from current
    if target_sample_rate is not None and sr != target_sample_rate:
        from mlx_audio.primitives import resample
        audio_array = resample(audio_array, sr, target_sample_rate)
        sr = target_sample_rate

    return audio_array, sr


def _load_from_file(
    path: "str | Path",
    sample_rate: int | None,
    mono: bool,
) -> tuple["mx.array", int]:
    """Load audio from file path."""
    from mlx_audio.types.audio import load_audio

    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"Audio file not found: {path}")

    try:
        audio_array, sr = load_audio(path, sample_rate=sample_rate, mono=mono)
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio from {path}: {e}") from e

    return audio_array, sr


def _load_from_numpy(
    audio: "np.ndarray",
    sample_rate: int | None,
    default_sample_rate: int,
    mono: bool,
) -> tuple["mx.array", int]:
    """Load audio from numpy array."""
    import mlx.core as mx
    import numpy as np

    # Validate array
    if audio.size == 0:
        raise AudioLoadError("Audio array is empty")

    if not np.isfinite(audio).all():
        raise AudioLoadError("Audio array contains NaN or Inf values")

    # Convert to float32 if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Convert to MLX
    audio_array = mx.array(audio)

    # Handle mono conversion
    if mono:
        audio_array = _to_mono(audio_array)

    sr = sample_rate if sample_rate is not None else default_sample_rate
    return audio_array, sr


def _load_from_mlx(
    audio: "mx.array",
    sample_rate: int | None,
    default_sample_rate: int,
    mono: bool,
) -> tuple["mx.array", int]:
    """Load audio from MLX array."""
    import mlx.core as mx

    # Validate array
    if audio.size == 0:
        raise AudioLoadError("Audio array is empty")

    # Handle mono conversion
    if mono:
        audio = _to_mono(audio)

    sr = sample_rate if sample_rate is not None else default_sample_rate
    return audio, sr


def _to_mono(audio: "mx.array") -> "mx.array":
    """Convert audio to mono by averaging channels."""
    import mlx.core as mx

    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        if audio.shape[0] <= 2:
            # [C, T] format - average over channels
            return mx.mean(audio, axis=0)
        else:
            # Might be [T, C] format or batched - assume first dim is channels if small
            return mx.mean(audio, axis=0)
    else:
        raise AudioLoadError(
            f"Unexpected audio shape: {audio.shape}. "
            "Expected [T], [C, T], or [B, C, T]."
        )


def ensure_batch_dim(audio: "mx.array") -> "mx.array":
    """Ensure audio has batch dimension [B, ...].

    Args:
        audio: Audio array of shape [T], [C, T], or [B, C, T]

    Returns:
        Audio array with batch dimension [B, C, T] or [B, T]
    """
    if audio.ndim == 1:
        # [T] -> [1, T]
        return audio[None, :]
    elif audio.ndim == 2:
        # Could be [C, T] or [B, T]
        # If first dim is 1-2, assume it's channels
        if audio.shape[0] <= 2:
            # [C, T] -> [1, C, T]
            return audio[None, :, :]
        else:
            # [B, T] - already has batch
            return audio
    else:
        # [B, C, T] - already has batch
        return audio


def ensure_mono_batch(audio: "mx.array") -> "mx.array":
    """Ensure audio is mono with batch dimension [B, T].

    Converts stereo to mono and adds batch dimension if missing.

    Args:
        audio: Audio array of shape [T], [C, T], [B, T], or [B, C, T]

    Returns:
        Audio array with shape [B, T]
    """
    import mlx.core as mx

    if audio.ndim == 1:
        # [T] -> [1, T]
        return audio[None, :]
    elif audio.ndim == 2:
        if audio.shape[0] <= 2:
            # [C, T] - stereo to mono, then add batch
            mono = mx.mean(audio, axis=0)
            return mono[None, :]
        else:
            # [B, T] - already correct
            return audio
    elif audio.ndim == 3:
        # [B, C, T] - stereo to mono
        return mx.mean(audio, axis=1)
    else:
        raise AudioLoadError(f"Unexpected audio shape: {audio.shape}")


def validate_audio(
    audio: "mx.array",
    min_samples: int = 0,
    max_samples: int | None = None,
    expected_channels: int | None = None,
) -> None:
    """Validate audio tensor.

    Args:
        audio: Audio tensor to validate
        min_samples: Minimum number of samples required
        max_samples: Maximum number of samples allowed (None = unlimited)
        expected_channels: Expected number of channels (None = any)

    Raises:
        AudioLoadError: If validation fails
    """
    import mlx.core as mx

    if audio.size == 0:
        raise AudioLoadError("Audio array is empty")

    # Get sample count (last dimension)
    num_samples = audio.shape[-1]

    if num_samples < min_samples:
        raise AudioLoadError(
            f"Audio too short: {num_samples} samples, "
            f"minimum {min_samples} required"
        )

    if max_samples is not None and num_samples > max_samples:
        raise AudioLoadError(
            f"Audio too long: {num_samples} samples, "
            f"maximum {max_samples} allowed"
        )

    if expected_channels is not None and audio.ndim >= 2:
        channels = audio.shape[-2] if audio.ndim == 2 else audio.shape[1]
        if channels != expected_channels:
            raise AudioLoadError(
                f"Expected {expected_channels} channels, got {channels}"
            )


__all__ = [
    "load_audio_input",
    "ensure_batch_dim",
    "ensure_mono_batch",
    "validate_audio",
]
