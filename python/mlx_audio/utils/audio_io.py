"""Audio file I/O utilities.

Provides common functions for loading and saving audio files
with multiple backend support.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx


def load_audio_file(
    path: str | Path,
    target_sr: int | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Load audio file with soundfile/librosa fallback.

    Tries soundfile first for speed, falls back to librosa for
    wider format support.

    Args:
        path: Path to audio file
        target_sr: Target sample rate for resampling (None to keep original)
        mono: Whether to convert to mono

    Returns:
        Tuple of (audio array [channels, samples], sample rate)

    Raises:
        ImportError: If neither soundfile nor librosa is installed
        FileNotFoundError: If audio file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio: np.ndarray
    sr: int

    # Try soundfile first (faster)
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")

        # Convert to [channels, samples] format
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        else:
            # soundfile returns [samples, channels]
            audio = audio.T

    except ImportError:
        # Fall back to librosa
        try:
            import librosa

            audio, sr = librosa.load(str(path), sr=None, mono=False)

            # librosa may return 1D for mono files
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

        except ImportError:
            raise ImportError(
                "Either soundfile or librosa is required for audio loading. "
                "Install with: pip install soundfile"
            )

    # Convert to mono if requested
    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to a target sample rate.

    Args:
        audio: Audio array with shape [channels, samples]
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    try:
        from scipy import signal

        num_samples = int(audio.shape[1] * target_sr / orig_sr)
        return signal.resample(audio, num_samples, axis=1)

    except ImportError:
        raise ImportError(
            "scipy is required for audio resampling. "
            "Install with: pip install scipy"
        )


def save_audio_file(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str | None = None,
) -> None:
    """Save audio to a file.

    Args:
        path: Output file path (format inferred from extension)
        audio: Audio array with shape [channels, samples]
        sample_rate: Sample rate in Hz
        subtype: Optional subtype for audio encoding (e.g., "PCM_16")
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for saving audio. "
            "Install with: pip install soundfile"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # soundfile expects [samples, channels] format
    if audio.ndim == 2:
        audio = audio.T
    elif audio.ndim == 1:
        # Keep 1D for mono
        pass

    sf.write(str(path), audio, sample_rate, subtype=subtype)


def normalize_audio_input(
    audio: str | Path | np.ndarray | mx.array,
    sample_rate: int | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Normalize various audio input formats to numpy array.

    Handles:
    - File paths (str or Path) - loads the file
    - numpy arrays - uses directly
    - MLX arrays - converts to numpy

    Args:
        audio: Audio input in various formats
        sample_rate: Target sample rate (used when loading files)
        mono: Whether to convert to mono

    Returns:
        Tuple of (numpy array [channels, samples], sample rate)
    """
    import mlx.core as mx

    # Handle file paths
    if isinstance(audio, (str, Path)):
        return load_audio_file(audio, target_sr=sample_rate, mono=mono)

    # Handle MLX arrays
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Handle numpy arrays
    if isinstance(audio, np.ndarray):
        # Ensure [channels, samples] format
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        if mono and audio.shape[0] > 1:
            audio = audio.mean(axis=0, keepdims=True)

        # We don't know the sample rate for raw arrays
        # Caller should provide it or use a default
        return audio, sample_rate or 0

    raise TypeError(
        f"Unsupported audio input type: {type(audio)}. "
        "Expected str, Path, np.ndarray, or mx.array."
    )


__all__ = [
    "load_audio_file",
    "resample_audio",
    "save_audio_file",
    "normalize_audio_input",
]
