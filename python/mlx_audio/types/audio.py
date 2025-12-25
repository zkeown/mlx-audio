"""Audio data types and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class AudioData:
    """Container for audio data with utilities.

    All audio is stored as MLX arrays internally for GPU efficiency.

    Attributes:
        array: Audio data as MLX array, shape [channels, samples] or [samples]
        sample_rate: Sample rate in Hz
    """

    array: mx.array
    sample_rate: int

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array(self.array)

    def to_mono(self) -> AudioData:
        """Convert to mono by averaging channels."""
        if len(self.array.shape) == 1:
            return self
        return AudioData(
            array=mx.mean(self.array, axis=0),
            sample_rate=self.sample_rate,
        )

    def resample(self, target_rate: int) -> AudioData:
        """Resample to target sample rate."""
        if target_rate == self.sample_rate:
            return self

        from mlx_audio.primitives import resample as _resample

        return AudioData(
            array=_resample(self.array, self.sample_rate, target_rate),
            sample_rate=target_rate,
        )

    def save(self, path: str | Path, format: str | None = None) -> Path:
        """Save audio to file.

        Args:
            path: Output file path
            format: Audio format (inferred from extension if None)

        Returns:
            Path to saved file
        """
        return save_audio(path, self.array, self.sample_rate, format=format)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        samples = self.array.shape[-1]
        return samples / self.sample_rate

    @property
    def channels(self) -> int:
        """Number of channels."""
        if len(self.array.shape) == 1:
            return 1
        return self.array.shape[0]

    def __len__(self) -> int:
        """Number of samples."""
        return self.array.shape[-1]


def load_audio(
    path: str | Path,
    sample_rate: int | None = None,
    mono: bool = False,
) -> tuple[mx.array, int]:
    """Load audio from file.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate (None = native rate)
        mono: Convert to mono

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio I/O. "
            "Install with: pip install soundfile"
        )

    path = Path(path)
    audio, sr = sf.read(str(path), always_2d=True)

    # Convert to [channels, samples]
    audio = audio.T.astype(np.float32)

    # Convert to mono if requested
    if mono and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True)

    # Resample if needed
    if sample_rate is not None and sample_rate != sr:
        try:
            from scipy import signal

            audio = signal.resample_poly(audio, sample_rate, sr, axis=-1)
            sr = sample_rate
        except ImportError:
            raise ImportError(
                "scipy is required for resampling. "
                "Install with: pip install scipy"
            )

    return mx.array(audio), sr


def save_audio(
    path: str | Path,
    audio: mx.array | np.ndarray,
    sample_rate: int,
    format: str | None = None,
) -> Path:
    """Save audio to file.

    Args:
        path: Output file path
        audio: Audio data [channels, samples] or [samples]
        sample_rate: Sample rate in Hz
        format: Audio format (inferred from extension if None)

    Returns:
        Path to saved file
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio I/O. "
            "Install with: pip install soundfile"
        )

    path = Path(path)

    # Convert to numpy if needed
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Ensure 2D
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    # Convert to [samples, channels] for soundfile
    audio = audio.T

    sf.write(str(path), audio, sample_rate, format=format)
    return path
