"""File-based audio source for streaming."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.constants import DEFAULT_CHUNK_SIZE
from mlx_audio.streaming.base import AudioSource
from mlx_audio.utils.audio_io import load_audio_file


class FileSource(AudioSource):
    """Audio source that reads from a file.

    Reads audio in chunks, useful for testing streaming pipelines
    without real-time constraints.

    Args:
        path: Path to the audio file
        chunk_samples: Number of samples per read (default: 4096)
        start_sample: Sample offset to start reading from (default: 0)
        end_sample: Sample to stop reading at (default: None = end of file)
        loop: Whether to loop when reaching end of file (default: False)

    Example:
        >>> source = FileSource("audio.wav", chunk_samples=4096)
        >>> source.start()
        >>> while (chunk := source.read(4096)) is not None:
        ...     process(chunk)
        >>> source.stop()
    """

    def __init__(
        self,
        path: str | Path,
        chunk_samples: int = DEFAULT_CHUNK_SIZE,
        start_sample: int = 0,
        end_sample: int | None = None,
        loop: bool = False,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self._chunk_samples = chunk_samples
        self._start_sample = start_sample
        self._end_sample = end_sample
        self._loop = loop

        # State
        self._audio: np.ndarray | None = None
        self._sample_rate: int = 0
        self._position: int = 0

    def _do_start(self) -> None:
        """Load the audio file and prepare for reading."""
        # Use the shared audio loading utility
        audio, sr = load_audio_file(str(self._path))
        self._sample_rate = sr
        self._audio = audio

        # Apply start/end bounds
        if self._end_sample is not None:
            self._audio = self._audio[:, : self._end_sample]
        if self._start_sample > 0:
            self._audio = self._audio[:, self._start_sample :]

        self._position = 0

    def _do_stop(self) -> None:
        """Release resources."""
        self._audio = None
        self._position = 0

    def read(self, num_samples: int | None = None) -> mx.array | None:
        """Read samples from the file.

        Args:
            num_samples: Number of samples to read (default: chunk_samples)

        Returns:
            Audio array with shape [channels, samples], or None if EOF
        """
        if not self._started or self._audio is None:
            raise RuntimeError("Source not started. Call start() first.")

        if num_samples is None:
            num_samples = self._chunk_samples

        total_samples = self._audio.shape[1]

        if self._position >= total_samples:
            if self._loop:
                self._position = 0
            else:
                return None

        # Calculate how many samples we can read
        end_pos = min(self._position + num_samples, total_samples)
        chunk = self._audio[:, self._position : end_pos]
        self._position = end_pos

        # Handle looping if we didn't get enough samples
        if self._loop and chunk.shape[1] < num_samples:
            remaining = num_samples - chunk.shape[1]
            self._position = 0
            end_pos = min(remaining, total_samples)
            extra = self._audio[:, :end_pos]
            self._position = end_pos
            chunk = np.concatenate([chunk, extra], axis=1)

        return mx.array(chunk)

    def seek(self, sample: int) -> None:
        """Seek to a specific sample position.

        Args:
            sample: Sample position to seek to
        """
        if self._audio is None:
            raise RuntimeError("Source not started. Call start() first.")

        total_samples = self._audio.shape[1]
        self._position = max(0, min(sample, total_samples))

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        if not self._started:
            raise RuntimeError("Source not started. Call start() first.")
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        if self._audio is None:
            raise RuntimeError("Source not started. Call start() first.")
        return self._audio.shape[0]

    @property
    def total_samples(self) -> int:
        """Total number of samples in the file."""
        if self._audio is None:
            raise RuntimeError("Source not started. Call start() first.")
        return self._audio.shape[1]

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds."""
        return self.total_samples / self.sample_rate

    @property
    def position(self) -> int:
        """Current read position in samples."""
        return self._position

    @property
    def remaining(self) -> int:
        """Number of samples remaining."""
        if self._audio is None:
            return 0
        return self._audio.shape[1] - self._position

    @property
    def is_eof(self) -> bool:
        """Whether we've reached the end of the file."""
        if self._audio is None:
            return True
        return self._position >= self._audio.shape[1] and not self._loop

    # __enter__ and __exit__ are inherited from StreamingComponent
