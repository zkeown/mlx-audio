"""Callback-based audio source for integration with external frameworks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mlx_audio.streaming.buffer import AudioRingBuffer

if TYPE_CHECKING:
    pass


class CallbackSource:
    """Audio source that receives audio via push callbacks.

    Useful for integrating with external audio frameworks like PyAudio,
    sounddevice callbacks, or custom audio capture systems.

    The external code pushes audio via push(), and the pipeline
    reads via read().

    Args:
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        buffer_seconds: Internal buffer duration in seconds

    Example:
        >>> source = CallbackSource(sample_rate=44100, channels=2)
        >>> source.start()
        >>>
        >>> # In audio callback from external framework:
        >>> def audio_callback(indata, frames, time_info, status):
        ...     source.push(indata)
        >>>
        >>> # Pipeline reads from source:
        >>> chunk = source.read(4096)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 2,
        buffer_seconds: float = 2.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._buffer_seconds = buffer_seconds

        # Internal buffer
        self._buffer: AudioRingBuffer | None = None
        self._started: bool = False
        self._eof: bool = False

    def start(self) -> None:
        """Initialize the source for receiving audio."""
        if self._started:
            return

        max_samples = int(self._buffer_seconds * self._sample_rate)
        self._buffer = AudioRingBuffer(
            max_samples=max_samples,
            channels=self._channels,
        )
        self._eof = False
        self._started = True

    def stop(self) -> None:
        """Stop the source and release resources."""
        if not self._started:
            return

        if self._buffer:
            self._buffer.shutdown()

        self._started = False

    def push(self, audio: mx.array | np.ndarray, timeout: float | None = 1.0) -> bool:
        """Push audio samples into the source.

        Called by external code (e.g., audio callback) to provide audio.

        Args:
            audio: Audio samples [channels, samples] or [samples, channels]
            timeout: Maximum time to wait if buffer is full

        Returns:
            True if push succeeded, False if buffer full or stopped
        """
        if not self._started or self._buffer is None:
            return False

        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Handle [samples, channels] format (common from audio callbacks)
        if audio.ndim == 2 and audio.shape[1] <= 8:  # Assume <= 8 channels
            # Likely [samples, channels], transpose to [channels, samples]
            if audio.shape[0] > audio.shape[1]:
                audio = audio.T

        return self._buffer.write(mx.array(audio), timeout=timeout)

    def read(self, num_samples: int) -> mx.array | None:
        """Read samples from the source.

        Called by the pipeline to get audio for processing.

        Args:
            num_samples: Number of samples to read

        Returns:
            Audio array [channels, samples], or None if stopped/empty
        """
        if not self._started or self._buffer is None:
            return None

        return self._buffer.read(num_samples, timeout=0.1)

    def signal_eof(self) -> None:
        """Signal that no more audio will be pushed.

        Call this when the audio stream ends to allow the pipeline
        to finish processing remaining buffered audio.
        """
        self._eof = True
        if self._buffer:
            self._buffer.shutdown()

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels

    @property
    def available(self) -> int:
        """Number of samples available to read."""
        if self._buffer is None:
            return 0
        return self._buffer.available

    @property
    def is_eof(self) -> bool:
        """Whether end-of-file has been signaled."""
        return self._eof

    def __enter__(self) -> CallbackSource:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
