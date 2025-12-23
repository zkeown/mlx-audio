"""Callback-based audio sink for integration with external frameworks."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mlx_audio.streaming.buffer import AudioRingBuffer

if TYPE_CHECKING:
    pass


class CallbackSink:
    """Audio sink that provides audio via pull callbacks.

    Useful for integrating with external audio frameworks that need
    to pull audio on their own schedule (e.g., sounddevice output callbacks).

    Two modes of operation:
    1. Callback mode: Provide a callback that receives each chunk
    2. Pull mode: External code calls pull() to get audio

    Args:
        sample_rate: Expected audio sample rate in Hz
        channels: Number of audio channels (after processing)
        buffer_seconds: Internal buffer duration in seconds
        callback: Optional callback function called for each chunk

    Example (callback mode):
        >>> def output_callback(audio):
        ...     play_audio(audio)  # Your playback code
        >>>
        >>> sink = CallbackSink(
        ...     sample_rate=44100,
        ...     callback=output_callback,
        ... )

    Example (pull mode):
        >>> sink = CallbackSink(sample_rate=44100)
        >>> sink.start()
        >>>
        >>> # In audio output callback:
        >>> def output_callback(outdata, frames, time_info, status):
        ...     audio = sink.pull(frames)
        ...     if audio is not None:
        ...         outdata[:] = audio
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 2,
        buffer_seconds: float = 2.0,
        callback: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._buffer_seconds = buffer_seconds
        self._callback = callback

        # Internal buffer for pull mode
        self._buffer: AudioRingBuffer | None = None
        self._started: bool = False

        # For callback mode
        self._callback_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the sink."""
        if self._started:
            return

        # Initialize buffer for pull mode or if callback provided
        max_samples = int(self._buffer_seconds * self._sample_rate)
        self._buffer = AudioRingBuffer(
            max_samples=max_samples,
            channels=self._channels,
        )

        self._stop_event.clear()

        # If callback provided, start callback thread
        if self._callback is not None:
            self._callback_thread = threading.Thread(
                target=self._callback_loop,
                name="CallbackSink-Callback",
                daemon=True,
            )
            self._callback_thread.start()

        self._started = True

    def stop(self) -> None:
        """Stop the sink and release resources."""
        if not self._started:
            return

        self._stop_event.set()

        if self._buffer:
            self._buffer.shutdown()

        if self._callback_thread and self._callback_thread.is_alive():
            self._callback_thread.join(timeout=1.0)

        self._started = False

    def write(self, audio: mx.array | np.ndarray) -> bool:
        """Write audio to the sink.

        Called by the pipeline to output processed audio.

        Args:
            audio: Audio samples [channels, samples]

        Returns:
            True if write succeeded
        """
        if not self._started or self._buffer is None:
            return False

        return self._buffer.write(audio, timeout=0.1)

    def pull(self, num_samples: int) -> np.ndarray | None:
        """Pull audio samples from the sink.

        Called by external code to retrieve processed audio.

        Args:
            num_samples: Number of samples to pull

        Returns:
            Audio array [samples, channels] (transposed for output callbacks),
            or None if no audio available
        """
        if not self._started or self._buffer is None:
            return None

        chunk = self._buffer.read(num_samples, timeout=0.01)
        if chunk is None:
            return None

        # Convert to numpy and transpose for typical output callback format
        audio = np.array(chunk)
        return audio.T  # Return [samples, channels]

    def _callback_loop(self) -> None:
        """Internal loop for callback mode."""
        chunk_samples = 1024  # Default chunk size for callbacks

        while not self._stop_event.is_set():
            if self._buffer is None:
                break

            chunk = self._buffer.read(chunk_samples, timeout=0.1)
            if chunk is None:
                if self._buffer.is_shutdown:
                    break
                continue

            # Convert to numpy and transpose for callback
            audio = np.array(chunk).T  # [samples, channels]

            try:
                self._callback(audio)
            except Exception:
                # Don't let callback errors stop the loop
                pass

    @property
    def sample_rate(self) -> int:
        """Expected sample rate in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Number of channels."""
        return self._channels

    @property
    def available(self) -> int:
        """Number of samples available to pull."""
        if self._buffer is None:
            return 0
        return self._buffer.available

    def __enter__(self) -> CallbackSink:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
