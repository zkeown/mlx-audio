"""Ring buffer for streaming audio samples."""

from __future__ import annotations

import threading

import mlx.core as mx
import numpy as np


class AudioRingBuffer:
    """Thread-safe ring buffer optimized for audio streaming.

    Unlike the generic PrefetchBuffer which handles discrete items,
    AudioRingBuffer manages contiguous audio samples with:
    - Variable-size writes (from audio sources)
    - Fixed-size reads (for model chunk requirements)
    - Optional overlap retention for seamless processing

    The buffer uses numpy internally for efficient indexing and converts
    to/from mx.array at the boundaries.

    Args:
        max_samples: Maximum number of samples to buffer
        channels: Number of audio channels
        overlap_samples: Samples to retain after reads for overlap processing
        dtype: Data type for the buffer (default: float32)

    Example:
        >>> buffer = AudioRingBuffer(max_samples=44100, channels=2)
        >>> buffer.write(audio_chunk)  # Variable size input
        >>> chunk = buffer.read(4096)  # Fixed size output
    """

    def __init__(
        self,
        max_samples: int,
        channels: int = 2,
        overlap_samples: int = 0,
        dtype: np.dtype = np.float32,
    ) -> None:
        self._max_samples = max_samples
        self._channels = channels
        self._overlap_samples = overlap_samples
        self._dtype = dtype

        # Internal buffer using numpy for efficient slicing
        # Shape: [channels, max_samples]
        self._buffer = np.zeros((channels, max_samples), dtype=dtype)

        # Read/write positions
        self._write_pos = 0
        self._read_pos = 0
        self._available = 0  # Samples available to read

        # Threading primitives
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._shutdown = False

    def write(
        self,
        audio: mx.array | np.ndarray,
        timeout: float | None = None,
    ) -> bool:
        """Write audio samples to the buffer.

        Args:
            audio: Audio samples with shape [channels, samples] or [samples]
            timeout: Maximum time to wait if buffer is full (None = block forever)

        Returns:
            True if write succeeded, False if shutdown or timeout
        """
        # Convert to numpy if needed
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Handle mono input
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Validate shape
        if audio.shape[0] != self._channels:
            raise ValueError(
                f"Expected {self._channels} channels, got {audio.shape[0]}"
            )

        num_samples = audio.shape[1]

        with self._not_full:
            if self._shutdown:
                return False

            # Wait for space
            while self._available + num_samples > self._max_samples:
                if self._shutdown:
                    return False
                if not self._not_full.wait(timeout):
                    return False  # Timeout
                if self._shutdown:
                    return False

            # Write samples (may wrap around)
            end_pos = self._write_pos + num_samples
            if end_pos <= self._max_samples:
                # No wrap
                self._buffer[:, self._write_pos:end_pos] = audio
            else:
                # Wrap around
                first_part = self._max_samples - self._write_pos
                self._buffer[:, self._write_pos:] = audio[:, :first_part]
                self._buffer[:, :end_pos - self._max_samples] = audio[:, first_part:]

            self._write_pos = end_pos % self._max_samples
            self._available += num_samples
            self._not_empty.notify_all()
            return True

    def read(
        self,
        num_samples: int,
        timeout: float | None = None,
    ) -> mx.array | None:
        """Read a fixed number of samples from the buffer.

        Args:
            num_samples: Number of samples to read
            timeout: Maximum time to wait if not enough samples (None = block forever)

        Returns:
            Audio array with shape [channels, num_samples], or None if shutdown/timeout
        """
        with self._not_empty:
            if self._shutdown and self._available == 0:
                return None

            # Wait for enough samples
            while self._available < num_samples:
                if self._shutdown:
                    # Return what we have on shutdown
                    if self._available > 0:
                        return self._read_available_unlocked()
                    return None
                if not self._not_empty.wait(timeout):
                    return None  # Timeout
                if self._shutdown and self._available == 0:
                    return None

            # Read samples (may wrap around)
            end_pos = self._read_pos + num_samples
            if end_pos <= self._max_samples:
                # No wrap - mx.array() will copy, so no need for numpy copy
                result = mx.array(self._buffer[:, self._read_pos:end_pos])
            else:
                # Wrap around - need to concatenate the two parts
                first_part = self._max_samples - self._read_pos
                result = np.empty((self._channels, num_samples), dtype=self._dtype)
                result[:, :first_part] = self._buffer[:, self._read_pos:]
                result[:, first_part:] = self._buffer[:, :end_pos - self._max_samples]
                result = mx.array(result)

            # Update read position, but retain overlap
            consume = num_samples - self._overlap_samples
            if consume > 0:
                self._read_pos = (self._read_pos + consume) % self._max_samples
                self._available -= consume
                self._not_full.notify_all()

            return result

    def read_available(self) -> mx.array | None:
        """Read all available samples without blocking.

        Returns:
            Audio array with all available samples, or None if empty
        """
        with self._not_empty:
            if self._available == 0:
                return None
            return self._read_available_unlocked()

    def _read_available_unlocked(self) -> mx.array:
        """Internal: read all available samples (must hold lock)."""
        num_samples = self._available
        end_pos = self._read_pos + num_samples

        if end_pos <= self._max_samples:
            # No wrap - mx.array() will copy, so no need for numpy copy
            result = mx.array(self._buffer[:, self._read_pos:end_pos])
        else:
            first_part = self._max_samples - self._read_pos
            result = np.empty((self._channels, num_samples), dtype=self._dtype)
            result[:, :first_part] = self._buffer[:, self._read_pos:]
            result[:, first_part:] = self._buffer[:, :end_pos - self._max_samples]
            result = mx.array(result)

        # Consume all but overlap
        consume = max(0, num_samples - self._overlap_samples)
        self._read_pos = (self._read_pos + consume) % self._max_samples
        self._available -= consume
        self._not_full.notify_all()

        return result

    def peek(self, num_samples: int) -> mx.array | None:
        """Read samples without consuming them (for lookahead).

        Args:
            num_samples: Number of samples to peek

        Returns:
            Audio array, or None if not enough samples available
        """
        with self._lock:
            if self._available < num_samples:
                return None

            end_pos = self._read_pos + num_samples
            if end_pos <= self._max_samples:
                # No wrap - mx.array() will copy, so no need for numpy copy
                return mx.array(self._buffer[:, self._read_pos:end_pos])
            else:
                first_part = self._max_samples - self._read_pos
                result = np.empty((self._channels, num_samples), dtype=self._dtype)
                result[:, :first_part] = self._buffer[:, self._read_pos:]
                result[:, first_part:] = self._buffer[:, :end_pos - self._max_samples]
                return mx.array(result)

    @property
    def available(self) -> int:
        """Number of samples available to read."""
        with self._lock:
            return self._available

    @property
    def space(self) -> int:
        """Number of samples that can be written."""
        with self._lock:
            return self._max_samples - self._available

    @property
    def channels(self) -> int:
        """Number of channels."""
        return self._channels

    @property
    def max_samples(self) -> int:
        """Maximum buffer capacity in samples."""
        return self._max_samples

    @property
    def is_shutdown(self) -> bool:
        """Whether the buffer has been shut down."""
        with self._lock:
            return self._shutdown

    def clear(self) -> None:
        """Clear all samples from the buffer."""
        with self._lock:
            self._read_pos = 0
            self._write_pos = 0
            self._available = 0
            self._not_full.notify_all()

    def shutdown(self) -> None:
        """Shutdown the buffer, unblocking all waiting threads."""
        with self._lock:
            self._shutdown = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def reset(self) -> None:
        """Reset the buffer to initial state (clears shutdown flag)."""
        with self._lock:
            self._shutdown = False
            self._read_pos = 0
            self._write_pos = 0
            self._available = 0
            self._buffer.fill(0)
            self._not_full.notify_all()

    def __len__(self) -> int:
        """Number of samples available to read."""
        return self.available
