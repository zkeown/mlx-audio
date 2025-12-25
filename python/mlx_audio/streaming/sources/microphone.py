"""Microphone audio source using sounddevice."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mlx_audio.streaming.buffer import AudioRingBuffer

if TYPE_CHECKING:
    pass


class MicrophoneSource:
    """Real-time audio source from microphone using sounddevice.

    Captures audio from the default or specified input device and
    buffers it for streaming pipeline consumption.

    Args:
        sample_rate: Audio sample rate in Hz (default: 44100)
        channels: Number of channels to capture (default: 2)
        device: Input device index or name (default: None = system default)
        buffer_seconds: Internal buffer duration in seconds
        blocksize: Samples per callback (default: 1024)

    Example:
        >>> source = MicrophoneSource(sample_rate=44100, channels=2)
        >>> source.start()
        >>> while True:
        ...     chunk = source.read(4096)
        ...     if chunk is not None:
        ...         process(chunk)
        >>> source.stop()

    Note:
        Requires sounddevice: pip install sounddevice
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 2,
        device: int | str | None = None,
        buffer_seconds: float = 2.0,
        blocksize: int = 1024,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._buffer_seconds = buffer_seconds
        self._blocksize = blocksize

        # State
        self._buffer: AudioRingBuffer | None = None
        self._stream = None
        self._started: bool = False
        self._error: Exception | None = None

    def start(self) -> None:
        """Start capturing audio from the microphone."""
        if self._started:
            return

        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for MicrophoneSource. "
                "Install with: pip install sounddevice"
            )

        # Initialize buffer
        max_samples = int(self._buffer_seconds * self._sample_rate)
        self._buffer = AudioRingBuffer(
            max_samples=max_samples,
            channels=self._channels,
        )

        # Create input stream
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            device=self._device,
            blocksize=self._blocksize,
            callback=self._audio_callback,
            dtype=np.float32,
        )

        self._error = None
        self._stream.start()
        self._started = True

    def stop(self) -> None:
        """Stop capturing audio."""
        if not self._started:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._buffer is not None:
            self._buffer.shutdown()

        self._started = False

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Sounddevice callback for incoming audio."""
        if status:
            # Handle status flags (overflow, etc.)
            pass

        if self._buffer is not None:
            # indata is [frames, channels], convert to [channels, frames]
            audio = indata.T.copy()
            self._buffer.write(mx.array(audio), timeout=0.0)

    def read(self, num_samples: int) -> mx.array | None:
        """Read samples from the microphone buffer.

        Args:
            num_samples: Number of samples to read

        Returns:
            Audio array [channels, samples], or None if not enough samples
        """
        if not self._started or self._buffer is None:
            return None

        return self._buffer.read(num_samples, timeout=0.1)

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
    def error(self) -> Exception | None:
        """Any error that occurred during capture."""
        return self._error

    @property
    def device_info(self) -> dict | None:
        """Information about the current input device."""
        try:
            import sounddevice as sd

            if self._device is not None:
                return sd.query_devices(self._device)
            return sd.query_devices(kind="input")
        except Exception:
            return None

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices.

        Returns:
            List of device info dictionaries
        """
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            # Filter to input devices
            return [
                {"index": i, **d}
                for i, d in enumerate(devices)
                if d.get("max_input_channels", 0) > 0
            ]
        except ImportError:
            raise ImportError(
                "sounddevice is required. Install with: pip install sounddevice"
            )

    def __enter__(self) -> MicrophoneSource:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
