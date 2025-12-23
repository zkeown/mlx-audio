"""Speaker audio sink using sounddevice."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mlx_audio.streaming.buffer import AudioRingBuffer

if TYPE_CHECKING:
    pass


class SpeakerSink:
    """Real-time audio output to speakers using sounddevice.

    Buffers processed audio and plays it through the default or
    specified output device.

    Args:
        sample_rate: Audio sample rate in Hz (default: 44100)
        channels: Number of output channels (default: 2)
        device: Output device index or name (default: None = system default)
        buffer_seconds: Internal buffer duration in seconds
        blocksize: Samples per callback (default: 1024)

    Example:
        >>> sink = SpeakerSink(sample_rate=44100, channels=2)
        >>> sink.start()
        >>> for chunk in processed_audio:
        ...     sink.write(chunk)
        >>> sink.stop()

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
        self._underrun_count: int = 0

    def start(self) -> None:
        """Start audio playback."""
        if self._started:
            return

        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice is required for SpeakerSink. "
                "Install with: pip install sounddevice"
            )

        # Initialize buffer
        max_samples = int(self._buffer_seconds * self._sample_rate)
        self._buffer = AudioRingBuffer(
            max_samples=max_samples,
            channels=self._channels,
        )

        # Create output stream
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            device=self._device,
            blocksize=self._blocksize,
            callback=self._audio_callback,
            dtype=np.float32,
        )

        self._underrun_count = 0
        self._stream.start()
        self._started = True

    def stop(self) -> None:
        """Stop audio playback and drain buffer."""
        if not self._started:
            return

        # Give time for buffer to drain
        if self._buffer is not None and self._buffer.available > 0:
            import time

            drain_time = self._buffer.available / self._sample_rate
            time.sleep(min(drain_time + 0.1, 2.0))

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._buffer is not None:
            self._buffer.shutdown()

        self._started = False

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Sounddevice callback for outgoing audio."""
        if status:
            # Handle status flags (underflow, etc.)
            pass

        if self._buffer is None:
            outdata.fill(0)
            return

        # Try to read from buffer
        chunk = self._buffer.read(frames, timeout=0.0)

        if chunk is None:
            # Buffer underrun - output silence
            outdata.fill(0)
            self._underrun_count += 1
            return

        # chunk is [channels, frames], outdata expects [frames, channels]
        audio = np.array(chunk).T

        # Handle shape mismatch
        if audio.shape[0] < frames:
            # Pad with zeros
            outdata[:audio.shape[0], :] = audio
            outdata[audio.shape[0]:, :] = 0
        elif audio.shape[0] > frames:
            # Truncate (shouldn't happen normally)
            outdata[:] = audio[:frames, :]
        else:
            outdata[:] = audio

    def write(self, audio: mx.array | np.ndarray) -> bool:
        """Write audio samples to the speaker buffer.

        Args:
            audio: Audio samples [channels, samples]

        Returns:
            True if write succeeded
        """
        if not self._started or self._buffer is None:
            return False

        # Handle multi-source output [sources, channels, samples]
        if isinstance(audio, mx.array):
            audio_np = np.array(audio)
        else:
            audio_np = audio

        if audio_np.ndim == 3:
            # Sum sources or select first source
            # For now, select first source (typically for single-stem playback)
            audio_np = audio_np[0]

        return self._buffer.write(mx.array(audio_np), timeout=0.1)

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Number of output channels."""
        return self._channels

    @property
    def buffer_level(self) -> float:
        """Current buffer level as fraction (0.0 to 1.0)."""
        if self._buffer is None:
            return 0.0
        return self._buffer.available / self._buffer.max_samples

    @property
    def underrun_count(self) -> int:
        """Number of buffer underruns (silence gaps)."""
        return self._underrun_count

    @property
    def device_info(self) -> dict | None:
        """Information about the current output device."""
        try:
            import sounddevice as sd

            if self._device is not None:
                return sd.query_devices(self._device)
            return sd.query_devices(kind="output")
        except Exception:
            return None

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio output devices.

        Returns:
            List of device info dictionaries
        """
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            # Filter to output devices
            return [
                {"index": i, **d}
                for i, d in enumerate(devices)
                if d.get("max_output_channels", 0) > 0
            ]
        except ImportError:
            raise ImportError(
                "sounddevice is required. Install with: pip install sounddevice"
            )

    def __enter__(self) -> SpeakerSink:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
