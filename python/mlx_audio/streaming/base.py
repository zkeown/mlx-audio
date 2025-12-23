"""Base classes for streaming audio components.

Provides abstract base classes for audio sources and sinks
with common lifecycle management.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self


class StreamingComponent(ABC):
    """Abstract base class for streaming audio components.

    Provides common start/stop lifecycle management and context manager
    support. Subclasses implement _do_start() and _do_stop() for their
    specific initialization and cleanup logic.

    The start/stop methods are idempotent - calling start() when already
    started or stop() when already stopped is a no-op.

    Example:
        >>> class MySource(StreamingComponent):
        ...     def _do_start(self) -> None:
        ...         self._device = open_audio_device()
        ...
        ...     def _do_stop(self) -> None:
        ...         self._device.close()
        ...
        >>> with MySource() as source:
        ...     data = source.read()
    """

    def __init__(self) -> None:
        """Initialize the streaming component."""
        self._started = False

    def start(self) -> None:
        """Start the streaming component.

        Calls _do_start() to perform actual initialization.
        Idempotent - does nothing if already started.
        """
        if self._started:
            return
        self._do_start()
        self._started = True

    def stop(self) -> None:
        """Stop the streaming component.

        Calls _do_stop() to perform actual cleanup.
        Idempotent - does nothing if already stopped.
        """
        if not self._started:
            return
        self._do_stop()
        self._started = False

    @abstractmethod
    def _do_start(self) -> None:
        """Perform actual start logic.

        Subclasses implement this to initialize resources,
        open devices, start threads, etc.
        """
        pass

    @abstractmethod
    def _do_stop(self) -> None:
        """Perform actual stop logic.

        Subclasses implement this to release resources,
        close devices, stop threads, etc.
        """
        pass

    def __enter__(self) -> Self:
        """Enter context manager, starting the component."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager, stopping the component."""
        self.stop()

    @property
    def is_running(self) -> bool:
        """Whether the component is currently started."""
        return self._started


class AudioSource(StreamingComponent):
    """Abstract base class for audio sources.

    Audio sources produce audio data that can be read in chunks.
    Subclasses must implement the read() method in addition to
    _do_start() and _do_stop().

    Properties like sample_rate, channels, etc. should be implemented
    by subclasses as appropriate.
    """

    @abstractmethod
    def read(self, num_samples: int | None = None):
        """Read audio samples from the source.

        Args:
            num_samples: Number of samples to read (None for default chunk)

        Returns:
            Audio array with shape [channels, samples], or None if EOF
        """
        pass


class AudioSink(StreamingComponent):
    """Abstract base class for audio sinks.

    Audio sinks consume audio data written to them.
    Subclasses must implement the write() method in addition to
    _do_start() and _do_stop().
    """

    @abstractmethod
    def write(self, audio) -> None:
        """Write audio samples to the sink.

        Args:
            audio: Audio array with shape [channels, samples]
        """
        pass


__all__ = [
    "StreamingComponent",
    "AudioSource",
    "AudioSink",
]
