"""Core types for streaming audio processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import mlx.core as mx


class StreamState(Enum):
    """State of a streaming pipeline or processor."""

    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class StreamChunk:
    """A chunk of audio data in a stream.

    Attributes:
        audio: Audio samples with shape [channels, samples] or [samples]
        timestamp: Position in the stream in seconds
        is_final: Whether this is the last chunk in the stream
        metadata: Optional metadata associated with this chunk
    """

    audio: mx.array
    timestamp: float
    is_final: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        """Number of samples in this chunk."""
        return self.audio.shape[-1]

    @property
    def num_channels(self) -> int:
        """Number of channels in this chunk."""
        if self.audio.ndim == 1:
            return 1
        return self.audio.shape[0]

    @property
    def duration(self) -> float | None:
        """Duration of this chunk in seconds, if sample rate is in metadata."""
        sample_rate = self.metadata.get("sample_rate")
        if sample_rate is not None:
            return self.num_samples / sample_rate
        return None


@dataclass
class StreamStats:
    """Statistics for a streaming session.

    Attributes:
        chunks_processed: Number of chunks processed
        samples_processed: Total samples processed
        total_duration: Total audio duration in seconds
        processing_time: Total time spent processing in seconds
        buffer_underruns: Number of times input buffer was empty
        buffer_overruns: Number of times output buffer was full
    """

    chunks_processed: int = 0
    samples_processed: int = 0
    total_duration: float = 0.0
    processing_time: float = 0.0
    buffer_underruns: int = 0
    buffer_overruns: int = 0

    @property
    def realtime_factor(self) -> float:
        """Ratio of audio duration to processing time (>1 means faster than real-time)."""
        if self.processing_time == 0:
            return float("inf")
        return self.total_duration / self.processing_time
