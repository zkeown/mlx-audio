"""Streaming audio processing for mlx-audio.

This module provides real-time streaming capabilities for audio processing,
including buffering, chunked processing with overlap-add, and integration
with audio I/O devices.

Example:
    >>> from mlx_audio.streaming import StreamingPipeline, HTDemucsStreamProcessor
    >>> from mlx_audio.streaming import FileSource, FileSink
    >>> from mlx_audio.models import HTDemucs
    >>>
    >>> model = HTDemucs.from_pretrained("mlx-community/htdemucs-ft")
    >>> pipeline = StreamingPipeline(
    ...     source=FileSource("song.mp3"),
    ...     processor=HTDemucsStreamProcessor(model),
    ...     sink=FileSink("vocals.wav", stem_index=3),
    ... )
    >>> pipeline.start()
    >>> pipeline.wait()
"""

from __future__ import annotations

# Core types
from mlx_audio.streaming._types import StreamChunk, StreamState, StreamStats

# State management
from mlx_audio.streaming.context import StreamingContext

# Buffering
from mlx_audio.streaming.buffer import AudioRingBuffer

# Processor base classes
from mlx_audio.streaming.processor import (
    GainProcessor,
    IdentityProcessor,
    Streamable,
    StreamProcessor,
)

# Pipeline
from mlx_audio.streaming.pipeline import AudioSink, AudioSource, StreamingPipeline

# Adapters
from mlx_audio.streaming.adapters.demucs import HTDemucsStreamProcessor

# Sources
from mlx_audio.streaming.sources.file import FileSource
from mlx_audio.streaming.sources.callback import CallbackSource
from mlx_audio.streaming.sources.microphone import MicrophoneSource

# Sinks
from mlx_audio.streaming.sinks.file import FileSink, MultiFileSink
from mlx_audio.streaming.sinks.callback import CallbackSink
from mlx_audio.streaming.sinks.speaker import SpeakerSink

# Metrics
from mlx_audio.streaming.metrics import (
    SeparationMetrics,
    correlation,
    mae,
    mse,
    sdr,
    si_sdr,
    snr,
)

__all__ = [
    # Types
    "StreamChunk",
    "StreamState",
    "StreamStats",
    # Context
    "StreamingContext",
    # Buffer
    "AudioRingBuffer",
    # Processor
    "StreamProcessor",
    "Streamable",
    "IdentityProcessor",
    "GainProcessor",
    # Pipeline
    "StreamingPipeline",
    "AudioSource",
    "AudioSink",
    # Adapters
    "HTDemucsStreamProcessor",
    # Sources
    "FileSource",
    "CallbackSource",
    "MicrophoneSource",
    # Sinks
    "FileSink",
    "MultiFileSink",
    "CallbackSink",
    "SpeakerSink",
    # Metrics
    "si_sdr",
    "sdr",
    "snr",
    "correlation",
    "mae",
    "mse",
    "SeparationMetrics",
]
