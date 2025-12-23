"""Audio sinks for streaming pipelines."""

from __future__ import annotations

from mlx_audio.streaming.sinks.callback import CallbackSink
from mlx_audio.streaming.sinks.file import FileSink, MultiFileSink
from mlx_audio.streaming.sinks.speaker import SpeakerSink

__all__ = ["FileSink", "MultiFileSink", "CallbackSink", "SpeakerSink"]
