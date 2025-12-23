"""Streaming adapters for mlx-audio models."""

from __future__ import annotations

from mlx_audio.streaming.adapters.demucs import HTDemucsStreamProcessor
from mlx_audio.streaming.adapters.vad import VADStreamProcessor

__all__ = ["HTDemucsStreamProcessor", "VADStreamProcessor"]
