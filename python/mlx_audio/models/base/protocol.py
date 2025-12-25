"""Protocol definitions for mlx-audio models.

This module defines the interface contracts that models should implement.
Using protocols enables structural subtyping (duck typing) while still
providing clear documentation of expected interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mlx.core as mx

    from mlx_audio.models.base.config import ModelConfig


@runtime_checkable
class AudioModel(Protocol):
    """Protocol defining the interface for audio models.

    All audio models should implement this interface to ensure
    consistent behavior across the library.

    Example:
        >>> def process_with_any_model(model: AudioModel, audio: mx.array):
        ...     # Works with any model implementing AudioModel protocol
        ...     return model(audio)
    """

    config: ModelConfig
    """Model configuration object."""

    def __call__(self, audio: mx.array, **kwargs: Any) -> mx.array:
        """Run inference on audio input.

        Args:
            audio: Input audio tensor
            **kwargs: Model-specific parameters

        Returns:
            Model output (varies by model type)
        """
        ...

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> AudioModel:
        """Load pretrained weights from a path or model name.

        Args:
            path: Path to model directory or model identifier
            **kwargs: Additional loading options

        Returns:
            Loaded model instance
        """
        ...

    @property
    def sample_rate(self) -> int:
        """Expected input sample rate in Hz.

        Returns:
            Sample rate (e.g., 16000, 44100, 48000)
        """
        ...


@runtime_checkable
class EncoderModel(Protocol):
    """Protocol for models that encode audio to embeddings."""

    def encode(self, audio: mx.array, **kwargs: Any) -> mx.array:
        """Encode audio to embedding vector(s).

        Args:
            audio: Input audio tensor
            **kwargs: Encoding options

        Returns:
            Embedding tensor [B, D] or [B, T, D]
        """
        ...


@runtime_checkable
class GenerativeModel(Protocol):
    """Protocol for models that generate audio."""

    def generate(
        self,
        prompt: Any,
        duration: float = 10.0,
        **kwargs: Any,
    ) -> mx.array:
        """Generate audio from a prompt.

        Args:
            prompt: Generation prompt (text, embeddings, etc.)
            duration: Target duration in seconds
            **kwargs: Generation options

        Returns:
            Generated audio tensor
        """
        ...


@runtime_checkable
class StreamingModel(Protocol):
    """Protocol for models that support streaming inference."""

    def process_chunk(
        self,
        chunk: mx.array,
        state: Any = None,
    ) -> tuple[mx.array, Any]:
        """Process a single audio chunk.

        Args:
            chunk: Audio chunk to process
            state: Previous state (None for first chunk)

        Returns:
            Tuple of (output, new_state)
        """
        ...

    def reset_state(self) -> None:
        """Reset any internal streaming state."""
        ...


@runtime_checkable
class SeparationModel(Protocol):
    """Protocol for source separation models."""

    @property
    def sources(self) -> list[str]:
        """List of source names this model can separate.

        Returns:
            Source names (e.g., ["drums", "bass", "other", "vocals"])
        """
        ...

    def separate(self, audio: mx.array, **kwargs: Any) -> dict[str, mx.array]:
        """Separate audio into sources.

        Args:
            audio: Mixed audio input
            **kwargs: Separation options

        Returns:
            Dictionary mapping source names to separated audio
        """
        ...


@runtime_checkable
class TranscriptionModel(Protocol):
    """Protocol for speech-to-text models."""

    def transcribe(
        self,
        audio: mx.array,
        language: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio input
            language: Language code (None for auto-detect)
            **kwargs: Transcription options

        Returns:
            Transcribed text
        """
        ...


__all__ = [
    "AudioModel",
    "EncoderModel",
    "GenerativeModel",
    "StreamingModel",
    "SeparationModel",
    "TranscriptionModel",
]
