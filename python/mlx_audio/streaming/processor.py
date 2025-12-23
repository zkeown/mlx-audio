"""Base classes and protocols for streaming audio processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import mlx.core as mx

from mlx_audio.streaming.context import StreamingContext

if TYPE_CHECKING:
    pass


@runtime_checkable
class Streamable(Protocol):
    """Protocol for objects that can be adapted for streaming.

    Any class implementing this protocol can be wrapped with a streaming
    adapter. Models like HTDemucs should implement this for seamless
    streaming integration.

    Example:
        >>> if isinstance(model, Streamable):
        ...     processor = GenericStreamAdapter(model)
    """

    def get_chunk_size(self) -> int:
        """Get the required input chunk size in samples.

        Returns:
            Minimum number of samples needed for processing
        """
        ...

    def get_overlap_size(self) -> int:
        """Get the required overlap between chunks in samples.

        Returns:
            Number of samples that should overlap between consecutive chunks
        """
        ...

    def process_chunk(
        self,
        audio: mx.array,
        context: StreamingContext,
    ) -> mx.array:
        """Process a single audio chunk.

        Args:
            audio: Input audio with shape [channels, samples]
            context: Streaming context (may be modified)

        Returns:
            Processed audio chunk
        """
        ...


class StreamProcessor(ABC):
    """Abstract base class for streaming audio processors.

    Subclasses must implement:
    - process_chunk: Process one audio chunk
    - get_chunk_size: Required input chunk size
    - get_overlap_size: Required overlap between chunks

    Optional overrides:
    - initialize_context: Set up initial state
    - finalize: Clean up and flush remaining audio

    Example:
        >>> class MyProcessor(StreamProcessor):
        ...     def process_chunk(self, audio, context):
        ...         return audio * 0.5  # Simple gain reduction
        ...
        ...     def get_chunk_size(self) -> int:
        ...         return 4096
        ...
        ...     def get_overlap_size(self) -> int:
        ...         return 0
    """

    @abstractmethod
    def process_chunk(
        self,
        audio: mx.array,
        context: StreamingContext,
    ) -> mx.array:
        """Process a single audio chunk.

        This method should:
        1. Apply processing to the input audio
        2. Update context state as needed
        3. Return the processed audio

        For models with output latency (like overlap-add), this may return
        fewer samples than the input, or buffer internally.

        Args:
            audio: Input chunk with shape [channels, samples]
            context: Streaming context (may be modified in place)

        Returns:
            Processed audio chunk. Shape depends on processor:
            - For effects: [channels, samples]
            - For source separation: [sources, channels, samples]
        """
        ...

    @abstractmethod
    def get_chunk_size(self) -> int:
        """Get the required input chunk size in samples.

        The pipeline will accumulate audio until this many samples
        are available before calling process_chunk.

        Returns:
            Minimum number of samples needed for processing
        """
        ...

    @abstractmethod
    def get_overlap_size(self) -> int:
        """Get the required overlap between chunks in samples.

        This many samples from the end of each chunk will be retained
        and prepended to the next chunk.

        Returns:
            Number of overlap samples (0 for no overlap)
        """
        ...

    def initialize_context(self, sample_rate: int) -> StreamingContext:
        """Create and initialize a streaming context for this processor.

        Override to set up processor-specific initial state.

        Args:
            sample_rate: Audio sample rate in Hz

        Returns:
            Initialized streaming context
        """
        return StreamingContext(sample_rate=sample_rate)

    def finalize(self, context: StreamingContext) -> mx.array | None:
        """Finalize processing and flush any remaining buffered audio.

        Called when the input stream ends. Override for processors that
        buffer audio internally (e.g., for overlap-add).

        Args:
            context: The streaming context

        Returns:
            Any remaining audio to output, or None if nothing to flush
        """
        return None

    @property
    def input_channels(self) -> int:
        """Expected number of input channels.

        Override if the processor expects a specific number of channels.

        Returns:
            Number of input channels (default: 2 for stereo)
        """
        return 2

    @property
    def output_channels(self) -> int:
        """Number of output channels.

        May differ from input_channels for processors that change
        the channel count (e.g., mono to stereo, or source separation).

        Returns:
            Number of output channels (default: same as input)
        """
        return self.input_channels

    @property
    def sample_rate(self) -> int | None:
        """Expected sample rate, if any.

        Override for processors that require a specific sample rate.

        Returns:
            Required sample rate in Hz, or None if any rate is accepted
        """
        return None

    @property
    def latency_samples(self) -> int:
        """Processing latency in samples.

        For processors with internal buffering (like overlap-add),
        this indicates how many samples of delay are introduced.

        Returns:
            Latency in samples (default: 0)
        """
        return 0


class IdentityProcessor(StreamProcessor):
    """A pass-through processor that returns input unchanged.

    Useful for testing pipelines without actual processing.

    Args:
        chunk_size: Chunk size to use (default: 4096)
        channels: Number of channels (default: 2)
    """

    def __init__(self, chunk_size: int = 4096, channels: int = 2) -> None:
        self._chunk_size = chunk_size
        self._channels = channels

    def process_chunk(
        self,
        audio: mx.array,
        context: StreamingContext,
    ) -> mx.array:
        """Return input unchanged."""
        context.advance(audio.shape[-1])
        return audio

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def get_overlap_size(self) -> int:
        return 0

    @property
    def input_channels(self) -> int:
        return self._channels

    @property
    def output_channels(self) -> int:
        return self._channels


class GainProcessor(StreamProcessor):
    """Simple gain adjustment processor.

    Useful for testing and as an example of a basic processor.

    Args:
        gain: Gain factor (1.0 = unity, 0.5 = -6dB, 2.0 = +6dB)
        chunk_size: Chunk size to use (default: 4096)
        channels: Number of channels (default: 2)
    """

    def __init__(
        self,
        gain: float = 1.0,
        chunk_size: int = 4096,
        channels: int = 2,
    ) -> None:
        self._gain = gain
        self._chunk_size = chunk_size
        self._channels = channels

    def process_chunk(
        self,
        audio: mx.array,
        context: StreamingContext,
    ) -> mx.array:
        """Apply gain to the audio."""
        context.advance(audio.shape[-1])
        return audio * self._gain

    def get_chunk_size(self) -> int:
        return self._chunk_size

    def get_overlap_size(self) -> int:
        return 0

    @property
    def input_channels(self) -> int:
        return self._channels

    @property
    def output_channels(self) -> int:
        return self._channels

    @property
    def gain(self) -> float:
        """Current gain value."""
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        """Set gain value."""
        self._gain = value
