"""Streaming adapter for VAD model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from mlx_audio.streaming.context import StreamingContext
from mlx_audio.streaming.processor import StreamProcessor

if TYPE_CHECKING:
    from mlx_audio.models.vad import SileroVAD


class VADStreamProcessor(StreamProcessor):
    """Streaming processor for Voice Activity Detection.

    Wraps a SileroVAD model for real-time speech detection, allowing
    downstream processors to check speech state via the streaming context.

    The processor stores the following state in the context:
    - "is_speech": bool - Whether current chunk contains speech
    - "speech_prob": float - Speech probability for current chunk
    - "speech_start": float | None - Start time of current speech segment
    - "in_speech": bool - Whether we're currently in a speech segment

    Args:
        model: SileroVAD model instance
        threshold: Speech probability threshold (default: 0.5)
        min_speech_duration: Minimum speech duration in seconds (default: 0.25)
        min_silence_duration: Minimum silence to end speech segment (default: 0.1)

    Example:
        >>> from mlx_audio.models.vad import SileroVAD
        >>> from mlx_audio.streaming import StreamingPipeline
        >>> from mlx_audio.streaming.adapters import VADStreamProcessor
        >>>
        >>> model = SileroVAD()
        >>> processor = VADStreamProcessor(model, threshold=0.5)
        >>> context = processor.initialize_context(16000)
        >>>
        >>> for chunk in audio_chunks:
        ...     output = processor.process_chunk(chunk, context)
        ...     if context.get_model_state("is_speech"):
        ...         print("Speech detected!")
    """

    def __init__(
        self,
        model: "SileroVAD",
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.1,
    ) -> None:
        self._model = model
        self._threshold = threshold
        self._min_speech_duration = min_speech_duration
        self._min_silence_duration = min_silence_duration

        # Get chunk size from model config
        self._chunk_samples = model.config.window_size_samples
        self._sample_rate_val = model.config.sample_rate

    def get_chunk_size(self) -> int:
        """Get the required input chunk size in samples."""
        return self._chunk_samples

    def get_overlap_size(self) -> int:
        """Get the required overlap between chunks in samples.

        VAD doesn't need overlap - each chunk is independent.
        """
        return 0

    def initialize_context(self, sample_rate: int) -> StreamingContext:
        """Initialize streaming context with VAD state.

        Args:
            sample_rate: Audio sample rate (should match model's expected rate)

        Returns:
            Initialized streaming context
        """
        if sample_rate != self._sample_rate_val:
            import warnings

            warnings.warn(
                f"Sample rate {sample_rate} differs from model's expected rate "
                f"{self._sample_rate_val}. Consider resampling input audio.",
                stacklevel=2,
            )

        ctx = StreamingContext(sample_rate=sample_rate)

        # Initialize VAD-specific state
        ctx.update_model_state("vad_model_state", None)  # LSTM state
        ctx.update_model_state("is_speech", False)
        ctx.update_model_state("speech_prob", 0.0)
        ctx.update_model_state("in_speech", False)
        ctx.update_model_state("speech_start", None)
        ctx.update_model_state("silence_start", None)
        ctx.update_model_state("segments", [])  # List of (start, end) tuples

        return ctx

    def process_chunk(
        self,
        audio: mx.array,
        context: StreamingContext,
    ) -> mx.array:
        """Process audio chunk through VAD.

        Updates context with speech detection state. The audio is passed
        through unchanged - VAD is a detector, not a transformer.

        Args:
            audio: Input audio chunk [channels, samples] or [samples]
            context: Streaming context (modified in place)

        Returns:
            Input audio unchanged
        """
        # Handle channel dimension - VAD expects mono
        if audio.ndim == 2:
            # Average channels if stereo
            audio_mono = mx.mean(audio, axis=0)
        else:
            audio_mono = audio

        # Get VAD model state
        vad_state = context.get_model_state("vad_model_state")

        # Run VAD model
        prob, new_vad_state = self._model(audio_mono, state=vad_state)

        # Update model state
        context.update_model_state("vad_model_state", new_vad_state)

        # Get probability as float
        prob_val = float(prob)
        is_speech = prob_val > self._threshold

        # Update context with current frame state
        context.update_model_state("speech_prob", prob_val)
        context.update_model_state("is_speech", is_speech)

        # Track speech segments
        current_time = context.position / context.sample_rate
        in_speech = context.get_model_state("in_speech", False)
        segments = context.get_model_state("segments", [])

        window_duration = self._chunk_samples / self._sample_rate_val

        if is_speech and not in_speech:
            # Speech onset
            context.update_model_state("in_speech", True)
            context.update_model_state("speech_start", current_time)
            context.update_model_state("silence_start", None)

        elif not is_speech and in_speech:
            # Potential speech offset - track silence duration
            silence_start = context.get_model_state("silence_start")
            if silence_start is None:
                context.update_model_state("silence_start", current_time)
            else:
                silence_duration = current_time - silence_start + window_duration
                if silence_duration >= self._min_silence_duration:
                    # End speech segment
                    speech_start = context.get_model_state("speech_start")
                    speech_duration = silence_start - speech_start
                    if speech_duration >= self._min_speech_duration:
                        segments.append((speech_start, silence_start))
                        context.update_model_state("segments", segments)

                    context.update_model_state("in_speech", False)
                    context.update_model_state("speech_start", None)
                    context.update_model_state("silence_start", None)

        elif is_speech and in_speech:
            # Continuing speech - reset silence tracking
            context.update_model_state("silence_start", None)

        # Advance context
        context.advance(audio.shape[-1])

        # Return audio unchanged
        return audio

    def finalize(self, context: StreamingContext) -> mx.array | None:
        """Finalize VAD processing and close any open speech segment.

        Args:
            context: The streaming context

        Returns:
            None (VAD doesn't produce additional audio)
        """
        # Close any open speech segment
        in_speech = context.get_model_state("in_speech", False)
        if in_speech:
            speech_start = context.get_model_state("speech_start")
            current_time = context.position / context.sample_rate

            if speech_start is not None:
                speech_duration = current_time - speech_start
                if speech_duration >= self._min_speech_duration:
                    segments = context.get_model_state("segments", [])
                    segments.append((speech_start, current_time))
                    context.update_model_state("segments", segments)

        return None

    @property
    def input_channels(self) -> int:
        """Expected number of input channels (mono)."""
        return 1

    @property
    def output_channels(self) -> int:
        """Number of output channels (same as input)."""
        return 1

    @property
    def sample_rate(self) -> int:
        """Expected sample rate."""
        return self._sample_rate_val

    @property
    def latency_samples(self) -> int:
        """Processing latency in samples.

        VAD has minimal latency - just the window size.
        """
        return self._chunk_samples

    @property
    def threshold(self) -> float:
        """Current speech probability threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set speech probability threshold."""
        self._threshold = value

    @property
    def model(self) -> "SileroVAD":
        """The underlying SileroVAD model."""
        return self._model


__all__ = ["VADStreamProcessor"]
