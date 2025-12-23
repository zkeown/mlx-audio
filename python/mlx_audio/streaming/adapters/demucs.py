"""Streaming adapter for HTDemucs source separation model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import mlx.core as mx

from mlx_audio.streaming.context import StreamingContext
from mlx_audio.streaming.processor import StreamProcessor

if TYPE_CHECKING:
    from mlx_audio.models.demucs.bag import BagOfModels
    from mlx_audio.models.demucs.model import HTDemucs

# Type alias for model parameter
ModelType = Union["HTDemucs", "BagOfModels"]


def _create_weight_window(length: int, power: float = 1.0) -> mx.array:
    """Create triangular weight window for overlap-add.

    This matches the implementation in models/demucs/inference.py.

    Args:
        length: Window length in samples
        power: Power to raise the window to (default: 1.0)

    Returns:
        Weight window array
    """
    half = length // 2
    ramp_up = mx.arange(1, half + 1, dtype=mx.float32)
    ramp_down = mx.arange(length - half, 0, -1, dtype=mx.float32)
    weight = mx.concatenate([ramp_up, ramp_down])
    weight = (weight / mx.max(weight)) ** power
    return weight


class HTDemucsStreamProcessor(StreamProcessor):
    """Streaming processor for HTDemucs source separation.

    Wraps an HTDemucs model or BagOfModels ensemble for chunk-by-chunk
    processing with overlap-add blending, enabling real-time source separation.

    The implementation adapts the chunked inference logic from
    models/demucs/inference.py for streaming use.

    Args:
        model: HTDemucs model instance or BagOfModels ensemble
        segment: Segment duration in seconds (default: from model config)
        overlap: Overlap ratio between segments (default: 0.25)

    Example:
        >>> model = HTDemucs.from_pretrained("mlx-community/htdemucs-ft")
        >>> processor = HTDemucsStreamProcessor(model, overlap=0.25)
        >>> context = processor.initialize_context(44100)
        >>> for chunk in audio_chunks:
        ...     output = processor.process_chunk(chunk, context)
        ...     # output has shape [sources, channels, stride_samples]

        >>> # Or with BagOfModels for better quality:
        >>> bag = BagOfModels.from_pretrained("path/to/htdemucs_ft_bag")
        >>> processor = HTDemucsStreamProcessor(bag, overlap=0.25)
    """

    def __init__(
        self,
        model: ModelType,
        segment: float | None = None,
        overlap: float = 0.25,
    ) -> None:
        self._model = model
        self._segment = segment if segment is not None else model.config.segment
        self._overlap = overlap

        # Compute sizes based on model config
        sr = model.config.samplerate
        self._chunk_samples = int(self._segment * sr)
        self._overlap_samples = int(self._chunk_samples * overlap)
        self._stride = self._chunk_samples - self._overlap_samples

        # Pre-compute weight window
        self._weight = _create_weight_window(self._chunk_samples)

        # Model properties
        self._num_sources = model.config.num_sources
        self._sample_rate = sr

    def get_chunk_size(self) -> int:
        """Get the required input chunk size in samples."""
        return self._chunk_samples

    def get_overlap_size(self) -> int:
        """Get the required overlap between chunks in samples."""
        return self._overlap_samples

    def initialize_context(self, sample_rate: int) -> StreamingContext:
        """Initialize streaming context with overlap-add buffers.

        Args:
            sample_rate: Audio sample rate (should match model's expected rate)

        Returns:
            Initialized streaming context
        """
        if sample_rate != self._sample_rate:
            import warnings

            warnings.warn(
                f"Sample rate {sample_rate} differs from model's expected rate "
                f"{self._sample_rate}. Consider resampling input audio.",
                stacklevel=2,
            )

        ctx = StreamingContext(sample_rate=sample_rate)

        # Initialize overlap-add state
        # We'll store partial results that need more chunks to complete
        ctx.update_model_state("pending_output", None)
        ctx.update_model_state("pending_weight", None)
        ctx.update_model_state("first_chunk", True)

        return ctx

    def process_chunk(
        self,
        audio: mx.array,
        context: StreamingContext,
    ) -> mx.array:
        """Process a chunk through HTDemucs with overlap-add blending.

        Uses the same overlap-add algorithm as batch inference (apply_model)
        to ensure identical outputs between streaming and batch modes.

        The algorithm maintains an accumulation buffer for positions that
        need contributions from multiple chunks. Each call outputs exactly
        `stride` samples (except possibly the last chunk).

        Chunk layout (overlap=25%, stride=75% of chunk):
        - Chunk 0: [0, chunk) -> output [0, stride), buffer [stride, chunk)
        - Chunk N: blend [0, overlap) with buffer, output stride, buffer new

        Args:
            audio: Input audio chunk [channels, samples] or [batch, C, T]
            context: Streaming context (modified in place)

        Returns:
            Separated stems for the stride portion that's now complete.
            Shape: [sources, channels, stride_samples] or
                   [batch, sources, channels, stride_samples]
        """
        # Add batch dimension if needed
        if audio.ndim == 2:
            audio = audio[None, ...]
            squeeze_batch = True
        else:
            squeeze_batch = False

        B, C, T = audio.shape
        S = self._num_sources

        # Pad if chunk is shorter than expected
        if T < self._chunk_samples:
            pad_amount = self._chunk_samples - T
            audio = mx.pad(audio, [(0, 0), (0, 0), (0, pad_amount)])
            actual_len = T
        else:
            actual_len = self._chunk_samples

        # Run model
        stems = self._model(audio)  # [B, S, C, chunk_samples]

        # Trim to actual length if we padded
        if actual_len < self._chunk_samples:
            stems = stems[:, :, :, :actual_len]
            chunk_weight = self._weight[:actual_len]
        else:
            chunk_weight = self._weight

        # Reshape weight for broadcasting: [1, 1, 1, chunk_len]
        weight_4d = chunk_weight.reshape(1, 1, 1, -1)

        # Apply weight to current chunk
        weighted_stems = stems * weight_4d
        weight_expanded = mx.broadcast_to(weight_4d, weighted_stems.shape)

        # Get accumulated buffer from context
        acc_output = context.get_model_state("acc_output")
        acc_weight = context.get_model_state("acc_weight")
        is_first = context.get_model_state("first_chunk", True)

        if is_first:
            context.update_model_state("first_chunk", False)

            # First chunk: output stride, buffer overlap for next
            output_end = min(self._stride, actual_len)
            stride_out = stems[:, :, :, :output_end]

            # Buffer the overlap region [stride:chunk] for blending with next
            if self._overlap_samples > 0 and actual_len > self._stride:
                context.update_model_state(
                    "acc_output",
                    weighted_stems[:, :, :, self._stride:],
                )
                context.update_model_state(
                    "acc_weight",
                    weight_expanded[:, :, :, self._stride:],
                )
        else:
            if acc_output is not None and acc_weight is not None:
                overlap_len = acc_output.shape[-1]

                # Blend the overlap region with current chunk's start
                current_overlap = weighted_stems[:, :, :, :overlap_len]
                current_weight = weight_expanded[:, :, :, :overlap_len]

                total_output = acc_output + current_overlap
                total_weight = acc_weight + current_weight
                blended = total_output / mx.maximum(total_weight, 1e-8)

                # The middle portion [overlap:stride] is ready (no further blend)
                middle_start = overlap_len
                middle_end = min(overlap_len + (self._stride - overlap_len), actual_len)
                if middle_end > middle_start:
                    middle = stems[:, :, :, middle_start:middle_end]
                    stride_out = mx.concatenate([blended, middle], axis=-1)
                else:
                    stride_out = blended

                # Buffer the new overlap [stride:chunk] for next chunk
                new_buffer_start = self._stride
                if new_buffer_start < actual_len:
                    context.update_model_state(
                        "acc_output",
                        weighted_stems[:, :, :, new_buffer_start:],
                    )
                    context.update_model_state(
                        "acc_weight",
                        weight_expanded[:, :, :, new_buffer_start:],
                    )
                else:
                    context.update_model_state("acc_output", None)
                    context.update_model_state("acc_weight", None)
            else:
                # No prior accumulator
                output_end = min(self._stride, actual_len)
                stride_out = stems[:, :, :, :output_end]

        # Ensure output is evaluated to prevent memory buildup
        mx.eval(stride_out)

        # Update context
        context.advance(stride_out.shape[-1])

        if squeeze_batch:
            stride_out = stride_out[0]

        return stride_out

    def finalize(self, context: StreamingContext) -> mx.array | None:
        """Flush any remaining audio in the overlap buffer.

        Args:
            context: The streaming context

        Returns:
            Remaining audio from overlap buffer, or None if empty.
            Shape: [sources, channels, remaining_samples] or
                   [batch, sources, channels, remaining_samples]
        """
        acc_output = context.get_model_state("acc_output")
        acc_weight = context.get_model_state("acc_weight")

        if acc_output is None or acc_weight is None:
            return None

        # Normalize the accumulated output
        output = acc_output / mx.maximum(acc_weight, 1e-8)

        # Clear state
        context.update_model_state("acc_output", None)
        context.update_model_state("acc_weight", None)

        # Remove batch dimension if original input was unbatched
        # Check if we're in unbatched mode by looking at shape
        if output.ndim == 4 and output.shape[0] == 1:
            output = output[0]

        return output

    @property
    def input_channels(self) -> int:
        """Expected number of input channels (stereo)."""
        return 2

    @property
    def output_channels(self) -> int:
        """Number of output channels per source."""
        return 2

    @property
    def num_sources(self) -> int:
        """Number of separated sources."""
        return self._num_sources

    @property
    def sample_rate(self) -> int:
        """Expected sample rate."""
        return self._sample_rate

    @property
    def latency_samples(self) -> int:
        """Processing latency in samples.

        Due to overlap-add, there's inherent latency of one overlap period.
        """
        return self._overlap_samples

    @property
    def segment(self) -> float:
        """Segment duration in seconds."""
        return self._segment

    @property
    def overlap(self) -> float:
        """Overlap ratio between segments."""
        return self._overlap

    @property
    def stride(self) -> int:
        """Stride in samples (chunk size - overlap)."""
        return self._stride

    @property
    def model(self) -> ModelType:
        """The underlying HTDemucs model or BagOfModels ensemble."""
        return self._model
