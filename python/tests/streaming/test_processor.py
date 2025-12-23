"""Tests for StreamProcessor and related classes."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_audio.streaming.context import StreamingContext
from mlx_audio.streaming.processor import (
    GainProcessor,
    IdentityProcessor,
    Streamable,
    StreamProcessor,
)


class TestStreamProcessor:
    """Tests for StreamProcessor ABC."""

    def test_cannot_instantiate_abstract(self):
        """Test that StreamProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StreamProcessor()

    def test_concrete_implementation(self):
        """Test that concrete implementations work."""

        class SimpleProcessor(StreamProcessor):
            def process_chunk(self, audio, context):
                return audio

            def get_chunk_size(self):
                return 1024

            def get_overlap_size(self):
                return 0

        proc = SimpleProcessor()
        assert proc.get_chunk_size() == 1024
        assert proc.get_overlap_size() == 0

    def test_default_properties(self):
        """Test default property values."""

        class MinimalProcessor(StreamProcessor):
            def process_chunk(self, audio, context):
                return audio

            def get_chunk_size(self):
                return 512

            def get_overlap_size(self):
                return 0

        proc = MinimalProcessor()
        assert proc.input_channels == 2
        assert proc.output_channels == 2
        assert proc.sample_rate is None
        assert proc.latency_samples == 0

    def test_initialize_context(self):
        """Test context initialization."""

        class TestProcessor(StreamProcessor):
            def process_chunk(self, audio, context):
                return audio

            def get_chunk_size(self):
                return 1024

            def get_overlap_size(self):
                return 128

        proc = TestProcessor()
        ctx = proc.initialize_context(sample_rate=44100)

        assert isinstance(ctx, StreamingContext)
        assert ctx.sample_rate == 44100
        assert ctx.position == 0

    def test_finalize_default(self):
        """Test default finalize returns None."""

        class TestProcessor(StreamProcessor):
            def process_chunk(self, audio, context):
                return audio

            def get_chunk_size(self):
                return 1024

            def get_overlap_size(self):
                return 0

        proc = TestProcessor()
        ctx = proc.initialize_context(44100)

        result = proc.finalize(ctx)
        assert result is None


class TestIdentityProcessor:
    """Tests for IdentityProcessor."""

    def test_pass_through(self):
        """Test that audio passes through unchanged."""
        proc = IdentityProcessor(chunk_size=512, channels=2)
        ctx = proc.initialize_context(44100)

        audio = mx.random.uniform(shape=(2, 512))
        result = proc.process_chunk(audio, ctx)

        assert result.shape == audio.shape
        # Identity should be exact
        assert mx.allclose(result, audio, atol=1e-7)

    def test_advances_context(self):
        """Test that context is advanced."""
        proc = IdentityProcessor(chunk_size=512)
        ctx = proc.initialize_context(44100)

        audio = mx.ones((2, 512))
        proc.process_chunk(audio, ctx)

        assert ctx.position == 512
        assert ctx.chunk_counter == 1

    def test_properties(self):
        """Test processor properties."""
        proc = IdentityProcessor(chunk_size=1024, channels=1)

        assert proc.get_chunk_size() == 1024
        assert proc.get_overlap_size() == 0
        assert proc.input_channels == 1
        assert proc.output_channels == 1


class TestGainProcessor:
    """Tests for GainProcessor."""

    def test_unity_gain(self):
        """Test unity gain (1.0) passes audio unchanged."""
        proc = GainProcessor(gain=1.0, chunk_size=512)
        ctx = proc.initialize_context(44100)

        audio = mx.random.uniform(shape=(2, 512))
        result = proc.process_chunk(audio, ctx)

        # Unity gain should be exact
        assert mx.allclose(result, audio, atol=1e-7)

    def test_gain_reduction(self):
        """Test gain reduction."""
        proc = GainProcessor(gain=0.5, chunk_size=512)
        ctx = proc.initialize_context(44100)

        audio = mx.ones((2, 512))
        result = proc.process_chunk(audio, ctx)

        expected = audio * 0.5
        # Simple multiplication should be exact
        assert mx.allclose(result, expected, atol=1e-7)

    def test_gain_boost(self):
        """Test gain boost."""
        proc = GainProcessor(gain=2.0, chunk_size=512)
        ctx = proc.initialize_context(44100)

        audio = mx.full((2, 512), vals=0.25)
        result = proc.process_chunk(audio, ctx)

        expected = audio * 2.0
        # Simple multiplication should be exact
        assert mx.allclose(result, expected, atol=1e-7)

    def test_gain_setter(self):
        """Test gain can be changed dynamically."""
        proc = GainProcessor(gain=1.0, chunk_size=512)
        ctx = proc.initialize_context(44100)

        audio = mx.ones((2, 512))

        # Initial gain
        result1 = proc.process_chunk(audio, ctx)
        assert mx.allclose(result1, audio, atol=1e-7)

        # Change gain
        proc.gain = 0.5
        assert proc.gain == 0.5

        result2 = proc.process_chunk(audio, ctx)
        assert mx.allclose(result2, audio * 0.5, atol=1e-7)


class TestStreamableProtocol:
    """Tests for Streamable protocol."""

    def test_protocol_compliance(self):
        """Test that compliant objects pass isinstance check."""

        class MyStreamable:
            def get_chunk_size(self) -> int:
                return 1024

            def get_overlap_size(self) -> int:
                return 128

            def process_chunk(self, audio, context):
                return audio

        obj = MyStreamable()
        assert isinstance(obj, Streamable)

    def test_protocol_non_compliance(self):
        """Test that non-compliant objects fail isinstance check."""

        class NotStreamable:
            def some_method(self):
                pass

        obj = NotStreamable()
        assert not isinstance(obj, Streamable)

    def test_streamprocessor_is_streamable(self):
        """Test that StreamProcessor subclasses are Streamable."""
        proc = IdentityProcessor()
        assert isinstance(proc, Streamable)


class TestCustomProcessor:
    """Tests for custom processor implementations."""

    def test_stateful_processor(self):
        """Test a processor that maintains state."""

        class SmoothingProcessor(StreamProcessor):
            """Simple exponential smoothing processor."""

            def __init__(self, alpha: float = 0.1):
                self.alpha = alpha

            def process_chunk(self, audio, context):
                prev = context.get_model_state("prev_sample")
                if prev is None:
                    prev = mx.zeros_like(audio[:, :1])

                # Simple smoothing (just for testing)
                result = self.alpha * audio + (1 - self.alpha) * prev
                context.update_model_state("prev_sample", result[:, -1:])
                context.advance(audio.shape[-1])
                return result

            def get_chunk_size(self):
                return 256

            def get_overlap_size(self):
                return 0

        proc = SmoothingProcessor(alpha=0.5)
        ctx = proc.initialize_context(44100)

        # Process first chunk
        audio1 = mx.ones((2, 256))
        result1 = proc.process_chunk(audio1, ctx)

        # State should be updated
        prev = ctx.get_model_state("prev_sample")
        assert prev is not None

        # Process second chunk with different values
        audio2 = mx.zeros((2, 256))
        result2 = proc.process_chunk(audio2, ctx)

        # Result should be smoothed (not all zeros due to previous state)
        assert not mx.allclose(result2, mx.zeros_like(result2))
