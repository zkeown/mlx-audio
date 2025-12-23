"""Integration tests for StreamingPipeline."""

from __future__ import annotations

import mlx.core as mx

from mlx_audio.streaming import (
    AudioRingBuffer,
    CallbackSink,
    CallbackSource,
    GainProcessor,
    IdentityProcessor,
    StreamingContext,
    StreamProcessor,
)


class TestCallbackSourceSink:
    """Tests for callback-based sources and sinks."""

    def test_callback_source_push_read(self):
        """Test pushing and reading from callback source."""
        source = CallbackSource(sample_rate=44100, channels=2)
        source.start()

        # Push some audio
        audio = mx.ones((2, 1000))
        assert source.push(audio)

        # Read it back
        result = source.read(1000)
        assert result is not None
        assert result.shape == (2, 1000)

        source.stop()

    def test_callback_sink_write_pull(self):
        """Test writing and pulling from callback sink."""
        sink = CallbackSink(sample_rate=44100, channels=2)
        sink.start()

        # Write some audio
        audio = mx.ones((2, 1000))
        assert sink.write(audio)

        # Pull it back
        result = sink.pull(1000)
        assert result is not None
        # Pull returns [samples, channels]
        assert result.shape == (1000, 2)

        sink.stop()


class TestProcessorWithContext:
    """Test processors with streaming context."""

    def test_identity_processor_stream(self):
        """Test streaming through identity processor."""
        proc = IdentityProcessor(chunk_size=512, channels=2)
        ctx = proc.initialize_context(44100)

        # Process multiple chunks
        for i in range(10):
            audio = mx.ones((2, 512))
            result = proc.process_chunk(audio, ctx)
            assert result.shape == (2, 512)

        # Context should track progress
        assert ctx.position == 512 * 10
        assert ctx.chunk_counter == 10

    def test_gain_processor_stream(self):
        """Test streaming through gain processor."""
        proc = GainProcessor(gain=0.5, chunk_size=256, channels=2)
        ctx = proc.initialize_context(44100)

        audio = mx.ones((2, 256))
        result = proc.process_chunk(audio, ctx)

        # Should be half amplitude (exact multiplication)
        assert mx.allclose(result, mx.full((2, 256), vals=0.5), atol=1e-7)

    def test_context_checkpoint_during_stream(self):
        """Test checkpointing context mid-stream."""
        proc = IdentityProcessor(chunk_size=512)
        ctx = proc.initialize_context(44100)

        # Process some chunks
        for _ in range(5):
            proc.process_chunk(mx.ones((2, 512)), ctx)

        # Checkpoint
        checkpoint = ctx.checkpoint()

        # Continue processing
        for _ in range(5):
            proc.process_chunk(mx.ones((2, 512)), ctx)

        # Restore from checkpoint
        ctx2 = StreamingContext.from_checkpoint(checkpoint)
        assert ctx2.position == 512 * 5
        assert ctx2.chunk_counter == 5


class TestBufferIntegration:
    """Test buffer integration with processors."""

    def test_buffer_to_processor_flow(self):
        """Test flowing audio through buffer to processor."""
        buffer = AudioRingBuffer(max_samples=10000, channels=2)
        proc = IdentityProcessor(chunk_size=512)
        ctx = proc.initialize_context(44100)

        # Write to buffer
        buffer.write(mx.ones((2, 2048)))

        # Read and process in chunks
        outputs = []
        while buffer.available >= 512:
            chunk = buffer.read(512)
            if chunk is not None:
                output = proc.process_chunk(chunk, ctx)
                outputs.append(output)

        # Should have processed 4 chunks
        assert len(outputs) == 4

    def test_sequential_produce_consume(self):
        """Test sequential produce-consume pattern with buffer.

        Note: MLX has threading limitations, so we test sequentially.
        """
        buffer = AudioRingBuffer(max_samples=5000, channels=2)
        proc = GainProcessor(gain=2.0, chunk_size=256)
        ctx = proc.initialize_context(44100)

        num_chunks = 10

        # Produce all chunks first
        for _ in range(num_chunks):
            audio = mx.full((2, 256), vals=0.25)
            buffer.write(audio, timeout=0.01)

        # Consume all chunks
        consumed = 0
        while buffer.available >= 256:
            chunk = buffer.read(256, timeout=0.01)
            if chunk is not None:
                output = proc.process_chunk(chunk, ctx)
                # Verify gain was applied (exact multiplication)
                assert mx.allclose(output, mx.full((2, 256), vals=0.5), atol=1e-7)
                consumed += 1
            else:
                break

        assert consumed == num_chunks


class TestCustomProcessor:
    """Tests for custom processor implementations."""

    def test_stateful_processor(self):
        """Test a processor that maintains state."""

        class CountingProcessor(StreamProcessor):
            """Counts samples processed."""

            def __init__(self):
                self.total_samples = 0

            def process_chunk(self, audio, context):
                self.total_samples += audio.shape[-1]
                context.advance(audio.shape[-1])
                return audio

            def get_chunk_size(self):
                return 256

            def get_overlap_size(self):
                return 0

        proc = CountingProcessor()
        ctx = proc.initialize_context(44100)

        for _ in range(10):
            proc.process_chunk(mx.ones((2, 256)), ctx)

        assert proc.total_samples == 256 * 10

    def test_filter_state_processor(self):
        """Test a processor that uses filter state."""

        class FilterStateProcessor(StreamProcessor):
            """Uses filter state for continuity."""

            def process_chunk(self, audio, context):
                # Get previous state
                prev = context.get_filter_state("accumulator")
                if prev is None:
                    prev = mx.zeros((2, 1))

                # Add previous value to first sample
                result = audio.at[:, 0].add(prev[:, 0])

                # Store last sample as state
                context.update_filter_state("accumulator", audio[:, -1:])
                context.advance(audio.shape[-1])

                return result

            def get_chunk_size(self):
                return 128

            def get_overlap_size(self):
                return 0

        proc = FilterStateProcessor()
        ctx = proc.initialize_context(44100)

        # Process first chunk
        audio1 = mx.ones((2, 128))
        _ = proc.process_chunk(audio1, ctx)

        # Check state was stored
        state = ctx.get_filter_state("accumulator")
        assert state is not None
        assert state.shape == (2, 1)

        # Process second chunk - first sample should have previous state added
        audio2 = mx.ones((2, 128))
        result2 = proc.process_chunk(audio2, ctx)

        # First sample should be 1 + 1 = 2 (exact addition)
        assert mx.allclose(result2[:, 0], mx.full((2,), vals=2.0), atol=1e-7)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_buffer_read(self):
        """Test reading from empty buffer."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)

        result = buffer.read(100, timeout=0.01)
        assert result is None

    def test_processor_with_variable_input_size(self):
        """Test processor handles different input sizes."""
        proc = IdentityProcessor(chunk_size=512)
        ctx = proc.initialize_context(44100)

        # Smaller than chunk_size should still work
        small = mx.ones((2, 256))
        result = proc.process_chunk(small, ctx)
        assert result.shape == (2, 256)

        # Larger than chunk_size should still work
        large = mx.ones((2, 1024))
        result = proc.process_chunk(large, ctx)
        assert result.shape == (2, 1024)

    def test_mono_audio_processing(self):
        """Test processing mono audio."""
        proc = IdentityProcessor(chunk_size=512, channels=1)
        ctx = proc.initialize_context(44100)

        audio = mx.ones((1, 512))
        result = proc.process_chunk(audio, ctx)

        assert result.shape == (1, 512)
        assert proc.input_channels == 1
        assert proc.output_channels == 1
