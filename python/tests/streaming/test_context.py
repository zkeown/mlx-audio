"""Tests for StreamingContext."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.streaming.context import StreamingContext


class TestStreamingContext:
    """Tests for StreamingContext."""

    def test_initialization(self):
        """Test basic initialization."""
        ctx = StreamingContext(sample_rate=44100)

        assert ctx.sample_rate == 44100
        assert ctx.position == 0
        assert ctx.chunk_counter == 0
        assert ctx.time_position == 0.0
        assert ctx.overlap_buffer is None
        assert ctx.weight_buffer is None

    def test_advance(self):
        """Test advancing the position."""
        ctx = StreamingContext(sample_rate=44100)

        ctx.advance(1000)
        assert ctx.position == 1000
        assert ctx.chunk_counter == 1
        assert abs(ctx.time_position - 1000 / 44100) < 1e-6

        ctx.advance(2000)
        assert ctx.position == 3000
        assert ctx.chunk_counter == 2

    def test_filter_state_management(self):
        """Test filter state update and retrieval."""
        ctx = StreamingContext(sample_rate=44100)

        # Initially no state
        assert ctx.get_filter_state("preemphasis") is None

        # Update state
        state = mx.array([0.5])
        ctx.update_filter_state("preemphasis", state)

        # Retrieve state
        retrieved = ctx.get_filter_state("preemphasis")
        assert retrieved is not None
        # State should be exact
        assert mx.allclose(retrieved, state, atol=1e-7)

    def test_model_state_management(self):
        """Test model state update and retrieval."""
        ctx = StreamingContext(sample_rate=44100)

        # Initially no state
        assert ctx.get_model_state("hidden") is None
        assert ctx.get_model_state("hidden", "default") == "default"

        # Update with various types
        ctx.update_model_state("array", mx.ones((10, 10)))
        ctx.update_model_state("scalar", 42)
        ctx.update_model_state("dict", {"a": 1, "b": 2})

        assert ctx.get_model_state("scalar") == 42
        assert ctx.get_model_state("dict") == {"a": 1, "b": 2}
        assert ctx.get_model_state("array").shape == (10, 10)

    def test_overlap_buffer_management(self):
        """Test overlap buffer setting."""
        ctx = StreamingContext(sample_rate=44100)

        output_buffer = mx.ones((4, 2, 1000))
        weight_buffer = mx.ones((1000,))

        ctx.set_overlap_buffer(output_buffer, weight_buffer)

        assert ctx.overlap_buffer is not None
        assert ctx.weight_buffer is not None
        assert ctx.overlap_buffer.shape == (4, 2, 1000)
        assert ctx.weight_buffer.shape == (1000,)

    def test_checkpoint_basic(self):
        """Test basic checkpoint and restore."""
        ctx = StreamingContext(sample_rate=44100)
        ctx.advance(5000)
        ctx.metadata["test_key"] = "test_value"

        checkpoint = ctx.checkpoint()

        # Verify checkpoint contents
        assert checkpoint["sample_rate"] == 44100
        assert checkpoint["position"] == 5000
        assert checkpoint["chunk_counter"] == 1
        assert checkpoint["metadata"]["test_key"] == "test_value"

        # Restore
        ctx2 = StreamingContext.from_checkpoint(checkpoint)
        assert ctx2.sample_rate == 44100
        assert ctx2.position == 5000
        assert ctx2.chunk_counter == 1
        assert ctx2.metadata["test_key"] == "test_value"

    def test_checkpoint_with_filter_states(self):
        """Test checkpoint includes filter states."""
        ctx = StreamingContext(sample_rate=44100)
        ctx.update_filter_state("preemphasis", mx.array([0.97]))
        ctx.update_filter_state("deemphasis", mx.array([0.5, 0.3]))

        checkpoint = ctx.checkpoint()
        ctx2 = StreamingContext.from_checkpoint(checkpoint)

        pre = ctx2.get_filter_state("preemphasis")
        de = ctx2.get_filter_state("deemphasis")

        assert pre is not None
        assert de is not None
        # Checkpoint state should be exact
        assert mx.allclose(pre, mx.array([0.97]), atol=1e-7)
        assert mx.allclose(de, mx.array([0.5, 0.3]), atol=1e-7)

    def test_checkpoint_with_model_states(self):
        """Test checkpoint includes model states."""
        ctx = StreamingContext(sample_rate=44100)
        ctx.update_model_state("hidden", mx.zeros((32, 64)))
        ctx.update_model_state("counter", 42)

        checkpoint = ctx.checkpoint()
        ctx2 = StreamingContext.from_checkpoint(checkpoint)

        hidden = ctx2.get_model_state("hidden")
        counter = ctx2.get_model_state("counter")

        assert hidden is not None
        assert hidden.shape == (32, 64)
        assert counter == 42

    def test_checkpoint_with_overlap_buffers(self):
        """Test checkpoint includes overlap buffers."""
        ctx = StreamingContext(sample_rate=44100)
        output_buffer = mx.arange(100, dtype=mx.float32).reshape(2, 50)
        weight_buffer = mx.ones((50,))
        ctx.set_overlap_buffer(output_buffer, weight_buffer)

        checkpoint = ctx.checkpoint()
        ctx2 = StreamingContext.from_checkpoint(checkpoint)

        assert ctx2.overlap_buffer is not None
        assert ctx2.weight_buffer is not None
        assert ctx2.overlap_buffer.shape == (2, 50)
        assert ctx2.weight_buffer.shape == (50,)

    def test_reset(self):
        """Test resetting context to initial state."""
        ctx = StreamingContext(sample_rate=44100)

        # Set up state
        ctx.advance(10000)
        ctx.update_filter_state("test", mx.ones((10,)))
        ctx.update_model_state("hidden", mx.zeros((5,)))
        ctx.set_overlap_buffer(mx.ones((2, 100)), mx.ones((100,)))
        ctx.metadata["key"] = "value"

        # Reset
        ctx.reset()

        # Verify reset
        assert ctx.position == 0
        assert ctx.chunk_counter == 0
        assert ctx.sample_rate == 44100  # Preserved
        assert ctx.get_filter_state("test") is None
        assert ctx.get_model_state("hidden") is None
        assert ctx.overlap_buffer is None
        assert ctx.weight_buffer is None
        assert "key" not in ctx.metadata

    def test_time_position_accuracy(self):
        """Test time position calculation at various sample rates."""
        for sample_rate in [8000, 16000, 22050, 44100, 48000, 96000]:
            ctx = StreamingContext(sample_rate=sample_rate)

            # Advance by exactly 1 second worth of samples
            ctx.advance(sample_rate)

            assert abs(ctx.time_position - 1.0) < 1e-9

            # Advance by 0.5 seconds
            ctx.advance(sample_rate // 2)

            assert abs(ctx.time_position - 1.5) < 1e-6

    def test_metadata(self):
        """Test metadata storage."""
        ctx = StreamingContext(sample_rate=44100)

        ctx.metadata["source"] = "microphone"
        ctx.metadata["timestamp"] = 12345.678
        ctx.metadata["config"] = {"gain": 0.5, "normalize": True}

        assert ctx.metadata["source"] == "microphone"
        assert ctx.metadata["timestamp"] == 12345.678
        assert ctx.metadata["config"]["gain"] == 0.5
