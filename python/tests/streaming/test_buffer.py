"""Tests for AudioRingBuffer."""

from __future__ import annotations

import threading
import time

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.streaming.buffer import AudioRingBuffer


class TestAudioRingBuffer:
    """Tests for AudioRingBuffer."""

    def test_basic_write_read(self):
        """Test basic write and read operations."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)

        # Write some audio
        audio = mx.ones((2, 100))
        assert buffer.write(audio)
        assert buffer.available == 100

        # Read it back
        result = buffer.read(100)
        assert result is not None
        assert result.shape == (2, 100)
        assert buffer.available == 0

    def test_variable_chunk_sizes(self):
        """Test writing small chunks and reading larger chunks."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)

        # Write multiple small chunks
        for i in range(5):
            audio = mx.full((2, 50), vals=float(i))
            assert buffer.write(audio)

        assert buffer.available == 250

        # Read larger chunk
        result = buffer.read(200)
        assert result is not None
        assert result.shape == (2, 200)
        assert buffer.available == 50

    def test_wrap_around(self):
        """Test buffer wrap-around behavior."""
        buffer = AudioRingBuffer(max_samples=100, channels=2)

        # Fill most of the buffer
        audio1 = mx.ones((2, 80))
        assert buffer.write(audio1)

        # Read some
        result1 = buffer.read(60)
        assert result1 is not None
        assert buffer.available == 20

        # Write more (will wrap around)
        audio2 = mx.full((2, 50), vals=2.0)
        assert buffer.write(audio2)
        assert buffer.available == 70

        # Read all
        result2 = buffer.read(70)
        assert result2 is not None
        assert result2.shape == (2, 70)

    def test_overlap_retention(self):
        """Test that overlap samples are retained after reads."""
        overlap = 20
        buffer = AudioRingBuffer(max_samples=1000, channels=2, overlap_samples=overlap)

        # Write audio with distinct values
        audio = mx.broadcast_to(mx.arange(100, dtype=mx.float32)[None, :], (2, 100))
        assert buffer.write(audio)

        # Read - overlap samples should be retained
        result1 = buffer.read(100)
        assert result1 is not None
        # After reading 100, we consumed 100-20=80, so 20 remain
        assert buffer.available == overlap

        # The retained samples should be the last 20 of the original
        retained = buffer.peek(overlap)
        assert retained is not None
        # Verify retained samples match the end of original audio
        expected = audio[:, -overlap:]
        # Buffer retention should be exact
        assert mx.allclose(retained, expected, atol=1e-7)

    def test_peek_does_not_consume(self):
        """Test that peek returns data without consuming it."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)

        audio = mx.ones((2, 100))
        buffer.write(audio)

        # Peek multiple times
        for _ in range(3):
            result = buffer.peek(50)
            assert result is not None
            assert result.shape == (2, 50)
            assert buffer.available == 100  # Unchanged

    def test_timeout_on_empty_read(self):
        """Test that read times out when buffer is empty."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)

        start = time.time()
        result = buffer.read(100, timeout=0.1)
        elapsed = time.time() - start

        assert result is None
        assert 0.09 <= elapsed < 0.5  # Should timeout around 0.1s

    def test_timeout_on_full_write(self):
        """Test that write times out when buffer is full."""
        buffer = AudioRingBuffer(max_samples=100, channels=2)

        # Fill the buffer
        audio = mx.ones((2, 100))
        assert buffer.write(audio)

        # Try to write more
        start = time.time()
        result = buffer.write(mx.ones((2, 10)), timeout=0.1)
        elapsed = time.time() - start

        assert result is False
        assert 0.09 <= elapsed < 0.5

    def test_shutdown_unblocks_readers(self):
        """Test that shutdown unblocks waiting readers."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)
        read_result = [None]

        def reader():
            read_result[0] = buffer.read(100, timeout=5.0)

        thread = threading.Thread(target=reader)
        thread.start()

        # Give thread time to start waiting
        time.sleep(0.1)

        # Shutdown should unblock
        buffer.shutdown()
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert read_result[0] is None

    def test_shutdown_unblocks_writers(self):
        """Test that shutdown unblocks waiting writers."""
        buffer = AudioRingBuffer(max_samples=100, channels=2)
        buffer.write(mx.ones((2, 100)))  # Fill buffer

        write_result = [None]

        def writer():
            write_result[0] = buffer.write(mx.ones((2, 10)), timeout=5.0)

        thread = threading.Thread(target=writer)
        thread.start()

        time.sleep(0.1)
        buffer.shutdown()
        thread.join(timeout=1.0)

        assert not thread.is_alive()
        assert write_result[0] is False

    def test_concurrent_read_write(self):
        """Test concurrent reading and writing."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)
        num_chunks = 100
        chunk_size = 50
        written_count = [0]
        read_count = [0]
        errors = []

        def writer():
            try:
                for i in range(num_chunks):
                    audio = mx.full((2, chunk_size), vals=float(i))
                    if buffer.write(audio, timeout=1.0):
                        written_count[0] += 1
            except Exception as e:
                errors.append(e)
            finally:
                buffer.shutdown()

        def reader():
            try:
                while True:
                    result = buffer.read(chunk_size, timeout=0.5)
                    if result is None:
                        if buffer.is_shutdown:
                            break
                        continue
                    read_count[0] += 1
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join(timeout=10.0)
        reader_thread.join(timeout=10.0)

        assert not errors, f"Errors: {errors}"
        assert written_count[0] == num_chunks
        assert read_count[0] == num_chunks

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)

        buffer.write(mx.ones((2, 500)))
        assert buffer.available == 500

        buffer.clear()
        assert buffer.available == 0

    def test_reset(self):
        """Test resetting the buffer after shutdown."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)

        buffer.write(mx.ones((2, 100)))
        buffer.shutdown()
        assert buffer.is_shutdown

        buffer.reset()
        assert not buffer.is_shutdown
        assert buffer.available == 0

        # Should work again
        assert buffer.write(mx.ones((2, 50)))
        assert buffer.available == 50

    def test_space_property(self):
        """Test the space property reports correct available space."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)

        assert buffer.space == 1000

        buffer.write(mx.ones((2, 300)))
        assert buffer.space == 700

        buffer.read(100)
        assert buffer.space == 800

    def test_mono_handling(self):
        """Test writing mono audio (1 channel)."""
        buffer = AudioRingBuffer(max_samples=1000, channels=1)

        audio = mx.ones((1, 100))
        assert buffer.write(audio)
        assert buffer.available == 100

        result = buffer.read(100)
        assert result is not None
        assert result.shape == (1, 100)

    def test_read_available(self):
        """Test reading all available samples."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)

        buffer.write(mx.ones((2, 123)))

        result = buffer.read_available()
        assert result is not None
        assert result.shape == (2, 123)
        assert buffer.available == 0

    def test_read_available_empty(self):
        """Test read_available on empty buffer."""
        buffer = AudioRingBuffer(max_samples=1000, channels=2)
        result = buffer.read_available()
        assert result is None
