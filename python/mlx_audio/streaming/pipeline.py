"""Streaming pipeline for real-time audio processing."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx

from mlx_audio.streaming._types import StreamState, StreamStats
from mlx_audio.streaming.buffer import AudioRingBuffer
from mlx_audio.streaming.context import StreamingContext
from mlx_audio.streaming.processor import StreamProcessor

if TYPE_CHECKING:
    pass


class AudioSource(Protocol):
    """Protocol for audio sources."""

    def start(self) -> None:
        """Start the source."""
        ...

    def stop(self) -> None:
        """Stop the source."""
        ...

    def read(self, num_samples: int) -> mx.array | None:
        """Read samples from the source."""
        ...

    @property
    def sample_rate(self) -> int:
        """Source sample rate."""
        ...

    @property
    def channels(self) -> int:
        """Number of channels."""
        ...


class AudioSink(Protocol):
    """Protocol for audio sinks."""

    def start(self) -> None:
        """Start the sink."""
        ...

    def stop(self) -> None:
        """Stop the sink."""
        ...

    def write(self, audio: mx.array) -> bool:
        """Write samples to the sink."""
        ...

    @property
    def sample_rate(self) -> int:
        """Expected sample rate."""
        ...


class StreamingPipeline:
    """Orchestrates real-time audio streaming through a processor.

    Connects an audio source, processor, and sink with buffering
    and threading for smooth real-time operation.

    Threading model:
    - Input thread: Reads from source, writes to input buffer
    - Process thread: Reads from input buffer, processes, writes to output buffer
    - Output thread: Reads from output buffer, writes to sink

    Args:
        source: Audio input source
        processor: Stream processor (e.g., HTDemucsStreamProcessor)
        sink: Audio output sink
        input_buffer_seconds: Input buffer duration in seconds
        output_buffer_seconds: Output buffer duration in seconds

    Example:
        >>> pipeline = StreamingPipeline(
        ...     source=FileSource("input.wav"),
        ...     processor=HTDemucsStreamProcessor(model),
        ...     sink=FileSink("output.wav", stem_index=3),
        ... )
        >>> pipeline.start()
        >>> pipeline.wait()  # Wait for completion
    """

    def __init__(
        self,
        source: AudioSource,
        processor: StreamProcessor,
        sink: AudioSink,
        *,
        input_buffer_seconds: float = 2.0,
        output_buffer_seconds: float = 2.0,
    ) -> None:
        self._source = source
        self._processor = processor
        self._sink = sink

        # Get parameters from processor
        self._chunk_size = processor.get_chunk_size()
        self._overlap_size = processor.get_overlap_size()

        # We'll initialize these on start() when we know the sample rate
        self._input_buffer: AudioRingBuffer | None = None
        self._output_buffer: AudioRingBuffer | None = None
        self._input_buffer_seconds = input_buffer_seconds
        self._output_buffer_seconds = output_buffer_seconds

        # Threading
        self._input_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        self._output_thread: threading.Thread | None = None

        # State
        self._state = StreamState.IDLE
        self._state_lock = threading.Lock()
        self._error: Exception | None = None
        self._context: StreamingContext | None = None

        # Statistics
        self._stats = StreamStats()
        self._start_time: float = 0.0

        # Events
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

    def start(self) -> None:
        """Start the streaming pipeline."""
        with self._state_lock:
            if self._state == StreamState.RUNNING:
                return
            if self._state == StreamState.PAUSED:
                self._pause_event.set()
                self._state = StreamState.RUNNING
                return

        # Start source to get sample rate
        self._source.start()
        sample_rate = self._source.sample_rate
        channels = self._source.channels

        # Initialize buffers
        input_samples = int(self._input_buffer_seconds * sample_rate)
        output_samples = int(self._output_buffer_seconds * sample_rate)

        self._input_buffer = AudioRingBuffer(
            max_samples=input_samples,
            channels=channels,
            overlap_samples=self._overlap_size,
        )

        # Output may have different channel count for separation
        output_channels = getattr(self._processor, "num_sources", 1) * channels
        self._output_buffer = AudioRingBuffer(
            max_samples=output_samples,
            channels=output_channels,
        )

        # Initialize context
        self._context = self._processor.initialize_context(sample_rate)

        # Reset state
        self._stop_event.clear()
        self._pause_event.set()
        self._error = None
        self._stats = StreamStats()
        self._start_time = time.time()

        # Start sink
        self._sink.start()

        # Start threads
        self._input_thread = threading.Thread(
            target=self._input_loop,
            name="StreamingPipeline-Input",
            daemon=True,
        )
        self._process_thread = threading.Thread(
            target=self._process_loop,
            name="StreamingPipeline-Process",
            daemon=True,
        )
        self._output_thread = threading.Thread(
            target=self._output_loop,
            name="StreamingPipeline-Output",
            daemon=True,
        )

        with self._state_lock:
            self._state = StreamState.RUNNING

        self._input_thread.start()
        self._process_thread.start()
        self._output_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the pipeline and wait for threads to finish.

        Args:
            timeout: Maximum time to wait for threads to stop
        """
        with self._state_lock:
            if self._state in (StreamState.IDLE, StreamState.STOPPED):
                return
            self._state = StreamState.STOPPED

        # Signal stop
        self._stop_event.set()
        self._pause_event.set()  # Unpause if paused

        # Shutdown buffers to unblock threads
        if self._input_buffer:
            self._input_buffer.shutdown()
        if self._output_buffer:
            self._output_buffer.shutdown()

        # Wait for threads
        for thread in [self._input_thread, self._process_thread, self._output_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=timeout / 3)

        # Stop source and sink
        self._source.stop()
        self._sink.stop()

    def pause(self) -> None:
        """Pause processing (audio continues to buffer)."""
        with self._state_lock:
            if self._state == StreamState.RUNNING:
                self._pause_event.clear()
                self._state = StreamState.PAUSED

    def resume(self) -> None:
        """Resume processing after pause."""
        with self._state_lock:
            if self._state == StreamState.PAUSED:
                self._pause_event.set()
                self._state = StreamState.RUNNING

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the pipeline to complete.

        Args:
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if completed, False if timeout
        """
        if self._input_thread:
            self._input_thread.join(timeout=timeout)
            if self._input_thread.is_alive():
                return False

        if self._process_thread:
            self._process_thread.join(timeout=timeout)
            if self._process_thread.is_alive():
                return False

        if self._output_thread:
            self._output_thread.join(timeout=timeout)
            if self._output_thread.is_alive():
                return False

        return True

    def _input_loop(self) -> None:
        """Read from source and write to input buffer."""
        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()

                # Read from source
                chunk = self._source.read(self._chunk_size)
                if chunk is None:
                    # End of source
                    break

                # Write to input buffer
                if not self._input_buffer.write(chunk, timeout=0.1):
                    if self._stop_event.is_set():
                        break
                    self._stats.buffer_overruns += 1

        except Exception as e:
            self._set_error(e)
        finally:
            # Signal end of input
            if self._input_buffer:
                self._input_buffer.shutdown()

    def _process_loop(self) -> None:
        """Read from input buffer, process, write to output buffer."""
        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()

                # Read from input buffer
                chunk = self._input_buffer.read(self._chunk_size, timeout=0.1)
                if chunk is None:
                    if self._input_buffer.is_shutdown:
                        break
                    self._stats.buffer_underruns += 1
                    continue

                # Process
                start_time = time.time()
                output = self._processor.process_chunk(chunk, self._context)
                process_time = time.time() - start_time

                # Update stats
                self._stats.chunks_processed += 1
                self._stats.samples_processed += chunk.shape[-1]
                self._stats.processing_time += process_time
                self._stats.total_duration = (
                    self._stats.samples_processed / self._source.sample_rate
                )

                # Reshape output for buffer if needed
                # [sources, channels, samples] -> [sources * channels, samples]
                if output.ndim == 3:
                    S, C, T = output.shape
                    output = output.reshape(S * C, T)

                # Write to output buffer
                if not self._output_buffer.write(output, timeout=0.1):
                    if self._stop_event.is_set():
                        break
                    self._stats.buffer_overruns += 1

            # Finalize processor
            if self._context:
                final = self._processor.finalize(self._context)
                if final is not None:
                    if final.ndim == 3:
                        S, C, T = final.shape
                        final = final.reshape(S * C, T)
                    self._output_buffer.write(final)

        except Exception as e:
            self._set_error(e)
        finally:
            # Signal end of processing
            if self._output_buffer:
                self._output_buffer.shutdown()

    def _output_loop(self) -> None:
        """Read from output buffer and write to sink."""
        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()

                # Read from output buffer
                chunk = self._output_buffer.read(self._chunk_size, timeout=0.1)
                if chunk is None:
                    if self._output_buffer.is_shutdown:
                        # Flush remaining
                        remaining = self._output_buffer.read_available()
                        if remaining is not None:
                            self._sink.write(remaining)
                        break
                    continue

                # Write to sink
                self._sink.write(chunk)

        except Exception as e:
            self._set_error(e)

    def _set_error(self, error: Exception) -> None:
        """Set error state and stop pipeline."""
        with self._state_lock:
            if self._error is None:
                self._error = error
                self._state = StreamState.ERROR

        # Trigger stop
        self._stop_event.set()
        if self._input_buffer:
            self._input_buffer.shutdown()
        if self._output_buffer:
            self._output_buffer.shutdown()

    @property
    def state(self) -> StreamState:
        """Current pipeline state."""
        with self._state_lock:
            return self._state

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is actively processing."""
        return self.state == StreamState.RUNNING

    @property
    def is_paused(self) -> bool:
        """Whether the pipeline is paused."""
        return self.state == StreamState.PAUSED

    @property
    def error(self) -> Exception | None:
        """Error that caused the pipeline to stop, if any."""
        with self._state_lock:
            return self._error

    @property
    def stats(self) -> StreamStats:
        """Current streaming statistics."""
        return self._stats

    @property
    def latency(self) -> float:
        """Estimated end-to-end latency in seconds."""
        if self._input_buffer is None:
            return 0.0

        sample_rate = self._source.sample_rate

        # Input buffer latency + processing latency + output buffer latency
        input_latency = self._input_buffer.available / sample_rate
        process_latency = self._processor.latency_samples / sample_rate
        output_latency = (
            self._output_buffer.available / sample_rate
            if self._output_buffer
            else 0.0
        )

        return input_latency + process_latency + output_latency

    @property
    def context(self) -> StreamingContext | None:
        """Current streaming context."""
        return self._context

    def get_checkpoint(self) -> dict[str, Any]:
        """Get current state for checkpointing.

        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = {
            "stats": {
                "chunks_processed": self._stats.chunks_processed,
                "samples_processed": self._stats.samples_processed,
                "total_duration": self._stats.total_duration,
                "processing_time": self._stats.processing_time,
            },
        }

        if self._context:
            checkpoint["context"] = self._context.checkpoint()

        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore from checkpoint.

        Args:
            checkpoint: Dictionary from get_checkpoint()
        """
        if "stats" in checkpoint:
            stats = checkpoint["stats"]
            self._stats.chunks_processed = stats.get("chunks_processed", 0)
            self._stats.samples_processed = stats.get("samples_processed", 0)
            self._stats.total_duration = stats.get("total_duration", 0.0)
            self._stats.processing_time = stats.get("processing_time", 0.0)

        if "context" in checkpoint and self._context:
            self._context = StreamingContext.from_checkpoint(checkpoint["context"])

    def __enter__(self) -> StreamingPipeline:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
