"""Benchmarks for mlx_audio streaming and data loading.

Measures performance of ring buffers, data loaders, and streaming pipelines.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float  # ops/sec or samples/sec
    peak_memory_mb: float
    iterations: int
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "throughput": self.throughput,
            "peak_memory_mb": self.peak_memory_mb,
            "iterations": self.iterations,
            "params": self.params,
        }


class BenchStreaming:
    """Benchmarks for streaming and data loading components."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.results: list[BenchmarkResult] = []

    def bench_ring_buffer_write(
        self,
        buffer_size: int = 16000,
        chunk_size: int = 1024,
        n_channels: int = 2,
        iterations: int = 1000,
    ) -> BenchmarkResult:
        """Benchmark ring buffer write operations."""
        from mlx_audio.streaming.buffer import AudioRingBuffer

        buffer = AudioRingBuffer(max_samples=buffer_size, channels=n_channels)
        chunk = np.random.randn(n_channels, chunk_size).astype(np.float32)

        # Warmup
        for _ in range(10):
            if buffer.space >= chunk_size:
                buffer.write(chunk)
            else:
                buffer.read(chunk_size)

        buffer.clear()

        times = []
        writes_completed = 0

        for _ in range(iterations):
            # Ensure space available
            if buffer.space < chunk_size:
                buffer.read(min(chunk_size, buffer.available))

            start = time.perf_counter()
            buffer.write(chunk)
            end = time.perf_counter()
            times.append(end - start)
            writes_completed += 1

        mean_time = np.mean(times)
        samples_per_write = chunk_size * n_channels
        throughput = samples_per_write / mean_time if mean_time > 0 else 0

        result = BenchmarkResult(
            name="ring_buffer_write",
            mean_time_ms=mean_time * 1000,
            std_time_ms=np.std(times) * 1000,
            min_time_ms=np.min(times) * 1000,
            max_time_ms=np.max(times) * 1000,
            throughput=throughput,
            peak_memory_mb=0.0,
            iterations=writes_completed,
            params={
                "buffer_size": buffer_size,
                "chunk_size": chunk_size,
                "n_channels": n_channels,
            },
        )
        self.results.append(result)
        return result

    def bench_ring_buffer_read(
        self,
        buffer_size: int = 16000,
        chunk_size: int = 1024,
        n_channels: int = 2,
        iterations: int = 1000,
    ) -> BenchmarkResult:
        """Benchmark ring buffer read operations."""
        from mlx_audio.streaming.buffer import AudioRingBuffer

        buffer = AudioRingBuffer(max_samples=buffer_size, channels=n_channels)

        # Pre-fill buffer
        chunk = np.random.randn(n_channels, chunk_size).astype(np.float32)
        while buffer.space >= chunk_size:
            buffer.write(chunk)

        times = []
        reads_completed = 0

        for _ in range(iterations):
            # Ensure data available
            if buffer.available < chunk_size:
                # Refill
                while buffer.space >= chunk_size:
                    buffer.write(chunk)

            start = time.perf_counter()
            _ = buffer.read(chunk_size)
            end = time.perf_counter()
            times.append(end - start)
            reads_completed += 1

        mean_time = np.mean(times)
        samples_per_read = chunk_size * n_channels
        throughput = samples_per_read / mean_time if mean_time > 0 else 0

        result = BenchmarkResult(
            name="ring_buffer_read",
            mean_time_ms=mean_time * 1000,
            std_time_ms=np.std(times) * 1000,
            min_time_ms=np.min(times) * 1000,
            max_time_ms=np.max(times) * 1000,
            throughput=throughput,
            peak_memory_mb=0.0,
            iterations=reads_completed,
            params={
                "buffer_size": buffer_size,
                "chunk_size": chunk_size,
                "n_channels": n_channels,
            },
        )
        self.results.append(result)
        return result

    def bench_ring_buffer_concurrent(
        self,
        buffer_size: int = 16000,
        chunk_size: int = 1024,
        n_channels: int = 2,
        duration_sec: float = 1.0,
    ) -> BenchmarkResult:
        """Benchmark concurrent read/write on ring buffer."""
        from mlx_audio.streaming.buffer import AudioRingBuffer

        buffer = AudioRingBuffer(max_samples=buffer_size, channels=n_channels)
        chunk = np.random.randn(n_channels, chunk_size).astype(np.float32)

        write_times: list[float] = []
        read_times: list[float] = []
        stop_event = threading.Event()

        def writer():
            while not stop_event.is_set():
                if buffer.space >= chunk_size:
                    start = time.perf_counter()
                    buffer.write(chunk)
                    write_times.append(time.perf_counter() - start)
                else:
                    time.sleep(0.0001)

        def reader():
            while not stop_event.is_set():
                if buffer.available >= chunk_size:
                    start = time.perf_counter()
                    buffer.read(chunk_size)
                    read_times.append(time.perf_counter() - start)
                else:
                    time.sleep(0.0001)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        time.sleep(duration_sec)
        stop_event.set()

        writer_thread.join()
        reader_thread.join()

        all_times = write_times + read_times
        if not all_times:
            all_times = [0.0]

        mean_time = np.mean(all_times)
        total_ops = len(write_times) + len(read_times)
        throughput = total_ops / duration_sec

        result = BenchmarkResult(
            name="ring_buffer_concurrent",
            mean_time_ms=mean_time * 1000,
            std_time_ms=np.std(all_times) * 1000,
            min_time_ms=np.min(all_times) * 1000,
            max_time_ms=np.max(all_times) * 1000,
            throughput=throughput,
            peak_memory_mb=0.0,
            iterations=total_ops,
            params={
                "buffer_size": buffer_size,
                "chunk_size": chunk_size,
                "n_channels": n_channels,
                "duration_sec": duration_sec,
                "writes": len(write_times),
                "reads": len(read_times),
            },
        )
        self.results.append(result)
        return result

    def run_all(
        self,
        buffer_sizes: list[int] | None = None,
        chunk_sizes: list[int] | None = None,
        iterations: int = 1000,
    ) -> list[BenchmarkResult]:
        """Run all streaming benchmarks."""
        if buffer_sizes is None:
            buffer_sizes = [8000, 16000, 48000]
        if chunk_sizes is None:
            chunk_sizes = [256, 512, 1024, 2048]

        for buffer_size in buffer_sizes:
            for chunk_size in chunk_sizes:
                if chunk_size <= buffer_size:
                    self.bench_ring_buffer_write(
                        buffer_size=buffer_size,
                        chunk_size=chunk_size,
                        iterations=iterations,
                    )
                    self.bench_ring_buffer_read(
                        buffer_size=buffer_size,
                        chunk_size=chunk_size,
                        iterations=iterations,
                    )

        # Concurrent benchmark with default settings
        self.bench_ring_buffer_concurrent(duration_sec=2.0)

        return self.results

    def summary(self) -> str:
        """Generate a summary of benchmark results."""
        lines = ["=" * 80, "Streaming Benchmark Summary", "=" * 80, ""]

        # Group by name
        by_name: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            by_name.setdefault(r.name, []).append(r)

        for name, results in by_name.items():
            lines.append(f"\n{name.upper()}")
            lines.append("-" * 40)
            header = f"{'BufSize':>10} {'Chunk':>8} {'Mean(us)':>10}"
            header += f" {'Std(us)':>10} {'Throughput':>15}"
            lines.append(header)

            for r in results:
                line = f"{r.params.get('buffer_size', 0):>10} "
                line += f"{r.params.get('chunk_size', 0):>8} "
                line += f"{r.mean_time_ms * 1000:>10.2f} "
                line += f"{r.std_time_ms * 1000:>10.2f} "
                line += f"{r.throughput:>12.0f} ops/s"
                lines.append(line)

        return "\n".join(lines)


if __name__ == "__main__":
    bench = BenchStreaming()
    bench.run_all(buffer_sizes=[16000], chunk_sizes=[512, 1024], iterations=500)
    print(bench.summary())
