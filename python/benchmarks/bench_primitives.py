"""Benchmarks for mlx_audio.primitives module.

Measures performance of STFT, Mel spectrogram, MFCC, and related operations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float  # samples/sec or frames/sec
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


def _warmup_and_sync():
    """Warmup MLX and synchronize."""
    # Simple warmup operation
    x = mx.ones((100, 100))
    _ = mx.matmul(x, x)
    mx.eval(_)


def _measure_time(
    fn, warmup: int = 3, iterations: int = 10
) -> tuple[list[float], float]:
    """Measure execution time with warmup.

    Returns:
        Tuple of (list of times in seconds, peak memory in MB)
    """
    # Warmup
    for _ in range(warmup):
        result = fn()
        mx.eval(result) if isinstance(result, mx.array) else None

    # Reset memory stats if available
    try:
        mx.reset_peak_memory()
    except AttributeError:
        try:
            mx.metal.reset_peak_memory()
        except AttributeError:
            pass

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        # Force evaluation to ensure timing is accurate
        if isinstance(result, mx.array):
            mx.eval(result)
        elif isinstance(result, tuple):
            mx.eval(*[r for r in result if isinstance(r, mx.array)])
        end = time.perf_counter()
        times.append(end - start)

    # Get peak memory if available
    try:
        peak_memory_mb = mx.metal.peak_memory() / (1024 * 1024)
    except AttributeError:
        peak_memory_mb = 0.0

    return times, peak_memory_mb


class BenchPrimitives:
    """Benchmarks for audio primitives."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.results: list[BenchmarkResult] = []

    def _generate_audio(
        self, duration_sec: float, batch_size: int = 1
    ) -> mx.array:
        """Generate random audio signal for benchmarking."""
        n_samples = int(duration_sec * self.sample_rate)
        np.random.seed(42)
        audio = np.random.randn(batch_size, n_samples).astype(np.float32)
        return mx.array(audio)

    def bench_stft(
        self,
        duration_sec: float = 10.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        batch_size: int = 1,
        iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark STFT computation."""
        from mlx_audio.primitives import stft

        audio = self._generate_audio(duration_sec, batch_size)
        n_samples = audio.shape[-1]

        def run_stft():
            return stft(audio, n_fft=n_fft, hop_length=hop_length)

        times, peak_memory = _measure_time(run_stft, iterations=iterations)

        mean_time = np.mean(times)
        throughput = (n_samples * batch_size) / mean_time  # samples/sec

        result = BenchmarkResult(
            name="stft",
            mean_time_ms=mean_time * 1000,
            std_time_ms=np.std(times) * 1000,
            min_time_ms=np.min(times) * 1000,
            max_time_ms=np.max(times) * 1000,
            throughput=throughput,
            peak_memory_mb=peak_memory,
            iterations=iterations,
            params={
                "duration_sec": duration_sec,
                "n_fft": n_fft,
                "hop_length": hop_length,
                "batch_size": batch_size,
                "n_samples": n_samples,
            },
        )
        self.results.append(result)
        return result

    def bench_istft(
        self,
        duration_sec: float = 10.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        batch_size: int = 1,
        iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark ISTFT computation."""
        from mlx_audio.primitives import istft, stft

        audio = self._generate_audio(duration_sec, batch_size)
        stft_matrix = stft(audio, n_fft=n_fft, hop_length=hop_length)
        mx.eval(stft_matrix)

        n_samples = audio.shape[-1]

        def run_istft():
            return istft(stft_matrix, hop_length=hop_length, n_fft=n_fft)

        times, peak_memory = _measure_time(run_istft, iterations=iterations)

        mean_time = np.mean(times)
        throughput = (n_samples * batch_size) / mean_time

        result = BenchmarkResult(
            name="istft",
            mean_time_ms=mean_time * 1000,
            std_time_ms=np.std(times) * 1000,
            min_time_ms=np.min(times) * 1000,
            max_time_ms=np.max(times) * 1000,
            throughput=throughput,
            peak_memory_mb=peak_memory,
            iterations=iterations,
            params={
                "duration_sec": duration_sec,
                "n_fft": n_fft,
                "hop_length": hop_length,
                "batch_size": batch_size,
            },
        )
        self.results.append(result)
        return result

    def bench_melspectrogram(
        self,
        duration_sec: float = 10.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        batch_size: int = 1,
        iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark Mel spectrogram computation."""
        from mlx_audio.primitives import melspectrogram

        audio = self._generate_audio(duration_sec, batch_size)
        n_samples = audio.shape[-1]

        def run_mel():
            return melspectrogram(
                audio,
                sr=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )

        times, peak_memory = _measure_time(run_mel, iterations=iterations)

        mean_time = np.mean(times)
        throughput = (n_samples * batch_size) / mean_time

        result = BenchmarkResult(
            name="melspectrogram",
            mean_time_ms=mean_time * 1000,
            std_time_ms=np.std(times) * 1000,
            min_time_ms=np.min(times) * 1000,
            max_time_ms=np.max(times) * 1000,
            throughput=throughput,
            peak_memory_mb=peak_memory,
            iterations=iterations,
            params={
                "duration_sec": duration_sec,
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "batch_size": batch_size,
            },
        )
        self.results.append(result)
        return result

    def bench_mfcc(
        self,
        duration_sec: float = 10.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 40,
        batch_size: int = 1,
        iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark MFCC computation."""
        from mlx_audio.primitives import mfcc

        audio = self._generate_audio(duration_sec, batch_size)
        n_samples = audio.shape[-1]

        def run_mfcc():
            return mfcc(
                audio,
                sr=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
            )

        times, peak_memory = _measure_time(run_mfcc, iterations=iterations)

        mean_time = np.mean(times)
        throughput = (n_samples * batch_size) / mean_time

        result = BenchmarkResult(
            name="mfcc",
            mean_time_ms=mean_time * 1000,
            std_time_ms=np.std(times) * 1000,
            min_time_ms=np.min(times) * 1000,
            max_time_ms=np.max(times) * 1000,
            throughput=throughput,
            peak_memory_mb=peak_memory,
            iterations=iterations,
            params={
                "duration_sec": duration_sec,
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "n_mfcc": n_mfcc,
                "batch_size": batch_size,
            },
        )
        self.results.append(result)
        return result

    def bench_stft_roundtrip(
        self,
        duration_sec: float = 10.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        batch_size: int = 1,
        iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark STFT -> ISTFT roundtrip."""
        from mlx_audio.primitives import istft, stft

        audio = self._generate_audio(duration_sec, batch_size)
        n_samples = audio.shape[-1]

        def run_roundtrip():
            s = stft(audio, n_fft=n_fft, hop_length=hop_length)
            return istft(s, hop_length=hop_length, n_fft=n_fft)

        times, peak_memory = _measure_time(run_roundtrip, iterations=iterations)

        mean_time = np.mean(times)
        throughput = (n_samples * batch_size) / mean_time

        result = BenchmarkResult(
            name="stft_roundtrip",
            mean_time_ms=mean_time * 1000,
            std_time_ms=np.std(times) * 1000,
            min_time_ms=np.min(times) * 1000,
            max_time_ms=np.max(times) * 1000,
            throughput=throughput,
            peak_memory_mb=peak_memory,
            iterations=iterations,
            params={
                "duration_sec": duration_sec,
                "n_fft": n_fft,
                "hop_length": hop_length,
                "batch_size": batch_size,
            },
        )
        self.results.append(result)
        return result

    def run_all(
        self,
        durations: list[float] | None = None,
        batch_sizes: list[int] | None = None,
        iterations: int = 10,
    ) -> list[BenchmarkResult]:
        """Run all primitive benchmarks with various configurations."""
        if durations is None:
            durations = [1.0, 5.0, 10.0, 30.0]
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]

        _warmup_and_sync()

        for duration in durations:
            for batch_size in batch_sizes:
                self.bench_stft(
                    duration_sec=duration,
                    batch_size=batch_size,
                    iterations=iterations,
                )
                self.bench_istft(
                    duration_sec=duration,
                    batch_size=batch_size,
                    iterations=iterations,
                )
                self.bench_melspectrogram(
                    duration_sec=duration,
                    batch_size=batch_size,
                    iterations=iterations,
                )
                self.bench_mfcc(
                    duration_sec=duration,
                    batch_size=batch_size,
                    iterations=iterations,
                )
                self.bench_stft_roundtrip(
                    duration_sec=duration,
                    batch_size=batch_size,
                    iterations=iterations,
                )

        return self.results

    def summary(self) -> str:
        """Generate a summary of benchmark results."""
        lines = ["=" * 80, "Primitives Benchmark Summary", "=" * 80, ""]

        # Group by name
        by_name: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            by_name.setdefault(r.name, []).append(r)

        for name, results in by_name.items():
            lines.append(f"\n{name.upper()}")
            lines.append("-" * 40)
            header = f"{'Duration':>10} {'Batch':>6} {'Mean(ms)':>10}"
            header += f" {'Std(ms)':>10} {'Throughput':>15}"
            lines.append(header)

            for r in results:
                line = f"{r.params.get('duration_sec', 0):>10.1f} "
                line += f"{r.params.get('batch_size', 1):>6} "
                line += f"{r.mean_time_ms:>10.2f} "
                line += f"{r.std_time_ms:>10.2f} "
                line += f"{r.throughput:>12.0f} s/s"
                lines.append(line)

        return "\n".join(lines)


if __name__ == "__main__":
    bench = BenchPrimitives()
    bench.run_all(durations=[1.0, 5.0], batch_sizes=[1], iterations=5)
    print(bench.summary())
