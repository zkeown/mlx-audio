"""Benchmarking suite for mlx-audio performance optimization."""

from benchmarks.bench_primitives import BenchPrimitives
from benchmarks.bench_streaming import BenchStreaming

__all__ = ["BenchPrimitives", "BenchStreaming"]
