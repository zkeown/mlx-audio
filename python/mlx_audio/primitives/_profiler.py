"""
Performance profiling infrastructure for mlx-audio-primitives.

Provides decorators and utilities for:
- Function-level timing with GPU sync
- GPU/CPU synchronization point detection
- CPU/GPU data transfer logging
- Cache hit/miss rate monitoring
"""

from __future__ import annotations

import functools
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


# Thread-local storage for profiler state
_profiler_local = threading.local()


@dataclass
class ProfileMetrics:
    """Metrics collected for a single profiled function call."""

    function_name: str
    wall_time_ms: float
    gpu_sync_count: int = 0
    cpu_to_gpu_transfers: int = 0
    gpu_to_cpu_transfers: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    peak_memory_bytes: int = 0
    allocation_count: int = 0


@dataclass
class ProfilerState:
    """Global profiler state."""

    enabled: bool = False
    metrics: list[ProfileMetrics] = field(default_factory=list)
    sync_points: list[tuple[str, float]] = field(default_factory=list)
    transfer_log: list[tuple[str, str, int]] = field(default_factory=list)
    cache_stats: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"hits": 0, "misses": 0})
    )


# Global profiler instance
_profiler = ProfilerState()


def enable_profiling() -> None:
    """Enable the profiler and clear previous data."""
    _profiler.enabled = True
    _profiler.metrics.clear()
    _profiler.sync_points.clear()
    _profiler.transfer_log.clear()
    _profiler.cache_stats.clear()


def disable_profiling() -> None:
    """Disable the profiler."""
    _profiler.enabled = False


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled."""
    return _profiler.enabled


def get_metrics() -> list[ProfileMetrics]:
    """Get collected metrics."""
    return _profiler.metrics.copy()


def get_sync_points() -> list[tuple[str, float]]:
    """Get logged synchronization points."""
    return _profiler.sync_points.copy()


def get_transfer_log() -> list[tuple[str, str, int]]:
    """Get CPU/GPU transfer log."""
    return _profiler.transfer_log.copy()


def get_cache_stats() -> dict[str, dict[str, int]]:
    """Get cache hit/miss statistics."""
    return dict(_profiler.cache_stats)


def clear_profiling_data() -> None:
    """Clear all profiling data without disabling."""
    _profiler.metrics.clear()
    _profiler.sync_points.clear()
    _profiler.transfer_log.clear()
    _profiler.cache_stats.clear()


@contextmanager
def profile_section(name: str) -> Generator[None, None, None]:
    """
    Context manager for profiling a code section.

    Parameters
    ----------
    name : str
        Name for this profiled section.

    Examples
    --------
    >>> with profile_section("STFT computation"):
    ...     result = stft(signal)
    """
    if not _profiler.enabled:
        yield
        return

    # Ensure GPU operations complete before timing
    mx.eval()
    start = time.perf_counter()

    yield

    # Sync again to capture GPU time
    mx.eval()
    end = time.perf_counter()

    _profiler.metrics.append(
        ProfileMetrics(
            function_name=name,
            wall_time_ms=(end - start) * 1000,
        )
    )


def log_sync_point(name: str) -> None:
    """
    Log a GPU synchronization point.

    Parameters
    ----------
    name : str
        Description of the synchronization point.
    """
    if _profiler.enabled:
        _profiler.sync_points.append((name, time.perf_counter()))


def log_transfer(direction: str, context: str, size_bytes: int) -> None:
    """
    Log a CPU/GPU data transfer.

    Parameters
    ----------
    direction : str
        "cpu_to_gpu" or "gpu_to_cpu"
    context : str
        Description of where transfer occurred
    size_bytes : int
        Size of transferred data in bytes
    """
    if _profiler.enabled:
        _profiler.transfer_log.append((direction, context, size_bytes))


def log_cache_access(cache_name: str, hit: bool) -> None:
    """
    Log a cache access.

    Parameters
    ----------
    cache_name : str
        Name of the cache being accessed.
    hit : bool
        True if cache hit, False if cache miss.
    """
    if _profiler.enabled:
        key = "hits" if hit else "misses"
        _profiler.cache_stats[cache_name][key] += 1


def profile(
    func: Callable | None = None,
    *,
    sync_before: bool = True,
    sync_after: bool = True,
) -> Callable:
    """
    Decorator to profile a function.

    Parameters
    ----------
    func : Callable
        Function to profile.
    sync_before : bool, default=True
        If True, call mx.eval() before timing.
    sync_after : bool, default=True
        If True, call mx.eval() after function to capture GPU time.

    Returns
    -------
    Callable
        Wrapped function.

    Examples
    --------
    >>> @profile
    ... def my_function(x):
    ...     return mx.sin(x)
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _profiler.enabled:
                return fn(*args, **kwargs)

            if sync_before:
                mx.eval()

            start = time.perf_counter()
            result = fn(*args, **kwargs)

            if sync_after:
                if isinstance(result, mx.array):
                    mx.eval(result)
                elif isinstance(result, tuple):
                    arrays = [r for r in result if isinstance(r, mx.array)]
                    if arrays:
                        mx.eval(*arrays)

            end = time.perf_counter()

            _profiler.metrics.append(
                ProfileMetrics(
                    function_name=fn.__name__,
                    wall_time_ms=(end - start) * 1000,
                )
            )

            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def tracked_np_array(mlx_arr: mx.array, context: str = "unknown") -> np.ndarray:
    """
    Convert mx.array to np.ndarray with transfer logging.

    Parameters
    ----------
    mlx_arr : mx.array
        MLX array to convert.
    context : str, default="unknown"
        Description of where the transfer occurred.

    Returns
    -------
    np.ndarray
        NumPy array with the same data.
    """
    size_bytes = mlx_arr.nbytes if hasattr(mlx_arr, "nbytes") else mlx_arr.size * 4
    log_transfer("gpu_to_cpu", context, size_bytes)
    log_sync_point(f"np.array() in {context}")
    return np.array(mlx_arr)


def tracked_mx_array(np_arr: np.ndarray, context: str = "unknown") -> mx.array:
    """
    Convert np.ndarray to mx.array with transfer logging.

    Parameters
    ----------
    np_arr : np.ndarray
        NumPy array to convert.
    context : str, default="unknown"
        Description of where the transfer occurred.

    Returns
    -------
    mx.array
        MLX array with the same data.
    """
    size_bytes = np_arr.nbytes
    log_transfer("cpu_to_gpu", context, size_bytes)
    return mx.array(np_arr)


def generate_text_report() -> str:
    """
    Generate a text summary of profiling results.

    Returns
    -------
    str
        Formatted text report.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MLX Audio Primitives - Performance Profile Report")
    lines.append("=" * 80)

    # Function timings
    metrics = get_metrics()
    if metrics:
        lines.append("\n## Function Timings")
        lines.append("-" * 40)

        # Aggregate by function name
        timing_map: dict[str, list[float]] = defaultdict(list)
        for m in metrics:
            timing_map[m.function_name].append(m.wall_time_ms)

        for func, times in sorted(timing_map.items(), key=lambda x: -sum(x[1])):
            total = sum(times)
            avg = total / len(times)
            lines.append(
                f"{func:40} total={total:8.2f}ms  avg={avg:6.2f}ms  calls={len(times)}"
            )

    # Sync points
    sync_points = get_sync_points()
    if sync_points:
        lines.append("\n## Synchronization Points")
        lines.append("-" * 40)
        # Aggregate sync points by name
        sync_counts: dict[str, int] = defaultdict(int)
        for name, _ in sync_points:
            sync_counts[name] += 1

        for name, count in sorted(sync_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}x")

    # Data transfers
    transfers = get_transfer_log()
    if transfers:
        lines.append("\n## CPU/GPU Data Transfers")
        lines.append("-" * 40)

        cpu_to_gpu = [(c, s) for d, c, s in transfers if d == "cpu_to_gpu"]
        gpu_to_cpu = [(c, s) for d, c, s in transfers if d == "gpu_to_cpu"]

        lines.append(
            f"CPU -> GPU: {len(cpu_to_gpu)} transfers, "
            f"{sum(s for _, s in cpu_to_gpu) / 1024**2:.2f} MB total"
        )
        lines.append(
            f"GPU -> CPU: {len(gpu_to_cpu)} transfers, "
            f"{sum(s for _, s in gpu_to_cpu) / 1024**2:.2f} MB total"
        )

        # Group by context
        if gpu_to_cpu:
            lines.append("\n  GPU -> CPU transfers by context:")
            context_totals: dict[str, tuple[int, int]] = {}
            for ctx, size in gpu_to_cpu:
                if ctx not in context_totals:
                    context_totals[ctx] = (0, 0)
                count, total = context_totals[ctx]
                context_totals[ctx] = (count + 1, total + size)

            for ctx, (count, total) in sorted(
                context_totals.items(), key=lambda x: -x[1][1]
            ):
                lines.append(f"    {ctx}: {count}x, {total / 1024**2:.2f} MB")

    # Cache stats
    cache_stats = get_cache_stats()
    if cache_stats:
        lines.append("\n## Cache Statistics")
        lines.append("-" * 40)
        for cache_name, stats in cache_stats.items():
            hits = stats["hits"]
            misses = stats["misses"]
            total = hits + misses
            hit_rate = hits / total * 100 if total > 0 else 0
            lines.append(
                f"{cache_name:30} hits={hits:4}  misses={misses:4}  "
                f"hit_rate={hit_rate:.1f}%"
            )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def export_json() -> dict[str, Any]:
    """
    Export profiling data as a dictionary (for JSON serialization).

    Returns
    -------
    dict
        Dictionary containing all profiling data.
    """
    return {
        "metrics": [
            {
                "function_name": m.function_name,
                "wall_time_ms": m.wall_time_ms,
                "gpu_sync_count": m.gpu_sync_count,
                "cpu_to_gpu_transfers": m.cpu_to_gpu_transfers,
                "gpu_to_cpu_transfers": m.gpu_to_cpu_transfers,
                "cache_hits": m.cache_hits,
                "cache_misses": m.cache_misses,
            }
            for m in get_metrics()
        ],
        "sync_points": get_sync_points(),
        "transfers": get_transfer_log(),
        "cache_stats": get_cache_stats(),
    }
