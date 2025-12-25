"""
Memory profiling utilities for mlx-audio-primitives.

Tracks peak memory usage and allocation patterns using MLX introspection.
"""

from __future__ import annotations

import gc
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class MemorySnapshot:
    """Memory state snapshot."""

    active_bytes: int
    peak_bytes: int
    cache_bytes: int

    @property
    def active_mb(self) -> float:
        """Active memory in MB."""
        return self.active_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        """Peak memory in MB."""
        return self.peak_bytes / (1024 * 1024)

    @property
    def cache_mb(self) -> float:
        """Cache memory in MB."""
        return self.cache_bytes / (1024 * 1024)


@dataclass
class MemoryProfile:
    """Memory profiling result."""

    peak_memory_mb: float
    allocated_mb: float
    output_size_mb: float
    efficiency: float  # output_size / peak_memory


def get_memory_snapshot() -> MemorySnapshot:
    """
    Get current memory state.

    Note: MLX doesn't expose detailed memory stats directly.
    This uses mx.metal.get_active_memory() and related functions
    when available (MLX >= 0.5).

    Returns
    -------
    MemorySnapshot
        Current memory state.
    """
    try:
        active = mx.metal.get_active_memory()
        peak = mx.metal.get_peak_memory()
        cache = mx.metal.get_cache_memory()
        return MemorySnapshot(active, peak, cache)
    except AttributeError:
        # Fallback for older MLX versions
        return MemorySnapshot(0, 0, 0)


def reset_peak_memory() -> None:
    """Reset peak memory tracker."""
    with suppress(AttributeError):
        mx.metal.reset_peak_memory()


def clear_cache() -> None:
    """Clear MLX memory cache."""
    with suppress(AttributeError):
        mx.metal.clear_cache()
    gc.collect()


def synchronize() -> None:
    """Synchronize GPU operations."""
    try:
        mx.synchronize()
    except AttributeError:
        # Fallback for older MLX versions - use eval on empty array
        mx.eval(mx.array([0.0]))


@contextmanager
def track_memory(label: str = "") -> Generator[MemorySnapshot, None, None]:
    """
    Context manager to track memory usage.

    Yields a MemorySnapshot that gets updated on exit with peak usage.

    Parameters
    ----------
    label : str, optional
        Label for this memory tracking section.

    Yields
    ------
    MemorySnapshot
        Snapshot that gets updated with peak memory on exit.

    Examples
    --------
    >>> with track_memory("STFT") as mem:
    ...     result = stft(signal)
    >>> print(f"Peak memory: {mem.peak_mb:.2f} MB")
    """
    clear_cache()
    reset_peak_memory()

    before = get_memory_snapshot()
    snapshot = MemorySnapshot(0, 0, 0)

    try:
        yield snapshot
    finally:
        synchronize()  # Ensure all ops complete
        after = get_memory_snapshot()
        snapshot.active_bytes = after.active_bytes - before.active_bytes
        snapshot.peak_bytes = after.peak_bytes
        snapshot.cache_bytes = after.cache_bytes


def profile_memory(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, MemoryProfile]:
    """
    Profile memory usage of a function.

    Parameters
    ----------
    fn : Callable
        Function to profile.
    *args, **kwargs
        Arguments to pass to function.

    Returns
    -------
    tuple[Any, MemoryProfile]
        Function result and memory profile.

    Examples
    --------
    >>> result, mem = profile_memory(stft, signal)
    >>> print(f"Peak memory: {mem.peak_memory_mb:.2f} MB")
    """
    reset_peak_memory()
    clear_cache()

    # Run function and capture result
    result = fn(*args, **kwargs)

    output_size_mb = 0.0
    if isinstance(result, mx.array):
        mx.eval(result)
        output_size_mb = result.nbytes / (1024 * 1024)
    elif isinstance(result, tuple):
        arrays = [r for r in result if isinstance(r, mx.array)]
        if arrays:
            mx.eval(*arrays)
            output_size_mb = sum(a.nbytes for a in arrays) / (1024 * 1024)

    synchronize()

    mem_info = get_memory_snapshot()
    peak_mb = mem_info.peak_mb

    return result, MemoryProfile(
        peak_memory_mb=peak_mb,
        allocated_mb=mem_info.active_mb,
        output_size_mb=output_size_mb,
        efficiency=output_size_mb / peak_mb if peak_mb > 0 else 0.0,
    )


def get_array_memory_info(arr: mx.array) -> dict[str, Any]:
    """
    Get memory information for an MLX array.

    Parameters
    ----------
    arr : mx.array
        Array to inspect.

    Returns
    -------
    dict
        Dictionary with shape, dtype, nbytes, and size information.
    """
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "nbytes": arr.nbytes if hasattr(arr, "nbytes") else arr.size * 4,
        "size": arr.size,
        "mb": (arr.nbytes if hasattr(arr, "nbytes") else arr.size * 4) / (1024 * 1024),
    }


def estimate_operation_memory(
    operation: str,
    signal_length: int,
    n_fft: int = 2048,
    hop_length: int | None = None,
    batch_size: int = 1,
    n_mels: int | None = None,
) -> dict[str, float]:
    """
    Estimate memory requirements for common operations.

    Parameters
    ----------
    operation : str
        One of: "stft", "istft", "melspectrogram", "mfcc"
    signal_length : int
        Length of input signal in samples.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, optional
        Hop length. Default: n_fft // 4
    batch_size : int, default=1
        Batch size.
    n_mels : int, optional
        Number of mel bands (for melspectrogram/mfcc).

    Returns
    -------
    dict[str, float]
        Estimated memory in MB for each stage.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    n_frames = 1 + (signal_length + n_fft - 1) // hop_length
    freq_bins = n_fft // 2 + 1

    estimates = {}
    bytes_per_float = 4  # float32
    bytes_per_complex = 8  # complex64

    # Input signal
    estimates["input_mb"] = (batch_size * signal_length * bytes_per_float) / (1024**2)

    if operation in ("stft", "istft", "melspectrogram", "mfcc"):
        # Padded signal (approximate)
        padded_length = signal_length + n_fft
        estimates["padded_mb"] = (batch_size * padded_length * bytes_per_float) / (
            1024**2
        )

        # Framed signal
        estimates["frames_mb"] = (batch_size * n_frames * n_fft * bytes_per_float) / (
            1024**2
        )

        # STFT output
        estimates["stft_mb"] = (
            batch_size * freq_bins * n_frames * bytes_per_complex
        ) / (1024**2)

    if operation in ("melspectrogram", "mfcc"):
        if n_mels is None:
            n_mels = 128

        # Magnitude spectrogram
        estimates["magnitude_mb"] = (
            batch_size * freq_bins * n_frames * bytes_per_float
        ) / (1024**2)

        # Mel filterbank
        estimates["filterbank_mb"] = (n_mels * freq_bins * bytes_per_float) / (1024**2)

        # Mel spectrogram
        estimates["mel_mb"] = (batch_size * n_mels * n_frames * bytes_per_float) / (
            1024**2
        )

    if operation == "mfcc":
        n_mfcc = 20  # typical
        estimates["mfcc_mb"] = (batch_size * n_mfcc * n_frames * bytes_per_float) / (
            1024**2
        )

    # Estimate peak (sum of concurrent allocations)
    if operation == "stft":
        estimates["estimated_peak_mb"] = (
            estimates["input_mb"]
            + estimates["padded_mb"]
            + estimates["frames_mb"]
            + estimates["stft_mb"]
        )
    elif operation == "melspectrogram":
        estimates["estimated_peak_mb"] = (
            estimates["input_mb"]
            + estimates["stft_mb"]
            + estimates["magnitude_mb"]
            + estimates["filterbank_mb"]
            + estimates["mel_mb"]
        )
    elif operation == "mfcc":
        estimates["estimated_peak_mb"] = (
            estimates["input_mb"]
            + estimates["stft_mb"]
            + estimates["mel_mb"]
            + estimates["mfcc_mb"]
        )

    return estimates
