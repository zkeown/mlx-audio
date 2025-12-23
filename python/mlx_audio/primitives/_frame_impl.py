"""
Shared optimized framing implementation.

Provides the core framing logic used by both stft.py and framing.py.
Uses the fastest available method:
1. C++ extension (if available)
2. mx.as_strided (MLX >= 0.5, zero-copy)
3. Gather-based fallback (always works)
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx

from ._extension import HAS_CPP_EXT, _ext

# Detect best available method at module load time (avoids per-call checks)
_HAS_AS_STRIDED = hasattr(mx, "as_strided")


def _get_framing_method() -> Literal["cpp", "strided", "gather"]:
    """Determine the best framing method available."""
    if HAS_CPP_EXT and _ext is not None:
        return "cpp"
    elif _HAS_AS_STRIDED:
        return "strided"
    else:
        return "gather"


# Cache the method selection at module load
_FRAMING_METHOD = _get_framing_method()


def _frame_cpp(
    y: mx.array, frame_length: int, hop_length: int, n_frames: int
) -> mx.array:
    """Frame using C++ extension."""
    return _ext.frame_signal(y, frame_length, hop_length)


def _frame_strided(
    y: mx.array, frame_length: int, hop_length: int, n_frames: int
) -> mx.array:
    """Frame using mx.as_strided (zero-copy)."""
    batch_size, signal_length = y.shape
    batch_stride = signal_length
    new_shape = (batch_size, n_frames, frame_length)
    new_strides = (batch_stride, hop_length, 1)
    return mx.as_strided(y, shape=new_shape, strides=new_strides)


def _frame_gather(
    y: mx.array, frame_length: int, hop_length: int, n_frames: int
) -> mx.array:
    """Frame using gather-based approach (fallback)."""
    batch_size = y.shape[0]
    frame_starts = mx.arange(n_frames) * hop_length
    sample_offsets = mx.arange(frame_length)
    indices = frame_starts[:, None] + sample_offsets[None, :]
    return mx.take(y, indices.flatten(), axis=1).reshape(
        batch_size, n_frames, frame_length
    )


# Map method name to implementation function
_FRAME_IMPLS = {
    "cpp": _frame_cpp,
    "strided": _frame_strided,
    "gather": _frame_gather,
}

# Bind the selected implementation at module load
_frame_impl = _FRAME_IMPLS[_FRAMING_METHOD]


def frame_signal_batched(
    y: mx.array,
    frame_length: int,
    hop_length: int,
) -> mx.array:
    """
    Frame a batched signal into overlapping windows.

    This is the core framing implementation used throughout the library.
    It automatically selects the fastest available method.

    Parameters
    ----------
    y : mx.array
        Signal of shape (batch, samples).
    frame_length : int
        Length of each frame in samples.
    hop_length : int
        Number of samples between frame starts.

    Returns
    -------
    mx.array
        Framed signal of shape (batch, n_frames, frame_length).

    Raises
    ------
    ValueError
        If parameters are invalid or signal is too short.
    """
    batch_size, signal_length = y.shape

    # Validate inputs
    if frame_length <= 0:
        raise ValueError(f"frame_length must be positive, got {frame_length}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if signal_length < frame_length:
        raise ValueError(
            f"Signal length ({signal_length}) must be >= frame_length "
            f"({frame_length}). Consider padding the signal."
        )

    n_frames = 1 + (signal_length - frame_length) // hop_length

    # Use pre-selected implementation (no per-call method detection)
    return _frame_impl(y, frame_length, hop_length, n_frames)
