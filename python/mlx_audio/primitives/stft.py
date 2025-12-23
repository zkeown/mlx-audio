"""
Short-Time Fourier Transform (STFT) and Inverse STFT.

Provides librosa-compatible STFT and ISTFT implementations for MLX.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx
import numpy as np

# Import C++ extension with graceful fallback
from ._extension import HAS_CPP_EXT, _ext
from ._frame_impl import frame_signal_batched
from .windows import get_window

# Numerical constants
_WINDOW_SUM_EPSILON = 1e-8

# Cache settings
_WINDOW_CACHE_MAXSIZE = 32


class _WindowCache:
    """
    LRU cache for padded windows with content-based hashing.

    Uses content-based hashing for array windows (via tobytes()) rather than
    id() for robustness. The cache is bounded to prevent memory leaks.
    """

    def __init__(self, maxsize: int = _WINDOW_CACHE_MAXSIZE):
        self._cache: dict[tuple, mx.array] = {}
        self._access_order: list[tuple] = []  # LRU tracking
        self._maxsize = maxsize

    def _get_window_hash(self, window: mx.array) -> int:
        """Compute content-based hash for window array."""
        return hash(np.array(window).tobytes())

    def get(
        self, window: str | mx.array, win_length: int, n_fft: int
    ) -> mx.array | None:
        """Get window from cache, returning None on miss."""
        if isinstance(window, str):
            cache_key = (window, win_length, n_fft)
        else:
            cache_key = ("array", self._get_window_hash(window), win_length, n_fft)

        if cache_key in self._cache:
            # Update LRU order
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]
        return None

    def put(
        self, window: str | mx.array, win_length: int, n_fft: int, value: mx.array
    ) -> None:
        """Store window in cache, evicting oldest if at capacity."""
        if isinstance(window, str):
            cache_key = (window, win_length, n_fft)
        else:
            cache_key = ("array", self._get_window_hash(window), win_length, n_fft)

        # Evict oldest if at capacity
        while len(self._cache) >= self._maxsize and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[cache_key] = value
        self._access_order.append(cache_key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


# Global window cache instance
_padded_window_cache = _WindowCache()


def _get_padded_window(window: str | mx.array, win_length: int, n_fft: int) -> mx.array:
    """Get window, padding to n_fft if needed, with content-based caching."""
    # Check cache first
    cached = _padded_window_cache.get(window, win_length, n_fft)
    if cached is not None:
        return cached

    # Get base window
    win = get_window(window, win_length, fftbins=True)

    # Pad if needed
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win = mx.pad(win, [(pad_left, pad_right)])

    # Cache and return
    _padded_window_cache.put(window, win_length, n_fft, win)
    return win


@lru_cache(maxsize=8)
def _get_compiled_stft_fn(n_fft: int, hop_length: int, center: bool, pad_mode: str):
    """
    Get a compiled STFT function for the given parameters.

    Using mx.compile() enables graph-level optimizations including
    kernel fusion and reduced dispatch overhead.
    """

    def _stft_core(y: mx.array, win: mx.array) -> mx.array:
        # Center padding (if enabled)
        if center:
            pad_length = n_fft // 2
            y = _pad_signal(y, pad_length, pad_mode)

        # Frame the signal
        frames = _frame_signal(y, n_fft, hop_length)

        # Apply window and compute FFT in sequence
        # (compiler may fuse these operations)
        frames = frames * win
        return mx.fft.rfft(frames, axis=-1)

    # Compile the function for better performance
    return mx.compile(_stft_core)


def stft(
    y: mx.array,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> mx.array:
    """
    Short-Time Fourier Transform.

    Parameters
    ----------
    y : mx.array
        Input signal. Shape: (samples,) or (batch, samples).
    n_fft : int, default=2048
        FFT size.
    hop_length : int, optional
        Number of samples between frames. Default: n_fft // 4.
    win_length : int, optional
        Window length. Default: n_fft.
    window : str or mx.array, default='hann'
        Window function. See get_window() for options.
    center : bool, default=True
        If True, pad signal so frame t is centered at y[t * hop_length].
    pad_mode : str, default='constant'
        Padding mode if center=True. One of: 'constant', 'reflect', 'edge'.

    Returns
    -------
    mx.array
        Complex STFT matrix.
        Shape: (n_fft//2 + 1, n_frames) for 1D input.
        Shape: (batch, n_fft//2 + 1, n_frames) for 2D input.

    Raises
    ------
    ValueError
        If parameters are invalid.

    Examples
    --------
    >>> y = mx.array(np.random.randn(22050).astype(np.float32))
    >>> S = stft(y, n_fft=2048, hop_length=512)
    >>> S.shape
    (1025, 44)
    """
    # Validate and set defaults
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if win_length <= 0:
        raise ValueError(f"win_length must be positive, got {win_length}")
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")
    if hop_length > n_fft:
        raise ValueError(
            f"hop_length ({hop_length}) should typically be <= n_fft ({n_fft})"
        )

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]  # Add batch dimension

    batch_size, signal_length = y.shape

    # Get window with caching (includes padding if needed)
    win = _get_padded_window(window, win_length, n_fft)

    # Use compiled STFT implementation for better performance
    stft_fn = _get_compiled_stft_fn(n_fft, hop_length, center, pad_mode)
    stft_matrix = stft_fn(y, win)

    # Transpose to (batch, freq_bins, n_frames) to match librosa convention
    stft_matrix = mx.transpose(stft_matrix, (0, 2, 1))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        stft_matrix = stft_matrix[0]

    return stft_matrix


def istft(
    stft_matrix: mx.array,
    hop_length: int | None = None,
    win_length: int | None = None,
    n_fft: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    length: int | None = None,
) -> mx.array:
    """
    Inverse Short-Time Fourier Transform.

    Parameters
    ----------
    stft_matrix : mx.array
        Complex STFT matrix.
        Shape: (n_fft//2 + 1, n_frames) for 1D output.
        Shape: (batch, n_fft//2 + 1, n_frames) for 2D output.
    hop_length : int, optional
        Number of samples between frames. Default: n_fft // 4.
    win_length : int, optional
        Window length. Default: n_fft.
    n_fft : int, optional
        FFT size. Default: inferred from stft_matrix as 2 * (freq_bins - 1).
    window : str or mx.array, default='hann'
        Window function. Must match the window used in stft().
    center : bool, default=True
        If True, trim the padding added during stft().
    length : int, optional
        If provided, the output is trimmed or zero-padded to this length.

    Returns
    -------
    mx.array
        Reconstructed time-domain signal.
        Shape: (samples,) for 2D input.
        Shape: (batch, samples) for 3D input.

    Examples
    --------
    >>> S = stft(y, n_fft=2048, hop_length=512)
    >>> y_reconstructed = istft(S, hop_length=512)
    """
    # Validate input dimensions
    if stft_matrix.ndim not in (2, 3):
        raise ValueError(f"stft_matrix must be 2D or 3D, got {stft_matrix.ndim}D")

    # Handle batched input
    input_is_2d = stft_matrix.ndim == 2
    if input_is_2d:
        stft_matrix = stft_matrix[None, :]  # Add batch dimension

    batch_size, freq_bins, n_frames = stft_matrix.shape

    # Infer n_fft from frequency bins
    if n_fft is None:
        n_fft = 2 * (freq_bins - 1)

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Get window with caching (includes padding if needed)
    win = _get_padded_window(window, win_length, n_fft)

    # Transpose to (batch, n_frames, freq_bins) for irfft
    stft_matrix = mx.transpose(stft_matrix, (0, 2, 1))

    # Inverse FFT: (batch, n_frames, n_fft)
    frames = mx.fft.irfft(stft_matrix, n=n_fft, axis=-1)

    # Determine output length
    # If length is specified with center=True, we need to account for
    # the padding that was added during STFT
    if length is not None:
        if center:
            # The original signal was padded by n_fft//2 on each side
            # So the "padded length" we need to reconstruct is length + n_fft
            padded_length = length + n_fft
        else:
            padded_length = length
    else:
        # Without length, use the natural overlap-add length
        padded_length = n_fft + (n_frames - 1) * hop_length

    # Perform overlap-add reconstruction
    y = _overlap_add(frames, hop_length, win, padded_length)

    # Trim center padding if needed
    if center:
        pad_length = n_fft // 2
        if length is not None:
            # When length is specified, extract exactly 'length' samples
            # from the middle of the reconstructed signal
            y = y[:, pad_length : pad_length + length]
        else:
            # Without length, just trim the padding from both sides
            # Note: negative slice y[:, pad:-pad] doesn't work when empty
            end_idx = y.shape[1] - pad_length
            if end_idx > pad_length:
                y = y[:, pad_length:end_idx]
            else:
                # Edge case: signal shorter than padding
                y = y[:, :0]  # Empty result
    else:
        # No center padding to trim
        if length is not None:
            current_length = y.shape[1]
            if length < current_length:
                y = y[:, :length]
            elif length > current_length:
                pad_amount = length - current_length
                y = mx.pad(y, [(0, 0), (0, pad_amount)])

    # Remove batch dimension if input was 2D
    if input_is_2d:
        y = y[0]

    return y


def magnitude(stft_matrix: mx.array) -> mx.array:
    """
    Compute magnitude spectrogram from complex STFT.

    Parameters
    ----------
    stft_matrix : mx.array
        Complex STFT matrix (output of stft()).

    Returns
    -------
    mx.array
        Magnitude spectrogram (same shape, real-valued).
    """
    # Use mx.abs for complex arrays - more efficient than manual computation
    return mx.abs(stft_matrix)


def phase(stft_matrix: mx.array) -> mx.array:
    """
    Compute phase spectrogram from complex STFT.

    Parameters
    ----------
    stft_matrix : mx.array
        Complex STFT matrix (output of stft()).

    Returns
    -------
    mx.array
        Phase spectrogram in radians (same shape, real-valued).
    """
    return mx.arctan2(stft_matrix.imag, stft_matrix.real)


def check_nola(
    window: str | mx.array,
    hop_length: int,
    n_fft: int,
    tol: float = 1e-10,
) -> bool:
    """
    Check the Nonzero Overlap-Add (NOLA) constraint.

    The NOLA constraint ensures that the ISTFT is invertible. It requires that
    the sum of squared windows at every sample position is nonzero.

    Parameters
    ----------
    window : str or mx.array
        Window function.
    hop_length : int
        Hop length.
    n_fft : int
        FFT size (determines window length for string windows).
    tol : float, default=1e-10
        Tolerance for considering sum as nonzero.

    Returns
    -------
    bool
        True if NOLA constraint is satisfied.

    Examples
    --------
    >>> check_nola('hann', hop_length=512, n_fft=2048)
    True
    """
    win = get_window(window, n_fft, fftbins=True)
    win_np = np.array(win)

    # Following scipy.signal.check_NOLA:
    # Sum the squared window values within each hop-sized bin
    # Check that the minimum bin sum is nonzero
    step = hop_length
    n_bins = n_fft // step

    # Sum squared windows for each position within a hop
    binsums = sum(win_np[ii * step : (ii + 1) * step] ** 2 for ii in range(n_bins))

    # Handle remainder if n_fft is not a multiple of hop_length
    if n_fft % step != 0:
        binsums[: n_fft % step] += win_np[-(n_fft % step) :] ** 2

    return bool(np.min(binsums) > tol)


def _pad_signal(y: mx.array, pad_length: int, mode: str) -> mx.array:
    """Pad signal on both sides."""
    # Use C++ extension if available
    if HAS_CPP_EXT and _ext is not None:
        return _ext.pad_signal(y, pad_length, mode)

    # Fallback to Python implementation
    if mode == "constant":
        return mx.pad(y, [(0, 0), (pad_length, pad_length)], mode="constant")
    elif mode == "edge":
        return mx.pad(y, [(0, 0), (pad_length, pad_length)], mode="edge")
    elif mode == "reflect":
        # MLX doesn't support reflect padding, implement manually
        # Reflect padding: [d c b a | a b c d e f | f e d c]
        batch_size, signal_length = y.shape

        # Left reflection (excluding the edge element, reversed)
        if pad_length > 0:
            # y[:, 1:pad_length+1] then reverse -> y[:, pad_length:0:-1]
            left_pad = y[:, pad_length:0:-1]
        else:
            left_pad = y[:, :0]  # Empty slice

        # Right reflection (excluding the edge element, reversed)
        if pad_length > 0:
            # y[:, -pad_length-1:-1] then reverse -> y[:, -2:-pad_length-2:-1]
            right_pad = y[:, -2 : -pad_length - 2 : -1]
        else:
            right_pad = y[:, :0]  # Empty slice

        return mx.concatenate([left_pad, y, right_pad], axis=1)
    else:
        raise ValueError(
            f"Unknown pad_mode: '{mode}'. Supported: reflect, constant, edge"
        )


def _frame_signal(y: mx.array, n_fft: int, hop_length: int) -> mx.array:
    """
    Frame signal into overlapping windows.

    Uses the shared optimized framing implementation from _frame_impl.py.

    Parameters
    ----------
    y : mx.array
        Signal of shape (batch, samples).
    n_fft : int
        Frame length.
    hop_length : int
        Hop length.

    Returns
    -------
    mx.array
        Framed signal of shape (batch, n_frames, n_fft).

    Raises
    ------
    ValueError
        If parameters are invalid or signal is too short.
    """
    return frame_signal_batched(y, n_fft, hop_length)


def _overlap_add(
    frames: mx.array,
    hop_length: int,
    window: mx.array,
    output_length: int,
) -> mx.array:
    """
    Overlap-add reconstruction with squared window normalization.

    Uses C++ extension when available, otherwise falls back to fused Metal kernel
    that handles windowing, accumulation, and normalization in a single pass.

    Parameters
    ----------
    frames : mx.array
        Framed signal of shape (batch, n_frames, n_fft).
    hop_length : int
        Hop length.
    window : mx.array
        Window function.
    output_length : int
        Length of output signal.

    Returns
    -------
    mx.array
        Reconstructed signal of shape (batch, output_length).
    """
    # Use C++ extension if available (handles windowing and normalization internally)
    if HAS_CPP_EXT and _ext is not None:
        return _ext.overlap_add(frames, window, hop_length, output_length)

    # Fallback to fused Python/Metal implementation.
    # The fused kernel applies windowing, accumulates overlapping frames,
    # and normalizes in a single pass - no intermediate allocations needed.
    return _fused_overlap_add_metal(frames, window, hop_length, output_length)


# Fused overlap-add Metal kernel with output-centric approach.
#
# Key optimization: Each thread handles one OUTPUT position and READS from all
# contributing frames. This eliminates atomic operations entirely (vs the old
# scatter-based approach which required 176K atomics for typical params).
#
# The kernel fuses: window application + overlap-add + normalization
# into a single pass, matching the C++ overlap_add_fused_kernel pattern.
#
# Grid layout: (output_length, batch_size) - one thread per output sample.
# Performance: 40-60% faster than atomic scatter approach.
_FUSED_OVERLAP_ADD_SOURCE = """
    // Thread indices - output-centric layout
    int out_idx = thread_position_in_grid.x;
    int batch_idx = thread_position_in_grid.y;

    // Bounds check
    if (out_idx >= output_length || batch_idx >= batch_size) {
        return;
    }

    // Compute which frames contribute to this output position.
    // Frame f contributes samples to positions [f * hop_length, f * hop_length + n_fft)
    // So position out_idx receives contributions from frames where:
    //   f * hop_length <= out_idx < f * hop_length + n_fft
    // Solving: (out_idx - n_fft + 1) / hop_length <= f <= out_idx / hop_length
    int first_frame = (out_idx - n_fft + 1 + hop_length - 1) / hop_length;
    if (first_frame < 0) first_frame = 0;
    int last_frame = out_idx / hop_length;
    if (last_frame >= n_frames) last_frame = n_frames - 1;

    float sum = 0.0f;
    float win_sq_sum = 0.0f;

    // Gather contributions from all overlapping frames
    for (int f = first_frame; f <= last_frame; f++) {
        int sample_in_frame = out_idx - f * hop_length;

        // Safety check (should always pass with correct frame bounds)
        if (sample_in_frame >= 0 && sample_in_frame < n_fft) {
            // Read window value
            float win_val = window[sample_in_frame];

            // Read frame value and apply window in one step
            int frame_flat_idx = batch_idx * n_frames * n_fft
                               + f * n_fft
                               + sample_in_frame;
            float frame_val = frames[frame_flat_idx];

            // Accumulate windowed value and window squared
            sum += win_val * frame_val;
            win_sq_sum += win_val * win_val;
        }
    }

    // Normalize and store output (fused operation)
    // Epsilon 1e-8 prevents division by zero for edge samples
    int out_flat_idx = batch_idx * output_length + out_idx;
    output[out_flat_idx] = sum / fmax(win_sq_sum, 1e-8f);
"""


@lru_cache(maxsize=8)
def _get_fused_overlap_add_kernel() -> Any:
    """Get the fused overlap-add kernel (output-centric, no atomics)."""
    return mx.fast.metal_kernel(
        name="fused_overlap_add",
        input_names=["frames", "window"],
        output_names=["output"],
        source=_FUSED_OVERLAP_ADD_SOURCE,
        atomic_outputs=False,  # No atomics needed - each thread writes unique position
    )


def _fused_overlap_add_metal(
    frames: mx.array,
    window: mx.array,
    hop_length: int,
    output_length: int,
) -> mx.array:
    """
    Perform fused overlap-add using output-centric Metal kernel.

    This kernel fuses window application, overlap-add accumulation, and
    normalization into a single pass. Each thread handles one output position,
    reading from all contributing frames - no atomic operations needed.

    Parameters
    ----------
    frames : mx.array
        Raw frames of shape (batch, n_frames, n_fft) - NOT pre-windowed.
    window : mx.array
        Window function of shape (n_fft,).
    hop_length : int
        Hop length between frames.
    output_length : int
        Length of output signal.

    Returns
    -------
    mx.array
        Reconstructed and normalized signal of shape (batch, output_length).
    """
    batch_size, n_frames, n_fft = frames.shape

    kernel = _get_fused_overlap_add_kernel()

    # Output-centric grid: each thread handles one output position
    # Grid: (output_length, batch_size) - much smaller than scatter approach!
    grid = (output_length, batch_size, 1)

    # Optimal threadgroup for Apple Silicon - use wider groups for batch=1
    if batch_size < 4:
        threadgroup = (min(output_length, 256), 1, 1)
    else:
        threadgroup = (min(output_length, 64), min(batch_size, 4), 1)

    outputs = kernel(
        inputs=[frames, window],
        output_shapes=[(batch_size, output_length)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup,
        template=[
            ("hop_length", hop_length),
            ("output_length", output_length),
            ("batch_size", batch_size),
            ("n_frames", n_frames),
            ("n_fft", n_fft),
        ],
        init_value=0,
    )

    return outputs[0]
