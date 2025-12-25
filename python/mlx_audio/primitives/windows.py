"""
Window functions for audio signal processing.

Provides window functions compatible with librosa/scipy conventions.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import mlx.core as mx
import numpy as np

# Import C++ extension with graceful fallback
from ._extension import HAS_CPP_EXT, _ext


def _generalized_cosine_window(
    n: int,
    coefficients: tuple,
    clamp_non_negative: bool = False,
) -> mx.array:
    """
    Generalized cosine window with arbitrary coefficients.

    w[k] = a0 - a1*cos(2*pi*k/(n-1)) + a2*cos(4*pi*k/(n-1)) - ...

    Parameters
    ----------
    n : int
        Window length.
    coefficients : tuple
        Tuple of (a0, a1, a2, ...) coefficients.
    clamp_non_negative : bool
        If True, clamp window values to be non-negative.

    Computed in float64 for precision, then cast to float32.
    This ensures perfect symmetry matching scipy.
    """
    if n <= 1:
        return mx.ones(n, dtype=mx.float32)

    k = np.arange(n, dtype=np.float64)
    denom = n - 1

    # Start with a0
    window = np.full(n, coefficients[0], dtype=np.float64)

    # Add cosine terms with alternating signs
    for i, coef in enumerate(coefficients[1:], 1):
        sign = -1 if i % 2 == 1 else 1
        window = window + sign * coef * np.cos(2 * i * np.pi * k / denom)

    if clamp_non_negative:
        window = np.maximum(window, 0.0)

    return mx.array(window.astype(np.float32))


# Generalized cosine window coefficients (a0, a1, a2, ...).
# Reference: Harris, F.J. (1978). "On the use of windows for harmonic analysis"
_COSINE_WINDOW_COEFFICIENTS = {
    "hann": (0.5, 0.5),
    "hamming": (0.54, 0.46),
    "blackman": (0.42, 0.5, 0.08),
}


def _hann(n: int) -> mx.array:
    """Hann window: w[k] = 0.5 - 0.5 * cos(2*pi*k/(n-1))."""
    return _generalized_cosine_window(n, _COSINE_WINDOW_COEFFICIENTS["hann"])


def _hamming(n: int) -> mx.array:
    """Hamming window: w[k] = 0.54 - 0.46 * cos(2*pi*k/(n-1))."""
    return _generalized_cosine_window(n, _COSINE_WINDOW_COEFFICIENTS["hamming"])


def _blackman(n: int) -> mx.array:
    """
    Blackman window.

    Clamped to non-negative since float64 can produce tiny negatives (~1e-17)
    at endpoints where theoretical value is exactly 0.
    """
    return _generalized_cosine_window(
        n, _COSINE_WINDOW_COEFFICIENTS["blackman"], clamp_non_negative=True
    )


def _bartlett(n: int) -> mx.array:
    """
    Bartlett (triangular) window: w[k] = 1 - |2*k/(n-1) - 1|

    Computed in float64 for precision, then cast to float32.
    """
    if n <= 1:
        return mx.ones(n, dtype=mx.float32)

    k = np.arange(n, dtype=np.float64)
    window = 1 - np.abs(2 * k / (n - 1) - 1)
    return mx.array(window.astype(np.float32))


def _rectangular(n: int) -> mx.array:
    """Rectangular (boxcar) window - all ones."""
    return mx.ones(n, dtype=mx.float32)


# Window function dispatch table for cleaner lookup
_WINDOW_FUNCTIONS: dict[str, Callable[[int], mx.array]] = {
    "hann": _hann,
    "hanning": _hann,  # Alias
    "hamming": _hamming,
    "blackman": _blackman,
    "bartlett": _bartlett,
    "triangular": _bartlett,  # Alias
    "rectangular": _rectangular,
    "boxcar": _rectangular,  # Alias
    "ones": _rectangular,  # Alias
}


# Caching Strategy:
# Window functions use a two-tier cache:
# 1. _get_window_cached: LRU cache storing bytes (hashable, for persistence)
# 2. _mlx_window_cache: Dict storing MLX arrays (avoids CPU→GPU transfer on hit)
#
# The MLX cache is checked first to avoid the overhead of:
# - bytes → np.frombuffer → mx.array conversion
# - CPU→GPU data transfer on every cache hit

# Secondary cache for MLX arrays (avoids CPU→GPU transfer on repeated access)
_mlx_window_cache: dict[tuple[str, int, bool], mx.array] = {}


@lru_cache(maxsize=128)
def _get_window_cached(
    window_name: str,
    n_fft: int,
    fftbins: bool,
) -> tuple[bytes, int]:
    """
    Compute window function with caching.

    Returns window data as bytes for efficient caching.
    """
    window_name = window_name.lower()

    # Use C++ extension if available
    if HAS_CPP_EXT and _ext is not None:
        # Map window names to C++ expected names
        cpp_name = window_name
        if window_name == "hanning":
            cpp_name = "hann"
        elif window_name == "triangular":
            cpp_name = "bartlett"
        elif window_name in ("boxcar", "ones"):
            cpp_name = "rectangular"

        w = _ext.generate_window(cpp_name, n_fft, fftbins)
        w_np = np.array(w, dtype=np.float32)
        return w_np.tobytes(), len(w_np)

    # Fallback to Python/NumPy implementation
    # For periodic (fftbins=True), we compute n_fft+1 points and drop the last
    # This matches scipy/librosa behavior for DFT-even windows
    n = n_fft + 1 if fftbins else n_fft

    # Use dispatch table for window function lookup
    window_func = _WINDOW_FUNCTIONS.get(window_name)
    if window_func is None:
        supported = sorted(set(_WINDOW_FUNCTIONS.keys()))
        raise ValueError(
            f"Unknown window type: '{window_name}'. Supported: {', '.join(supported)}"
        )
    w = window_func(n)

    # For periodic windows, drop the last sample
    if fftbins and n > n_fft:
        w = w[:n_fft]

    # Convert to bytes for caching
    w_np = np.array(w, dtype=np.float32)
    return w_np.tobytes(), len(w_np)


def get_window(
    window: str | mx.array,
    n_fft: int,
    fftbins: bool = True,
) -> mx.array:
    """
    Get a window function.

    Results are cached for repeated calls with identical parameters.

    Parameters
    ----------
    window : str or mx.array
        Window specification. If string, one of:
        - 'hann' or 'hanning': Hann window
        - 'hamming': Hamming window
        - 'blackman': Blackman window
        - 'bartlett' or 'triangular': Bartlett (triangular) window
        - 'rectangular' or 'boxcar' or 'ones': Rectangular window (all ones)
        If mx.array, used directly (must have length n_fft).
    n_fft : int
        Length of the window.
    fftbins : bool, default=True
        If True, create a periodic window for use with FFT (DFT-even).
        If False, create a symmetric window.

    Returns
    -------
    mx.array
        Window of shape (n_fft,) with dtype float32.

    Raises
    ------
    ValueError
        If window string is not recognized or array has wrong length.

    Examples
    --------
    >>> window = get_window('hann', 2048)
    >>> window.shape
    (2048,)
    """
    if isinstance(window, mx.array):
        if window.shape[0] != n_fft:
            raise ValueError(
                f"Window array length ({window.shape[0]}) must match n_fft ({n_fft})"
            )
        return window.astype(mx.float32)

    if not isinstance(window, str):
        raise TypeError(f"window must be str or mx.array, got {type(window).__name__}")

    # Check MLX cache first (avoids CPU→GPU transfer)
    cache_key = (window.lower(), n_fft, fftbins)
    if cache_key in _mlx_window_cache:
        return _mlx_window_cache[cache_key]

    # Get cached window data (bytes)
    window_bytes, length = _get_window_cached(window, n_fft, fftbins)

    # Convert from bytes to MLX array and cache
    w = np.frombuffer(window_bytes, dtype=np.float32)
    result = mx.array(w)
    _mlx_window_cache[cache_key] = result
    return result
