"""
Time-domain audio primitives.

Provides signal framing, RMS energy computation, and pre-emphasis filtering.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ._frame_impl import frame_signal_batched
from ._validation import validate_positive


def frame(
    y: mx.array,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
) -> mx.array:
    """
    Frame an audio signal into overlapping windows.

    Uses the fastest available implementation:
    1. C++ extension (if available)
    2. mx.as_strided (MLX >= 0.5, zero-copy)
    3. Gather-based fallback

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,) or (batch, samples).
    frame_length : int
        Length of each frame in samples.
    hop_length : int
        Number of samples between frame starts.
    axis : int, default=-1
        Axis along which to frame. Must be -1 (last axis).

    Returns
    -------
    mx.array
        Framed signal.
        Shape: (n_frames, frame_length) for 1D input.
        Shape: (batch, n_frames, frame_length) for 2D input.

    Raises
    ------
    ValueError
        If parameters are invalid or signal is too short.

    Examples
    --------
    >>> y = mx.array(np.random.randn(22050).astype(np.float32))
    >>> frames = frame(y, frame_length=2048, hop_length=512)
    >>> frames.shape
    (40, 2048)
    """
    validate_positive(frame_length, "frame_length")
    validate_positive(hop_length, "hop_length")

    if axis != -1:
        raise ValueError(f"axis must be -1, got {axis}")

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]  # Add batch dimension

    # Use shared optimized framing implementation
    frames = frame_signal_batched(y, frame_length, hop_length)

    # Remove batch dimension if input was 1D
    if input_is_1d:
        frames = frames[0]

    return frames


def rms(
    y: mx.array,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = "constant",
) -> mx.array:
    """
    Compute root-mean-square (RMS) energy per frame.

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,) or (batch, samples).
    frame_length : int, default=2048
        Length of each frame in samples.
    hop_length : int, default=512
        Number of samples between frame starts.
    center : bool, default=True
        If True, center-pad the signal before framing.
    pad_mode : str, default='constant'
        Padding mode if center=True. One of: 'constant', 'edge'.

    Returns
    -------
    mx.array
        RMS energy per frame.
        Shape: (1, n_frames) for 1D input (matches librosa).
        Shape: (batch, 1, n_frames) for 2D input.

    Examples
    --------
    >>> y = mx.array(np.random.randn(22050).astype(np.float32))
    >>> energy = rms(y, frame_length=2048, hop_length=512)
    >>> energy.shape
    (1, 44)
    """
    validate_positive(frame_length, "frame_length")
    validate_positive(hop_length, "hop_length")

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]  # Add batch dimension

    # Center padding
    if center:
        pad_length = frame_length // 2
        if pad_mode == "constant":
            y = mx.pad(y, [(0, 0), (pad_length, pad_length)], mode="constant")
        elif pad_mode == "edge":
            y = mx.pad(y, [(0, 0), (pad_length, pad_length)], mode="edge")
        else:
            raise ValueError(
                f"Unknown pad_mode: '{pad_mode}'. Supported: 'constant', 'edge'"
            )

    # Frame the signal: (batch, n_frames, frame_length)
    frames = frame(y, frame_length, hop_length)

    # Compute RMS: sqrt(mean(x^2))
    rms_energy = mx.sqrt(mx.mean(frames**2, axis=-1, keepdims=True))

    # Transpose to (batch, 1, n_frames) to match librosa convention
    rms_energy = mx.transpose(rms_energy, (0, 2, 1))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        rms_energy = rms_energy[0]

    return rms_energy


def _preemphasis_mlx(
    y: mx.array,
    coef: float,
    zi: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """
    MLX-native preemphasis implementation.

    This is a vectorized FIR filter: y_out[n] = y[n] - coef * y[n-1]
    Much faster than scipy.signal.lfilter as it avoids CPU<->GPU transfer.

    The key insight is that scipy.signal.lfilter with zi adds zi directly
    to the first output (it's the filter state, not the previous sample).
    So: y_out[0] = y[0] + zi, y_out[n] = y[n] - coef * y[n-1] for n >= 1.

    To match librosa's default zi = 2*y[0] - y[1], we compute:
    y_out[0] = y[0] + zi = y[0] + (2*y[0] - y[1]) = 3*y[0] - y[1]
    """
    # Compute initial state (linear extrapolation to match librosa)
    if zi is None:
        zi = 2 * y[..., 0:1] - y[..., 1:2]
    elif zi.ndim == 0:
        zi = zi[None]

    # For n >= 1: y_out[n] = y[n] - coef * y[n-1]
    # We compute this for all positions, then fix up position 0
    y_shifted = mx.concatenate([y[..., 0:1], y[..., :-1]], axis=-1)
    y_out = y - coef * y_shifted

    # Fix position 0: y_out[0] = y[0] + zi (scipy lfilter behavior)
    # Currently y_out[0] = y[0] - coef * y[0], we need y[0] + zi
    y_out_0 = y[..., 0:1] + zi
    y_out = mx.concatenate([y_out_0, y_out[..., 1:]], axis=-1)

    # Final state is the last sample (for streaming continuation)
    zf = y[..., -1:]

    return y_out, zf


def preemphasis(
    y: mx.array,
    coef: float = 0.97,
    zi: mx.array | None = None,
    return_zf: bool = False,
    use_mlx: bool = True,
) -> mx.array | tuple[mx.array, mx.array]:
    """
    Apply pre-emphasis filter to emphasize high frequencies.

    The pre-emphasis filter is defined as:
        y_out[n] = y[n] - coef * y[n-1]

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,) or (batch, samples).
    coef : float, default=0.97
        Pre-emphasis coefficient. Typical values are 0.95-0.97.
    zi : mx.array, optional
        Initial filter state. If None, uses linear extrapolation
        (zi = 2*y[0] - y[1]) to match librosa behavior.
    return_zf : bool, default=False
        If True, return the final filter state.
    use_mlx : bool, default=True
        If True, use MLX-native implementation (faster, matches librosa exactly).
        If False, use scipy.signal.lfilter.

    Returns
    -------
    mx.array or tuple
        Pre-emphasized signal. Same shape as input.
        If return_zf=True, returns (y_out, zf) where zf is final state.

    Examples
    --------
    >>> y = mx.array(np.random.randn(22050).astype(np.float32))
    >>> y_emph = preemphasis(y, coef=0.97)
    >>> y_emph.shape
    (22050,)
    """
    if not 0.0 <= coef <= 1.0:
        raise ValueError(f"coef must be in [0, 1], got {coef}")

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, signal_length = y.shape

    if use_mlx:
        # Fast MLX-native implementation
        # Handle zi shape for MLX path
        if zi is not None:
            if isinstance(zi, mx.array):
                if zi.ndim == 0:
                    zi = mx.broadcast_to(zi[None, None], (batch_size, 1))
                elif zi.ndim == 1:
                    if zi.shape[0] != batch_size:
                        zi = mx.broadcast_to(zi[None, :], (batch_size, 1))
                    else:
                        zi = zi[:, None]
            else:
                zi = mx.array([[zi]] * batch_size)

        y_out, zf = _preemphasis_mlx(y, coef, zi)
    else:
        # Use scipy.signal.lfilter for exact librosa compatibility
        from scipy import signal

        # Filter coefficients
        b = np.array([1.0, -coef], dtype=np.float32)
        a = np.array([1.0], dtype=np.float32)

        y_np = np.array(y)

        # Default zi: linear extrapolation (matches librosa)
        if zi is None:
            zi_np = 2 * y_np[..., 0:1] - y_np[..., 1:2]
        else:
            zi_np = np.atleast_1d(np.array(zi))
            if zi_np.ndim == 1 and zi_np.shape[0] != batch_size:
                zi_np = np.broadcast_to(zi_np, (batch_size, 1))
            elif zi_np.ndim == 1:
                zi_np = zi_np[:, None]

        y_out_np, zf_np = signal.lfilter(
            b, a, y_np, zi=zi_np.astype(np.float32), axis=-1
        )

        y_out = mx.array(y_out_np.astype(np.float32))
        zf = mx.array(zf_np.astype(np.float32))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        y_out = y_out[0]
        zf = zf[0]

    if return_zf:
        return y_out, zf
    return y_out


def deemphasis(
    y: mx.array,
    coef: float = 0.97,
    zi: mx.array | None = None,
    return_zf: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """
    Apply de-emphasis filter (inverse of pre-emphasis).

    The de-emphasis filter is defined as:
        y_out[n] = y[n] + coef * y_out[n-1]

    Parameters
    ----------
    y : mx.array
        Pre-emphasized audio signal. Shape: (samples,) or (batch, samples).
    coef : float, default=0.97
        De-emphasis coefficient (same as pre-emphasis).
    zi : mx.array, optional
        Initial filter state. If None, applies correction to match
        librosa's preemphasis default initialization.
    return_zf : bool, default=False
        If True, return the final filter state.

    Returns
    -------
    mx.array or tuple
        De-emphasized signal. Same shape as input.
        If return_zf=True, returns (y_out, zf) where zf is final state.

    Examples
    --------
    >>> y_emph = preemphasis(y, coef=0.97)
    >>> y_recovered = deemphasis(y_emph, coef=0.97)
    """
    if not 0.0 <= coef <= 1.0:
        raise ValueError(f"coef must be in [0, 1], got {coef}")

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, signal_length = y.shape

    # De-emphasis is a recursive filter (IIR)
    # y_out[n] = y[n] + coef * y_out[n-1]
    # Use scipy.signal.lfilter for efficiency and correctness
    from scipy import signal

    # Filter coefficients: H(z) = 1 / (1 - coef * z^-1)
    b = np.array([1.0], dtype=np.float32)
    a = np.array([1.0, -coef], dtype=np.float32)

    y_np = np.array(y)

    if zi is not None:
        # Use provided initial state
        zi_np = np.array(zi)
        if zi_np.ndim == 0:
            zi_np = np.array([[zi_np]] * batch_size)
        elif zi_np.ndim == 1:
            if zi_np.shape[0] == batch_size:
                zi_np = zi_np[:, None]
            else:
                zi_np = np.broadcast_to(zi_np[None, :], (batch_size, 1))

        y_out_np, zf_np = signal.lfilter(b, a, y_np, zi=zi_np, axis=-1)
    else:
        # Apply correction to match librosa's preemphasis default zi
        # librosa preemphasis uses zi = 2*y[0] - y[1], so deemphasis
        # needs to compensate for this
        zi_zeros = np.zeros((batch_size, 1), dtype=np.float32)
        y_out_np, zf_np = signal.lfilter(b, a, y_np, zi=zi_zeros, axis=-1)

        # Apply correction factor (matches librosa implementation)
        # The preemphasis filter used zi = 2*y[0] - y[1], which adds an offset
        # to the filtered signal. To recover the original, we subtract this offset
        # (scaled by the IIR filter's decaying impulse response).
        # Derivation: corr = ((2-c)*y[0] - y[1]) / (3-c), then decay = c^n
        corr = ((2 - coef) * y_np[:, 0:1] - y_np[:, 1:2]) / (3 - coef)
        decay = coef ** np.arange(signal_length, dtype=np.float32)
        y_out_np = y_out_np - corr * decay

    y_out = mx.array(y_out_np.astype(np.float32))
    zf = mx.array(zf_np.astype(np.float32))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        y_out = y_out[0]
        zf = zf[0]

    if return_zf:
        return y_out, zf
    return y_out
