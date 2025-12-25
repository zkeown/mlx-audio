"""
Audio resampling.

Provides FFT-based and polyphase resampling for converting audio between
sample rates.

Note: FFT operations in MLX run on CPU, so resampling very long signals
may benefit from pre-processing with librosa/scipy on the host.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from ._validation import validate_positive


def resample(
    y: mx.array,
    orig_sr: int,
    target_sr: int,
    res_type: str = "fft",
    fix: bool = True,
    scale: bool = False,
    axis: int = -1,
) -> mx.array:
    """
    Resample audio to a different sample rate.

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,) or (batch, samples).
    orig_sr : int
        Original sample rate.
    target_sr : int
        Target sample rate.
    res_type : str, default='fft'
        Resampling method:
        - 'fft': FFT-based bandlimited resampling (default)
        - 'linear': Fast linear interpolation (not bandlimited, may alias)
    fix : bool, default=True
        Adjust output length to exactly target_sr/orig_sr * len(y).
    scale : bool, default=False
        Scale output amplitude by the resampling ratio.
    axis : int, default=-1
        Axis along which to resample.

    Returns
    -------
    mx.array
        Resampled audio signal.

    Notes
    -----
    FFT-based resampling is bandlimited and prevents aliasing, but runs
    on CPU in MLX. For very long signals, consider using librosa.resample
    for better performance.

    Examples
    --------
    >>> y_22k = resample(y_44k, orig_sr=44100, target_sr=22050)
    >>> y_22k.shape
    (11025,)  # Half the original length
    """
    validate_positive(orig_sr, "orig_sr")
    validate_positive(target_sr, "target_sr")

    # Handle same sample rate
    if orig_sr == target_sr:
        return y

    if res_type == "fft":
        return _resample_fft(y, orig_sr, target_sr, fix, scale, axis)
    elif res_type == "linear":
        return _resample_linear(y, orig_sr, target_sr, fix, scale, axis)
    else:
        raise ValueError(f"Unknown res_type: '{res_type}'. Supported: 'fft', 'linear'")


def _resample_fft(
    y: mx.array,
    orig_sr: int,
    target_sr: int,
    fix: bool,
    scale: bool,
    axis: int,
) -> mx.array:
    """
    FFT-based bandlimited resampling using scipy.signal.resample.

    This provides high-quality resampling matching librosa's 'fft' method.
    """
    from scipy.signal import resample as scipy_resample

    # Move axis to last position
    if axis != -1:
        y = mx.moveaxis(y, axis, -1)

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, orig_length = y.shape

    # Compute target length
    ratio = target_sr / orig_sr
    target_length = int(np.round(orig_length * ratio)) if fix else int(np.ceil(orig_length * ratio))

    if target_length == orig_length:
        result = y
    else:
        y_np = np.array(y)

        # Use scipy's resample for high-quality FFT-based resampling
        y_new = scipy_resample(y_np, target_length, axis=-1)

        # Scale if requested
        if scale:
            y_new *= ratio

        result = mx.array(y_new.astype(np.float32))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        result = result[0]

    # Move axis back if needed
    if axis != -1:
        result = mx.moveaxis(result, -1, axis)

    return result


def _resample_linear(
    y: mx.array,
    orig_sr: int,
    target_sr: int,
    fix: bool,
    scale: bool,
    axis: int,
) -> mx.array:
    """
    Linear interpolation resampling.

    WARNING: Linear interpolation is NOT bandlimited. When downsampling,
    frequencies above the new Nyquist (target_sr / 2) fold back into
    the audible range as aliasing artifacts. This is acceptable for:
    - Quick previews where quality doesn't matter
    - Signals known to be bandlimited below target Nyquist
    - Non-audio data (control signals, envelopes)

    For quality audio downsampling, use res_type='fft' instead.
    """
    # Move axis to last position
    if axis != -1:
        y = mx.moveaxis(y, axis, -1)

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, orig_length = y.shape

    # Compute target length
    ratio = target_sr / orig_sr
    target_length = int(np.round(orig_length * ratio)) if fix else int(np.ceil(orig_length * ratio))

    if target_length == orig_length:
        result = y
    else:
        # Linear interpolation using numpy
        y_np = np.array(y)

        # Create interpolation indices
        # Map target indices [0, target_length-1] to source [0, orig_length-1]
        target_indices = np.linspace(0, orig_length - 1, target_length)

        # Get integer and fractional parts
        idx_low = np.floor(target_indices).astype(np.int32)
        idx_high = np.minimum(idx_low + 1, orig_length - 1)
        frac = target_indices - idx_low

        # Linear interpolation
        y_new = (1 - frac) * y_np[:, idx_low] + frac * y_np[:, idx_high]

        # Scale if requested
        if scale:
            y_new *= ratio

        result = mx.array(y_new.astype(np.float32))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        result = result[0]

    # Move axis back if needed
    if axis != -1:
        result = mx.moveaxis(result, -1, axis)

    return result


def resample_poly(
    y: mx.array,
    up: int,
    down: int,
    axis: int = -1,
    padtype: str = "constant",
) -> mx.array:
    """
    Resample using polyphase filtering (integer ratio).

    This is more efficient than FFT-based resampling for simple integer
    ratios like 2:1 or 1:2.

    Parameters
    ----------
    y : mx.array
        Audio signal.
    up : int
        Upsampling factor.
    down : int
        Downsampling factor.
    axis : int, default=-1
        Axis along which to resample.
    padtype : str, default='constant'
        Padding type for filtering.

    Returns
    -------
    mx.array
        Resampled signal.

    Examples
    --------
    >>> y_22k = resample_poly(y_44k, up=1, down=2)  # 44100 -> 22050
    >>> y_48k = resample_poly(y_16k, up=3, down=1)  # 16000 -> 48000
    """
    validate_positive(up, "up")
    validate_positive(down, "down")

    # Simplify ratio
    g = math.gcd(up, down)
    up = up // g
    down = down // g

    if up == 1 and down == 1:
        return y

    # Move axis to last position
    if axis != -1:
        y = mx.moveaxis(y, axis, -1)

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    # NOTE: Uses scipy.signal.resample_poly for correctness.
    # Polyphase filtering requires designing an anti-aliasing FIR filter
    # with cutoff at min(1, down/up) * Nyquist, then efficiently computing
    # the convolution via polyphase decomposition. Getting this right
    # (filter design, phase alignment, edge handling) is non-trivial.
    y_np = np.array(y)

    try:
        from scipy.signal import resample_poly as scipy_resample_poly

        y_resampled = scipy_resample_poly(y_np, up, down, axis=-1, padtype=padtype)
    except ImportError:
        # Fallback to simple implementation
        # Upsample by up, then downsample by down
        if up > 1:
            # Zero-insert upsampling
            new_length = y_np.shape[-1] * up
            y_up = np.zeros((*y_np.shape[:-1], new_length), dtype=y_np.dtype)
            y_up[..., ::up] = y_np
            y_np = y_up

        if down > 1:
            # Simple decimation (not ideal without anti-alias filter)
            y_np = y_np[..., ::down]

        y_resampled = y_np

    result = mx.array(y_resampled.astype(np.float32))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        result = result[0]

    # Move axis back if needed
    if axis != -1:
        result = mx.moveaxis(result, -1, axis)

    return result
