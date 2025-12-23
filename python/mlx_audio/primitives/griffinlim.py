"""
Griffin-Lim phase reconstruction algorithm.

Reconstructs audio from magnitude spectrograms using iterative
phase estimation.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ._validation import validate_positive, validate_range
from .stft import istft, magnitude, phase, stft


def griffinlim(
    S: mx.array,
    n_iter: int = 32,
    hop_length: int | None = None,
    win_length: int | None = None,
    n_fft: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    length: int | None = None,
    pad_mode: str = "constant",
    momentum: float = 0.99,
    init: str = "random",
    random_state: int | None = None,
) -> mx.array:
    """
    Reconstruct audio from magnitude spectrogram using Griffin-Lim.

    The Griffin-Lim algorithm iteratively estimates phase by:
    1. Starting with random or zero phase
    2. Reconstructing signal via ISTFT
    3. Computing STFT of reconstructed signal
    4. Replacing magnitude with target, keeping estimated phase
    5. Repeating until convergence

    Parameters
    ----------
    S : mx.array
        Magnitude spectrogram.
        Shape: (n_fft//2+1, n_frames) or (batch, n_fft//2+1, n_frames).
    n_iter : int, default=32
        Number of iterations.
    hop_length : int, optional
        Hop length for STFT/ISTFT. Default: n_fft // 4.
    win_length : int, optional
        Window length. Default: n_fft.
    n_fft : int, optional
        FFT size. Default: 2 * (freq_bins - 1).
    window : str or mx.array, default='hann'
        Window function.
    center : bool, default=True
        Center padding for STFT.
    length : int, optional
        Output signal length.
    pad_mode : str, default='constant'
        Padding mode for STFT.
    momentum : float, default=0.99
        Momentum for accelerated convergence (Perraudin et al. 2013).
        Set to 0 for classic Griffin-Lim.
    init : str, default='random'
        Phase initialization:
        - 'random': Random uniform phase in [-pi, pi]
        - 'zeros': Zero phase (all phases start at 0)
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    mx.array
        Reconstructed audio signal.
        Shape: (samples,) or (batch, samples).

    Notes
    -----
    The Griffin-Lim algorithm converges to a local minimum and may not
    perfectly reconstruct the original signal. For better quality,
    consider neural vocoder approaches.

    The momentum parameter significantly speeds up convergence.
    Values close to 1 (like 0.99) work well in practice.

    Examples
    --------
    >>> S = magnitude(stft(y))
    >>> y_reconstructed = griffinlim(S, n_iter=32)
    """
    validate_positive(n_iter, "n_iter")
    validate_range(momentum, "momentum", min_val=0.0, max_val=1.0, max_inclusive=False)

    # Handle batched vs non-batched
    is_batched = S.ndim == 3
    if not is_batched:
        S = S[None, :]  # Add batch dimension

    batch_size, freq_bins, n_frames = S.shape

    # Infer n_fft from frequency bins
    if n_fft is None:
        n_fft = 2 * (freq_bins - 1)

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Initialize phase
    rng = np.random.default_rng(random_state)
    if init == "random":
        angles_np = rng.uniform(-np.pi, np.pi, (batch_size, freq_bins, n_frames))
        angles = mx.array(angles_np.astype(np.float32))
    elif init == "zeros":
        angles = mx.zeros((batch_size, freq_bins, n_frames))
    else:
        raise ValueError(f"Unknown init: '{init}'. Supported: 'random', 'zeros'")

    # Build initial complex spectrogram
    # S_complex = S * exp(j * angles)
    rebuilt = S * mx.exp(1j * angles.astype(mx.complex64))

    # Store previous for momentum
    tprev = rebuilt

    # Iterative reconstruction
    for _ in range(n_iter):
        # Inverse STFT to get time domain signal
        y_estimate = istft(
            rebuilt,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            length=length,
        )

        # Forward STFT to get new phase estimate
        rebuilt_new = stft(
            y_estimate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )

        # Handle shape mismatch between STFT output and target magnitude
        # This can occur due to rounding in frame count calculation:
        # n_frames_stft = 1 + (length - n_fft) // hop_length
        # The mismatch is typically ±1 frame at the edges.
        new_n_frames = rebuilt_new.shape[-1]
        if new_n_frames > n_frames:
            rebuilt_new = rebuilt_new[..., :n_frames]
        elif new_n_frames < n_frames:
            pad_width = n_frames - new_n_frames
            rebuilt_new = mx.pad(
                rebuilt_new,
                [(0, 0)] * (rebuilt_new.ndim - 1) + [(0, pad_width)],
                mode="constant",
            )

        # Extract phase and apply magnitude constraint
        angles = phase(rebuilt_new)
        rebuilt_new = S * mx.exp(1j * angles.astype(mx.complex64))

        # Apply momentum acceleration (Perraudin et al. 2013)
        # "A fast Griffin-Lim algorithm" - IEEE WASPAA
        # Momentum adds (rebuilt_new - tprev) term, which pushes the estimate
        # in the direction it was already moving. This dramatically speeds
        # convergence (often 2-3x fewer iterations for same quality).
        if momentum > 0:
            rebuilt = rebuilt_new + momentum * (rebuilt_new - tprev)
            tprev = rebuilt_new
        else:
            rebuilt = rebuilt_new

    # Final reconstruction
    y = istft(
        rebuilt,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        length=length,
    )

    if not is_batched:
        y = y[0]

    return y


def griffinlim_iter(
    S: mx.array,
    angles: mx.array,
    hop_length: int,
    win_length: int,
    n_fft: int,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    momentum: float = 0.99,
    tprev: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Perform a single Griffin-Lim iteration.

    This is useful for implementing custom stopping criteria or
    monitoring convergence.

    Parameters
    ----------
    S : mx.array
        Magnitude spectrogram.
    angles : mx.array
        Current phase estimate.
    hop_length : int
        Hop length.
    win_length : int
        Window length.
    n_fft : int
        FFT size.
    window : str or mx.array, default='hann'
        Window function.
    center : bool, default=True
        Center padding.
    pad_mode : str, default='constant'
        Padding mode.
    momentum : float, default=0.99
        Momentum parameter.
    tprev : mx.array, optional
        Previous estimate for momentum.

    Returns
    -------
    tuple
        (new_angles, new_rebuilt, error) where error is reconstruction MSE.
    """
    # Build complex spectrogram
    rebuilt = S * mx.exp(1j * angles.astype(mx.complex64))

    # Inverse STFT
    y_estimate = istft(
        rebuilt,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
    )

    # Forward STFT
    rebuilt_new = stft(
        y_estimate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Compute reconstruction error
    mag_new = magnitude(rebuilt_new)
    error = mx.mean((S - mag_new) ** 2)

    # Extract phase
    new_angles = phase(rebuilt_new)

    # Apply magnitude constraint with momentum
    rebuilt_new = S * mx.exp(1j * new_angles.astype(mx.complex64))

    if momentum > 0 and tprev is not None:
        rebuilt_out = rebuilt_new + momentum * (rebuilt_new - tprev)
    else:
        rebuilt_out = rebuilt_new

    return new_angles, rebuilt_out, error
