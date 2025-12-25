"""
Onset detection primitives.

Provides onset strength envelope and onset detection for beat tracking
and rhythm analysis.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ._validation import validate_positive
from .mel import melspectrogram


def onset_strength(
    y: mx.array | None = None,
    sr: int = 22050,
    S: mx.array | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    lag: int = 1,
    max_size: int = 1,
    detrend: bool = False,
    aggregate: str = "mean",
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> mx.array:
    """
    Compute onset strength envelope from spectral flux.

    The onset strength is computed as the positive first-order difference
    of a mel-frequency spectrogram. This follows librosa's implementation.

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    S : mx.array, optional
        Pre-computed magnitude spectrogram. If provided, y is ignored.
        Shape: (n_freq, n_frames) or (batch, n_freq, n_frames).
    n_fft : int, default=2048
        FFT size.
    hop_length : int, default=512
        Hop length for STFT.
    win_length : int, optional
        Window length. Default: n_fft.
    window : str or mx.array, default='hann'
        Window function.
    center : bool, default=True
        Center padding for STFT.
    pad_mode : str, default='constant'
        Padding mode for STFT.
    lag : int, default=1
        Time lag for computing differences.
    max_size : int, default=1
        Size of the local max filter for suppression. 1 means no filtering.
    detrend : bool, default=False
        If True, subtract a running local mean.
    aggregate : str, default='mean'
        Aggregation across frequency: 'mean', 'median', or 'max'.
    n_mels : int, default=128
        Number of mel bands.
    fmin : float, default=0.0
        Minimum frequency for mel bands.
    fmax : float, optional
        Maximum frequency for mel bands. Default: sr/2.

    Returns
    -------
    mx.array
        Onset strength envelope.
        Shape: (n_frames,) for 1D input.
        Shape: (batch, n_frames) for batched input.

    Notes
    -----
    This is compatible with librosa.onset.onset_strength().
    For best results with beat tracking, use the default parameters.

    Examples
    --------
    >>> envelope = onset_strength(y, sr=22050)
    >>> envelope.shape
    (87,)  # depends on signal length

    >>> # With pre-computed mel spectrogram
    >>> S = melspectrogram(y, sr=22050)
    >>> envelope = onset_strength(S=S, sr=22050)
    """
    validate_positive(lag, "lag")
    validate_positive(max_size, "max_size")

    # Get mel spectrogram if not provided
    if S is None:
        if y is None:
            raise ValueError("Either y (audio) or S (spectrogram) must be provided")

        S = melspectrogram(
            y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=1.0,  # Use magnitude, not power
        )

    # Handle batched input
    input_is_1d = S.ndim == 2
    if input_is_1d:
        S = S[None, :]  # Add batch dimension

    # Convert to dB scale (log compression)
    # Using log1p for numerical stability
    S_db = mx.log1p(S * 1e4)

    # Compute spectral flux (positive differences across time)
    # diff[..., t] = S[..., t] - S[..., t-lag]
    S_diff = S_db[..., lag:] - S_db[..., :-lag]

    # Half-wave rectification (keep only positive increases)
    S_diff = mx.maximum(S_diff, 0.0)

    # Aggregate across frequency bands
    if aggregate == "mean":
        onset_env = mx.mean(S_diff, axis=1)
    elif aggregate == "median":
        # MLX doesn't have median, use numpy
        S_diff_np = np.array(S_diff)
        onset_env = mx.array(np.median(S_diff_np, axis=1).astype(np.float32))
    elif aggregate == "max":
        onset_env = mx.max(S_diff, axis=1)
    else:
        raise ValueError(f"Unknown aggregate: '{aggregate}'. Use 'mean', 'median', or 'max'")

    # Local max filtering for suppression (if max_size > 1)
    if max_size > 1:
        # Use numpy for max pooling
        onset_np = np.array(onset_env)
        from scipy.ndimage import maximum_filter1d

        onset_filtered = maximum_filter1d(onset_np, size=max_size, axis=-1)
        onset_env = mx.array(onset_filtered.astype(np.float32))

    # Detrend by subtracting local mean
    if detrend:
        # Simple running mean detrending
        kernel_size = max(1, onset_env.shape[-1] // 10)
        onset_np = np.array(onset_env)
        from scipy.ndimage import uniform_filter1d

        local_mean = uniform_filter1d(onset_np, size=kernel_size, axis=-1)
        onset_env = mx.array((onset_np - local_mean).astype(np.float32))
        onset_env = mx.maximum(onset_env, 0.0)

    # Pad to match original frame count (lost `lag` frames at start)
    pad_width = [(0, 0)] * (onset_env.ndim - 1) + [(lag, 0)]
    onset_env = mx.pad(onset_env, pad_width, mode="constant")

    # Remove batch dimension if input was 2D
    if input_is_1d:
        onset_env = onset_env[0]

    return onset_env


def onset_detect(
    y: mx.array | None = None,
    sr: int = 22050,
    onset_envelope: mx.array | None = None,
    hop_length: int = 512,
    backtrack: bool = False,
    units: str = "frames",
    pre_max: int = 1,
    post_max: int = 1,
    pre_avg: int = 1,
    post_avg: int = 1,
    delta: float = 0.07,
    wait: int = 1,
    **kwargs,
) -> mx.array:
    """
    Detect onset events using peak-picking on onset envelope.

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    onset_envelope : mx.array, optional
        Pre-computed onset strength envelope. If provided, y is ignored.
    hop_length : int, default=512
        Hop length (for converting to time/samples).
    backtrack : bool, default=False
        If True, backtrack onset positions to the nearest local minimum.
    units : str, default='frames'
        Output units: 'frames', 'samples', or 'time'.
    pre_max : int, default=1
        Number of frames before current frame to search for local max.
    post_max : int, default=1
        Number of frames after current frame to search for local max.
    pre_avg : int, default=1
        Number of frames before current frame for local average.
    post_avg : int, default=1
        Number of frames after current frame for local average.
    delta : float, default=0.07
        Threshold offset for peak picking (onset must exceed mean + delta).
    wait : int, default=1
        Minimum frames between detected onsets.
    **kwargs
        Additional arguments passed to onset_strength if y is provided.

    Returns
    -------
    mx.array
        Detected onset positions.
        Shape: (n_onsets,) for 1D input.

    Notes
    -----
    This is compatible with librosa.onset.onset_detect().

    Examples
    --------
    >>> onsets = onset_detect(y, sr=22050, units='time')
    >>> print(f"First onset at {onsets[0]:.2f} seconds")

    >>> # With pre-computed envelope
    >>> env = onset_strength(y, sr=22050)
    >>> onsets = onset_detect(onset_envelope=env, sr=22050)
    """
    validate_positive(hop_length, "hop_length")

    # Get onset envelope if not provided
    if onset_envelope is None:
        if y is None:
            raise ValueError(
                "Either y (audio) or onset_envelope must be provided"
            )
        onset_envelope = onset_strength(y, sr=sr, hop_length=hop_length, **kwargs)

    # Convert to numpy for peak picking
    onset_np = np.array(onset_envelope)

    # Handle batched input - process first batch only for now
    if onset_np.ndim > 1:
        onset_np = onset_np[0]

    n_frames = len(onset_np)

    # Peak picking algorithm (matches librosa)
    # An onset is detected at frame i if:
    # 1. onset_env[i] >= onset_env[i-pre_max:i+post_max+1].max()
    # 2. onset_env[i] >= onset_env[i-pre_avg:i+post_avg+1].mean() + delta
    # 3. i >= last_onset + wait

    onsets = []
    last_onset = -wait - 1

    for i in range(n_frames):
        # Check local maximum condition
        start_max = max(0, i - pre_max)
        end_max = min(n_frames, i + post_max + 1)
        local_max = np.max(onset_np[start_max:end_max])

        if onset_np[i] < local_max:
            continue

        # Check threshold condition (mean + delta)
        start_avg = max(0, i - pre_avg)
        end_avg = min(n_frames, i + post_avg + 1)
        local_mean = np.mean(onset_np[start_avg:end_avg])

        if onset_np[i] < local_mean + delta:
            continue

        # Check wait condition
        if i < last_onset + wait:
            continue

        onsets.append(i)
        last_onset = i

    onsets = np.array(onsets, dtype=np.int64)

    # Backtrack to local minima if requested
    if backtrack and len(onsets) > 0:
        # For each onset, find the local minimum in the preceding region
        for i, onset in enumerate(onsets):
            # Search region before onset
            start = max(0, onset - pre_max * 2)
            if start < onset:
                region = onset_np[start:onset]
                min_idx = np.argmin(region)
                onsets[i] = start + min_idx

    # Convert units
    if units == "frames":
        result = onsets
    elif units == "samples":
        result = onsets * hop_length
    elif units == "time":
        result = (onsets * hop_length) / sr
    else:
        raise ValueError(f"Unknown units: '{units}'. Use 'frames', 'samples', or 'time'")

    return mx.array(result)


def onset_strength_multi(
    y: mx.array,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    channels: list[tuple[float, float]] | None = None,
    aggregate: str = "mean",
    **kwargs,
) -> mx.array:
    """
    Compute multi-band onset strength.

    This computes onset strength separately for different frequency bands,
    useful for separating percussive and tonal onsets.

    Parameters
    ----------
    y : mx.array
        Audio waveform. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, default=512
        Hop length.
    channels : list of (fmin, fmax), optional
        Frequency bands to compute. Default: [(20, 200), (200, 2000), (2000, sr/2)].
    aggregate : str, default='mean'
        Aggregation within each band.
    **kwargs
        Additional arguments passed to onset_strength.

    Returns
    -------
    mx.array
        Onset strength for each channel.
        Shape: (n_channels, n_frames) for 1D input.
        Shape: (batch, n_channels, n_frames) for batched input.

    Examples
    --------
    >>> # Default 3-band: low (bass), mid (instruments), high (cymbals)
    >>> multi_env = onset_strength_multi(y, sr=22050)
    >>> multi_env.shape
    (3, 87)

    >>> # Custom bands
    >>> channels = [(20, 100), (100, 1000), (1000, 4000), (4000, 11025)]
    >>> multi_env = onset_strength_multi(y, sr=22050, channels=channels)
    """
    if channels is None:
        # Default: low/mid/high bands
        nyquist = sr / 2
        channels = [
            (20.0, 200.0),      # Bass/kick drum
            (200.0, 2000.0),    # Mid-range
            (2000.0, nyquist),  # High frequencies/cymbals
        ]

    # Compute mel spectrogram once
    S = melspectrogram(
        y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        power=1.0,
        **kwargs,
    )

    # Handle batched input
    input_is_1d = S.ndim == 2
    if input_is_1d:
        S = S[None, :]

    # Compute onset strength for each channel
    envelopes = []
    for fmin, fmax in channels:
        env = onset_strength(
            S=S,
            sr=sr,
            hop_length=hop_length,
            aggregate=aggregate,
            fmin=fmin,
            fmax=fmax,
            n_mels=128,
        )
        envelopes.append(env)

    # Stack channels
    result = mx.stack(envelopes, axis=-2)

    if input_is_1d:
        result = result[0]

    return result
