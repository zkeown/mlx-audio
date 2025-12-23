"""
Spectral feature extraction.

Provides spectral centroid, bandwidth, rolloff, flatness, contrast,
and zero crossing rate computations.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ._extension import HAS_CPP_EXT, _ext
from ._validation import validate_positive, validate_range
from .framing import frame
from .stft import magnitude, stft


def _get_frequencies(sr: int, n_fft: int) -> mx.array:
    """Get frequency bin centers for STFT."""
    return mx.linspace(0, sr / 2.0, n_fft // 2 + 1)


def _compute_spectrogram(
    y: mx.array | None,
    S: mx.array | None,
    n_fft: int,
    hop_length: int,
    win_length: int | None,
    window: str | mx.array,
    center: bool,
    pad_mode: str,
    power: float = 1.0,
) -> mx.array:
    """Compute magnitude spectrogram if not provided."""
    if S is not None:
        return S

    if y is None:
        raise ValueError("Either y (audio) or S (spectrogram) must be provided")

    S_complex = stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    S = magnitude(S_complex)
    if power != 1.0:
        S = mx.power(S, power)
    return S


def spectral_centroid(
    y: mx.array | None = None,
    sr: int = 22050,
    S: mx.array | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    freq: mx.array | None = None,
) -> mx.array:
    """
    Compute spectral centroid (center of mass of spectrum).

    The spectral centroid indicates where the "center of mass" of the
    spectrum is located. It is associated with the brightness of a sound.

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    S : mx.array, optional
        Pre-computed magnitude spectrogram. If provided, y is ignored.
        Shape: (n_fft//2+1, n_frames) or (batch, n_fft//2+1, n_frames).
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
    freq : mx.array, optional
        Pre-computed frequency bin centers.

    Returns
    -------
    mx.array
        Spectral centroid for each frame.
        Shape: (1, n_frames) for 1D input.
        Shape: (batch, 1, n_frames) for batched input.

    Examples
    --------
    >>> centroid = spectral_centroid(y, sr=22050)
    >>> centroid.shape
    (1, 44)
    """
    S = _compute_spectrogram(
        y, S, n_fft, hop_length, win_length, window, center, pad_mode
    )

    if freq is None:
        freq = _get_frequencies(sr, n_fft)

    # Handle batched vs non-batched
    is_batched = S.ndim == 3
    if not is_batched:
        S = S[None, :]  # Add batch dimension

    # S shape: (batch, freq_bins, n_frames), freq shape: (freq_bins,)
    # centroid = sum(f * S) / sum(S)
    freq_expanded = freq[:, None]
    weighted_sum = mx.sum(freq_expanded * S, axis=1, keepdims=True)
    total_sum = mx.sum(S, axis=1, keepdims=True) + 1e-10
    centroid = weighted_sum / total_sum

    if not is_batched:
        centroid = centroid[0]

    return centroid


def spectral_bandwidth(
    y: mx.array | None = None,
    sr: int = 22050,
    S: mx.array | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    freq: mx.array | None = None,
    centroid: mx.array | None = None,
    p: float = 2.0,
    norm: bool = True,
) -> mx.array:
    """
    Compute spectral bandwidth (spread around centroid).

    The spectral bandwidth is the weighted average of the differences
    between the spectral components and the centroid.

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform.
    sr : int, default=22050
        Sample rate.
    S : mx.array, optional
        Pre-computed magnitude spectrogram.
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
    freq : mx.array, optional
        Pre-computed frequency bin centers.
    centroid : mx.array, optional
        Pre-computed spectral centroid.
    p : float, default=2.0
        Power for computing bandwidth (p=2 gives standard deviation).
    norm : bool, default=True
        Normalize by sum of spectrogram.

    Returns
    -------
    mx.array
        Spectral bandwidth for each frame.
        Shape: (1, n_frames) for 1D input.
        Shape: (batch, 1, n_frames) for batched input.

    Examples
    --------
    >>> bandwidth = spectral_bandwidth(y, sr=22050)
    >>> bandwidth.shape
    (1, 44)
    """
    S = _compute_spectrogram(
        y, S, n_fft, hop_length, win_length, window, center, pad_mode
    )

    if freq is None:
        freq = _get_frequencies(sr, n_fft)

    is_batched = S.ndim == 3
    if not is_batched:
        S = S[None, :]

    if centroid is None:
        centroid = spectral_centroid(
            S=S, sr=sr, n_fft=n_fft, freq=freq
        )  # (batch, 1, n_frames)

    # Ensure centroid has correct shape for broadcasting
    if centroid.ndim == 2:
        centroid = centroid[None, :]

    # S: (batch, freq_bins, n_frames)
    # centroid: (batch, 1, n_frames)
    # deviation = |f - centroid|
    freq_expanded = freq[None, :, None]  # (1, freq_bins, 1)
    deviation = mx.abs(freq_expanded - centroid)  # (batch, freq_bins, n_frames)

    # bandwidth = (sum(S * |f - centroid|^p) / sum(S))^(1/p)
    if norm:
        weighted = mx.sum(S * mx.power(deviation, p), axis=1, keepdims=True)
        normalizer = mx.sum(S, axis=1, keepdims=True) + 1e-10
        bandwidth = mx.power(weighted / normalizer, 1.0 / p)
    else:
        bandwidth = mx.power(
            mx.sum(S * mx.power(deviation, p), axis=1, keepdims=True), 1.0 / p
        )

    if not is_batched:
        bandwidth = bandwidth[0]

    return bandwidth


def _spectral_rolloff_numpy(
    S: mx.array,
    freq: mx.array,
    roll_percent: float,
) -> mx.array:
    """
    NumPy fallback implementation of spectral rolloff.

    Uses nested loops with np.searchsorted. Slower but exact librosa behavior.
    """
    # Compute cumulative energy
    cumsum = mx.cumsum(S, axis=1)  # (batch, freq_bins, n_frames)
    total_energy = cumsum[:, -1:, :]  # (batch, 1, n_frames)
    threshold = roll_percent * total_energy  # (batch, 1, n_frames)

    # Find the first bin that exceeds threshold
    cumsum_np = np.array(cumsum)
    threshold_np = np.array(threshold)
    freq_np = np.array(freq)

    batch_size, n_bins, n_frames = cumsum_np.shape
    rolloff_np = np.zeros((batch_size, 1, n_frames), dtype=np.float32)

    for b in range(batch_size):
        for t in range(n_frames):
            idx = np.searchsorted(cumsum_np[b, :, t], threshold_np[b, 0, t])
            idx = min(idx, n_bins - 1)
            rolloff_np[b, 0, t] = freq_np[idx]

    return mx.array(rolloff_np)


def spectral_rolloff(
    y: mx.array | None = None,
    sr: int = 22050,
    S: mx.array | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    freq: mx.array | None = None,
    roll_percent: float = 0.85,
    use_cpp: bool = True,
) -> mx.array:
    """
    Compute spectral rolloff frequency.

    The rolloff frequency is the frequency below which a specified
    percentage of the total spectral energy lies.

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform.
    sr : int, default=22050
        Sample rate.
    S : mx.array, optional
        Pre-computed magnitude spectrogram.
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
    freq : mx.array, optional
        Pre-computed frequency bin centers.
    roll_percent : float, default=0.85
        Fraction of energy below rolloff frequency.
    use_cpp : bool, default=True
        If True and C++ extension is available, use optimized C++ implementation.
        If False, use NumPy fallback (exact librosa compatibility).

    Returns
    -------
    mx.array
        Rolloff frequency for each frame.
        Shape: (1, n_frames) for 1D input.
        Shape: (batch, 1, n_frames) for batched input.

    Examples
    --------
    >>> rolloff = spectral_rolloff(y, sr=22050, roll_percent=0.85)
    >>> rolloff.shape
    (1, 44)
    """
    validate_range(roll_percent, "roll_percent", min_val=0.0, max_val=1.0)

    S = _compute_spectrogram(
        y, S, n_fft, hop_length, win_length, window, center, pad_mode
    )

    if freq is None:
        freq = _get_frequencies(sr, n_fft)

    is_batched = S.ndim == 3
    if not is_batched:
        S = S[None, :]

    # Use C++ extension if available and requested
    if use_cpp and HAS_CPP_EXT and _ext is not None:
        # C++ extension expects shape (batch, freq_bins, n_frames)
        # which is already our shape after adding batch dim if needed
        rolloff = _ext.spectral_rolloff(S, freq, roll_percent)
    else:
        # Fall back to NumPy implementation
        rolloff = _spectral_rolloff_numpy(S, freq, roll_percent)

    if not is_batched:
        rolloff = rolloff[0]

    return rolloff


def spectral_flatness(
    y: mx.array | None = None,
    S: mx.array | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    amin: float = 1e-10,
) -> mx.array:
    """
    Compute spectral flatness (Wiener entropy).

    Spectral flatness measures how noise-like a signal is.
    A value close to 1 indicates white noise (flat spectrum),
    while a value close to 0 indicates a tonal signal.

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform.
    S : mx.array, optional
        Pre-computed magnitude spectrogram.
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
    power : float, default=2.0
        Exponent for the magnitude spectrogram (2.0 for power).
    amin : float, default=1e-10
        Minimum amplitude to avoid log(0).

    Returns
    -------
    mx.array
        Spectral flatness for each frame.
        Shape: (1, n_frames) for 1D input.
        Shape: (batch, 1, n_frames) for batched input.

    Examples
    --------
    >>> flatness = spectral_flatness(y)
    >>> flatness.shape
    (1, 44)
    """
    S = _compute_spectrogram(
        y, S, n_fft, hop_length, win_length, window, center, pad_mode, power
    )

    is_batched = S.ndim == 3
    if not is_batched:
        S = S[None, :]

    # Clamp to minimum amplitude
    S = mx.maximum(S, amin)

    # Geometric mean: exp(mean(log(S)))
    log_S = mx.log(S)
    gmean = mx.exp(mx.mean(log_S, axis=1, keepdims=True))

    # Arithmetic mean
    amean = mx.mean(S, axis=1, keepdims=True)

    # Flatness = geometric_mean / arithmetic_mean
    flatness = gmean / (amean + 1e-10)

    if not is_batched:
        flatness = flatness[0]

    return flatness


def spectral_contrast(
    y: mx.array | None = None,
    sr: int = 22050,
    S: mx.array | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    freq: mx.array | None = None,
    fmin: float = 200.0,
    n_bands: int = 6,
    quantile: float = 0.02,
    linear: bool = False,
) -> mx.array:
    """
    Compute spectral contrast per frequency band.

    Spectral contrast measures the difference between peaks and valleys
    in each frequency band. It's useful for music classification.

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform.
    sr : int, default=22050
        Sample rate.
    S : mx.array, optional
        Pre-computed magnitude spectrogram.
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
    freq : mx.array, optional
        Pre-computed frequency bin centers.
    fmin : float, default=200.0
        Minimum frequency for band computation.
    n_bands : int, default=6
        Number of octave bands.
    quantile : float, default=0.02
        Quantile for peak/valley estimation.
    linear : bool, default=False
        If True, return linear contrast. If False, return log contrast.

    Returns
    -------
    mx.array
        Spectral contrast for each band and frame.
        Shape: (n_bands + 1, n_frames) for 1D input.
        Shape: (batch, n_bands + 1, n_frames) for batched input.

    Examples
    --------
    >>> contrast = spectral_contrast(y, sr=22050, n_bands=6)
    >>> contrast.shape
    (7, 44)
    """
    validate_positive(n_bands, "n_bands")
    validate_range(quantile, "quantile", min_val=0.0, max_val=1.0)

    S = _compute_spectrogram(
        y, S, n_fft, hop_length, win_length, window, center, pad_mode
    )

    if freq is None:
        freq = _get_frequencies(sr, n_fft)

    is_batched = S.ndim == 3
    if not is_batched:
        S = S[None, :]

    # NOTE: Uses numpy for complex band logic - octave band iteration,
    # per-band quantile sorting, and neighbor bin extension require operations
    # (selective indexing, partial sorting) not efficiently expressible in MLX.
    # This matches librosa's exact algorithm for reproducibility.
    S_np = np.array(S)
    freq_np = np.array(freq)

    batch_size, n_bins, n_frames = S_np.shape

    # Define octave bands: [0, fmin], [fmin, 2*fmin], [2*fmin, 4*fmin], ...
    # librosa creates n_bands + 2 edge points
    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0 ** np.arange(0, n_bands + 1))

    # Output shape
    valley = np.zeros((batch_size, n_bands + 1, n_frames), dtype=np.float32)
    peak = np.zeros_like(valley)

    for k, (f_low, f_high) in enumerate(zip(octa[:-1], octa[1:], strict=True)):
        # Find bins in current band
        current_band = np.logical_and(freq_np >= f_low, freq_np <= f_high)
        idx = np.flatnonzero(current_band)

        if len(idx) == 0:
            continue

        # Include neighbor bin at lower edge (except for first band)
        if k > 0 and idx[0] > 0:
            current_band[idx[0] - 1] = True

        # Extend last band to Nyquist
        if k == n_bands and idx[-1] + 1 < len(current_band):
            current_band[idx[-1] + 1 :] = True

        # Calculate n_quantile BEFORE removing last bin (matches librosa)
        n_quantile = int(np.maximum(np.rint(quantile * np.sum(current_band)), 1))

        # Extract sub-band spectrogram
        sub_band = S_np[:, current_band, :]

        # Remove last bin for all bands except the last
        if k < n_bands and sub_band.shape[1] > 1:
            sub_band = sub_band[:, :-1, :]

        # Sort along frequency axis
        sorted_sub = np.sort(sub_band, axis=1)

        # Valley (bottom quantile) and peak (top quantile)
        valley[:, k, :] = np.mean(sorted_sub[:, :n_quantile, :], axis=1)
        peak[:, k, :] = np.mean(sorted_sub[:, -n_quantile:, :], axis=1)

    # Compute contrast
    if linear:
        contrast_np = peak - valley
    else:
        # Use power_to_db like librosa (10 * log10)
        ref = 1.0
        amin = 1e-10
        peak_db = 10.0 * np.log10(np.maximum(peak, amin) / ref)
        valley_db = 10.0 * np.log10(np.maximum(valley, amin) / ref)
        contrast_np = peak_db - valley_db

    contrast = mx.array(contrast_np.astype(np.float32))

    if not is_batched:
        contrast = contrast[0]

    return contrast


def _zero_crossing_rate_mlx(frames: mx.array) -> mx.array:
    """
    MLX-native zero crossing rate computation.

    Parameters
    ----------
    frames : mx.array
        Framed signal. Shape: (batch, n_frames, frame_length)

    Returns
    -------
    mx.array
        Zero crossing rate. Shape: (batch, n_frames, 1)
    """
    # Sign of current samples: True if >= 0, False if < 0
    sign_current = frames >= 0

    # Sign of previous samples (shifted right by 1, pad with first sample's sign)
    sign_prev = mx.concatenate([sign_current[..., :1], sign_current[..., :-1]], axis=-1)

    # A crossing occurs when the sign changes
    crossings = (sign_current != sign_prev).astype(mx.float32)

    # ZCR = mean(crossings) over the frame
    # The first element is always 0 (no crossing at start), matching librosa's pad=False
    zcr = mx.mean(crossings, axis=-1, keepdims=True)

    return zcr


def zero_crossing_rate(
    y: mx.array,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = "edge",
    use_mlx: bool = True,
) -> mx.array:
    """
    Compute zero crossing rate per frame.

    The zero crossing rate is the rate at which the signal changes sign.
    It's a simple measure often used in speech processing.

    Parameters
    ----------
    y : mx.array
        Audio waveform. Shape: (samples,) or (batch, samples).
    frame_length : int, default=2048
        Length of each frame.
    hop_length : int, default=512
        Hop length between frames.
    center : bool, default=True
        Center padding for framing.
    pad_mode : str, default='edge'
        Padding mode if center=True. Uses 'edge' to match librosa.
    use_mlx : bool, default=True
        If True, use MLX-native implementation (faster).
        If False, use NumPy implementation (exact librosa compatibility).

    Returns
    -------
    mx.array
        Zero crossing rate for each frame.
        Shape: (1, n_frames) for 1D input.
        Shape: (batch, 1, n_frames) for batched input.

    Examples
    --------
    >>> zcr = zero_crossing_rate(y, frame_length=2048, hop_length=512)
    >>> zcr.shape
    (1, 44)
    """
    validate_positive(frame_length, "frame_length")
    validate_positive(hop_length, "hop_length")

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    # Center padding (librosa uses edge mode by default)
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

    # Frame the signal
    frames = frame(y, frame_length, hop_length)  # (batch, n_frames, frame_length)

    if use_mlx:
        # Fast MLX-native implementation
        zcr = _zero_crossing_rate_mlx(frames)
    else:
        # NumPy implementation for exact librosa compatibility
        # Zero crossings using signbit (matches librosa exactly)
        # A crossing occurs when signbit differs between consecutive samples
        y_np = np.array(frames)

        # Compute sign changes (diff reduces length by 1)
        sign_changes = np.abs(np.diff(np.signbit(y_np), axis=-1))

        # Pad with False at the start to match librosa's pad=False behavior
        # This keeps the shape as (batch, n_frames, frame_length)
        crossings = np.concatenate(
            [np.zeros((*y_np.shape[:-1], 1), dtype=sign_changes.dtype), sign_changes],
            axis=-1,
        )

        # ZCR = mean(crossings) over the frame
        zcr_np = np.mean(crossings, axis=-1, keepdims=True)

        zcr = mx.array(zcr_np.astype(np.float32))

    # Transpose to (batch, 1, n_frames) to match librosa convention
    zcr = mx.transpose(zcr, (0, 2, 1))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        zcr = zcr[0]

    return zcr
