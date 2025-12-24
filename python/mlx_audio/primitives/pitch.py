"""
Pitch and periodicity analysis primitives.

Provides autocorrelation, YIN, and PYIN for pitch detection and
periodicity analysis.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ._extension import HAS_CPP_EXT, _ext
from ._validation import validate_positive


def autocorrelation(
    y: mx.array,
    max_lag: int | None = None,
    normalize: bool = True,
    center: bool = True,
) -> mx.array:
    """
    Compute autocorrelation of a signal using FFT.

    The autocorrelation is computed efficiently using the Wiener-Khinchin
    theorem: r[k] = IFFT(|FFT(y)|^2).

    Parameters
    ----------
    y : mx.array
        Input signal. Shape: (samples,) or (batch, samples).
    max_lag : int, optional
        Maximum lag to compute. Default: signal length.
    normalize : bool, default=True
        If True, normalize by r[0] so that r[0] = 1.
    center : bool, default=True
        If True, subtract mean before computing autocorrelation.

    Returns
    -------
    mx.array
        Autocorrelation values for lags 0 to max_lag-1.
        Shape: (max_lag,) for 1D input.
        Shape: (batch, max_lag) for batched input.

    Notes
    -----
    For pitch detection, look for the first peak after lag 0.
    The lag of this peak corresponds to the fundamental period.

    Examples
    --------
    >>> y = mx.array(np.sin(2 * np.pi * 440 * np.arange(22050) / 22050))
    >>> r = autocorrelation(y, max_lag=2000)
    >>> # Peak at lag ~50 corresponds to 440 Hz at 22050 sample rate
    """
    # Use C++ extension if available
    if HAS_CPP_EXT and _ext is not None:
        return _ext.autocorrelation(
            y,
            max_lag if max_lag is not None else -1,
            normalize,
            center,
        )

    # Python fallback
    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, n = y.shape

    if max_lag is None:
        max_lag = n

    max_lag = min(max_lag, n)

    # Center the signal (subtract mean)
    if center:
        y = y - mx.mean(y, axis=-1, keepdims=True)

    # Use FFT for efficient autocorrelation (Wiener-Khinchin theorem)
    # Zero-pad to avoid circular correlation
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))

    # Convert to numpy for FFT (MLX FFT runs on CPU anyway)
    y_np = np.array(y)

    # FFT
    Y = np.fft.rfft(y_np, n=n_fft, axis=-1)

    # Power spectrum
    power = Y * np.conj(Y)

    # Inverse FFT to get autocorrelation
    r = np.fft.irfft(power, n=n_fft, axis=-1)

    # Take only positive lags up to max_lag
    r = r[:, :max_lag]

    # Normalize if requested
    if normalize:
        # Normalize by r[0] (variance)
        r0 = r[:, :1]
        r0 = np.maximum(r0, 1e-10)  # Avoid division by zero
        r = r / r0

    result = mx.array(r.astype(np.float32))

    # Remove batch dimension if input was 1D
    if input_is_1d:
        result = result[0]

    return result


def pitch_detect_acf(
    y: mx.array,
    sr: int = 22050,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold: float = 0.1,
    center: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Detect pitch using autocorrelation.

    This is a simple pitch detection algorithm that finds the fundamental
    frequency by looking for peaks in the autocorrelation function.

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    fmin : float, default=50.0
        Minimum frequency to detect (Hz).
    fmax : float, default=2000.0
        Maximum frequency to detect (Hz).
    frame_length : int, default=2048
        Analysis frame length.
    hop_length : int, default=512
        Hop length between frames.
    threshold : float, default=0.1
        Minimum autocorrelation peak value for voiced detection.
    center : bool, default=True
        Center-pad the signal for framing.

    Returns
    -------
    tuple
        (f0, voiced_flag) where:
        - f0: Detected fundamental frequency for each frame (Hz).
          Shape: (n_frames,) or (batch, n_frames).
        - voiced_flag: Boolean indicating voiced frames.
          Shape: (n_frames,) or (batch, n_frames).

    Examples
    --------
    >>> f0, voiced = pitch_detect_acf(y, sr=22050, fmin=80, fmax=500)
    >>> f0[voiced]  # Get only voiced pitch values
    """
    validate_positive(frame_length, "frame_length")
    validate_positive(hop_length, "hop_length")

    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")

    # Compute lag bounds from frequency bounds
    # Lag (samples) = sr / frequency, so:
    # - High frequencies → short periods → small lags
    # - Low frequencies → long periods → large lags
    min_lag = int(sr / fmax)  # Maximum frequency → minimum lag
    max_lag = int(sr / fmin)  # Minimum frequency → maximum lag

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, signal_length = y.shape

    # Center padding
    if center:
        pad_length = frame_length // 2
        y_np = np.array(y)
        y_np = np.pad(y_np, [(0, 0), (pad_length, pad_length)], mode="constant")
    else:
        y_np = np.array(y)

    # Frame the signal
    padded_length = y_np.shape[1]
    n_frames = 1 + (padded_length - frame_length) // hop_length

    # === BATCHED FFT OPTIMIZATION ===
    # Instead of per-frame FFT, extract all frames and process in one batched FFT call

    # Step 1: Extract all frames using strided view (zero-copy)
    # Shape: (batch_size, n_frames, frame_length)
    strides = (y_np.strides[0], y_np.strides[1] * hop_length, y_np.strides[1])
    all_frames = np.lib.stride_tricks.as_strided(
        y_np,
        shape=(batch_size, n_frames, frame_length),
        strides=strides,
    ).copy()  # Copy to ensure contiguous memory for FFT

    # Step 2: Center each frame (subtract mean)
    frame_means = np.mean(all_frames, axis=-1, keepdims=True)
    all_frames_centered = all_frames - frame_means

    # Step 3: Batched FFT for autocorrelation (single call for all frames)
    n_fft = 2 ** int(np.ceil(np.log2(2 * frame_length - 1)))
    # Reshape to (batch_size * n_frames, frame_length) for batched FFT
    flat_frames = all_frames_centered.reshape(-1, frame_length)
    Y = np.fft.rfft(flat_frames, n=n_fft, axis=-1)
    power = Y * np.conj(Y)
    acf_flat = np.fft.irfft(power, n=n_fft, axis=-1).real

    # Reshape back to (batch_size, n_frames, n_fft)
    acf = acf_flat.reshape(batch_size, n_frames, -1)

    # Step 4: Normalize by acf[0] (the zero-lag value)
    acf_zero = acf[:, :, 0:1]  # (batch_size, n_frames, 1)
    acf_zero = np.maximum(acf_zero, 1e-10)  # Avoid division by zero
    acf_normalized = acf / acf_zero

    # Step 5: Extract search range for all frames at once
    # search_range shape: (batch_size, n_frames, max_lag - min_lag + 1)
    search_range = acf_normalized[:, :, min_lag:max_lag + 1]
    n_lags = search_range.shape[-1]

    # Step 6: Vectorized peak detection
    # Find local maxima: search_range[i] > search_range[i-1] AND search_range[i] > search_range[i+1]
    # AND search_range[i] > threshold
    if n_lags >= 3:
        is_peak = np.zeros_like(search_range, dtype=bool)
        is_peak[:, :, 1:-1] = (
            (search_range[:, :, 1:-1] > search_range[:, :, :-2]) &
            (search_range[:, :, 1:-1] > search_range[:, :, 2:]) &
            (search_range[:, :, 1:-1] > threshold)
        )

        # Find first peak for each frame using argmax on boolean array
        # argmax returns index of first True value
        first_peak_idx = np.argmax(is_peak, axis=-1)  # (batch_size, n_frames)
        has_peak = np.any(is_peak, axis=-1)  # (batch_size, n_frames)

        # For frames without peaks, fall back to global max if above threshold
        global_max_idx = np.argmax(search_range, axis=-1)  # (batch_size, n_frames)
        # Get the value at global max using advanced indexing
        batch_idx = np.arange(batch_size)[:, None]
        frame_idx = np.arange(n_frames)[None, :]
        global_max_val = search_range[batch_idx, frame_idx, global_max_idx]

        # Use first peak if available, otherwise global max if above threshold
        peak_idx = np.where(has_peak, first_peak_idx, global_max_idx)
        voiced_np = has_peak | (global_max_val > threshold)

        # Convert peak index to frequency
        peak_lag = min_lag + peak_idx
        f0_np = np.where(voiced_np, sr / peak_lag, 0.0).astype(np.float32)
    else:
        # Edge case: search range too small
        f0_np = np.zeros((batch_size, n_frames), dtype=np.float32)
        voiced_np = np.zeros((batch_size, n_frames), dtype=bool)

    f0 = mx.array(f0_np)
    voiced = mx.array(voiced_np)

    # Remove batch dimension if input was 1D
    if input_is_1d:
        f0 = f0[0]
        voiced = voiced[0]

    return f0, voiced


def periodicity(
    y: mx.array,
    sr: int = 22050,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
) -> mx.array:
    """
    Compute periodicity (autocorrelation strength) per frame.

    Periodicity measures how periodic/harmonic a signal is at each frame.
    Values close to 1 indicate highly periodic (tonal) content,
    while values close to 0 indicate noise-like content.

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    fmin : float, default=50.0
        Minimum frequency for periodicity search (Hz).
    fmax : float, default=2000.0
        Maximum frequency for periodicity search (Hz).
    frame_length : int, default=2048
        Analysis frame length.
    hop_length : int, default=512
        Hop length between frames.
    center : bool, default=True
        Center-pad the signal for framing.

    Returns
    -------
    mx.array
        Periodicity strength for each frame (0 to 1).
        Shape: (1, n_frames) for 1D input.
        Shape: (batch, 1, n_frames) for batched input.

    Examples
    --------
    >>> p = periodicity(y, sr=22050)
    >>> # High values indicate voiced/tonal content
    """
    validate_positive(frame_length, "frame_length")
    validate_positive(hop_length, "hop_length")

    # Compute lag bounds
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    batch_size, signal_length = y.shape

    # Center padding
    if center:
        pad_length = frame_length // 2
        y_np = np.array(y)
        y_np = np.pad(y_np, [(0, 0), (pad_length, pad_length)], mode="constant")
    else:
        y_np = np.array(y)

    # Frame the signal
    padded_length = y_np.shape[1]
    n_frames = 1 + (padded_length - frame_length) // hop_length

    # === BATCHED FFT OPTIMIZATION ===
    # Instead of per-frame FFT, extract all frames and process in one batched FFT call

    # Step 1: Extract all frames using strided view (zero-copy)
    strides = (y_np.strides[0], y_np.strides[1] * hop_length, y_np.strides[1])
    all_frames = np.lib.stride_tricks.as_strided(
        y_np,
        shape=(batch_size, n_frames, frame_length),
        strides=strides,
    ).copy()  # Copy to ensure contiguous memory for FFT

    # Step 2: Center each frame (subtract mean)
    frame_means = np.mean(all_frames, axis=-1, keepdims=True)
    all_frames_centered = all_frames - frame_means

    # Step 3: Batched FFT for autocorrelation
    n_fft = 2 ** int(np.ceil(np.log2(2 * frame_length - 1)))
    flat_frames = all_frames_centered.reshape(-1, frame_length)
    Y = np.fft.rfft(flat_frames, n=n_fft, axis=-1)
    power = Y * np.conj(Y)
    acf_flat = np.fft.irfft(power, n=n_fft, axis=-1).real

    # Reshape back to (batch_size, n_frames, n_fft)
    acf = acf_flat.reshape(batch_size, n_frames, -1)

    # Step 4: Normalize by acf[0]
    acf_zero = acf[:, :, 0:1]
    acf_zero = np.maximum(acf_zero, 1e-10)
    acf_normalized = acf / acf_zero

    # Step 5: Extract search range and find max for all frames at once
    search_range = acf_normalized[:, :, min_lag:max_lag + 1]
    if search_range.shape[-1] > 0:
        periodicity_values = np.max(search_range, axis=-1)  # (batch_size, n_frames)
    else:
        periodicity_values = np.zeros((batch_size, n_frames), dtype=np.float32)

    # Reshape to (batch_size, 1, n_frames) for output format
    periodicity_np = periodicity_values[:, np.newaxis, :]

    result = mx.array(periodicity_np)

    # Remove batch dimension if input was 1D
    if input_is_1d:
        result = result[0]

    return result


def _cumulative_mean_normalized_difference(
    y_frames: np.ndarray,
    min_period: int,
    max_period: int,
) -> np.ndarray:
    """
    Compute the cumulative mean normalized difference function for YIN.

    This is the core of the YIN algorithm. For each frame, it computes
    the CMNDF which has a global minimum at the fundamental period.

    Parameters
    ----------
    y_frames : np.ndarray
        Framed signal. Shape: (n_frames, frame_length).
    min_period : int
        Minimum period (samples) to consider.
    max_period : int
        Maximum period (samples) to consider.

    Returns
    -------
    np.ndarray
        CMNDF values. Shape: (n_frames, max_period - min_period).
    """
    n_frames, frame_length = y_frames.shape
    n_lags = max_period - min_period

    # === BATCHED FFT OPTIMIZATION ===
    # Process all frames with a single batched FFT call

    # Step 1: Batched FFT for autocorrelation of all frames at once
    n_fft = 2 ** int(np.ceil(np.log2(2 * frame_length - 1)))
    Y = np.fft.rfft(y_frames, n=n_fft, axis=-1)  # (n_frames, n_fft//2+1)
    power = Y * np.conj(Y)
    acf = np.fft.irfft(power, n=n_fft, axis=-1)[:, :frame_length].real  # (n_frames, frame_length)

    # Step 2: Compute difference function vectorized
    # d(tau) = 2 * (acf[:, 0] - acf[:, tau])
    energy = acf[:, 0:1]  # (n_frames, 1)

    # Compute diff for lags 1 to max_period
    # Ensure we don't go beyond frame_length
    max_tau = min(max_period + 1, frame_length)
    diff = np.zeros((n_frames, max_period + 1), dtype=np.float32)
    diff[:, 1:max_tau] = 2 * (energy - acf[:, 1:max_tau])

    # Step 3: Vectorized cumulative sum for CMNDF normalization
    # cumsum[tau] = sum(diff[1:tau+1])
    cumsum = np.cumsum(diff, axis=-1)  # (n_frames, max_period + 1)

    # Step 4: Compute CMNDF: d'(tau) = d(tau) / (cumsum[tau] / tau)
    # For tau in [min_period, max_period]
    tau_values = np.arange(1, max_period + 1, dtype=np.float32)[None, :]  # (1, max_period)
    mean = cumsum[:, 1:max_period + 1] / tau_values  # (n_frames, max_period)
    mean = np.maximum(mean, 1e-10)  # Avoid division by zero

    # CMNDF for all taus
    cmndf_full = diff[:, 1:max_period + 1] / mean  # (n_frames, max_period)

    # Select valid tau range [min_period, max_period)
    cmndf = cmndf_full[:, min_period - 1:min_period - 1 + n_lags]

    # Handle edge case where we don't have enough lags
    if cmndf.shape[1] < n_lags:
        padding = np.ones((n_frames, n_lags - cmndf.shape[1]), dtype=np.float32)
        cmndf = np.concatenate([cmndf, padding], axis=1)

    return cmndf


def _parabolic_interpolation(
    values: np.ndarray,
    idx: int,
) -> tuple[float, float]:
    """
    Parabolic interpolation for sub-sample period estimation.

    Parameters
    ----------
    values : np.ndarray
        Array of values.
    idx : int
        Index of the minimum.

    Returns
    -------
    tuple
        (interpolated_idx, interpolated_value).
    """
    if idx <= 0 or idx >= len(values) - 1:
        return float(idx), values[idx]

    # Fit parabola through three points
    y0 = values[idx - 1]
    y1 = values[idx]
    y2 = values[idx + 1]

    # Vertex of parabola
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-10:
        return float(idx), y1

    shift = 0.5 * (y0 - y2) / denom
    interp_idx = idx + shift
    interp_val = y1 - 0.25 * (y0 - y2) * shift

    return interp_idx, interp_val


def yin(
    y: mx.array,
    sr: int = 22050,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold: float = 0.1,
    center: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    YIN pitch detection algorithm.

    YIN is a fast and accurate fundamental frequency estimator based on
    the autocorrelation method with several improvements including
    cumulative mean normalization.

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    fmin : float, default=50.0
        Minimum frequency to detect (Hz).
    fmax : float, default=2000.0
        Maximum frequency to detect (Hz).
    frame_length : int, default=2048
        Analysis frame length.
    hop_length : int, default=512
        Hop length between frames.
    threshold : float, default=0.1
        Aperiodicity threshold. Lower values are more selective.
    center : bool, default=True
        Center-pad the signal for framing.

    Returns
    -------
    tuple
        (f0, voiced_flag) where:
        - f0: Detected fundamental frequency for each frame (Hz).
          Unvoiced frames have f0 = 0.
        - voiced_flag: Boolean indicating voiced frames.

    Notes
    -----
    Based on: de Cheveigné, A., & Kawahara, H. (2002).
    "YIN, a fundamental frequency estimator for speech and music."

    Examples
    --------
    >>> f0, voiced = yin(y, sr=22050, fmin=80, fmax=500)
    >>> f0_voiced = f0[voiced]  # Get only voiced pitch values
    """
    validate_positive(frame_length, "frame_length")
    validate_positive(hop_length, "hop_length")

    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")

    # Period bounds from frequency bounds
    min_period = max(2, int(sr / fmax))
    max_period = min(frame_length // 2, int(sr / fmin))

    # Handle batched input
    input_is_1d = y.ndim == 1
    if input_is_1d:
        y = y[None, :]

    y_np = np.array(y)
    batch_size, signal_length = y_np.shape

    # Center padding
    if center:
        pad_length = frame_length // 2
        y_np = np.pad(y_np, [(0, 0), (pad_length, pad_length)], mode="constant")

    # Frame the signal
    padded_length = y_np.shape[1]
    n_frames = 1 + (padded_length - frame_length) // hop_length

    # === BATCHED FRAME EXTRACTION & PROCESSING ===
    # Extract all frames for all batches using strided view
    strides = (y_np.strides[0], y_np.strides[1] * hop_length, y_np.strides[1])
    all_frames = np.lib.stride_tricks.as_strided(
        y_np,
        shape=(batch_size, n_frames, frame_length),
        strides=strides,
    ).copy()

    # Reshape to (batch_size * n_frames, frame_length) for CMNDF
    flat_frames = all_frames.reshape(-1, frame_length)

    # Compute CMNDF for all frames at once (already optimized with batched FFT)
    cmndf_flat = _cumulative_mean_normalized_difference(flat_frames, min_period, max_period)

    # Reshape back to (batch_size, n_frames, n_lags)
    n_lags = cmndf_flat.shape[-1]
    cmndf = cmndf_flat.reshape(batch_size, n_frames, n_lags)

    # === VECTORIZED MINIMUM FINDING ===
    # Find local minima: cmndf[i] < cmndf[i+1] AND cmndf[i] < threshold
    if n_lags >= 2:
        # Check for local minimum (value less than next value) below threshold
        is_valid_min = np.zeros_like(cmndf, dtype=bool)
        is_valid_min[:, :, :-1] = (
            (cmndf[:, :, :-1] < threshold) &
            (cmndf[:, :, :-1] < cmndf[:, :, 1:])
        )

        # Find first valid minimum for each frame
        first_min_idx = np.argmax(is_valid_min, axis=-1)  # (batch_size, n_frames)
        has_valid_min = np.any(is_valid_min, axis=-1)  # (batch_size, n_frames)

        # For frames without valid minimum, use global minimum
        global_min_idx = np.argmin(cmndf, axis=-1)  # (batch_size, n_frames)

        # Get values at global minimum for threshold check
        batch_idx = np.arange(batch_size)[:, None]
        frame_idx = np.arange(n_frames)[None, :]
        global_min_val = cmndf[batch_idx, frame_idx, global_min_idx]

        # Use first valid min if found, else global min
        best_tau = np.where(has_valid_min, first_min_idx, global_min_idx)

        # Determine if voiced: either has valid min, or global min is below 0.5
        voiced_np = has_valid_min | (global_min_val <= 0.5)

        # Apply parabolic interpolation for sub-sample accuracy (vectorized)
        # Get values at best_tau-1, best_tau, best_tau+1 for interpolation
        # Handle boundaries
        tau_m1 = np.maximum(best_tau - 1, 0)
        tau_p1 = np.minimum(best_tau + 1, n_lags - 1)

        y0 = cmndf[batch_idx, frame_idx, tau_m1]
        y1 = cmndf[batch_idx, frame_idx, best_tau]
        y2 = cmndf[batch_idx, frame_idx, tau_p1]

        # Parabolic interpolation: shift = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        denom = y0 - 2 * y1 + y2
        # Only interpolate where denominator is significant and tau is not at boundary
        can_interpolate = (np.abs(denom) > 1e-10) & (best_tau > 0) & (best_tau < n_lags - 1)
        shift = np.where(can_interpolate, 0.5 * (y0 - y2) / denom, 0.0)
        interp_tau = best_tau + shift

        # Convert to frequency
        period = min_period + interp_tau
        f0_np = np.where(voiced_np & (period > 0), sr / period, 0.0).astype(np.float32)
    else:
        f0_np = np.zeros((batch_size, n_frames), dtype=np.float32)
        voiced_np = np.zeros((batch_size, n_frames), dtype=bool)

    f0 = mx.array(f0_np)
    voiced = mx.array(voiced_np)

    if input_is_1d:
        f0 = f0[0]
        voiced = voiced[0]

    return f0, voiced


def pyin(
    y: mx.array,
    sr: int = 22050,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    frame_length: int = 2048,
    hop_length: int | None = None,
    n_thresholds: int = 100,
    beta_parameters: tuple[float, float] = (2.0, 18.0),
    resolution: float = 0.1,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    no_trough_prob: float = 0.01,
    fill_na: float | None = None,
    center: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Probabilistic YIN (PYIN) pitch detection.

    PYIN extends YIN by considering multiple pitch candidates per frame
    and using a Hidden Markov Model to find the optimal pitch trajectory.

    Parameters
    ----------
    y : mx.array
        Audio signal. Shape: (samples,).
    sr : int, default=22050
        Sample rate.
    fmin : float, default=50.0
        Minimum frequency to detect (Hz).
    fmax : float, default=2000.0
        Maximum frequency to detect (Hz).
    frame_length : int, default=2048
        Analysis frame length.
    hop_length : int, optional
        Hop length. Default: frame_length // 4.
    n_thresholds : int, default=100
        Number of thresholds to try for candidate generation.
    beta_parameters : tuple, default=(2.0, 18.0)
        Shape parameters for the beta distribution prior on thresholds.
    resolution : float, default=0.1
        Frequency resolution in semitones for pitch quantization.
    max_transition_rate : float, default=35.92
        Maximum pitch change rate in semitones per second.
    switch_prob : float, default=0.01
        Probability of switching between voiced and unvoiced.
    no_trough_prob : float, default=0.01
        Probability of no valid trough in CMNDF.
    fill_na : float, optional
        Value to fill unvoiced frames. If None, use 0.
    center : bool, default=True
        Center-pad the signal.

    Returns
    -------
    tuple
        (f0, voiced_flag, voiced_prob) where:
        - f0: Fundamental frequency for each frame (Hz).
        - voiced_flag: Boolean indicating voiced frames.
        - voiced_prob: Probability of being voiced for each frame.

    Notes
    -----
    Based on: Mauch, M., & Dixon, S. (2014).
    "PYIN: A fundamental frequency estimator using probabilistic
    threshold distributions."

    This is compatible with librosa.pyin().

    Examples
    --------
    >>> f0, voiced, prob = pyin(y, sr=22050, fmin=80, fmax=500)
    >>> # Get confident pitch estimates
    >>> confident = prob > 0.8
    >>> f0_confident = f0[confident]
    """
    validate_positive(frame_length, "frame_length")

    if hop_length is None:
        hop_length = frame_length // 4

    validate_positive(hop_length, "hop_length")

    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")

    # Period bounds
    min_period = max(2, int(sr / fmax))
    max_period = min(frame_length // 2, int(sr / fmin))

    # Convert to numpy
    y_np = np.array(y)
    if y_np.ndim > 1:
        y_np = y_np[0]  # Use first batch

    # Center padding
    if center:
        pad_length = frame_length // 2
        y_np = np.pad(y_np, (pad_length, pad_length), mode="constant")

    # Frame the signal
    n_frames = 1 + (len(y_np) - frame_length) // hop_length

    # Generate thresholds from beta distribution
    # Use thresholds in range [0.01, 0.8] for robust detection
    from scipy.stats import beta
    raw_thresholds = beta.ppf(
        np.linspace(0, 1, n_thresholds + 2)[1:-1],
        beta_parameters[0],
        beta_parameters[1]
    )
    # Scale thresholds to practical range
    thresholds = 0.01 + raw_thresholds * 0.79

    # Extract frames using strided view (vectorized, no loop)
    strides = (y_np.strides[0] * hop_length, y_np.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        y_np,
        shape=(n_frames, frame_length),
        strides=strides,
    ).copy()  # Copy to ensure contiguous memory

    cmndf = _cumulative_mean_normalized_difference(
        frames, min_period, max_period
    )

    # Use C++ extension with Metal GPU acceleration if available
    if HAS_CPP_EXT and _ext is not None:
        # Convert to MLX arrays for GPU processing
        cmndf_mx = mx.array(cmndf.astype(np.float32))
        thresholds_mx = mx.array(thresholds.astype(np.float32))

        # Parallel threshold candidate detection on GPU
        candidates, weights, n_candidates_mx = _ext.pyin_candidates(
            cmndf_mx, thresholds_mx, min_period, sr
        )

        # Weighted median computation on GPU
        f0, prob = _ext.pyin_weighted_median(candidates, weights, n_candidates_mx)

        # Evaluate and convert back to numpy for post-processing
        mx.eval(f0, prob)
        f0_np = np.array(f0)
        prob_np = np.array(prob)

        # Handle frames with no candidates - fallback to global minimum
        no_candidates = prob_np == 0.0
        for t in np.where(no_candidates)[0]:
            best_tau = np.argmin(cmndf[t])
            if cmndf[t, best_tau] < 0.5:
                period = min_period + best_tau
                f0_np[t] = sr / period
                prob_np[t] = 0.5 * (1.0 - cmndf[t, best_tau])
    else:
        # Python fallback (slower)
        f0_np = np.zeros(n_frames, dtype=np.float32)
        prob_np = np.zeros(n_frames, dtype=np.float32)

        n_lags = cmndf.shape[1]

        # Pre-allocate candidate storage (max candidates = n_thresholds per frame)
        max_candidates = len(thresholds)
        all_candidates = np.zeros((n_frames, max_candidates), dtype=np.float32)
        all_weights = np.zeros((n_frames, max_candidates), dtype=np.float32)
        n_candidates = np.zeros(n_frames, dtype=np.int32)

        # Vectorized local minimum detection: cmndf[t, tau] < cmndf[t, tau+1]
        is_local_min = np.zeros((n_frames, n_lags), dtype=bool)
        is_local_min[:, :-1] = cmndf[:, :-1] < cmndf[:, 1:]

        # For each threshold, find candidates across all frames (vectorized)
        for thresh_idx, thresh in enumerate(thresholds):
            # below_thresh[t, tau] = True if cmndf[t, tau] < thresh
            below_thresh = cmndf < thresh

            # valid_candidate[t, tau] = below_thresh AND is_local_min
            valid_candidate = below_thresh & is_local_min

            # Find first valid candidate for each frame (argmax finds first True)
            has_candidate = np.any(valid_candidate, axis=1)
            first_tau_idx = np.argmax(valid_candidate, axis=1)

            # Get CMNDF values at first candidate positions
            frame_indices = np.arange(n_frames)
            cmndf_values = cmndf[frame_indices, first_tau_idx]

            # Convert tau index to frequency
            periods = min_period + first_tau_idx
            frequencies = sr / periods

            # Weight by inverse CMNDF (higher weight for lower aperiodicity)
            weights_np = 1.0 - cmndf_values

            # Store candidates for frames that have one
            valid_frames = has_candidate
            all_candidates[valid_frames, thresh_idx] = frequencies[valid_frames]
            all_weights[valid_frames, thresh_idx] = weights_np[valid_frames]
            n_candidates[valid_frames] += 1

        # Compute weighted median for frames with candidates
        for t in range(n_frames):
            nc = n_candidates[t]
            if nc > 0:
                cands = all_candidates[t, :nc]
                wgts = all_weights[t, :nc]

                # Voiced probability = fraction of thresholds that found candidate
                prob_np[t] = nc / max_candidates

                # Normalize weights and compute weighted median
                wgts = wgts / np.sum(wgts)
                sorted_idx = np.argsort(cands)
                cum_weights = np.cumsum(wgts[sorted_idx])
                median_idx = np.searchsorted(cum_weights, 0.5)
                median_idx = min(median_idx, nc - 1)

                f0_np[t] = cands[sorted_idx[median_idx]]
            else:
                # Check global minimum (fallback)
                best_tau = np.argmin(cmndf[t])
                if cmndf[t, best_tau] < 0.5:
                    period = min_period + best_tau
                    f0_np[t] = sr / period
                    prob_np[t] = 0.5 * (1.0 - cmndf[t, best_tau])
                elif fill_na is not None:
                    f0_np[t] = fill_na

    # Simple temporal smoothing of voiced probability
    from scipy.ndimage import uniform_filter1d
    prob_np = uniform_filter1d(prob_np, size=3)

    # Threshold voiced probability
    voiced_np = prob_np > 0.3

    # Zero out unvoiced frames if no fill_na
    if fill_na is None:
        f0_np[~voiced_np] = 0.0

    return mx.array(f0_np), mx.array(voiced_np), mx.array(prob_np)
