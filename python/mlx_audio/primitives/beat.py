"""
Beat tracking and tempo estimation primitives.

Provides tempo estimation and beat tracking for rhythm analysis.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ._validation import validate_positive
from .onset import onset_strength


def tempo(
    y: mx.array | None = None,
    sr: int = 22050,
    onset_envelope: mx.array | None = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    std_bpm: float = 1.0,
    ac_size: float = 8.0,
    max_tempo: float = 320.0,
    aggregate: str | None = None,
    prior: str | None = "lognormal",
) -> mx.array:
    """
    Estimate tempo (BPM) from onset strength envelope.

    Uses autocorrelation of the onset envelope to find the dominant
    period, then converts to BPM.

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    onset_envelope : mx.array, optional
        Pre-computed onset strength envelope.
    hop_length : int, default=512
        Hop length (for time conversion).
    start_bpm : float, default=120.0
        Prior tempo estimate (center of tempo prior).
    std_bpm : float, default=1.0
        Standard deviation of tempo prior (in octaves).
    ac_size : float, default=8.0
        Length of autocorrelation window in seconds.
    max_tempo : float, default=320.0
        Maximum tempo to consider (BPM).
    aggregate : str, optional
        If 'mean' or 'median', aggregate multiple tempo estimates.
        If None, return dominant tempo.
    prior : str, optional
        Tempo prior: 'lognormal' (default), 'uniform', or None.

    Returns
    -------
    mx.array
        Estimated tempo in BPM.
        Shape: () or (1,) for single estimate.
        Shape: (2,) for (tempo1, tempo2) if aggregate is None.

    Notes
    -----
    This is compatible with librosa.beat.tempo().

    Examples
    --------
    >>> bpm = tempo(y, sr=22050)
    >>> print(f"Tempo: {float(bpm):.1f} BPM")

    >>> # Get top 2 tempo estimates
    >>> tempos = tempo(y, sr=22050)  # Returns (tempo1, tempo2)
    """
    validate_positive(hop_length, "hop_length")
    validate_positive(max_tempo, "max_tempo")

    # Get onset envelope if not provided
    if onset_envelope is None:
        if y is None:
            raise ValueError(
                "Either y (audio) or onset_envelope must be provided"
            )
        onset_envelope = onset_strength(y, sr=sr, hop_length=hop_length)

    # Convert to numpy for processing
    onset_np = np.array(onset_envelope)

    # Handle batched input - use first batch
    if onset_np.ndim > 1:
        onset_np = onset_np[0]

    # Compute autocorrelation size in frames
    frame_rate = sr / hop_length
    ac_frames = int(ac_size * frame_rate)
    ac_frames = min(ac_frames, len(onset_np))

    # Compute autocorrelation
    # Use FFT-based autocorrelation
    n_fft = 2 ** int(np.ceil(np.log2(2 * len(onset_np) - 1)))
    onset_centered = onset_np - np.mean(onset_np)
    Y = np.fft.rfft(onset_centered, n=n_fft)
    power = Y * np.conj(Y)
    ac = np.fft.irfft(power, n=n_fft)[:ac_frames]

    # Normalize
    if ac[0] > 1e-10:
        ac = ac / ac[0]

    # Convert lag to BPM
    # BPM = 60 * frame_rate / lag
    # Min lag corresponds to max_tempo
    min_lag = max(1, int(60.0 * frame_rate / max_tempo))

    # Create tempo prior
    if prior == "lognormal":
        # Log-normal prior centered at start_bpm
        lags = np.arange(min_lag, ac_frames)
        bpms = 60.0 * frame_rate / lags
        log_bpm = np.log2(bpms / start_bpm)
        prior_weights = np.exp(-0.5 * (log_bpm / std_bpm) ** 2)
    elif prior == "uniform":
        lags = np.arange(min_lag, ac_frames)
        prior_weights = np.ones(len(lags))
    else:
        lags = np.arange(min_lag, ac_frames)
        prior_weights = np.ones(len(lags))

    # Weight autocorrelation by prior
    ac_weighted = ac[min_lag:ac_frames] * prior_weights

    # Find peaks in weighted autocorrelation
    # Simple peak detection: find local maxima
    peaks = []
    for i in range(1, len(ac_weighted) - 1):
        if ac_weighted[i] > ac_weighted[i - 1] and ac_weighted[i] > ac_weighted[i + 1]:
            peaks.append((i, ac_weighted[i]))

    # Sort by weight and get top 2
    peaks.sort(key=lambda x: x[1], reverse=True)

    if len(peaks) >= 2:
        tempo1_lag = min_lag + peaks[0][0]
        tempo2_lag = min_lag + peaks[1][0]
        tempo1 = 60.0 * frame_rate / tempo1_lag
        tempo2 = 60.0 * frame_rate / tempo2_lag
    elif len(peaks) == 1:
        tempo1_lag = min_lag + peaks[0][0]
        tempo1 = 60.0 * frame_rate / tempo1_lag
        tempo2 = tempo1
    else:
        # Fallback: use global max
        best_lag = min_lag + np.argmax(ac_weighted)
        tempo1 = 60.0 * frame_rate / best_lag
        tempo2 = tempo1

    # Aggregate if requested
    if aggregate == "mean":
        result = np.array([(tempo1 + tempo2) / 2], dtype=np.float32)
    elif aggregate == "median":
        result = np.array([np.median([tempo1, tempo2])], dtype=np.float32)
    else:
        result = np.array([tempo1, tempo2], dtype=np.float32)

    return mx.array(result)


def beat_track(
    y: mx.array | None = None,
    sr: int = 22050,
    onset_envelope: mx.array | None = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100.0,
    trim: bool = True,
    units: str = "frames",
    **kwargs,
) -> tuple[mx.array, mx.array]:
    """
    Track beats using dynamic programming.

    Uses the algorithm from Ellis (2007): "Beat Tracking by Dynamic Programming".

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform. Shape: (samples,).
    sr : int, default=22050
        Sample rate.
    onset_envelope : mx.array, optional
        Pre-computed onset strength envelope.
    hop_length : int, default=512
        Hop length.
    start_bpm : float, default=120.0
        Initial tempo estimate for beat tracking.
    tightness : float, default=100.0
        Tightness of beat distribution around tempo.
        Higher values enforce stricter tempo adherence.
    trim : bool, default=True
        If True, trim beats before first onset.
    units : str, default='frames'
        Output units: 'frames', 'samples', or 'time'.
    **kwargs
        Additional arguments passed to onset_strength.

    Returns
    -------
    tuple
        (tempo, beats) where:
        - tempo: Estimated tempo in BPM.
        - beats: Beat positions in specified units.

    Notes
    -----
    This is compatible with librosa.beat.beat_track().
    The algorithm finds the path through the onset envelope that
    maximizes the total onset strength while maintaining a consistent
    tempo.

    Examples
    --------
    >>> tempo, beats = beat_track(y, sr=22050)
    >>> print(f"Tempo: {float(tempo):.1f} BPM")
    >>> print(f"Found {len(beats)} beats")

    >>> # Get beat times in seconds
    >>> tempo, beat_times = beat_track(y, sr=22050, units='time')
    """
    validate_positive(hop_length, "hop_length")
    validate_positive(tightness, "tightness")

    # Get onset envelope if not provided
    if onset_envelope is None:
        if y is None:
            raise ValueError(
                "Either y (audio) or onset_envelope must be provided"
            )
        onset_envelope = onset_strength(y, sr=sr, hop_length=hop_length, **kwargs)

    # Estimate tempo
    tempo_estimate = tempo(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
    )
    bpm = float(tempo_estimate[0])

    # Convert to numpy for DP
    onset_np = np.array(onset_envelope)
    if onset_np.ndim > 1:
        onset_np = onset_np[0]

    n_frames = len(onset_np)

    # Convert tempo to period in frames
    frame_rate = sr / hop_length
    period = 60.0 * frame_rate / bpm

    # Dynamic programming for beat tracking
    # Score function: onset strength + transition penalty
    # Transition penalty encourages consistent beat spacing

    # Initialize DP arrays
    scores = np.zeros(n_frames)
    backlinks = np.full(n_frames, -1, dtype=np.int64)

    # Window for searching previous beats
    window = int(period * 2)

    for i in range(n_frames):
        # Current frame's onset strength
        onset_score = onset_np[i]

        # Search for best previous beat
        best_score = onset_score
        best_prev = -1

        start = max(0, i - window)
        for j in range(start, i):
            # Expected interval
            interval = i - j

            # Transition penalty: log-Gaussian around expected period
            deviation = np.log2(interval / period) if interval > 0 else 0
            transition_penalty = -tightness * (deviation ** 2)

            # Total score
            score = scores[j] + onset_score + transition_penalty

            if score > best_score:
                best_score = score
                best_prev = j

        scores[i] = best_score
        backlinks[i] = best_prev

    # Backtrack from the best final frame
    # Look in the last few periods for the best ending
    search_start = max(0, n_frames - int(period * 2))
    best_end = search_start + np.argmax(scores[search_start:])

    # Trace back
    beats = []
    current = best_end
    while current >= 0:
        beats.append(current)
        current = backlinks[current]

    beats = np.array(beats[::-1], dtype=np.int64)

    # Trim beats before first significant onset
    if trim and len(beats) > 0:
        # Find first significant onset
        threshold = np.mean(onset_np) + np.std(onset_np)
        first_onset = np.argmax(onset_np > threshold)

        # Keep only beats at or after first onset
        beats = beats[beats >= first_onset]

    # Convert units
    if units == "frames":
        beats_out = beats
    elif units == "samples":
        beats_out = beats * hop_length
    elif units == "time":
        beats_out = (beats * hop_length) / sr
    else:
        raise ValueError(f"Unknown units: '{units}'. Use 'frames', 'samples', or 'time'")

    return mx.array([bpm]), mx.array(beats_out)


def plp(
    onset_envelope: mx.array,
    sr: int = 22050,
    hop_length: int = 512,
    tempo_min: float = 30.0,
    tempo_max: float = 300.0,
    prior: str = "lognormal",
) -> mx.array:
    """
    Compute predominant local pulse (tempo over time).

    This estimates the local tempo at each frame, useful for music
    with tempo variations.

    Parameters
    ----------
    onset_envelope : mx.array
        Onset strength envelope. Shape: (n_frames,).
    sr : int, default=22050
        Sample rate.
    hop_length : int, default=512
        Hop length.
    tempo_min : float, default=30.0
        Minimum tempo (BPM).
    tempo_max : float, default=300.0
        Maximum tempo (BPM).
    prior : str, default='lognormal'
        Tempo prior distribution.

    Returns
    -------
    mx.array
        Local pulse (tempo) at each frame.
        Shape: (n_frames,).

    Notes
    -----
    This is compatible with librosa.beat.plp().

    Examples
    --------
    >>> env = onset_strength(y, sr=22050)
    >>> local_tempo = plp(env, sr=22050)
    >>> # local_tempo varies over time for music with tempo changes
    """
    validate_positive(hop_length, "hop_length")
    validate_positive(tempo_min, "tempo_min")
    validate_positive(tempo_max, "tempo_max")

    if tempo_min >= tempo_max:
        raise ValueError(f"tempo_min ({tempo_min}) must be < tempo_max ({tempo_max})")

    # Convert to numpy
    onset_np = np.array(onset_envelope)
    if onset_np.ndim > 1:
        onset_np = onset_np[0]

    n_frames = len(onset_np)
    frame_rate = sr / hop_length

    # Lag bounds from tempo bounds
    min_lag = max(1, int(60.0 * frame_rate / tempo_max))
    max_lag = int(60.0 * frame_rate / tempo_min)

    # Window size for local analysis
    window_size = max_lag * 2

    # Compute local tempo at each frame
    local_tempo = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        # Extract local window
        start = max(0, i - window_size // 2)
        end = min(n_frames, i + window_size // 2)
        local_env = onset_np[start:end]

        if len(local_env) < max_lag:
            # Not enough data, use global estimate
            local_tempo[i] = 120.0  # Default
            continue

        # Compute local autocorrelation
        local_centered = local_env - np.mean(local_env)
        n_fft = 2 ** int(np.ceil(np.log2(2 * len(local_env) - 1)))
        Y = np.fft.rfft(local_centered, n=n_fft)
        power = Y * np.conj(Y)
        ac = np.fft.irfft(power, n=n_fft)

        # Normalize
        if ac[0] > 1e-10:
            ac = ac / ac[0]

        # Apply tempo prior and find peak
        valid_lags = np.arange(min_lag, min(max_lag, len(ac)))
        if len(valid_lags) == 0:
            local_tempo[i] = 120.0
            continue

        ac_valid = ac[min_lag:min(max_lag, len(ac))]

        if prior == "lognormal":
            bpms = 60.0 * frame_rate / valid_lags
            log_bpm = np.log2(bpms / 120.0)  # Center at 120 BPM
            weights = np.exp(-0.5 * log_bpm ** 2)
            ac_valid = ac_valid * weights

        # Find best lag
        best_idx = np.argmax(ac_valid)
        best_lag = min_lag + best_idx
        local_tempo[i] = 60.0 * frame_rate / best_lag

    return mx.array(local_tempo)
