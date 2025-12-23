"""ERB (Equivalent Rectangular Bandwidth) filterbank operations."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def hz_to_erb(f: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency in Hz to ERB scale.

    ERB scale models human auditory perception better than linear frequency.

    Parameters
    ----------
    f : float or array
        Frequency in Hz.

    Returns
    -------
    float or array
        ERB-rate value.
    """
    return 9.265 * np.log(1 + f / (24.7 * 9.265))


def erb_to_hz(erb: float | np.ndarray) -> float | np.ndarray:
    """Convert ERB scale to frequency in Hz.

    Parameters
    ----------
    erb : float or array
        ERB-rate value.

    Returns
    -------
    float or array
        Frequency in Hz.
    """
    return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)


def erb_filterbank(
    n_fft: int,
    sample_rate: int,
    n_bands: int = 32,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> mx.array:
    """Create ERB-scale filterbank matrix.

    Creates a filterbank that maps linear frequency bins to ERB bands.
    Each row represents one ERB band.

    Parameters
    ----------
    n_fft : int
        FFT size.
    sample_rate : int
        Sample rate in Hz.
    n_bands : int, default=32
        Number of ERB bands.
    f_min : float, default=0.0
        Minimum frequency in Hz.
    f_max : float, optional
        Maximum frequency in Hz. Default: sample_rate / 2.

    Returns
    -------
    mx.array
        Filterbank matrix of shape (n_bands, n_fft // 2 + 1).
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    n_freqs = n_fft // 2 + 1

    # Frequency bins
    freqs = np.linspace(0, sample_rate / 2, n_freqs)

    # ERB scale center frequencies
    erb_min = hz_to_erb(f_min)
    erb_max = hz_to_erb(f_max)
    erb_centers = np.linspace(erb_min, erb_max, n_bands + 2)
    center_freqs = erb_to_hz(erb_centers)

    # Build filterbank with triangular filters
    filterbank = np.zeros((n_bands, n_freqs), dtype=np.float32)

    for i in range(n_bands):
        f_low = center_freqs[i]
        f_center = center_freqs[i + 1]
        f_high = center_freqs[i + 2]

        # Rising slope
        rising = (freqs - f_low) / (f_center - f_low + 1e-10)
        rising = np.clip(rising, 0, 1)

        # Falling slope
        falling = (f_high - freqs) / (f_high - f_center + 1e-10)
        falling = np.clip(falling, 0, 1)

        # Combine
        filterbank[i] = np.minimum(rising, falling)

    # Normalize each band to sum to 1
    row_sums = filterbank.sum(axis=1, keepdims=True)
    filterbank = filterbank / (row_sums + 1e-10)

    return mx.array(filterbank)


def erb_inverse_filterbank(
    n_fft: int,
    sample_rate: int,
    n_bands: int = 32,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> mx.array:
    """Create inverse ERB filterbank for reconstruction.

    Maps ERB bands back to linear frequency bins.

    Parameters
    ----------
    n_fft : int
        FFT size.
    sample_rate : int
        Sample rate.
    n_bands : int, default=32
        Number of ERB bands.
    f_min : float, default=0.0
        Minimum frequency.
    f_max : float, optional
        Maximum frequency.

    Returns
    -------
    mx.array
        Inverse filterbank matrix of shape (n_fft // 2 + 1, n_bands).
    """
    fb = erb_filterbank(n_fft, sample_rate, n_bands, f_min, f_max)

    # Transpose and normalize columns
    fb_np = np.array(fb).T  # (n_freqs, n_bands)

    # Normalize so that fb @ fb_inv ~ identity
    col_sums = fb_np.sum(axis=0, keepdims=True)
    fb_inv = fb_np / (col_sums + 1e-10)

    return mx.array(fb_inv.astype(np.float32))
