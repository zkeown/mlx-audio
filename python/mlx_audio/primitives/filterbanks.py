"""
Filterbank construction beyond mel scale.

Provides linear-scale and Bark-scale filterbanks for audio analysis.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from mlx_audio.constants import FILTERBANK_EPSILON

from ._validation import validate_non_negative, validate_positive


def hz_to_bark(frequencies: np.ndarray, formula: str = "zwicker") -> np.ndarray:
    """
    Convert Hz to Bark scale.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies in Hz.
    formula : str, default='zwicker'
        Bark scale formula to use:
        - 'zwicker': z = 13*arctan(0.00076*f) + 3.5*arctan((f/7500)^2)
        - 'traunmuller': z = (26.81*f)/(1960+f) - 0.53

    Returns
    -------
    np.ndarray
        Frequencies in Bark scale.
    """
    frequencies = np.asarray(frequencies)

    if formula == "zwicker":
        # Zwicker & Terhardt (1980): "Analytical expressions for critical-band rate
        # and critical bandwidth as a function of frequency"
        # JASA 68(5): 1523-1525. Two-term arctan approximation for critical bands.
        return 13.0 * np.arctan(0.00076 * frequencies) + 3.5 * np.arctan(
            (frequencies / 7500.0) ** 2
        )
    elif formula == "traunmuller":
        # Traunmuller (1990): "Analytical expressions for the tonotopic sensory scale"
        # JASA 88(1): 97-100. Simpler formula with edge corrections.
        bark = (26.81 * frequencies) / (1960.0 + frequencies) - 0.53
        # Adjustments for edge cases
        bark = np.where(bark < 2, bark + 0.15 * (2 - bark), bark)
        bark = np.where(bark > 20.1, bark + 0.22 * (bark - 20.1), bark)
        return bark
    else:
        raise ValueError(
            f"Unknown formula: '{formula}'. Supported: 'zwicker', 'traunmuller'"
        )


def bark_to_hz(bark: np.ndarray, formula: str = "zwicker") -> np.ndarray:
    """
    Convert Bark scale to Hz.

    Parameters
    ----------
    bark : np.ndarray
        Frequencies in Bark scale.
    formula : str, default='zwicker'
        Bark scale formula to use (must match hz_to_bark).

    Returns
    -------
    np.ndarray
        Frequencies in Hz.
    """
    bark = np.asarray(bark)

    if formula == "zwicker":
        # Numerical inversion via Newton-Raphson for Zwicker formula.
        # The Zwicker formula has no closed-form inverse due to the two arctan terms,
        # so we use iterative refinement. Initial guess from sinh approximation
        # (valid for small bark values), then Newton-Raphson converges in ~5 steps.
        hz = 600.0 * np.sinh(bark / 6.0)

        # Newton-Raphson: hz_new = hz - f(hz)/f'(hz) where f(hz) = bark_est - bark
        for _ in range(5):
            bark_est = hz_to_bark(hz, formula="zwicker")
            # Derivative approximation
            eps = 1e-6
            deriv = (hz_to_bark(hz + eps, formula="zwicker") - bark_est) / eps
            deriv = np.maximum(deriv, 1e-10)  # Avoid division by zero
            hz = hz - (bark_est - bark) / deriv
            hz = np.maximum(hz, 0)  # Ensure non-negative

        return hz
    elif formula == "traunmuller":
        # Reverse adjustments
        bark = np.asarray(bark, dtype=np.float64)
        bark = np.where(bark < 2, bark - 0.15 * (2 - bark) / 1.15, bark)
        bark = np.where(bark > 20.1, bark - 0.22 * (bark - 20.1) / 1.22, bark)
        # Inverse formula
        return 1960.0 * (bark + 0.53) / (26.28 - bark)
    else:
        raise ValueError(
            f"Unknown formula: '{formula}'. Supported: 'zwicker', 'traunmuller'"
        )


# Secondary cache for MLX arrays (avoids CPU→GPU transfer on repeated access)
_mlx_bark_filterbank_cache: dict[tuple, mx.array] = {}
_mlx_linear_filterbank_cache: dict[tuple, mx.array] = {}


@lru_cache(maxsize=64)
def _compute_bark_filterbank_np(
    sr: int,
    n_fft: int,
    n_bands: int,
    fmin: float,
    fmax: float,
    formula: str,
    norm: str | None,
) -> tuple[bytes, tuple[int, int]]:
    """
    Compute Bark filterbank as a cacheable tuple structure.

    Returns the filterbank as bytes (hashable) along with shape info.
    """
    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)

    # Bark scale boundaries
    bark_min = hz_to_bark(np.array([fmin]), formula=formula)[0]
    bark_max = hz_to_bark(np.array([fmax]), formula=formula)[0]

    # Bark points: n_bands + 2 points (including edges)
    bark_points = np.linspace(bark_min, bark_max, n_bands + 2)
    hz_points = bark_to_hz(bark_points, formula=formula)

    # Create filterbank using triangular filters (same pattern as mel)
    f_lower = hz_points[:-2, np.newaxis]
    f_center = hz_points[1:-1, np.newaxis]
    f_upper = hz_points[2:, np.newaxis]
    freqs = fft_freqs[np.newaxis, :]

    lower_slope = (freqs - f_lower) / (f_center - f_lower + FILTERBANK_EPSILON)
    upper_slope = (f_upper - freqs) / (f_upper - f_center + FILTERBANK_EPSILON)

    filterbank = np.maximum(0, np.minimum(lower_slope, upper_slope)).astype(np.float32)

    # Normalize
    if norm == "slaney":
        enorm = 2.0 / (hz_points[2 : n_bands + 2] - hz_points[:n_bands])
        filterbank *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"Unknown norm: '{norm}'. Supported: 'slaney', None")

    return filterbank.tobytes(), filterbank.shape


def bark_filterbank(
    sr: int,
    n_fft: int,
    n_bands: int = 24,
    fmin: float = 0.0,
    fmax: float | None = None,
    formula: str = "zwicker",
    norm: str | None = "slaney",
) -> mx.array:
    """
    Create a Bark-scale filterbank matrix.

    The Bark scale is a psychoacoustic scale that corresponds to the
    critical bands of human hearing.

    Results are cached for repeated calls with identical parameters.

    Parameters
    ----------
    sr : int
        Sample rate of the audio.
    n_fft : int
        FFT size.
    n_bands : int, default=24
        Number of Bark bands (typically 24 for full audible range).
    fmin : float, default=0.0
        Minimum frequency (Hz).
    fmax : float, optional
        Maximum frequency (Hz). Default: sr / 2.
    formula : str, default='zwicker'
        Bark scale formula: 'zwicker' or 'traunmuller'.
    norm : str or None, default='slaney'
        Normalization mode:
        - 'slaney': Divide each filter by its bandwidth (area normalization).
        - None: No normalization.

    Returns
    -------
    mx.array
        Bark filterbank matrix of shape (n_bands, n_fft // 2 + 1).

    Examples
    --------
    >>> bark_fb = bark_filterbank(sr=22050, n_fft=2048, n_bands=24)
    >>> bark_fb.shape
    (24, 1025)
    """
    validate_positive(n_bands, "n_bands")
    validate_non_negative(fmin, "fmin")

    if fmax is None:
        fmax = sr / 2.0

    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
    if fmax > sr / 2.0:
        raise ValueError(f"fmax ({fmax}) cannot exceed Nyquist frequency ({sr / 2.0})")

    # Check MLX cache first
    cache_key = (sr, n_fft, n_bands, fmin, fmax, formula, norm)
    if cache_key in _mlx_bark_filterbank_cache:
        return _mlx_bark_filterbank_cache[cache_key]

    # Get cached filterbank data
    filterbank_bytes, shape = _compute_bark_filterbank_np(
        sr, n_fft, n_bands, fmin, fmax, formula, norm
    )

    # Convert from bytes to MLX array and cache
    filterbank = np.frombuffer(filterbank_bytes, dtype=np.float32).reshape(shape)
    result = mx.array(filterbank)
    _mlx_bark_filterbank_cache[cache_key] = result
    return result


@lru_cache(maxsize=64)
def _compute_linear_filterbank_np(
    sr: int,
    n_fft: int,
    n_bands: int,
    fmin: float,
    fmax: float,
    norm: str | None,
) -> tuple[bytes, tuple[int, int]]:
    """
    Compute linear filterbank as a cacheable tuple structure.
    """
    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0, sr / 2.0, n_freqs)

    # Linear frequency points: n_bands + 2 points (including edges)
    hz_points = np.linspace(fmin, fmax, n_bands + 2)

    # Create filterbank using triangular filters
    f_lower = hz_points[:-2, np.newaxis]
    f_center = hz_points[1:-1, np.newaxis]
    f_upper = hz_points[2:, np.newaxis]
    freqs = fft_freqs[np.newaxis, :]

    lower_slope = (freqs - f_lower) / (f_center - f_lower + FILTERBANK_EPSILON)
    upper_slope = (f_upper - freqs) / (f_upper - f_center + FILTERBANK_EPSILON)

    filterbank = np.maximum(0, np.minimum(lower_slope, upper_slope)).astype(np.float32)

    # Normalize
    if norm == "slaney":
        enorm = 2.0 / (hz_points[2 : n_bands + 2] - hz_points[:n_bands])
        filterbank *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"Unknown norm: '{norm}'. Supported: 'slaney', None")

    return filterbank.tobytes(), filterbank.shape


def linear_filterbank(
    sr: int,
    n_fft: int,
    n_bands: int = 64,
    fmin: float = 0.0,
    fmax: float | None = None,
    norm: str | None = "slaney",
) -> mx.array:
    """
    Create a linear-scale filterbank matrix.

    Unlike mel or Bark filterbanks, linear filterbanks have equal-width
    frequency bands in Hz.

    Results are cached for repeated calls with identical parameters.

    Parameters
    ----------
    sr : int
        Sample rate of the audio.
    n_fft : int
        FFT size.
    n_bands : int, default=64
        Number of frequency bands.
    fmin : float, default=0.0
        Minimum frequency (Hz).
    fmax : float, optional
        Maximum frequency (Hz). Default: sr / 2.
    norm : str or None, default='slaney'
        Normalization mode:
        - 'slaney': Divide each filter by its bandwidth (area normalization).
        - None: No normalization.

    Returns
    -------
    mx.array
        Linear filterbank matrix of shape (n_bands, n_fft // 2 + 1).

    Examples
    --------
    >>> linear_fb = linear_filterbank(sr=22050, n_fft=2048, n_bands=64)
    >>> linear_fb.shape
    (64, 1025)
    """
    validate_positive(n_bands, "n_bands")
    validate_non_negative(fmin, "fmin")

    if fmax is None:
        fmax = sr / 2.0

    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
    if fmax > sr / 2.0:
        raise ValueError(f"fmax ({fmax}) cannot exceed Nyquist frequency ({sr / 2.0})")

    # Check MLX cache first
    cache_key = (sr, n_fft, n_bands, fmin, fmax, norm)
    if cache_key in _mlx_linear_filterbank_cache:
        return _mlx_linear_filterbank_cache[cache_key]

    # Get cached filterbank data
    filterbank_bytes, shape = _compute_linear_filterbank_np(
        sr, n_fft, n_bands, fmin, fmax, norm
    )

    # Convert from bytes to MLX array and cache
    filterbank = np.frombuffer(filterbank_bytes, dtype=np.float32).reshape(shape)
    result = mx.array(filterbank)
    _mlx_linear_filterbank_cache[cache_key] = result
    return result
