"""
Mel-frequency cepstral coefficients (MFCC) and delta features.

Provides MFCC extraction with DCT computed via cached NumPy basis matrices.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from ._extension import HAS_CPP_EXT, _ext
from ._validation import validate_positive
from .convert import power_to_db
from .mel import melspectrogram

# Secondary cache for MLX DCT matrices (avoids CPU→GPU transfer on repeated access)
_mlx_dct_cache: dict[tuple, mx.array] = {}


@lru_cache(maxsize=32)
def _compute_dct_matrix_np(
    n_mfcc: int,
    n_mels: int,
    norm: str | None,
) -> tuple[bytes, tuple[int, int]]:
    """
    Compute DCT-II basis matrix via NumPy, return cacheable bytes.

    The DCT-II is defined as:
        C[k, n] = cos(pi * k * (2n + 1) / (2N))
    for k = 0, ..., K-1 and n = 0, ..., N-1

    Parameters
    ----------
    n_mfcc : int
        Number of MFCCs to return.
    n_mels : int
        Number of mel bands.
    norm : str or None
        Normalization mode: 'ortho' for orthonormal, None for standard.

    Returns
    -------
    tuple
        (bytes, shape) for caching.
    """
    # Create DCT-II basis matrix
    # The DCT-II decorrelates mel band energies into cepstral coefficients.
    # Lower coefficients capture overall spectral shape (formants),
    # higher coefficients capture fine spectral detail (pitch harmonics).
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)

    # DCT-II formula: C[k, n] = cos(π * k * (2n + 1) / (2N))
    # This is equivalent to taking real part of DFT of symmetrically extended input.
    dct_basis = np.cos(np.pi * k[:, None] * (2 * n + 1) / (2 * n_mels))

    if norm == "ortho":
        # Orthonormal scaling
        dct_basis[0, :] *= 1.0 / np.sqrt(n_mels)
        dct_basis[1:, :] *= np.sqrt(2.0 / n_mels)

    return dct_basis.astype(np.float32).tobytes(), dct_basis.shape


def dct(
    x: mx.array,
    type: int = 2,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = "ortho",
) -> mx.array:
    """
    Discrete Cosine Transform (DCT-II).

    Computes the DCT using a pre-computed basis matrix.

    Parameters
    ----------
    x : mx.array
        Input signal.
    type : int, default=2
        DCT type. Only type 2 is supported.
    n : int, optional
        Number of output coefficients. Default: input size along axis.
    axis : int, default=-1
        Axis along which to compute DCT.
    norm : str or None, default='ortho'
        Normalization mode: 'ortho' for orthonormal, None for standard.

    Returns
    -------
    mx.array
        DCT coefficients.

    Raises
    ------
    ValueError
        If type is not 2.
    """
    if type != 2:
        raise ValueError(f"Only DCT type 2 is supported, got {type}")

    # Get input size along axis
    input_size = x.shape[axis]
    if n is None:
        n = input_size

    # Use C++ extension if available
    if HAS_CPP_EXT and _ext is not None:
        norm_str = norm if norm is not None else ""
        return _ext.dct(x, n, axis, norm_str)

    # Python fallback
    # Get cached DCT matrix
    cache_key = (n, input_size, norm)
    if cache_key in _mlx_dct_cache:
        dct_matrix = _mlx_dct_cache[cache_key]
    else:
        dct_bytes, shape = _compute_dct_matrix_np(n, input_size, norm)
        dct_matrix = mx.array(np.frombuffer(dct_bytes, dtype=np.float32).reshape(shape))
        _mlx_dct_cache[cache_key] = dct_matrix

    # Apply DCT: output = dct_matrix @ x
    # Handle axis by moving it to the right position
    if axis != -1 and axis != x.ndim - 1:
        x = mx.moveaxis(x, axis, -1)

    # dct_matrix: (n, input_size)
    # x: (..., input_size)
    # output: (..., n)
    result = mx.matmul(x, dct_matrix.T)

    if axis != -1 and axis != x.ndim - 1:
        result = mx.moveaxis(result, -1, axis)

    return result


def mfcc(
    y: mx.array | None = None,
    sr: int = 22050,
    S: mx.array | None = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: str | None = "ortho",
    lifter: int = 0,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str | mx.array = "hann",
    center: bool = True,
    pad_mode: str = "constant",
    power: float = 2.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    mel_norm: str | None = "slaney",
) -> mx.array:
    """
    Compute Mel-frequency cepstral coefficients (MFCCs).

    MFCCs are computed from a power mel spectrogram by applying
    the discrete cosine transform (DCT) to the log-power mel spectrogram.

    Parameters
    ----------
    y : mx.array, optional
        Audio waveform. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    S : mx.array, optional
        Pre-computed log-power mel spectrogram. If provided, y is ignored.
        Shape: (n_mels, n_frames) or (batch, n_mels, n_frames).
    n_mfcc : int, default=20
        Number of MFCCs to return.
    dct_type : int, default=2
        DCT type (only 2 is supported).
    norm : str or None, default='ortho'
        DCT normalization. Use 'ortho' for orthonormal DCT.
    lifter : int, default=0
        If > 0, apply liftering (cepstral filtering) with the given
        coefficient. Liftering emphasizes higher-order coefficients.
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
        Exponent for mel spectrogram.
    n_mels : int, default=128
        Number of mel bands.
    fmin : float, default=0.0
        Minimum frequency for mel filterbank.
    fmax : float, optional
        Maximum frequency for mel filterbank. Default: sr / 2.
    htk : bool, default=False
        Use HTK formula for mel scale.
    mel_norm : str or None, default='slaney'
        Normalization for mel filterbank.

    Returns
    -------
    mx.array
        MFCC features.
        Shape: (n_mfcc, n_frames) for 1D input.
        Shape: (batch, n_mfcc, n_frames) for batched input.

    Examples
    --------
    >>> mfccs = mfcc(y, sr=22050, n_mfcc=13)
    >>> mfccs.shape
    (13, 44)
    """
    validate_positive(n_mfcc, "n_mfcc")

    # Track if S was provided directly (assumed to be log-power already)
    s_was_provided = S is not None

    if S is None:
        # Compute mel spectrogram from waveform
        S = melspectrogram(
            y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            norm=mel_norm,
        )

    # Handle batched vs non-batched
    is_batched = S.ndim == 3
    if not is_batched:
        S = S[None, :]  # Add batch dimension

    # Convert to log scale (dB) only if we computed the mel spectrogram
    # If S was provided directly, it's assumed to be log-power already
    if s_was_provided:
        S_db = S  # Already in log-power format
    else:
        S_db = power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0)

    # S_db shape: (batch, n_mels, n_frames)
    # Apply DCT along the mel axis (axis=1) directly
    # This avoids two transpose operations vs transposing to/from (batch, n_frames, n_mels)
    M = dct(S_db, type=dct_type, n=n_mfcc, axis=1, norm=norm)

    # Apply liftering if specified
    # Liftering is cepstral domain windowing that de-emphasizes higher coefficients.
    # This smooths the spectral envelope, reducing sensitivity to pitch harmonics
    # while preserving formant information. Common lifter values: 22-26 for ASR.
    if lifter > 0:
        # Liftering formula: M[n] *= 1 + (L/2) * sin(π(n+1)/L)
        n = np.arange(n_mfcc)
        lift = 1 + (lifter / 2.0) * np.sin(np.pi * (n + 1) / lifter)
        lift = mx.array(lift.astype(np.float32))[:, None]  # (n_mfcc, 1)
        M = M * lift

    if not is_batched:
        M = M[0]

    return M


def delta(
    data: mx.array,
    width: int = 9,
    order: int = 1,
    axis: int = -1,
    mode: str = "interp",
    **kwargs,
) -> mx.array:
    """
    Compute delta (derivative) features.

    Delta features capture temporal dynamics by computing local
    derivatives using Savitzky-Golay filtering (matching librosa exactly).

    Parameters
    ----------
    data : mx.array
        Input feature matrix (e.g., MFCCs).
        Shape: (n_features, n_frames) or (batch, n_features, n_frames).
    width : int, default=9
        Width of the delta window. Must be odd and >= 3.
    order : int, default=1
        Order of the derivative:
        - 1: First derivative (delta)
        - 2: Second derivative (delta-delta or acceleration)
    axis : int, default=-1
        Axis along which to compute deltas (time axis).
    mode : str, default='interp'
        Edge handling mode. Passed to scipy.signal.savgol_filter.
        - 'interp': Linear interpolation for edge frames
        - 'nearest': Replicate edge values
        - 'mirror': Reflect values
        - 'constant': Pad with zeros
        - 'wrap': Circular wrap
    **kwargs
        Additional arguments passed to scipy.signal.savgol_filter.

    Returns
    -------
    mx.array
        Delta features. Same shape as input.

    Examples
    --------
    >>> mfccs = mfcc(y, sr=22050, n_mfcc=13)
    >>> delta1 = delta(mfccs, width=9, order=1)
    >>> delta2 = delta(mfccs, width=9, order=2)
    >>> features = mx.concatenate([mfccs, delta1, delta2], axis=0)
    """
    from scipy.signal import savgol_filter

    validate_positive(width, "width")
    validate_positive(order, "order")

    if width < 3:
        raise ValueError(f"width must be >= 3, got {width}")
    if width % 2 == 0:
        raise ValueError(f"width must be odd, got {width}")

    data_np = np.atleast_1d(np.array(data))

    if mode == "interp" and width > data_np.shape[axis]:
        raise ValueError(
            f"when mode='interp', width={width} "
            f"cannot exceed data.shape[axis]={data_np.shape[axis]}"
        )

    # NOTE: Uses scipy.signal.savgol_filter for librosa compatibility.
    # Savitzky-Golay computes smoothed derivatives via local polynomial fitting,
    # which is more robust than simple finite differences at frame boundaries.
    # librosa switched to this method in v0.8.0 for better edge handling.
    kwargs.pop("deriv", None)
    kwargs.setdefault("polyorder", order)

    result_np = savgol_filter(
        data_np, width, deriv=order, axis=axis, mode=mode, **kwargs
    )

    return mx.array(result_np.astype(np.float32))
