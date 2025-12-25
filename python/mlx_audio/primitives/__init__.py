"""
mlx-audio-primitives: Foundational audio DSP primitives for Apple's MLX framework.

This library provides librosa-compatible audio processing primitives for MLX,
enabling efficient audio ML pipelines on Apple Silicon.

Core Operations
---------------
stft : Short-Time Fourier Transform
istft : Inverse Short-Time Fourier Transform
magnitude : Compute magnitude from complex STFT
phase : Compute phase from complex STFT

Window Functions
----------------
get_window : Get a window function (hann, hamming, blackman, etc.)

Mel-Scale Operations
--------------------
mel_filterbank : Create mel-scale filterbank matrix
melspectrogram : Compute mel spectrogram from waveform

Filterbanks
-----------
linear_filterbank : Create linear-scale filterbank matrix
bark_filterbank : Create Bark-scale filterbank matrix
hz_to_bark : Convert Hz to Bark scale
bark_to_hz : Convert Bark scale to Hz

Spectral Features
-----------------
spectral_centroid : Compute spectral centroid
spectral_bandwidth : Compute spectral bandwidth
spectral_rolloff : Compute spectral rolloff frequency
spectral_flatness : Compute spectral flatness
spectral_contrast : Compute spectral contrast
zero_crossing_rate : Compute zero crossing rate

MFCC
----
mfcc : Compute Mel-frequency cepstral coefficients
delta : Compute delta (derivative) features
dct : Discrete Cosine Transform (Type II)

Time-Domain
-----------
frame : Frame signal into overlapping windows
rms : Compute root-mean-square energy per frame
preemphasis : Apply pre-emphasis filter
deemphasis : Apply de-emphasis filter (inverse)

Resampling
----------
resample : Resample audio to different sample rate
resample_poly : Polyphase resampling for integer ratios

Phase Reconstruction
--------------------
griffinlim : Griffin-Lim phase reconstruction

Pitch/Periodicity
-----------------
autocorrelation : Compute autocorrelation
pitch_detect_acf : Detect pitch using autocorrelation
periodicity : Compute periodicity strength

Decibel Conversions
-------------------
power_to_db : Convert power spectrogram to decibels
db_to_power : Convert decibels to power spectrogram
amplitude_to_db : Convert amplitude spectrogram to decibels
db_to_amplitude : Convert decibels to amplitude spectrogram

Utilities
---------
check_nola : Check NOLA constraint for STFT invertibility
hz_to_mel : Convert Hz to mel scale
mel_to_hz : Convert mel scale to Hz
"""

# Import MLX first to ensure library paths are set up for C++ extension
import mlx.core as _mx  # noqa: F401

# Get version from package metadata (single source of truth in pyproject.toml)
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _get_version

    __version__ = _get_version("mlx-audio-primitives")
except (ImportError, PackageNotFoundError):
    __version__ = "1.0.0"  # Fallback for editable installs

# Import C++ extension availability flag for external use
from ._extension import HAS_CPP_EXT as _HAS_CPP_EXT  # noqa: F401

# Beat tracking
from .beat import (
    beat_track,
    plp,
    tempo,
)

# Decibel conversions
from .convert import (
    amplitude_to_db,
    db_to_amplitude,
    db_to_power,
    power_to_db,
)

# Spectral features
from .features import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_contrast,
    spectral_flatness,
    spectral_rolloff,
    zero_crossing_rate,
)

# Filterbanks (linear and Bark scale)
from .filterbanks import (
    bark_filterbank,
    bark_to_hz,
    hz_to_bark,
    linear_filterbank,
)

# Time-domain primitives
from .framing import (
    deemphasis,
    frame,
    preemphasis,
    rms,
)

# Phase reconstruction
from .griffinlim import (
    griffinlim,
)

# Mel-scale operations
from .mel import (
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    melspectrogram,
)

# MFCC and delta
from .mfcc import (
    dct,
    delta,
    mfcc,
)

# Onset detection
from .onset import (
    onset_detect,
    onset_strength,
    onset_strength_multi,
)

# Pitch and periodicity
from .pitch import (
    autocorrelation,
    periodicity,
    pitch_detect_acf,
    pyin,
    yin,
)

# Resampling
from .resample import (
    resample,
    resample_poly,
)

# Spectral gating (noise reduction)
from .spectral_gate import (
    spectral_gate,
    spectral_gate_adaptive,
)

# Core STFT operations
from .stft import (
    check_nola,
    istft,
    magnitude,
    phase,
    stft,
)

# Window functions
from .windows import get_window

__all__ = [
    # Version
    "__version__",
    # STFT
    "stft",
    "istft",
    "magnitude",
    "phase",
    "check_nola",
    # Windows
    "get_window",
    # Mel
    "mel_filterbank",
    "melspectrogram",
    "hz_to_mel",
    "mel_to_hz",
    # Filterbanks
    "linear_filterbank",
    "bark_filterbank",
    "hz_to_bark",
    "bark_to_hz",
    # Spectral features
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "spectral_contrast",
    "zero_crossing_rate",
    # MFCC
    "mfcc",
    "delta",
    "dct",
    # Time-domain
    "frame",
    "rms",
    "preemphasis",
    "deemphasis",
    # Resampling
    "resample",
    "resample_poly",
    # Phase reconstruction
    "griffinlim",
    # Pitch/periodicity
    "autocorrelation",
    "pitch_detect_acf",
    "periodicity",
    "yin",
    "pyin",
    # Onset detection
    "onset_strength",
    "onset_detect",
    "onset_strength_multi",
    # Beat tracking
    "tempo",
    "beat_track",
    "plp",
    # Spectral gating
    "spectral_gate",
    "spectral_gate_adaptive",
    # Conversions
    "power_to_db",
    "db_to_power",
    "amplitude_to_db",
    "db_to_amplitude",
]
