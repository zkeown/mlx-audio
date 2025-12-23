"""Algorithm-specific constants and magic numbers.

These constants define parameters for various audio processing
algorithms used throughout mlx-audio.
"""

from __future__ import annotations

# Pre-emphasis filtering
PRE_EMPHASIS_COEF = 0.97
"""Standard pre-emphasis coefficient for speech processing."""

# Griffin-Lim algorithm
GRIFFIN_LIM_MOMENTUM = 0.99
"""Default momentum for Griffin-Lim phase reconstruction."""

GRIFFIN_LIM_ITERATIONS = 32
"""Default number of Griffin-Lim iterations."""

# Spectral feature extraction
SPECTRAL_ROLLOFF_PERCENTILE = 0.85
"""Default percentile for spectral rolloff computation."""

SPECTRAL_CONTRAST_QUANTILE = 0.02
"""Default quantile for spectral contrast computation."""

# Whisper spectrogram normalization
WHISPER_LOG_CLIP_VALUE = 8.0
"""Whisper log mel spectrogram clipping value."""

WHISPER_NORMALIZATION_SCALE = 4.0
"""Whisper spectrogram normalization scale factor."""

# Numerical stability
LAYER_NORM_EPS = 1e-5
"""Default epsilon for layer normalization."""

DEFAULT_AMIN = 1e-10
"""Default minimum amplitude for logarithmic operations."""

CONVERT_AMIN = 1e-5
"""Amplitude floor for dB conversion."""

DIVISION_EPSILON = 1e-8
"""Epsilon for preventing division by zero in general operations."""

FILTERBANK_EPSILON = 1e-10
"""Epsilon for filterbank slope calculations (tight precision needed)."""

# CLAP-specific
CLAP_LOGIT_SCALE_INIT = 2.6592
"""CLAP contrastive learning logit scale (log(14.2857))."""

# Weight initialization
DCONV_INIT_SCALE = 1e-4
"""Demucs DConv weight initialization scale."""

__all__ = [
    "PRE_EMPHASIS_COEF",
    "GRIFFIN_LIM_MOMENTUM",
    "GRIFFIN_LIM_ITERATIONS",
    "SPECTRAL_ROLLOFF_PERCENTILE",
    "SPECTRAL_CONTRAST_QUANTILE",
    "WHISPER_LOG_CLIP_VALUE",
    "WHISPER_NORMALIZATION_SCALE",
    "LAYER_NORM_EPS",
    "DEFAULT_AMIN",
    "CONVERT_AMIN",
    "DIVISION_EPSILON",
    "FILTERBANK_EPSILON",
    "CLAP_LOGIT_SCALE_INIT",
    "DCONV_INIT_SCALE",
]
