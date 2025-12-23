"""Spectral analysis constants for mel filterbanks and spectrograms.

These constants define parameters for mel-frequency analysis
and spectrogram computation.
"""

from __future__ import annotations

import math

# Mel filterbank sizes by model
WHISPER_N_MELS = 80
"""Whisper v1/v2 mel filterbank bins."""

WHISPER_V3_N_MELS = 128
"""Whisper v3 mel filterbank bins."""

CLAP_N_MELS = 64
"""CLAP mel filterbank bins."""

DEFAULT_N_MELS = 128
"""Default mel filterbank bins for general use."""

# Frequency bounds
WHISPER_FMAX = 8000.0
"""Whisper maximum frequency for mel filterbank (Nyquist at 16kHz)."""

CLAP_SPEC_SIZE = 256
"""CLAP spectrogram size."""

# Slaney mel formula constants
# These match librosa's default 'slaney' mel scale
SLANEY_F_MIN = 0.0
"""Slaney mel scale minimum frequency."""

SLANEY_F_SP = 200.0 / 3
"""Slaney mel scale frequency spacing (66.67 Hz)."""

SLANEY_MIN_LOG_HZ = 1000.0
"""Slaney mel scale transition to log spacing."""

SLANEY_LOGSTEP_DIVISOR = 27.0
"""Slaney mel scale log step divisor."""

SLANEY_LOGSTEP = math.log(6.4) / SLANEY_LOGSTEP_DIVISOR
"""Slaney mel scale log step size."""

# HTK mel formula constants
HTK_MEL_FACTOR = 2595.0
"""HTK mel scale conversion factor."""

HTK_MEL_BASE = 700.0
"""HTK mel scale base frequency."""

__all__ = [
    "WHISPER_N_MELS",
    "WHISPER_V3_N_MELS",
    "CLAP_N_MELS",
    "DEFAULT_N_MELS",
    "WHISPER_FMAX",
    "CLAP_SPEC_SIZE",
    "SLANEY_F_MIN",
    "SLANEY_F_SP",
    "SLANEY_MIN_LOG_HZ",
    "SLANEY_LOGSTEP_DIVISOR",
    "SLANEY_LOGSTEP",
    "HTK_MEL_FACTOR",
    "HTK_MEL_BASE",
]
