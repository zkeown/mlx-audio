"""Constants module for mlx-audio.

This module centralizes magic numbers, configuration values, and
hardcoded constants used throughout the codebase.

Submodules:
    sample_rates: Audio sample rate constants
    audio_processing: STFT, buffer, and chunk size constants
    spectral: Mel filterbank and spectrogram constants
    algorithms: Algorithm-specific magic numbers
    paths: Cache directories and file operation constants
"""

from __future__ import annotations

from mlx_audio.constants.algorithms import *
from mlx_audio.constants.audio_processing import *
from mlx_audio.constants.paths import *
from mlx_audio.constants.sample_rates import *
from mlx_audio.constants.spectral import *

__all__ = [
    # Sample rates
    "WHISPER_SAMPLE_RATE",
    "DEMUCS_SAMPLE_RATE",
    "MUSICGEN_SAMPLE_RATE",
    "CLAP_SAMPLE_RATE",
    "ENCODEC_24KHZ_RATE",
    "ENCODEC_32KHZ_RATE",
    "ENCODEC_48KHZ_RATE",
    "DEFAULT_STREAMING_RATE",
    # Audio processing
    "WHISPER_N_FFT",
    "WHISPER_HOP_LENGTH",
    "DEMUCS_N_FFT",
    "DEMUCS_HOP_LENGTH",
    "CLAP_N_FFT",
    "CLAP_HOP_LENGTH",
    "CLAP_WINDOW_LENGTH",
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_BLOCKSIZE",
    "DEFAULT_CHANNELS",
    "WHISPER_CHUNK_LENGTH",
    "WHISPER_N_AUDIO_CTX",
    "WHISPER_N_TEXT_CTX",
    # Spectral
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
    # Algorithms
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
    "CLAP_LOGIT_SCALE_INIT",
    "DCONV_INIT_SCALE",
    # Paths
    "CACHE_DIR",
    "MODELS_CACHE_DIR",
    "WHISPER_CACHE_DIR",
    "DEMUCS_CACHE_DIR",
    "ENCODEC_CACHE_DIR",
    "MUSICGEN_CACHE_DIR",
    "CLAP_CACHE_DIR",
    "WINDOW_CACHE_MAXSIZE",
    "FILE_DOWNLOAD_CHUNK_SIZE",
]
