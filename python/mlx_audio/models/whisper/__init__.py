"""Whisper speech recognition model for MLX.

This module provides a complete implementation of OpenAI's Whisper
speech recognition model optimized for Apple Silicon using MLX.

Example:
    >>> from mlx_audio.models.whisper import Whisper, WhisperTokenizer
    >>> from mlx_audio.models.whisper.inference import apply_model
    >>>
    >>> # Load model
    >>> model = Whisper.from_pretrained("path/to/whisper")
    >>> tokenizer = WhisperTokenizer(multilingual=True)
    >>>
    >>> # Transcribe
    >>> result = apply_model(model, audio, tokenizer)
    >>> print(result.text)
"""

from mlx_audio.models.whisper.config import WhisperConfig
from mlx_audio.models.whisper.convert import convert_whisper_weights, download_and_convert
from mlx_audio.models.whisper.inference import (
    DecodingOptions,
    apply_model,
    beam_search_decode,
    compute_log_mel_spectrogram,
    greedy_decode,
    pad_or_trim,
    transcribe_segment,
    transcribe_with_chunks,
)
from mlx_audio.models.whisper.model import Whisper
from mlx_audio.models.whisper.tokenizer import LANGUAGES, WhisperTokenizer, get_tokenizer

__all__ = [
    # Config
    "WhisperConfig",
    # Model
    "Whisper",
    # Tokenizer
    "WhisperTokenizer",
    "get_tokenizer",
    "LANGUAGES",
    # Conversion
    "convert_whisper_weights",
    "download_and_convert",
    # Inference
    "DecodingOptions",
    "apply_model",
    "greedy_decode",
    "beam_search_decode",
    "transcribe_segment",
    "transcribe_with_chunks",
    "compute_log_mel_spectrogram",
    "pad_or_trim",
]
