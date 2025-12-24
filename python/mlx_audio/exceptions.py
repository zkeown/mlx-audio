"""Custom exception hierarchy for mlx-audio.

All mlx-audio specific exceptions inherit from MLXAudioError,
making it easy to catch any library-specific error.
"""

from __future__ import annotations


class MLXAudioError(Exception):
    """Base exception for all mlx-audio errors.

    Example:
        try:
            result = mlx_audio.separate("audio.mp3")
        except MLXAudioError as e:
            print(f"mlx-audio error: {e}")
    """

    pass


class AudioLoadError(MLXAudioError):
    """Failed to load or process audio file.

    Raised when:
        - File does not exist or cannot be read
        - Audio format is unsupported
        - Audio data is corrupted or invalid
    """

    pass


class ModelNotFoundError(MLXAudioError):
    """Model weights not found in cache or hub.

    Raised when:
        - Model name is not in the registry
        - Weights cannot be downloaded from HuggingFace Hub
        - Local model path does not exist
    """

    pass


class ConfigurationError(MLXAudioError):
    """Invalid model or operation configuration.

    Raised when:
        - Config parameters are out of valid range
        - Required config fields are missing
        - Incompatible config combinations are specified
    """

    pass


class InferenceError(MLXAudioError):
    """Error during model inference.

    Raised when:
        - Input tensor has wrong shape or dtype
        - Model encounters numerical issues (NaN, Inf)
        - GPU memory is insufficient
    """

    pass


class WeightConversionError(MLXAudioError):
    """Error converting weights from PyTorch to MLX.

    Raised when:
        - Weight file format is unrecognized
        - Weight shapes don't match expected architecture
        - Required weight keys are missing
    """

    pass


class StreamingError(MLXAudioError):
    """Error in audio streaming pipeline.

    Raised when:
        - Audio source/sink fails to initialize
        - Buffer overflow/underflow occurs
        - Pipeline connection fails
    """

    pass


class TokenizationError(MLXAudioError):
    """Error tokenizing text input.

    Raised when:
        - Text contains unsupported characters
        - Tokenizer model cannot be loaded
        - Token sequence exceeds maximum length
    """

    pass


__all__ = [
    "MLXAudioError",
    "AudioLoadError",
    "ModelNotFoundError",
    "ConfigurationError",
    "InferenceError",
    "WeightConversionError",
    "StreamingError",
    "TokenizationError",
]
