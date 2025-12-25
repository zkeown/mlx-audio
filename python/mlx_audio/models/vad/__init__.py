"""Voice Activity Detection model.

This module provides a Silero VAD-compatible voice activity detection model
for detecting speech in audio signals.

Example:
    >>> from mlx_audio.models.vad import SileroVAD, VADConfig
    >>>
    >>> # Create model with default config (16kHz)
    >>> model = SileroVAD()
    >>>
    >>> # Process audio (512 samples = 32ms at 16kHz)
    >>> import mlx.core as mx
    >>> audio = mx.random.normal((512,))
    >>> prob, state = model(audio)
    >>> print(f"Speech probability: {float(prob):.3f}")
    >>>
    >>> # Streaming usage
    >>> state = None
    >>> for chunk in audio_chunks:
    ...     prob, state = model(chunk, state=state)
    ...     if float(prob) > 0.5:
    ...         print("Speech detected")

    >>> # Use 8kHz config
    >>> config = VADConfig.silero_vad_8k()
    >>> model = SileroVAD(config)
"""

from mlx_audio.models.vad.config import VADConfig
from mlx_audio.models.vad.layers import StackedLSTM, VADDecoder, VADEncoder
from mlx_audio.models.vad.model import SileroVAD

__all__ = [
    "VADConfig",
    "SileroVAD",
    "VADEncoder",
    "VADDecoder",
    "StackedLSTM",
]
