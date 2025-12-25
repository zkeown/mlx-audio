"""VAD model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from mlx_audio.models.base import ModelConfig


@dataclass
class VADConfig(ModelConfig):
    """Configuration for Voice Activity Detection model.

    Silero VAD-compatible configuration for detecting speech in audio.
    The model uses LSTM-based architecture optimized for real-time inference.

    Attributes:
        sample_rate: Audio sample rate in Hz (8000 or 16000)
        window_size_samples: Number of samples per processing window
        context_size_samples: Number of context samples retained between windows
        hidden_size: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        threshold: Default speech probability threshold
    """

    # Audio parameters
    sample_rate: int = 16000

    # Window and context sizes (derived from sample rate)
    # At 16kHz: 512 samples = 32ms window, 64 samples = 4ms context
    # At 8kHz: 256 samples = 32ms window, 32 samples = 4ms context
    window_size_samples: int = 512
    context_size_samples: int = 64

    # Model architecture
    hidden_size: int = 128
    num_layers: int = 2

    # Default inference parameters
    threshold: float = 0.5

    # Registry for preset configs
    _config_registry: ClassVar[dict[str, str]] = {
        "silero_vad": "silero_vad_16k",
        "silero_vad_16k": "silero_vad_16k",
        "silero_vad_8k": "silero_vad_8k",
    }

    @classmethod
    def silero_vad_16k(cls) -> VADConfig:
        """Silero VAD configuration for 16kHz audio."""
        return cls(
            sample_rate=16000,
            window_size_samples=512,
            context_size_samples=64,
            hidden_size=128,
            num_layers=2,
        )

    @classmethod
    def silero_vad_8k(cls) -> VADConfig:
        """Silero VAD configuration for 8kHz audio."""
        return cls(
            sample_rate=8000,
            window_size_samples=256,
            context_size_samples=32,
            hidden_size=128,
            num_layers=2,
        )

    @property
    def window_duration_ms(self) -> float:
        """Window duration in milliseconds."""
        return (self.window_size_samples / self.sample_rate) * 1000

    @property
    def context_duration_ms(self) -> float:
        """Context duration in milliseconds."""
        return (self.context_size_samples / self.sample_rate) * 1000

    @property
    def min_speech_duration_samples(self) -> int:
        """Minimum speech segment duration in samples (250ms default)."""
        return int(0.25 * self.sample_rate)

    @property
    def min_silence_duration_samples(self) -> int:
        """Minimum silence duration to split segments (100ms default)."""
        return int(0.1 * self.sample_rate)


__all__ = ["VADConfig"]
