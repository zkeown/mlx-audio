"""Whisper model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_audio.constants import (
    WHISPER_SAMPLE_RATE,
    WHISPER_N_FFT,
    WHISPER_HOP_LENGTH,
    WHISPER_CHUNK_LENGTH,
    WHISPER_N_MELS,
    WHISPER_V3_N_MELS,
    WHISPER_N_AUDIO_CTX,
    WHISPER_N_TEXT_CTX,
)
from mlx_audio.models.base import ModelConfig


@dataclass
class WhisperConfig(ModelConfig):
    """Configuration for Whisper speech recognition model.

    Whisper is an encoder-decoder transformer model trained on multilingual
    speech recognition, translation, language identification, and voice activity
    detection tasks.

    Attributes:
        n_mels: Number of mel filterbank bins (80 for v1/v2, 128 for v3)
        n_audio_ctx: Audio context length (encoder output frames)
        n_audio_state: Encoder hidden dimension
        n_audio_head: Number of encoder attention heads
        n_audio_layer: Number of encoder transformer layers
        n_text_ctx: Text context length (max tokens)
        n_text_state: Decoder hidden dimension
        n_text_head: Number of decoder attention heads
        n_text_layer: Number of decoder transformer layers
        n_vocab: Vocabulary size
        sample_rate: Expected audio sample rate
        n_fft: FFT window size
        hop_length: STFT hop length
        chunk_length: Audio chunk length in seconds
    """

    # Audio processing
    n_mels: int = WHISPER_N_MELS
    sample_rate: int = WHISPER_SAMPLE_RATE
    n_fft: int = WHISPER_N_FFT  # 25ms window at 16kHz
    hop_length: int = WHISPER_HOP_LENGTH  # 10ms hop at 16kHz
    chunk_length: int = WHISPER_CHUNK_LENGTH  # 30 second chunks

    # Encoder architecture
    n_audio_ctx: int = WHISPER_N_AUDIO_CTX  # 30s * 16000 / 160 / 2 (due to conv stride)
    n_audio_state: int = 384
    n_audio_head: int = 6
    n_audio_layer: int = 4

    # Decoder architecture
    n_text_ctx: int = WHISPER_N_TEXT_CTX
    n_text_state: int = 384
    n_text_head: int = 6
    n_text_layer: int = 4

    # Vocabulary
    n_vocab: int = 51865  # Multilingual tokenizer size

    @property
    def n_samples(self) -> int:
        """Number of audio samples per chunk."""
        return self.chunk_length * self.sample_rate

    @property
    def n_frames(self) -> int:
        """Number of mel frames per chunk (before conv downsampling)."""
        return self.n_samples // self.hop_length

    @classmethod
    def tiny(cls) -> "WhisperConfig":
        """Whisper tiny configuration (39M parameters)."""
        return cls(
            n_mels=WHISPER_N_MELS,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
        )

    @classmethod
    def tiny_en(cls) -> "WhisperConfig":
        """Whisper tiny.en configuration (English-only, 39M parameters)."""
        config = cls.tiny()
        config.n_vocab = 51864  # English-only vocab
        return config

    @classmethod
    def base(cls) -> "WhisperConfig":
        """Whisper base configuration (74M parameters)."""
        return cls(
            n_mels=WHISPER_N_MELS,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            n_text_state=512,
            n_text_head=8,
            n_text_layer=6,
        )

    @classmethod
    def base_en(cls) -> "WhisperConfig":
        """Whisper base.en configuration (English-only, 74M parameters)."""
        config = cls.base()
        config.n_vocab = 51864
        return config

    @classmethod
    def small(cls) -> "WhisperConfig":
        """Whisper small configuration (244M parameters)."""
        return cls(
            n_mels=WHISPER_N_MELS,
            n_audio_state=768,
            n_audio_head=12,
            n_audio_layer=12,
            n_text_state=768,
            n_text_head=12,
            n_text_layer=12,
        )

    @classmethod
    def small_en(cls) -> "WhisperConfig":
        """Whisper small.en configuration (English-only, 244M parameters)."""
        config = cls.small()
        config.n_vocab = 51864
        return config

    @classmethod
    def medium(cls) -> "WhisperConfig":
        """Whisper medium configuration (769M parameters)."""
        return cls(
            n_mels=WHISPER_N_MELS,
            n_audio_state=1024,
            n_audio_head=16,
            n_audio_layer=24,
            n_text_state=1024,
            n_text_head=16,
            n_text_layer=24,
        )

    @classmethod
    def medium_en(cls) -> "WhisperConfig":
        """Whisper medium.en configuration (English-only, 769M parameters)."""
        config = cls.medium()
        config.n_vocab = 51864
        return config

    @classmethod
    def large(cls) -> "WhisperConfig":
        """Whisper large configuration (1.5B parameters)."""
        return cls(
            n_mels=WHISPER_N_MELS,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )

    @classmethod
    def large_v2(cls) -> "WhisperConfig":
        """Whisper large-v2 configuration (1.5B parameters, improved training)."""
        return cls.large()

    @classmethod
    def large_v3(cls) -> "WhisperConfig":
        """Whisper large-v3 configuration (1.5B parameters, 128 mel bins)."""
        config = cls.large()
        config.n_mels = WHISPER_V3_N_MELS
        return config

    @classmethod
    def large_v3_turbo(cls) -> "WhisperConfig":
        """Whisper large-v3-turbo configuration (809M parameters).

        Uses large-v3 encoder (32 layers) with only 4 decoder layers,
        providing ~8x faster inference with minimal quality loss.
        """
        return cls(
            n_mels=WHISPER_V3_N_MELS,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=4,  # Reduced from 32 to 4
        )

    # Aliases
    @classmethod
    def turbo(cls) -> "WhisperConfig":
        """Alias for large_v3_turbo."""
        return cls.large_v3_turbo()

    @classmethod
    def from_name(cls, name: str) -> "WhisperConfig":
        """Create config from model name.

        Args:
            name: Model name (e.g., "tiny", "base.en", "large-v3-turbo")

        Returns:
            WhisperConfig for the specified model

        Raises:
            ValueError: If model name is not recognized
        """
        # Normalize name
        name = name.lower().replace("-", "_").replace(".", "_")

        # Remove common prefixes
        for prefix in ("whisper_", "openai_whisper_", "openai/whisper_"):
            if name.startswith(prefix):
                name = name[len(prefix):]

        configs = {
            "tiny": cls.tiny,
            "tiny_en": cls.tiny_en,
            "base": cls.base,
            "base_en": cls.base_en,
            "small": cls.small,
            "small_en": cls.small_en,
            "medium": cls.medium,
            "medium_en": cls.medium_en,
            "large": cls.large,
            "large_v1": cls.large,
            "large_v2": cls.large_v2,
            "large_v3": cls.large_v3,
            "large_v3_turbo": cls.large_v3_turbo,
            "turbo": cls.turbo,
        }

        if name not in configs:
            available = ", ".join(sorted(configs.keys()))
            raise ValueError(
                f"Unknown Whisper model: {name!r}. Available: {available}"
            )

        return configs[name]()

    # from_dict and to_dict are inherited from ModelConfig

    @property
    def is_multilingual(self) -> bool:
        """Whether this is a multilingual model."""
        return self.n_vocab >= 51865

    @property
    def is_v3(self) -> bool:
        """Whether this uses v3 architecture (128 mel bins)."""
        return self.n_mels == WHISPER_V3_N_MELS
