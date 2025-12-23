"""Parler-TTS model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParlerTTSConfig:
    """Configuration for Parler-TTS text-to-speech model.

    Parler-TTS is a decoder-only transformer that generates speech audio tokens
    conditioned on text input and optional voice description. It uses EnCodec
    for audio tokenization/detokenization.

    Attributes:
        # Audio codec settings
        num_codebooks: Number of audio codebooks (from DAC/EnCodec)
        codebook_size: Vocabulary size per codebook
        audio_channels: Number of audio channels (1=mono)
        sample_rate: Audio sample rate in Hz
        frame_rate: Audio codec frame rate (tokens per second)

        # Text encoder settings
        text_encoder_name: Name of the text encoder model (T5)
        text_hidden_size: Dimension of text encoder hidden states
        max_text_length: Maximum text sequence length

        # Description encoder settings
        description_encoder_name: Name of the description encoder (T5)
        description_hidden_size: Dimension of description encoder hidden states
        max_description_length: Maximum description sequence length

        # Decoder transformer settings
        hidden_size: Decoder hidden dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        intermediate_size: FFN intermediate dimension
        activation_function: Activation function in FFN
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
        layer_norm_eps: Layer normalization epsilon

        # Generation settings
        max_duration: Maximum generation duration in seconds
        use_cache: Whether to use KV caching during generation

        # Special tokens
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
    """

    # Audio codec settings (DAC 24kHz)
    num_codebooks: int = 9
    codebook_size: int = 1024
    audio_channels: int = 1
    sample_rate: int = 24000
    frame_rate: int = 75  # DAC frame rate at 24kHz

    # Text encoder settings (T5)
    text_encoder_name: str = "google/flan-t5-large"
    text_hidden_size: int = 1024
    max_text_length: int = 600

    # Description encoder settings (T5)
    description_encoder_name: str = "google/flan-t5-large"
    description_hidden_size: int = 1024
    max_description_length: int = 256

    # Decoder transformer settings
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16  # Same as num_attention_heads for MHA
    intermediate_size: int = 4096
    activation_function: str = "gelu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-5

    # Position embedding settings
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0

    # Generation settings
    max_duration: float = 30.0  # seconds
    use_cache: bool = True

    # Special tokens
    pad_token_id: int = 1024  # codebook_size
    bos_token_id: int = 1025  # codebook_size + 1
    eos_token_id: int = 1024  # Same as pad

    @property
    def max_new_tokens(self) -> int:
        """Maximum number of new tokens to generate."""
        return int(self.max_duration * self.frame_rate)

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self.codebook_size + 2  # +2 for pad and bos

    @classmethod
    def mini(cls) -> "ParlerTTSConfig":
        """Parler-TTS Mini configuration (~880M parameters).

        Fast inference, good for real-time applications.
        """
        return cls(
            num_codebooks=9,
            codebook_size=1024,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=4096,
            text_encoder_name="google/flan-t5-large",
            text_hidden_size=1024,
            description_encoder_name="google/flan-t5-large",
            description_hidden_size=1024,
        )

    @classmethod
    def large(cls) -> "ParlerTTSConfig":
        """Parler-TTS Large configuration (~2.3B parameters).

        Best quality, slower inference.
        """
        return cls(
            num_codebooks=9,
            codebook_size=1024,
            hidden_size=1536,
            num_hidden_layers=36,
            num_attention_heads=24,
            num_key_value_heads=24,
            intermediate_size=6144,
            text_encoder_name="google/flan-t5-large",
            text_hidden_size=1024,
            description_encoder_name="google/flan-t5-large",
            description_hidden_size=1024,
        )

    @classmethod
    def from_name(cls, name: str) -> "ParlerTTSConfig":
        """Create config from model name.

        Args:
            name: Model name (e.g., "mini", "large")

        Returns:
            ParlerTTSConfig for the specified model

        Raises:
            ValueError: If model name is not recognized
        """
        name = name.lower().replace("-", "_").replace("parler_tts_", "")

        configs = {
            "mini": cls.mini,
            "large": cls.large,
        }

        if name not in configs:
            available = ", ".join(sorted(configs.keys()))
            raise ValueError(
                f"Unknown Parler-TTS model: {name!r}. Available: {available}"
            )

        return configs[name]()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParlerTTSConfig":
        """Create config from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict

        return asdict(self)
