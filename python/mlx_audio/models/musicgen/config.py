"""MusicGen model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MusicGenConfig:
    """Configuration for MusicGen text-to-music generation model.

    MusicGen is a decoder-only transformer that generates audio tokens
    conditioned on text descriptions. It uses a delay pattern to handle
    multiple codebooks from the audio codec.

    Attributes:
        # Audio codec settings
        num_codebooks: Number of audio codebooks (from EnCodec)
        codebook_size: Vocabulary size per codebook
        audio_channels: Number of audio channels (1=mono, 2=stereo)
        sample_rate: Audio sample rate in Hz
        frame_rate: Audio codec frame rate (tokens per second)

        # Text encoder settings
        text_encoder_name: Name of the text encoder model (T5)
        text_hidden_size: Dimension of text encoder hidden states
        max_text_length: Maximum text sequence length

        # Decoder transformer settings
        hidden_size: Decoder hidden dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
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

    # Audio codec settings
    num_codebooks: int = 4
    codebook_size: int = 2048
    audio_channels: int = 1
    sample_rate: int = 32000
    frame_rate: int = 50  # frames per second

    # Text encoder settings (T5)
    text_encoder_name: str = "t5-base"
    text_hidden_size: int = 768
    max_text_length: int = 256

    # Decoder transformer settings
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    activation_function: str = "gelu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-5

    # Generation settings
    max_duration: float = 30.0  # seconds
    use_cache: bool = True

    # Special tokens (per codebook)
    pad_token_id: int = 2048  # codebook_size
    bos_token_id: int = 2048  # codebook_size (same as pad for musicgen)
    eos_token_id: int = 2048  # codebook_size

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
        """Total vocabulary size (matches HuggingFace MusicGen)."""
        return self.codebook_size  # HF uses codebook_size directly

    @classmethod
    def small(cls) -> MusicGenConfig:
        """MusicGen small configuration (~300M parameters)."""
        return cls(
            num_codebooks=4,
            codebook_size=2048,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            text_encoder_name="t5-base",
            text_hidden_size=768,
        )

    @classmethod
    def medium(cls) -> MusicGenConfig:
        """MusicGen medium configuration (~1.5B parameters)."""
        return cls(
            num_codebooks=4,
            codebook_size=2048,
            hidden_size=1536,
            num_hidden_layers=48,
            num_attention_heads=24,
            intermediate_size=6144,
            text_encoder_name="t5-base",
            text_hidden_size=768,
        )

    @classmethod
    def large(cls) -> MusicGenConfig:
        """MusicGen large configuration (~3.3B parameters)."""
        return cls(
            num_codebooks=4,
            codebook_size=2048,
            hidden_size=2048,
            num_hidden_layers=48,
            num_attention_heads=32,
            intermediate_size=8192,
            text_encoder_name="t5-base",
            text_hidden_size=768,
        )

    @classmethod
    def melody(cls) -> MusicGenConfig:
        """MusicGen melody configuration (with melody conditioning)."""
        config = cls.medium()
        # Melody variant uses additional conditioning
        return config

    @classmethod
    def from_name(cls, name: str) -> MusicGenConfig:
        """Create config from model name.

        Args:
            name: Model name (e.g., "small", "medium", "large", "melody")

        Returns:
            MusicGenConfig for the specified model

        Raises:
            ValueError: If model name is not recognized
        """
        name = name.lower().replace("-", "_").replace("musicgen_", "")

        configs = {
            "small": cls.small,
            "medium": cls.medium,
            "large": cls.large,
            "melody": cls.melody,
        }

        if name not in configs:
            available = ", ".join(sorted(configs.keys()))
            raise ValueError(
                f"Unknown MusicGen model: {name!r}. Available: {available}"
            )

        return configs[name]()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MusicGenConfig:
        """Create config from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
