"""EnCodec model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnCodecConfig:
    """Configuration for EnCodec neural audio codec.

    EnCodec is a neural audio codec that compresses audio into discrete tokens
    using a convolutional encoder, residual vector quantization, and a
    convolutional decoder.

    Attributes:
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels (1 for mono, 2 for stereo)
        num_codebooks: Number of codebooks for residual vector quantization
        codebook_size: Number of entries per codebook (vocabulary size)
        codebook_dim: Dimension of codebook vectors
        hidden_size: Base hidden dimension for encoder/decoder
        num_filters: Number of output filters in first conv layer
        num_residual_layers: Number of residual layers in each block
        ratios: Downsampling/upsampling ratios for each encoder/decoder stage
        norm_type: Normalization type ("weight_norm" or "time_group_norm")
        kernel_size: Kernel size for convolutional layers
        last_kernel_size: Kernel size for final conv layer
        residual_kernel_size: Kernel size for residual layers
        dilation_base: Base for dilated convolutions
        causal: Whether to use causal convolutions
        pad_mode: Padding mode for convolutions
        compress: Compression factor for hidden dimension
        lstm_layers: Number of LSTM layers (0 to disable)
        disable_norm_outer_blocks: Number of outer blocks to skip normalization
        trim_right_ratio: Ratio of right padding to trim
    """

    # Audio settings
    sample_rate: int = 32000
    channels: int = 1

    # Quantization settings
    num_codebooks: int = 4
    codebook_size: int = 2048
    codebook_dim: int = 128

    # Architecture settings
    hidden_size: int = 128
    num_filters: int = 32
    num_residual_layers: int = 1
    ratios: tuple[int, ...] = (8, 5, 4, 2)
    norm_type: str = "weight_norm"
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    causal: bool = True
    pad_mode: str = "constant"
    compress: int = 2
    lstm_layers: int = 2
    disable_norm_outer_blocks: int = 0
    trim_right_ratio: float = 1.0

    @property
    def frame_rate(self) -> float:
        """Number of frames per second of audio."""
        hop_length = 1
        for r in self.ratios:
            hop_length *= r
        return self.sample_rate / hop_length

    @property
    def hop_length(self) -> int:
        """Total downsampling factor."""
        result = 1
        for r in self.ratios:
            result *= r
        return result

    @classmethod
    def encodec_24khz(cls) -> "EnCodecConfig":
        """EnCodec 24kHz mono configuration (default for MusicGen)."""
        return cls(
            sample_rate=24000,
            channels=1,
            num_codebooks=8,
            codebook_size=1024,
            codebook_dim=128,
            hidden_size=128,
            num_filters=32,
            ratios=(8, 5, 4, 2),
            causal=True,
            lstm_layers=2,
        )

    @classmethod
    def encodec_32khz(cls) -> "EnCodecConfig":
        """EnCodec 32kHz configuration for MusicGen."""
        return cls(
            sample_rate=32000,
            channels=1,
            num_codebooks=4,
            codebook_size=2048,
            codebook_dim=128,
            hidden_size=128,
            num_filters=64,
            ratios=(8, 5, 4, 4),
            causal=True,
            lstm_layers=2,
        )

    @classmethod
    def encodec_48khz_stereo(cls) -> "EnCodecConfig":
        """EnCodec 48kHz stereo configuration."""
        return cls(
            sample_rate=48000,
            channels=2,
            num_codebooks=8,
            codebook_size=1024,
            codebook_dim=128,
            hidden_size=128,
            num_filters=32,
            ratios=(8, 5, 4, 2),
            causal=True,
            lstm_layers=2,
        )

    @classmethod
    def from_name(cls, name: str) -> "EnCodecConfig":
        """Create config from model name.

        Args:
            name: Model name (e.g., "encodec_24khz", "encodec_32khz")

        Returns:
            EnCodecConfig for the specified model

        Raises:
            ValueError: If model name is not recognized
        """
        name = name.lower().replace("-", "_")

        configs = {
            "encodec_24khz": cls.encodec_24khz,
            "encodec_32khz": cls.encodec_32khz,
            "encodec_48khz_stereo": cls.encodec_48khz_stereo,
            "24khz": cls.encodec_24khz,
            "32khz": cls.encodec_32khz,
            "48khz": cls.encodec_48khz_stereo,
        }

        if name not in configs:
            available = ", ".join(sorted(configs.keys()))
            raise ValueError(
                f"Unknown EnCodec model: {name!r}. Available: {available}"
            )

        return configs[name]()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnCodecConfig":
        """Create config from dictionary."""
        # Handle ratios as list -> tuple
        if "ratios" in d and isinstance(d["ratios"], list):
            d = d.copy()
            d["ratios"] = tuple(d["ratios"])

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict

        d = asdict(self)
        # Convert tuple to list for JSON serialization
        d["ratios"] = list(d["ratios"])
        return d
