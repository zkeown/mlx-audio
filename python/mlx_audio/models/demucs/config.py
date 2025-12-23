"""Configuration for HTDemucs model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HTDemucsConfig:
    """HTDemucs model configuration.

    Attributes:
        sources: List of source names to separate
        audio_channels: Number of audio channels (1=mono, 2=stereo)
        samplerate: Expected sample rate
        segment: Segment duration in seconds for chunked processing

        channels: Initial hidden channels
        growth: Channel growth factor per layer
        depth: Number of encoder/decoder layers

        nfft: FFT size for STFT
        hop_length: Hop length for STFT
        freq_emb: Frequency embedding scale

        t_depth: Number of transformer layers
        t_heads: Number of attention heads
        t_dropout: Transformer dropout rate
        t_hidden_scale: FFN hidden dimension multiplier
        t_pos_embedding: Positional embedding type ("sin", "scaled", "cape")

        dconv_depth: Depth of dilated conv residual blocks
        dconv_lstm: Number of LSTM layers in DConv (0 = none)
        dconv_attn: Number of attention heads in DConv (0 = none)

        cac: Complex-as-channels mode
    """

    sources: list[str] = field(
        default_factory=lambda: ["drums", "bass", "other", "vocals"]
    )
    audio_channels: int = 2
    samplerate: int = 44100
    segment: float = 6.0

    # Encoder/decoder
    channels: int = 48
    growth: float = 2.0
    depth: int = 4
    kernel_size: int = 8
    stride: int = 4

    # Spectrogram
    nfft: int = 4096
    hop_length: int = 1024
    freq_emb: float = 0.2

    # Transformer
    t_depth: int = 5
    t_heads: int = 8
    t_dropout: float = 0.0
    t_hidden_scale: float = 4.0
    t_pos_embedding: str = "sin"
    bottom_channels: int = 512  # Transformer dimension (0 = use encoder output)

    # DConv
    dconv_depth: int = 2
    dconv_comp: int = 8  # Compression factor for hidden channels
    dconv_lstm: int = 0
    dconv_attn: int = 0

    # Output mode
    cac: bool = True

    @property
    def num_sources(self) -> int:
        """Number of sources to separate."""
        return len(self.sources)

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins in STFT."""
        return self.nfft // 2 + 1

    @classmethod
    def htdemucs_ft(cls) -> HTDemucsConfig:
        """Fine-tuned HTDemucs configuration (default pretrained)."""
        return cls(
            sources=["drums", "bass", "other", "vocals"],
            channels=48,
            depth=4,
            t_depth=5,
            t_heads=8,
        )

    @classmethod
    def htdemucs_6s(cls) -> HTDemucsConfig:
        """6-source HTDemucs configuration."""
        return cls(
            sources=["drums", "bass", "other", "vocals", "guitar", "piano"],
            channels=48,
            depth=4,
            t_depth=5,
            t_heads=8,
        )

    @classmethod
    def from_dict(cls, d: dict) -> HTDemucsConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
