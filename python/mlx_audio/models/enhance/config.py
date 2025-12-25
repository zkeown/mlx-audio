"""Configuration for DeepFilterNet model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeepFilterNetConfig:
    """DeepFilterNet model configuration.

    Attributes:
        sample_rate: Input sample rate (16000 or 48000)
        frame_size: STFT frame size in samples
        hop_size: Hop size in samples
        fft_size: FFT size (usually 2x frame_size for zero-padding)
        erb_bands: Number of ERB bands
        df_order: Deep filter order (temporal context)
        df_bins: Number of frequency bins for deep filtering
        hidden_size: GRU/Linear hidden dimension
        num_groups: Number of groups for grouped layers
        enc_layers: Number of encoder GRU layers
        dec_layers: Number of decoder layers
        post_filter: Whether to apply post-processing filter
    """

    sample_rate: int = 48000
    frame_size: int = 960
    hop_size: int = 480
    fft_size: int = 1920
    erb_bands: int = 32
    df_order: int = 5
    df_bins: int = 96
    hidden_size: int = 256
    num_groups: int = 1
    enc_layers: int = 3
    dec_layers: int = 2
    post_filter: bool = True

    @property
    def n_freqs(self) -> int:
        """Number of frequency bins."""
        return self.fft_size // 2 + 1

    @classmethod
    def deepfilternet2(cls) -> DeepFilterNetConfig:
        """DeepFilterNet2 configuration (48kHz, best quality)."""
        return cls(
            sample_rate=48000,
            frame_size=960,
            hop_size=480,
            fft_size=1920,
            erb_bands=32,
            df_order=5,
            df_bins=96,
            hidden_size=256,
            enc_layers=3,
        )

    @classmethod
    def deepfilternet2_16k(cls) -> DeepFilterNetConfig:
        """16kHz version for telephony applications."""
        return cls(
            sample_rate=16000,
            frame_size=320,
            hop_size=160,
            fft_size=640,
            erb_bands=24,
            df_order=5,
            df_bins=64,
            hidden_size=192,
            enc_layers=2,
        )

    @classmethod
    def from_dict(cls, d: dict) -> DeepFilterNetConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
