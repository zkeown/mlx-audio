"""Configuration for Banquet query-based source separation model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BanquetConfig:
    """Banquet model configuration.

    Banquet uses a reference audio query to extract matching sounds from a mixture.
    This configuration matches the ev-pre-aug checkpoint by default.

    Attributes:
        sample_rate: Expected sample rate in Hz
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
        win_length: Window length for STFT (defaults to n_fft if None)
        in_channel: Number of input audio channels (1=mono, 2=stereo)

        n_bands: Number of frequency bands for band splitting
        band_type: Band type specification ("musical")

        emb_dim: Embedding dimension for band features
        rnn_dim: Hidden dimension for RNN layers
        n_sqm_modules: Number of sequential band modelling modules
        mlp_dim: MLP hidden dimension for mask estimation
        bidirectional: Whether to use bidirectional RNN
        rnn_type: RNN type ("LSTM" or "GRU")
        complex_mask: Whether to use complex-valued masks
        use_freq_weights: Whether to use frequency weights in mask estimation

        cond_emb_dim: Conditioning embedding dimension (PaSST output: 768)
        film_additive: Whether to use additive modulation (beta)
        film_multiplicative: Whether to use multiplicative modulation (gamma)
        film_depth: Depth of FiLM modulation networks
        channels_per_group: Channels per group for GroupNorm

        hidden_activation: Hidden activation function
    """

    # Audio settings
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int | None = None
    in_channel: int = 2

    # Band split settings
    n_bands: int = 64
    band_type: str = "musical"

    # Model architecture
    emb_dim: int = 128
    rnn_dim: int = 256
    n_sqm_modules: int = 12
    mlp_dim: int = 512
    bidirectional: bool = True
    rnn_type: str = "LSTM"
    complex_mask: bool = True
    use_freq_weights: bool = True

    # FiLM settings
    cond_emb_dim: int = 768
    film_additive: bool = True
    film_multiplicative: bool = True
    film_depth: int = 2
    channels_per_group: int = 16

    # Mask estimation
    hidden_activation: str = "Tanh"

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins in STFT."""
        return self.n_fft // 2 + 1

    @property
    def effective_win_length(self) -> int:
        """Effective window length."""
        return self.win_length if self.win_length is not None else self.n_fft

    @classmethod
    def banquet(cls) -> BanquetConfig:
        """Default Banquet configuration (ev-pre-aug checkpoint)."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> BanquetConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "in_channel": self.in_channel,
            "n_bands": self.n_bands,
            "band_type": self.band_type,
            "emb_dim": self.emb_dim,
            "rnn_dim": self.rnn_dim,
            "n_sqm_modules": self.n_sqm_modules,
            "mlp_dim": self.mlp_dim,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
            "complex_mask": self.complex_mask,
            "use_freq_weights": self.use_freq_weights,
            "cond_emb_dim": self.cond_emb_dim,
            "film_additive": self.film_additive,
            "film_multiplicative": self.film_multiplicative,
            "film_depth": self.film_depth,
            "channels_per_group": self.channels_per_group,
            "hidden_activation": self.hidden_activation,
        }


@dataclass
class PaSSTConfig:
    """PaSST (Patchout Spectrogram Transformer) query encoder configuration.

    PaSST encodes reference audio into a 768-dimensional embedding
    for query-based separation.

    Attributes:
        sample_rate: PaSST expected sample rate (32kHz)
        n_mels: Number of mel filterbank bins
        n_fft: FFT size for mel spectrogram
        hop_length: Hop length for mel spectrogram
        win_length: Window length for mel spectrogram
        n_time_frames: Number of expected time frames

        patch_size: Patch size for patch embedding (freq, time)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_ratio: MLP ratio (hidden_dim = embed_dim * mlp_ratio)
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
    """

    # Audio settings
    sample_rate: int = 32000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 320
    win_length: int = 800
    n_time_frames: int = 998

    # Model architecture
    patch_size: Tuple[int, int] = (16, 16)
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0

    @property
    def num_freq_patches(self) -> int:
        """Number of frequency patches."""
        return self.n_mels // self.patch_size[0]

    @property
    def num_time_patches(self) -> int:
        """Number of time patches."""
        return self.n_time_frames // self.patch_size[1]

    @property
    def num_patches(self) -> int:
        """Total number of patches."""
        return self.num_freq_patches * self.num_time_patches

    @classmethod
    def passt(cls) -> PaSSTConfig:
        """Default PaSST configuration (openmic architecture)."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> PaSSTConfig:
        """Create config from dictionary."""
        # Handle patch_size as list
        if "patch_size" in d and isinstance(d["patch_size"], list):
            d = d.copy()
            d["patch_size"] = tuple(d["patch_size"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_time_frames": self.n_time_frames,
            "patch_size": list(self.patch_size),
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
        }


@dataclass
class BandSpec:
    """Band specification for frequency band splitting.

    Attributes:
        start_bin: Start frequency bin (inclusive)
        end_bin: End frequency bin (exclusive)
    """

    start_bin: int
    end_bin: int

    @property
    def bandwidth(self) -> int:
        """Bandwidth in bins."""
        return self.end_bin - self.start_bin
