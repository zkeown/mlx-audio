"""CLAP model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CLAPAudioConfig:
    """Configuration for CLAP audio encoder (HTSAT).

    Attributes:
        sample_rate: Audio sample rate in Hz
        n_mels: Number of mel filterbank bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        patch_size: Patch size for patch embedding
        embed_dim: Initial embedding dimension
        depths: Number of blocks in each stage
        num_heads: Number of attention heads in each stage
        window_size: Window size for local attention
        mlp_ratio: MLP expansion ratio
        qkv_bias: Whether to use bias in QKV projection
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        hidden_size: Output hidden size before projection
    """

    sample_rate: int = 48000
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 480  # 10ms at 48kHz
    window_length: int = 1024

    # HTSAT architecture
    patch_size: int = 4
    patch_stride: tuple[int, int] = (4, 4)
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (4, 8, 16, 32)
    window_size: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    # Output
    hidden_size: int = 768  # 1024 for base

    # Fusion for variable-length audio
    enable_fusion: bool = True
    fusion_type: str = "aff_2d"

    # Spec augmentation
    spec_size: int = 256
    freq_ratio: float = 0.5


@dataclass
class CLAPTextConfig:
    """Configuration for CLAP text encoder (RoBERTa).

    Attributes:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        hidden_act: Activation function
        hidden_dropout_prob: Dropout probability
        attention_probs_dropout_prob: Attention dropout probability
        max_position_embeddings: Maximum sequence length
        type_vocab_size: Number of token type IDs
        layer_norm_eps: Layer normalization epsilon
        pad_token_id: Padding token ID
    """

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1


@dataclass
class CLAPConfig:
    """Full CLAP model configuration.

    Attributes:
        audio: Audio encoder configuration
        text: Text encoder configuration
        projection_dim: Shared projection dimension
        logit_scale_init: Initial value for logit scale
    """

    audio: CLAPAudioConfig = field(default_factory=CLAPAudioConfig)
    text: CLAPTextConfig = field(default_factory=CLAPTextConfig)

    # Shared projection space
    projection_dim: int = 512
    logit_scale_init: float = 2.6592  # log(14.2857)

    @classmethod
    def htsat_tiny(cls) -> "CLAPConfig":
        """HTSAT-tiny configuration (laion/clap-htsat-fused)."""
        return cls(
            audio=CLAPAudioConfig(
                embed_dim=96,
                hidden_size=768,
            ),
        )

    @classmethod
    def htsat_base(cls) -> "CLAPConfig":
        """HTSAT-base configuration (larger_clap_*)."""
        return cls(
            audio=CLAPAudioConfig(
                embed_dim=128,
                hidden_size=1024,
            ),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CLAPConfig":
        """Create config from dictionary.

        Handles nested audio/text configs.
        """
        audio_dict = d.pop("audio", {})
        text_dict = d.pop("text", {})

        # Filter to valid fields
        audio_fields = {f.name for f in CLAPAudioConfig.__dataclass_fields__.values()}
        text_fields = {f.name for f in CLAPTextConfig.__dataclass_fields__.values()}
        config_fields = {f.name for f in cls.__dataclass_fields__.values()}

        # Handle patch_stride as list or int
        if "patch_stride" in audio_dict:
            ps = audio_dict["patch_stride"]
            if isinstance(ps, list):
                audio_dict["patch_stride"] = tuple(ps)
            elif isinstance(ps, int):
                audio_dict["patch_stride"] = (ps, ps)

        # Handle depths and num_heads as lists
        if "depths" in audio_dict and isinstance(audio_dict["depths"], list):
            audio_dict["depths"] = tuple(audio_dict["depths"])
        if "num_heads" in audio_dict and isinstance(audio_dict["num_heads"], list):
            audio_dict["num_heads"] = tuple(audio_dict["num_heads"])

        audio_config = CLAPAudioConfig(
            **{k: v for k, v in audio_dict.items() if k in audio_fields}
        )
        text_config = CLAPTextConfig(
            **{k: v for k, v in text_dict.items() if k in text_fields}
        )

        return cls(
            audio=audio_config,
            text=text_config,
            **{k: v for k, v in d.items() if k in config_fields},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
