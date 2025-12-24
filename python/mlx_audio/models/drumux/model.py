"""MLX port of the Drumux drum transcription model.

Architecture:
    Input: Mel-spectrogram (batch, time, n_mels, 1) - NHWC format
    -> CNN Encoder: Extract local spectral features  
    -> Transformer: Model temporal dependencies
    -> Dual Head: Predict onsets and velocities
    Output: onset_logits (batch, time, num_classes)
            velocity (batch, time, num_classes)
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

NUM_CLASSES = 14


@dataclass
class DrumTranscriberConfig:
    """Configuration for DrumTranscriber model."""

    # Input
    n_mels: int = 128
    num_classes: int = NUM_CLASSES

    # Encoder
    encoder_type: str = "standard"  # "standard" or "lightweight"
    embed_dim: int = 512

    # Transformer
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    use_local_attention: bool = False
    window_size: int = 64
    max_seq_len: int = 2048

    # Heads
    head_hidden_dim: int | None = None
    share_head_layers: bool = False

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    encoder_dropout: float = 0.1


# =============================================================================
# Encoder Components
# =============================================================================


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and SiLU activation.
    
    MLX uses NHWC format: (batch, height, width, channels)
    For spectrograms: (batch, time, freq, channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        use_bn: bool = True,
    ):
        super().__init__()
        
        # Normalize to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        
        # Compute "same" padding if not specified
        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif isinstance(padding, int):
            padding = (padding, padding)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,
        )
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return nn.silu(x)


class SpectrogramEncoder(nn.Module):
    """CNN encoder for mel-spectrogram feature extraction.

    Simplified architecture - no depthwise convs, just efficient channel scaling.
    Designed to hit ~8M params to match PyTorch version's total of ~21M.

    Input: (batch, time, freq, 1) mel-spectrogram in NHWC
    Output: (batch, time, embed_dim) feature sequence
    """

    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 512,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.embed_dim = embed_dim

        # Use smaller channel counts to stay within param budget
        # (B, T, 128, 1) -> (B, T, 128, 32)
        self.conv1 = ConvBlock(1, 32, kernel_size=(3, 7), stride=1, padding=(1, 3))
        
        # (B, T, 128, 32) -> (B, T, 64, 64)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=(1, 2))
        self.conv2b = ConvBlock(64, 64, kernel_size=3, stride=1)
        
        # (B, T, 64, 64) -> (B, T, 32, 128)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=(1, 2))
        self.conv3b = ConvBlock(128, 128, kernel_size=3, stride=1)
        
        # (B, T, 32, 128) -> (B, T, 16, 256)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=(1, 2))
        self.conv4b = ConvBlock(256, 256, kernel_size=3, stride=1)
        
        # (B, T, 16, 256) -> (B, T, 8, 384)
        self.conv5 = ConvBlock(256, 384, kernel_size=3, stride=(1, 2))
        self.conv5b = ConvBlock(384, 384, kernel_size=3, stride=1)
        
        self.dropout = nn.Dropout(drop_rate)
        
        # Project to embed_dim
        self.proj = nn.Linear(384, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input spectrogram (batch, time, freq, 1) NHWC

        Returns:
            Feature sequence (batch, time, embed_dim)
        """
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.conv2b(x)
        
        x = self.conv3(x)
        x = self.conv3b(x)
        
        x = self.conv4(x)
        x = self.conv4b(x)
        
        x = self.conv5(x)
        x = self.conv5b(x)
        
        x = self.dropout(x)

        # Average pool over frequency dimension: (B, T, F', C) -> (B, T, C)
        x = mx.mean(x, axis=2)

        # Project to embed_dim
        x = self.proj(x)

        return x


class LightweightEncoder(nn.Module):
    """Lighter CNN encoder for faster training/inference."""

    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 256,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        self.conv1 = ConvBlock(1, 32, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.conv2 = ConvBlock(32, 64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=(1, 2))
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=(1, 2))
        self.dropout = nn.Dropout(drop_rate)
        self.proj = nn.Linear(256, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        
        # Pool over frequency
        x = mx.mean(x, axis=2)  # (B, T, C)
        x = self.proj(x)
        return x


# =============================================================================
# Transformer Components
# =============================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.max_len = max_len
        self._pe = None

    def _get_pe(self, seq_len: int) -> mx.array:
        """Compute positional encoding lazily."""
        if self._pe is None or self._pe.shape[0] < seq_len:
            position = mx.arange(self.max_len)[:, None]
            div_term = mx.exp(
                mx.arange(0, self.embed_dim, 2) * (-mx.log(mx.array(10000.0)) / self.embed_dim)
            )
            pe = mx.zeros((self.max_len, self.embed_dim))
            pe = pe.at[:, 0::2].add(mx.sin(position * div_term))
            pe = pe.at[:, 1::2].add(mx.cos(position * div_term))
            self._pe = pe
        return self._pe[:seq_len]

    def __call__(self, x: mx.array) -> mx.array:
        seq_len = x.shape[1]
        pe = self._get_pe(seq_len)
        x = x + pe
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = mx.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale

        if mask is not None:
            attn = mx.where(mask == 0, mx.array(float("-inf")), attn)

        attn = mx.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        # Apply attention
        x = attn @ v  # (B, heads, N, head_dim)
        x = mx.transpose(x, (0, 2, 1, 3))  # (B, N, heads, head_dim)
        x = mx.reshape(x, (B, N, C))
        x = self.proj(x)

        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, hidden_dim=int(embed_dim * mlp_ratio), dropout=dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal sequence modeling."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()

        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len, dropout)
        self.layers = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =============================================================================
# Prediction Heads
# =============================================================================


class DualHead(nn.Module):
    """Combined onset and velocity prediction head."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 14,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = hidden_dim or embed_dim // 2

        # Onset head
        self.onset_fc1 = nn.Linear(embed_dim, hidden_dim)
        self.onset_fc2 = nn.Linear(hidden_dim, num_classes)
        self.onset_dropout = nn.Dropout(dropout)

        # Velocity head
        self.vel_fc1 = nn.Linear(embed_dim, hidden_dim)
        self.vel_fc2 = nn.Linear(hidden_dim, num_classes)
        self.vel_dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # Onset path
        onset = nn.gelu(self.onset_fc1(x))
        onset = self.onset_dropout(onset)
        onset_logits = self.onset_fc2(onset)

        # Velocity path
        vel = nn.gelu(self.vel_fc1(x))
        vel = self.vel_dropout(vel)
        velocity = mx.sigmoid(self.vel_fc2(vel))

        return onset_logits, velocity


# =============================================================================
# Main Model
# =============================================================================


class DrumTranscriber(nn.Module):
    """Drum transcription model - MLX version."""

    def __init__(self, config: DrumTranscriberConfig | None = None):
        super().__init__()

        self.config = config or DrumTranscriberConfig()

        # Build encoder
        if self.config.encoder_type == "lightweight":
            self.encoder = LightweightEncoder(
                n_mels=self.config.n_mels,
                embed_dim=self.config.embed_dim,
                drop_rate=self.config.encoder_dropout,
            )
        else:
            self.encoder = SpectrogramEncoder(
                n_mels=self.config.n_mels,
                embed_dim=self.config.embed_dim,
                drop_rate=self.config.encoder_dropout,
            )

        # Build transformer
        self.transformer = TemporalTransformer(
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            dropout=self.config.dropout,
            max_len=self.config.max_seq_len,
        )

        # Build prediction head
        self.head = DualHead(
            embed_dim=self.config.embed_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.head_hidden_dim,
            dropout=self.config.dropout,
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Mel-spectrogram (batch, time, freq, 1) NHWC format

        Returns:
            Tuple of:
                - onset_logits (batch, time, num_classes)
                - velocity (batch, time, num_classes) in [0, 1]
        """
        features = self.encoder(x)
        features = self.transformer(features)
        onset_logits, velocity = self.head(features)
        return onset_logits, velocity

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))


def create_model(preset: str = "standard", **kwargs) -> DrumTranscriber:
    """Create a model with preset configuration."""
    presets = {
        "standard": DrumTranscriberConfig(
            encoder_type="standard",
            embed_dim=512,
            num_layers=4,
            num_heads=8,
        ),
        "lightweight": DrumTranscriberConfig(
            encoder_type="lightweight",
            embed_dim=256,
            num_layers=3,
            num_heads=4,
        ),
        "fast": DrumTranscriberConfig(
            encoder_type="lightweight",
            embed_dim=256,
            num_layers=2,
            num_heads=4,
            use_local_attention=True,
            window_size=32,
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    config = presets[preset]
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return DrumTranscriber(config)
