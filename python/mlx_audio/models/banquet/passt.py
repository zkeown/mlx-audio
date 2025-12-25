"""PaSST (Patchout Spectrogram Transformer) query encoder for Banquet.

PaSST encodes reference audio into a 768-dimensional embedding for query-based
source separation. This implementation ports the hear21passt model to MLX.

Architecture:
    Input: mel spectrogram [batch, 1, 128, 998]
    Patch embedding: Conv2d (1 → 768, kernel=16x16, stride=10x10)
    Position: Separate time (99) and frequency (12) embeddings
    Tokens: CLS + DIST tokens prepended
    Transformer: 12 blocks, 12 heads, mlp_ratio=4
    Output: Average of CLS and DIST tokens [batch, 768]
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import PaSSTConfig


class Attention(nn.Module):
    """Multi-head self-attention.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else lambda x: x
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else lambda x: x

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = mx.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = mx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # Output
        x = attn @ v  # [B, num_heads, N, head_dim]
        x = mx.transpose(x, (0, 2, 1, 3))  # [B, N, num_heads, head_dim]
        x = mx.reshape(x, (B, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (default: 4x input)
        out_features: Output dimension (default: same as input)
        drop: Dropout rate
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else lambda x: x

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture.

    Architecture: x = x + attn(norm1(x)); x = x + mlp(norm2(x))

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: Whether to use bias in QKV
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Patch embedding using Conv2d.

    Converts mel spectrogram to patch embeddings.

    Args:
        in_channels: Input channels (1 for mel spectrogram)
        embed_dim: Embedding dimension
        patch_size: Patch size (height, width)
        stride: Stride for patch extraction
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 768,
        patch_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (10, 10),
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input [batch, channels, height, width]

        Returns:
            Patch embeddings [batch, embed_dim, n_freq_patches, n_time_patches]
        """
        # MLX Conv2d expects [batch, height, width, channels]
        x = mx.transpose(x, (0, 2, 3, 1))
        x = self.proj(x)
        # Back to [batch, channels, height, width]
        x = mx.transpose(x, (0, 3, 1, 2))
        return x


class PaSST(nn.Module):
    """PaSST (Patchout Spectrogram Transformer) encoder.

    Encodes mel spectrogram into a 768-dimensional embedding for query-based
    source separation.

    Input: mel spectrogram [batch, 1, 128, 998]
    Output: embedding [batch, 768]

    Args:
        config: PaSST configuration
    """

    def __init__(self, config: PaSSTConfig | None = None):
        super().__init__()
        config = config or PaSSTConfig()
        self.config = config

        embed_dim = config.embed_dim
        num_heads = config.num_heads
        num_layers = config.num_layers
        mlp_ratio = config.mlp_ratio
        dropout = config.dropout
        attn_dropout = config.attention_dropout

        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=1,
            embed_dim=embed_dim,
            patch_size=config.patch_size,
            stride=(10, 10),  # Fixed stride for hear21passt compatibility
        )

        # Number of patches: 12 x 99 = 1188 for (128, 998) input
        # with patch_size=(16,16), stride=(10,10)
        n_freq_patches = 12
        n_time_patches = 99

        # Special tokens
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.dist_token = mx.zeros((1, 1, embed_dim))

        # Separate time and frequency position embeddings
        # time_pos_embed: [1, embed_dim, 1, n_time_patches]
        # freq_pos_embed: [1, embed_dim, n_freq_patches, 1]
        self.time_pos_embed = mx.zeros((1, embed_dim, 1, n_time_patches))
        self.freq_pos_embed = mx.zeros((1, embed_dim, n_freq_patches, 1))

        # Position embedding for CLS and DIST tokens
        self.pos_embed = mx.zeros((1, 2, embed_dim))

        self.pos_drop = nn.Dropout(dropout) if dropout > 0 else lambda x: x

        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                attn_drop=attn_dropout,
            )
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Mel spectrogram [batch, 1, 128, 998]

        Returns:
            Embedding [batch, 768]
        """
        B = x.shape[0]

        # Patch embedding: [B, 768, 12, 99]
        x = self.patch_embed(x)

        # Add position embeddings (broadcast over spatial dimensions)
        x = x + self.time_pos_embed + self.freq_pos_embed

        # Flatten patches: [B, 768, 12, 99] -> [B, 1188, 768]
        x = mx.reshape(x, (B, self.config.embed_dim, -1))
        x = mx.transpose(x, (0, 2, 1))

        # Expand special tokens for batch
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.config.embed_dim))
        dist_tokens = mx.broadcast_to(self.dist_token, (B, 1, self.config.embed_dim))

        # Concatenate: [B, 1190, 768]
        x = mx.concatenate([cls_tokens, dist_tokens, x], axis=1)

        # Add position embedding for special tokens
        pos_embed = mx.broadcast_to(self.pos_embed, (B, 2, self.config.embed_dim))
        # Update first two positions
        x_special = x[:, :2, :] + pos_embed
        x_patches = x[:, 2:, :]
        x = mx.concatenate([x_special, x_patches], axis=1)

        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Average CLS and DIST tokens for final embedding
        embedding = (x[:, 0] + x[:, 1]) / 2

        return embedding

    @staticmethod
    def from_config(config: PaSSTConfig) -> PaSST:
        """Create PaSST from configuration.

        Args:
            config: PaSST configuration

        Returns:
            PaSST model
        """
        return PaSST(config)
