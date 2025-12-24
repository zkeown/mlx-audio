"""Transformer layers for temporal modeling.

Captures long-range temporal dependencies in drum performances:
- Crash cymbal decay patterns (distinguishing from ride)
- Ghost note patterns around main snare hits
- Hi-hat patterns and their rhythmic context
- Fill patterns spanning multiple beats

Ported from PyTorch implementation.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Precompute positional encodings at init time for compile compatibility
        self._pe = self._compute_pe(max_len)

    def _compute_pe(self, length: int) -> mx.array:
        """Compute positional encodings."""
        position = mx.arange(0, length)[:, None]
        div_term = mx.exp(
            mx.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim)
        )

        pe_sin = mx.sin(position * div_term)
        pe_cos = mx.cos(position * div_term)

        # Stack and interleave sin/cos: [sin0, cos0, sin1, cos1, ...]
        # Shape: (length, embed_dim/2) for each
        # Interleave to (length, embed_dim)
        pe = mx.concatenate([pe_sin[:, :, None], pe_cos[:, :, None]], axis=2)
        pe = pe.reshape(length, self.embed_dim)

        return pe

    def __call__(self, x: mx.array) -> mx.array:
        """Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.shape[1]
        pe = self._pe[:seq_len]
        x = x + pe[None, :, :]
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
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, N, 3 * embed_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale  # (B, heads, N, N)

        if mask is not None:
            # Expand mask for heads dimension
            if mask.ndim == 2:
                mask = mask[None, None, :, :]  # (1, 1, N, N)
            attn = mx.where(mask == 0, -1e9, attn)

        attn = mx.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = attn @ v  # (B, heads, N, head_dim)
        x = mx.transpose(x, (0, 2, 1, 3))  # (B, N, heads, head_dim)
        x = x.reshape(B, N, C)

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
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(
            embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # Pre-norm architecture (more stable training)
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal sequence modeling.

    Takes CNN-encoded features and models temporal dependencies.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        """Initialize transformer.

        Args:
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension multiplier
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()

        self.embed_dim = embed_dim

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len, dropout)

        # Transformer blocks
        self.layers = [
            TransformerBlock(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input features (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output features (batch, seq_len, embed_dim)
        """
        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.norm(x)

        return x


class LocalAttentionTransformer(nn.Module):
    """Transformer with local attention windows.

    More efficient for long sequences by limiting attention
    to local windows. Good for real-time applications.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        window_size: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()

        self.window_size = window_size
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len, dropout)

        self.layers = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        x = self.pos_encoding(x)

        # Create local attention mask
        mask = self._create_local_mask(N)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

    def _create_local_mask(self, seq_len: int) -> mx.array:
        """Create attention mask for local windows."""
        # Create a mask where 1 means attend, 0 means don't attend
        # Use vectorized operations for compile compatibility
        i_idx = mx.arange(seq_len)[:, None]  # (seq_len, 1)
        j_idx = mx.arange(seq_len)[None, :]  # (1, seq_len)
        # Distance between positions
        dist = mx.abs(i_idx - j_idx)
        # Mask is 1 where distance <= window_size/2
        half_window = self.window_size // 2
        mask = (dist <= half_window).astype(mx.float32)
        return mask
