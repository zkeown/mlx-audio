"""Whisper audio encoder."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.whisper.layers.attention import ResidualAttentionBlock

if TYPE_CHECKING:
    from mlx_audio.models.whisper.config import WhisperConfig


def sinusoids(length: int, dim: int, max_timescale: float = 10000.0) -> mx.array:
    """Create sinusoidal positional embeddings.

    Args:
        length: Sequence length
        dim: Embedding dimension
        max_timescale: Maximum timescale for encoding

    Returns:
        Positional embeddings [length, dim]
    """
    half_dim = dim // 2
    log_timescale = math.log(max_timescale) / (half_dim - 1)
    inv_timescales = mx.exp(-log_timescale * mx.arange(half_dim, dtype=mx.float32))

    positions = mx.arange(length, dtype=mx.float32)[:, None]
    scaled_time = positions * inv_timescales[None, :]

    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1)


class AudioEncoder(nn.Module):
    """Whisper audio encoder.

    Processes log-mel spectrograms into audio features using:
    1. Two 1D convolutions for initial processing and downsampling
    2. Sinusoidal positional encoding
    3. Stack of transformer blocks

    The encoder output is used as cross-attention context for the decoder.

    Attributes:
        conv1: First convolution (n_mels -> n_state)
        conv2: Second convolution with stride 2 (n_state -> n_state)
        positional_embedding: Sinusoidal position embeddings
        blocks: Transformer encoder blocks
        ln_post: Final layer normalization
    """

    def __init__(self, config: WhisperConfig):
        """Initialize audio encoder.

        Args:
            config: Whisper configuration
        """
        super().__init__()

        n_mels = config.n_mels
        n_state = config.n_audio_state
        n_head = config.n_audio_head
        n_layer = config.n_audio_layer
        n_ctx = config.n_audio_ctx

        # Initial convolutions
        # Conv1: (n_mels, n_state, kernel=3, padding=1)
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        # Conv2: (n_state, n_state, kernel=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        # Positional embedding (not learned, sinusoidal)
        self.positional_embedding = sinusoids(n_ctx, n_state)

        # Transformer blocks
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=False)
            for _ in range(n_layer)
        ]

        # Final layer norm
        self.ln_post = nn.LayerNorm(n_state)

    def __call__(self, mel: mx.array) -> mx.array:
        """Encode audio features.

        Args:
            mel: Log-mel spectrogram [B, n_mels, T] or [n_mels, T]

        Returns:
            Audio features [B, T//2, n_state]
        """
        # Handle unbatched input
        if mel.ndim == 2:
            mel = mel[None, :, :]

        # MLX Conv1d expects [B, T, C] (channels-last)
        # Input mel is [B, n_mels, T], so transpose to [B, T, n_mels]
        x = mel.transpose(0, 2, 1)

        # First conv + GELU
        x = nn.gelu(self.conv1(x))

        # Second conv (stride 2 for downsampling) + GELU
        x = nn.gelu(self.conv2(x))

        # x is now [B, T//2, n_state] (channels-last after conv)

        # Add positional embedding
        # x: [B, T, n_state], positional_embedding: [n_ctx, n_state]
        # Only use the first T positions
        T = x.shape[1]
        x = x + self.positional_embedding[:T, :]

        # Apply transformer blocks (encoder blocks don't use KV cache)
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_post(x)

        return x
