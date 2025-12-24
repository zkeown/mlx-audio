"""Embedding layers for MusicGen."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.musicgen.config import MusicGenConfig


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding.

    Creates position embeddings using sine and cosine functions
    at different frequencies.
    """

    def __init__(
        self,
        hidden_size: int,
        max_length: int = 8192,
    ):
        """Initialize positional embedding.

        Args:
            hidden_size: Embedding dimension
            max_length: Maximum sequence length
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        # Pre-compute embeddings
        self._embeddings = self._create_embeddings(max_length, hidden_size)

    def _create_embeddings(self, max_length: int, hidden_size: int) -> mx.array:
        """Create sinusoidal position embeddings.

        Args:
            max_length: Maximum sequence length
            hidden_size: Embedding dimension

        Returns:
            Position embeddings [max_length, hidden_size]
        """
        positions = mx.arange(max_length, dtype=mx.float32)
        dim_indices = mx.arange(hidden_size // 2, dtype=mx.float32)

        # Compute frequencies
        freqs = mx.exp(-math.log(10000.0) * dim_indices / (hidden_size // 2))

        # Compute embeddings
        angles = positions[:, None] * freqs[None, :]
        sin_emb = mx.sin(angles)
        cos_emb = mx.cos(angles)

        # Interleave sin and cos
        embeddings = mx.concatenate([sin_emb, cos_emb], axis=-1)

        return embeddings

    def __call__(self, positions: mx.array) -> mx.array:
        """Get positional embeddings for given positions.

        Args:
            positions: Position indices [B, T] or [T]

        Returns:
            Position embeddings [B, T, D] or [T, D]
        """
        return self._embeddings[positions]


class CodebookEmbeddings(nn.Module):
    """Embeddings for multiple codebooks.

    Each codebook has its own embedding table. The embeddings from
    all codebooks are summed together to form the input representation.
    """

    def __init__(self, config: "MusicGenConfig"):
        """Initialize codebook embeddings.

        Args:
            config: MusicGen configuration
        """
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.hidden_size = config.hidden_size

        # Separate embedding table for each codebook
        # HuggingFace uses codebook_size + 1 for embeddings (includes pad token)
        vocab_size = config.codebook_size + 1
        self.embeddings = [
            nn.Embedding(vocab_size, config.hidden_size)
            for _ in range(config.num_codebooks)
        ]

        # Positional embeddings
        max_length = int(config.max_duration * config.frame_rate) + 100
        self.position_embeddings = SinusoidalPositionalEmbedding(
            config.hidden_size,
            max_length=max_length,
        )

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        """Compute embeddings for codebook tokens.

        Args:
            input_ids: Token IDs [B, K, T] where K is num_codebooks
            position_ids: Optional position IDs [B, T] or None

        Returns:
            Token embeddings [B, T, D]
        """
        batch_size, num_codebooks, seq_length = input_ids.shape

        # Sum embeddings from all codebooks
        embeddings = mx.zeros((batch_size, seq_length, self.hidden_size))
        for k in range(num_codebooks):
            codebook_ids = input_ids[:, k, :]  # [B, T]
            embeddings = embeddings + self.embeddings[k](codebook_ids)

        # Add positional embeddings
        if position_ids is None:
            position_ids = mx.arange(seq_length)
            position_ids = mx.broadcast_to(
                position_ids[None, :],
                (batch_size, seq_length)
            )

        pos_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + pos_embeddings

        return embeddings

    def get_codebook_embedding(self, codebook_idx: int, token_ids: mx.array) -> mx.array:
        """Get embeddings for a specific codebook.

        Args:
            codebook_idx: Index of the codebook (0 to num_codebooks-1)
            token_ids: Token IDs [B, T] or [T]

        Returns:
            Embeddings [B, T, D] or [T, D]
        """
        return self.embeddings[codebook_idx](token_ids)
