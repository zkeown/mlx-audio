"""Embedding layers for Parler-TTS."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.tts.config import ParlerTTSConfig


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary position embeddings to query and key tensors.
    This allows the model to learn relative positions through
    the attention mechanism.
    """

    def __init__(
        self,
        head_dim: int,
        max_length: int = 8192,
        base: float = 10000.0,
    ):
        """Initialize rotary embeddings.

        Args:
            head_dim: Dimension of each attention head
            max_length: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        self.head_dim = head_dim
        self.max_length = max_length
        self.base = base

        # Compute frequencies
        inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        self._inv_freq = inv_freq

    def _compute_cos_sin(self, seq_length: int, offset: int = 0) -> tuple[mx.array, mx.array]:
        """Compute cos and sin for rotary embedding.

        Args:
            seq_length: Sequence length
            offset: Position offset for cached positions

        Returns:
            Tuple of (cos, sin) each [seq_length, head_dim]
        """
        positions = mx.arange(offset, offset + seq_length, dtype=mx.float32)
        # [seq_length, head_dim // 2]
        freqs = positions[:, None] * self._inv_freq[None, :]
        # [seq_length, head_dim]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """Apply rotary embeddings to query and key.

        Args:
            query: Query tensor [B, num_heads, T, head_dim]
            key: Key tensor [B, num_heads, S, head_dim]
            offset: Position offset for cached positions

        Returns:
            Tuple of rotated (query, key)
        """
        seq_length = query.shape[2]
        cos, sin = self._compute_cos_sin(seq_length, offset)

        # Reshape for broadcasting: [1, 1, T, head_dim]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # Apply rotation to query
        query_rotated = self._rotate(query, cos, sin)

        # Apply rotation to key (may have different length for cross-attention)
        key_seq_length = key.shape[2]
        if key_seq_length != seq_length:
            cos_k, sin_k = self._compute_cos_sin(key_seq_length, 0)
            cos_k = cos_k[None, None, :, :]
            sin_k = sin_k[None, None, :, :]
            key_rotated = self._rotate(key, cos_k, sin_k)
        else:
            key_rotated = self._rotate(key, cos, sin)

        return query_rotated, key_rotated

    def _rotate(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        """Apply rotation to tensor.

        Args:
            x: Input tensor [B, num_heads, T, head_dim]
            cos: Cosine values [1, 1, T, head_dim]
            sin: Sine values [1, 1, T, head_dim]

        Returns:
            Rotated tensor
        """
        # Split into halves
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]

        # Rotate: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        cos1 = cos[..., : self.head_dim // 2]
        cos2 = cos[..., self.head_dim // 2 :]
        sin1 = sin[..., : self.head_dim // 2]
        sin2 = sin[..., self.head_dim // 2 :]

        rotated = mx.concatenate(
            [x1 * cos1 - x2 * sin1, x2 * cos2 + x1 * sin2],
            axis=-1,
        )
        return rotated


class CodebookEmbeddings(nn.Module):
    """Embeddings for multiple audio codebooks.

    Each codebook has its own embedding table. The embeddings from
    all codebooks are summed together to form the input representation.
    """

    def __init__(self, config: "ParlerTTSConfig"):
        """Initialize codebook embeddings.

        Args:
            config: Parler-TTS configuration
        """
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.hidden_size = config.hidden_size

        # Separate embedding table for each codebook
        # +2 for special tokens (pad, bos)
        vocab_size = config.codebook_size + 2
        self.embeddings = [
            nn.Embedding(vocab_size, config.hidden_size)
            for _ in range(config.num_codebooks)
        ]

        # Input projection for first codebook (special handling)
        self.input_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        input_ids: mx.array,
    ) -> mx.array:
        """Compute embeddings for codebook tokens.

        Args:
            input_ids: Token IDs [B, K, T] where K is num_codebooks

        Returns:
            Token embeddings [B, T, D]
        """
        batch_size, num_codebooks, seq_length = input_ids.shape

        # Sum embeddings from all codebooks
        embeddings = mx.zeros((batch_size, seq_length, self.hidden_size))
        for k in range(min(num_codebooks, self.num_codebooks)):
            codebook_ids = input_ids[:, k, :]  # [B, T]
            codebook_emb = self.embeddings[k](codebook_ids)
            embeddings = embeddings + codebook_emb

        return embeddings

    def get_codebook_embedding(
        self,
        codebook_idx: int,
        token_ids: mx.array,
    ) -> mx.array:
        """Get embeddings for a specific codebook.

        Args:
            codebook_idx: Index of the codebook (0 to num_codebooks-1)
            token_ids: Token IDs [B, T] or [T]

        Returns:
            Embeddings [B, T, D] or [T, D]
        """
        return self.embeddings[codebook_idx](token_ids)
