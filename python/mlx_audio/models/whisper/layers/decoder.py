"""Whisper text decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.whisper.layers.attention import ResidualAttentionBlock

if TYPE_CHECKING:
    from mlx_audio.models.whisper.config import WhisperConfig


class TextDecoder(nn.Module):
    """Whisper text decoder.

    Generates text tokens autoregressively using:
    1. Token embeddings + learned positional embeddings
    2. Stack of transformer blocks with causal self-attention
       and cross-attention to encoder output
    3. Final layer norm and projection to vocabulary

    The decoder uses KV caching for efficient incremental decoding.

    Attributes:
        token_embedding: Token embedding lookup table
        positional_embedding: Learned position embeddings
        blocks: Transformer decoder blocks
        ln: Final layer normalization
    """

    def __init__(self, config: "WhisperConfig"):
        """Initialize text decoder.

        Args:
            config: Whisper configuration
        """
        super().__init__()

        n_vocab = config.n_vocab
        n_ctx = config.n_text_ctx
        n_state = config.n_text_state
        n_head = config.n_text_head
        n_layer = config.n_text_layer

        self.n_state = n_state
        self.n_ctx = n_ctx

        # Token embedding
        self.token_embedding = nn.Embedding(n_vocab, n_state)

        # Learned positional embedding
        self.positional_embedding = mx.zeros((n_ctx, n_state))

        # Transformer blocks (with cross-attention)
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ]

        # Final layer norm
        self.ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        tokens: mx.array,
        audio_features: mx.array,
        kv_cache: list[tuple[mx.array, mx.array]] | None = None,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Generate logits for next token prediction.

        Args:
            tokens: Input token IDs [B, T]
            audio_features: Encoder output [B, S, D]
            kv_cache: List of (key, value) tuples for each layer,
                from previous decoding steps. Length = n_layers.

        Returns:
            Tuple of:
                - Logits [B, T, n_vocab]
                - Updated KV cache for all layers
        """
        B, T = tokens.shape

        # Token + positional embeddings
        # For incremental decoding, offset is based on cache length
        if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0] is not None:
            offset = kv_cache[0][0].shape[1]  # Length of cached keys
        else:
            offset = 0

        x = self.token_embedding(tokens)
        x = x + self.positional_embedding[offset : offset + T, :]

        # Create causal mask for self-attention
        # Mask shape: [T, T + offset] where cached positions are visible
        mask = self._create_causal_mask(T, offset)

        # Apply transformer blocks
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None and i < len(kv_cache) else None
            x, new_cache = block(
                x,
                xa=audio_features,
                mask=mask,
                kv_cache=layer_cache,
            )
            new_kv_cache.append(new_cache)

        # Final layer norm
        x = self.ln(x)

        # Project to vocabulary using tied embeddings
        # x: [B, T, n_state], token_embedding.weight: [n_vocab, n_state]
        logits = x @ self.token_embedding.weight.T

        return logits, new_kv_cache

    def _create_causal_mask(self, T: int, offset: int = 0) -> mx.array:
        """Create causal attention mask.

        Args:
            T: Query sequence length
            offset: Number of cached positions

        Returns:
            Causal mask [T, T + offset] where -inf indicates masked positions
        """
        # Total key/value length including cache
        S = T + offset

        # Create mask where True means "attend to this position"
        # For position i, can attend to positions 0..i+offset
        query_pos = mx.arange(T)[:, None] + offset
        key_pos = mx.arange(S)[None, :]

        # Allow attending to positions <= current position
        mask = key_pos <= query_pos

        # Convert to additive mask (0 for attend, -inf for masked)
        mask = mx.where(mask, mx.array(0.0), mx.array(float("-inf")))

        return mask
