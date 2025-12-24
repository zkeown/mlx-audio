"""Transformer decoder for Parler-TTS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.tts.config import ParlerTTSConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional KV caching.

    Supports both self-attention and cross-attention modes.
    Uses Grouped Query Attention (GQA) when num_key_value_heads < num_attention_heads.

    Note: Position embeddings are applied externally (sinusoidal embeddings added
    to inputs before the decoder), matching the PyTorch Parler-TTS architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialize multi-head attention.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads (for GQA). If None, uses num_heads.
            dropout: Attention dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        # Number of query heads per KV head (for GQA)
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
        position_offset: int = 0,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass.

        Args:
            hidden_states: Query input [B, T, D]
            key_value_states: Key/value source for cross-attention [B, S, D]
            attention_mask: Attention mask [T, S] or [B, T, S]
            kv_cache: Cached (key, value) from previous steps
            position_offset: Position offset for RoPE (from cached positions)

        Returns:
            Tuple of:
                - Output tensor [B, T, D]
                - Updated KV cache (self-attention only)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Compute query
        query = self.q_proj(hidden_states)
        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        query = query.transpose(0, 2, 1, 3)  # [B, num_heads, T, head_dim]

        # Determine if this is cross-attention
        is_cross_attention = key_value_states is not None

        if is_cross_attention:
            # Cross-attention: use external key/value states
            key = self.k_proj(key_value_states)
            value = self.v_proj(key_value_states)
            kv_seq_length = key_value_states.shape[1]

            key = key.reshape(batch_size, kv_seq_length, self.num_kv_heads, self.head_dim)
            key = key.transpose(0, 2, 1, 3)

            value = value.reshape(batch_size, kv_seq_length, self.num_kv_heads, self.head_dim)
            value = value.transpose(0, 2, 1, 3)

            # No RoPE for cross-attention keys
            new_kv_cache = None
        else:
            # Self-attention
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

            key = key.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
            key = key.transpose(0, 2, 1, 3)

            value = value.reshape(batch_size, seq_length, self.num_kv_heads, self.head_dim)
            value = value.transpose(0, 2, 1, 3)

            # Handle KV cache (no RoPE - sinusoidal embeddings applied externally)
            if kv_cache is not None:
                key_cache, value_cache = kv_cache
                key = mx.concatenate([key_cache, key], axis=2)
                value = mx.concatenate([value_cache, value], axis=2)

            new_kv_cache = (key, value)

        kv_seq_length = key.shape[2]

        # Expand KV heads for GQA
        if self.num_heads_per_kv > 1:
            # Repeat KV heads to match query heads
            key = mx.repeat(key, self.num_heads_per_kv, axis=1)
            value = mx.repeat(value, self.num_heads_per_kv, axis=1)

        # Use Flash Attention if available (O(T) memory vs O(T²))
        if hasattr(mx.fast, "scaled_dot_product_attention"):
            attn_output = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.scale, mask=attention_mask
            )
        else:
            # Fallback: standard O(T²) attention
            attn_weights = (query @ key.transpose(0, 1, 3, 2)) * self.scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = mx.softmax(attn_weights, axis=-1)
            attn_output = attn_weights @ value

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, new_kv_cache


class ParlerTTSDecoderBlock(nn.Module):
    """Single transformer decoder block for Parler-TTS.

    Structure:
        - LayerNorm -> Self-Attention -> Residual
        - LayerNorm -> Cross-Attention -> Residual
        - LayerNorm -> FFN (GELU) -> Residual

    Note: Position embeddings are applied externally (sinusoidal embeddings
    added to inputs before the decoder), matching the PyTorch Parler-TTS.
    """

    def __init__(self, config: "ParlerTTSConfig", layer_idx: int = 0):
        """Initialize decoder block.

        Args:
            config: Parler-TTS configuration
            layer_idx: Layer index (for debugging)
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Self-attention
        self.self_attn = MultiHeadAttention(
            config.hidden_size,
            config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

        # Cross-attention to text prompt
        self.encoder_attn = MultiHeadAttention(
            config.hidden_size,
            config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

        # FFN (2-layer GELU, matching PyTorch Parler-TTS)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        encoder_attention_mask: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
        position_offset: int = 0,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, T, D]
            encoder_hidden_states: Conditioning [B, S, D]
            attention_mask: Causal self-attention mask
            encoder_attention_mask: Cross-attention mask
            kv_cache: Cached (key, value) for self-attention
            position_offset: Position offset for cached positions

        Returns:
            Tuple of:
                - Output tensor [B, T, D]
                - Updated KV cache
        """
        residual = hidden_states

        # Self-attention
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, new_kv_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_offset=position_offset,
        )
        hidden_states = residual + hidden_states

        # Cross-attention (if encoder states provided)
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, _ = self.encoder_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = residual + hidden_states

        # FFN (GELU)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


class ParlerTTSDecoder(nn.Module):
    """Full Parler-TTS transformer decoder.

    Generates audio tokens conditioned on text and description embeddings.
    """

    def __init__(self, config: "ParlerTTSConfig"):
        """Initialize decoder.

        Args:
            config: Parler-TTS configuration
        """
        super().__init__()
        self.config = config

        # Transformer blocks
        self.layers = [
            ParlerTTSDecoderBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]

        # Final layer norm
        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        encoder_attention_mask: mx.array | None = None,
        kv_cache: list[tuple[mx.array, mx.array]] | None = None,
        position_offset: int = 0,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Forward pass.

        Args:
            hidden_states: Input embeddings [B, T, D]
            encoder_hidden_states: Conditioning [B, S, D]
            attention_mask: Causal self-attention mask
            encoder_attention_mask: Cross-attention mask
            kv_cache: List of (key, value) tuples for each layer
            position_offset: Position offset for cached positions

        Returns:
            Tuple of:
                - Output hidden states [B, T, D]
                - Updated KV cache for all layers
        """
        new_kv_cache = []

        for i, layer in enumerate(self.layers):
            layer_cache = None
            if kv_cache is not None and i < len(kv_cache):
                layer_cache = kv_cache[i]

            hidden_states, layer_kv_cache = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                kv_cache=layer_cache,
                position_offset=position_offset,
            )
            new_kv_cache.append(layer_kv_cache)

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, new_kv_cache

    def create_causal_mask(
        self,
        seq_length: int,
        offset: int = 0,
    ) -> mx.array:
        """Create causal attention mask.

        Args:
            seq_length: Query sequence length
            offset: Number of cached positions

        Returns:
            Causal mask [seq_length, seq_length + offset]
        """
        total_length = seq_length + offset

        # Create mask where position i can attend to positions 0..i+offset
        query_pos = mx.arange(seq_length)[:, None] + offset
        key_pos = mx.arange(total_length)[None, :]

        mask = key_pos <= query_pos
        mask = mx.where(mask, mx.array(0.0), mx.array(float("-inf")))

        return mask
