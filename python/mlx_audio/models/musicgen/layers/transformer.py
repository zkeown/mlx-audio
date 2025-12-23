"""Transformer decoder for MusicGen."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.musicgen.config import MusicGenConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional KV caching.

    Supports both self-attention and cross-attention modes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """Initialize multi-head attention.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            dropout: Attention dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass.

        Args:
            hidden_states: Query input [B, T, D]
            key_value_states: Key/value source for cross-attention [B, S, D]
            attention_mask: Attention mask [T, S] or [B, T, S]
            kv_cache: Cached (key, value) from previous steps

        Returns:
            Tuple of:
                - Output tensor [B, T, D]
                - Updated KV cache (self-attention only)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Compute query
        query = self.q_proj(hidden_states)

        # Determine if this is cross-attention
        is_cross_attention = key_value_states is not None

        if is_cross_attention:
            # Cross-attention: use external key/value states
            key = self.k_proj(key_value_states)
            value = self.v_proj(key_value_states)
            new_kv_cache = None
        else:
            # Self-attention
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

            # Handle KV cache
            if kv_cache is not None:
                key_cache, value_cache = kv_cache
                key = mx.concatenate([key_cache, key], axis=1)
                value = mx.concatenate([value_cache, value], axis=1)

            new_kv_cache = (key, value)

        kv_seq_length = key.shape[1]

        # Reshape for multi-head attention
        # [B, T, D] -> [B, num_heads, T, head_dim]
        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        query = query.transpose(0, 2, 1, 3)

        key = key.reshape(batch_size, kv_seq_length, self.num_heads, self.head_dim)
        key = key.transpose(0, 2, 1, 3)

        value = value.reshape(batch_size, kv_seq_length, self.num_heads, self.head_dim)
        value = value.transpose(0, 2, 1, 3)

        # Compute attention scores
        attn_weights = (query @ key.transpose(0, 1, 3, 2)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply to values
        attn_output = attn_weights @ value

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, new_kv_cache


class MusicGenDecoderBlock(nn.Module):
    """Single transformer decoder block for MusicGen.

    Structure:
        - LayerNorm -> Self-Attention -> Residual
        - LayerNorm -> Cross-Attention -> Residual
        - LayerNorm -> FFN -> Residual
    """

    def __init__(self, config: "MusicGenConfig"):
        """Initialize decoder block.

        Args:
            config: MusicGen configuration
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention
        self.self_attn = MultiHeadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

        # Cross-attention to text conditioning
        self.cross_attn = MultiHeadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

        # FFN
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

        # Activation
        if config.activation_function == "gelu":
            self.activation_fn = nn.gelu
        elif config.activation_function == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.gelu

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        encoder_attention_mask: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, T, D]
            encoder_hidden_states: Text conditioning [B, S, D]
            attention_mask: Causal self-attention mask
            encoder_attention_mask: Cross-attention mask
            kv_cache: Cached (key, value) for self-attention

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
        )
        hidden_states = residual + hidden_states

        # Cross-attention (if encoder states provided)
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)
            hidden_states, _ = self.cross_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv_cache


class MusicGenDecoder(nn.Module):
    """Full MusicGen transformer decoder.

    Generates audio tokens conditioned on text embeddings.
    """

    def __init__(self, config: "MusicGenConfig"):
        """Initialize decoder.

        Args:
            config: MusicGen configuration
        """
        super().__init__()
        self.config = config

        # Transformer blocks
        self.layers = [
            MusicGenDecoderBlock(config)
            for _ in range(config.num_hidden_layers)
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
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Forward pass.

        Args:
            hidden_states: Input embeddings [B, T, D]
            encoder_hidden_states: Text conditioning [B, S, D]
            attention_mask: Causal self-attention mask
            encoder_attention_mask: Cross-attention mask
            kv_cache: List of (key, value) tuples for each layer

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
