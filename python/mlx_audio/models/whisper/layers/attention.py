"""Multi-head attention with KV caching for Whisper."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    pass


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional KV caching.

    Supports both self-attention and cross-attention modes.
    KV caching is used for efficient autoregressive decoding.

    Attributes:
        n_state: Hidden dimension
        n_head: Number of attention heads
        scale: Attention scaling factor
    """

    def __init__(self, n_state: int, n_head: int):
        """Initialize multi-head attention.

        Args:
            n_state: Hidden dimension (must be divisible by n_head)
            n_head: Number of attention heads
        """
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.head_dim = n_state // n_head
        self.scale = self.head_dim ** -0.5

        # Separate projections for Q, K, V (Whisper style)
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(
        self,
        x: mx.array,
        xa: mx.array | None = None,
        mask: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass with optional KV caching.

        Args:
            x: Query input [B, T, D]
            xa: Key/value source for cross-attention [B, S, D].
                If None, performs self-attention.
            mask: Attention mask [T, S] or [B, T, S]
            kv_cache: Cached (key, value) from previous steps for
                incremental decoding. Only used for self-attention.

        Returns:
            Tuple of:
                - Output tensor [B, T, D]
                - Updated KV cache (key, value) for self-attention,
                  or None for cross-attention
        """
        B, T, _ = x.shape

        # Compute query
        q = self.query(x)

        if xa is None:
            # Self-attention
            k = self.key(x)
            v = self.value(x)

            # Handle KV cache for incremental decoding
            if kv_cache is not None:
                k_cache, v_cache = kv_cache
                k = mx.concatenate([k_cache, k], axis=1)
                v = mx.concatenate([v_cache, v], axis=1)

            new_kv_cache = (k, v)
        else:
            # Cross-attention (no caching needed, encoder output is fixed)
            k = self.key(xa)
            v = self.value(xa)
            new_kv_cache = None

        S = k.shape[1]

        # Reshape for multi-head attention
        # [B, T, D] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores
        # [B, n_head, T, head_dim] @ [B, n_head, head_dim, S] -> [B, n_head, T, S]
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn = attn + mask

        # Softmax and apply to values
        attn = mx.softmax(attn, axis=-1)

        # [B, n_head, T, S] @ [B, n_head, S, head_dim] -> [B, n_head, T, head_dim]
        out = attn @ v

        # Reshape back
        # [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, D]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_state)

        # Output projection
        out = self.out(out)

        return out, new_kv_cache


class ResidualAttentionBlock(nn.Module):
    """Residual attention block for Whisper transformer.

    Structure:
        - LayerNorm -> Multi-head Self-Attention -> Residual
        - (Optional) LayerNorm -> Multi-head Cross-Attention -> Residual
        - LayerNorm -> MLP -> Residual

    Attributes:
        attn: Self-attention layer
        cross_attn: Cross-attention layer (decoder only)
        mlp: Feed-forward network
    """

    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
    ):
        """Initialize residual attention block.

        Args:
            n_state: Hidden dimension
            n_head: Number of attention heads
            cross_attention: Whether to include cross-attention (decoder)
        """
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head

        # Self-attention
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        # Cross-attention (decoder only)
        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_attn = MultiHeadAttention(n_state, n_head)
            self.cross_attn_ln = nn.LayerNorm(n_state)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4),
            nn.GELU(),
            nn.Linear(n_state * 4, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(
        self,
        x: mx.array,
        xa: mx.array | None = None,
        mask: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass.

        Args:
            x: Input tensor [B, T, D]
            xa: Encoder output for cross-attention [B, S, D]
            mask: Causal attention mask
            kv_cache: Cached (key, value) for incremental decoding

        Returns:
            Tuple of:
                - Output tensor [B, T, D]
                - Updated KV cache (self-attention only)
        """
        # Self-attention
        attn_out, new_kv_cache = self.attn(
            self.attn_ln(x),
            mask=mask,
            kv_cache=kv_cache,
        )
        x = x + attn_out

        # Cross-attention (if applicable)
        if self.cross_attention and xa is not None:
            cross_out, _ = self.cross_attn(
                self.cross_attn_ln(x),
                xa=xa,
            )
            x = x + cross_out

        # MLP
        x = x + self.mlp(self.mlp_ln(x))

        return x, new_kv_cache
