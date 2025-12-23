"""Transformer layers for HTDemucs.

Matches PyTorch demucs.transformer exactly.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.demucs.layers.dconv import LayerScale


def create_sin_embedding(
    length: int, dim: int, shift: int = 0, max_period: float = 10000.0
) -> mx.array:
    """Create 1D sinusoidal positional embedding.

    Matches PyTorch demucs create_sin_embedding exactly.
    Returns TBC format: [length, 1, dim].

    Args:
        length: Sequence length
        dim: Embedding dimension
        shift: Position shift (default 0)
        max_period: Maximum period for sinusoidal encoding

    Returns:
        Positional embedding [length, 1, dim]
    """
    pos = (shift + mx.arange(length, dtype=mx.float32))[:, None, None]  # [T, 1, 1]
    half_dim = dim // 2
    adim = mx.arange(half_dim, dtype=mx.float32)[None, None, :]  # [1, 1, D/2]
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    emb = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)  # [T, 1, D]
    return emb


def create_2d_sin_embedding(
    d_model: int, height: int, width: int, max_period: float = 10000.0
) -> mx.array:
    """Create 2D sinusoidal positional embedding.

    Matches PyTorch demucs create_2d_sin_embedding exactly:
    - First half of channels: encodes width with sin/cos interleaved
    - Second half of channels: encodes height with sin/cos interleaved

    Args:
        d_model: Embedding dimension (must be divisible by 4)
        height: Height (freq bins)
        width: Width (time frames)
        max_period: Maximum period

    Returns:
        Positional embedding [1, d_model, height, width]
    """
    half_d = d_model // 2
    quarter_d = half_d // 2

    # div_term: step by 2 to match PyTorch
    div_term = mx.exp(
        mx.arange(0.0, half_d, 2, dtype=mx.float32)
        * -(math.log(max_period) / half_d)
    )  # [D/4]

    pos_w = mx.arange(width, dtype=mx.float32)[:, None]  # [W, 1]
    pos_h = mx.arange(height, dtype=mx.float32)[:, None]  # [H, 1]

    # Width encoding
    w_args = pos_w * div_term[None, :]  # [W, D/4]
    sin_w = mx.sin(w_args)  # [W, D/4]
    cos_w = mx.cos(w_args)  # [W, D/4]

    # Height encoding
    h_args = pos_h * div_term[None, :]  # [H, D/4]
    sin_h = mx.sin(h_args)  # [H, D/4]
    cos_h = mx.cos(h_args)  # [H, D/4]

    # Interleave sin/cos: [W, D/4] -> stack -> [W, D/4, 2] -> reshape [W, D/2]
    width_emb = mx.stack([sin_w, cos_w], axis=-1)  # [W, D/4, 2]
    width_emb = width_emb.reshape(width, half_d)  # [W, D/2]
    width_emb = width_emb.T  # [D/2, W]
    # Broadcast over height: [D/2, W] -> [D/2, H, W]
    width_emb = width_emb[:, None, :]
    width_emb = mx.broadcast_to(width_emb, (half_d, height, width))

    height_emb = mx.stack([sin_h, cos_h], axis=-1)  # [H, D/4, 2]
    height_emb = height_emb.reshape(height, half_d)  # [H, D/2]
    height_emb = height_emb.T  # [D/2, H]
    # Broadcast over width: [D/2, H] -> [D/2, H, W]
    height_emb = height_emb[:, :, None]
    height_emb = mx.broadcast_to(height_emb, (half_d, height, width))

    pe = mx.concatenate([width_emb, height_emb], axis=0)  # [D, H, W]
    return pe[None, :, :, :]  # [1, D, H, W]


class MyGroupNorm(nn.Module):
    """GroupNorm that matches PyTorch's MyGroupNorm (1 group).

    PyTorch MyGroupNorm with groups=1:
    1. Transposes [B, T, C] -> [B, C, T]
    2. Normalizes over ALL of C and T together (not just C)
    3. Transposes back

    This is equivalent to normalizing over axes (1, 2) of [B, T, C].
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C]
        # GroupNorm(1, C) normalizes over all of T and C together
        mean = mx.mean(x, axis=(1, 2), keepdims=True)
        var = mx.var(x, axis=(1, 2), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x * self.weight + self.bias


class MultiheadAttention(nn.Module):
    """Multihead attention matching PyTorch's nn.MultiheadAttention.

    Uses combined in_proj for Q, K, V (in_proj_weight, in_proj_bias).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection (MLX format: [in, out])
        # PyTorch [3*D, D] -> transposed to MLX [D, 3*D]
        self.in_proj_weight = mx.zeros((embed_dim, 3 * embed_dim))
        self.in_proj_bias = mx.zeros((3 * embed_dim,))

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            query: [B, T, D]
            key: [B, S, D]
            value: [B, S, D]

        Returns:
            Output [B, T, D]
        """
        B, T, _ = query.shape
        S = key.shape[1]

        # Project Q, K, V using combined weight
        # in_proj_weight is [D, 3*D], split along axis 1
        w_q = self.in_proj_weight[:, :self.embed_dim]
        w_k = self.in_proj_weight[:, self.embed_dim:2*self.embed_dim]
        w_v = self.in_proj_weight[:, 2*self.embed_dim:]

        b_q = self.in_proj_bias[:self.embed_dim]
        b_k = self.in_proj_bias[self.embed_dim:2*self.embed_dim]
        b_v = self.in_proj_bias[2*self.embed_dim:]

        # query @ w_q: [B, T, D] @ [D, D] -> [B, T, D]
        q = query @ w_q + b_q
        k = key @ w_k + b_k
        v = value @ w_v + b_v

        # Reshape for multi-head attention
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)

        # Output
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)

        return self.out_proj(out)


class MyTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer matching PyTorch's MyTransformerEncoderLayer.

    Structure (self-attention variant):
        self_attn: MultiheadAttention
        linear1: Linear
        dropout: (no params)
        linear2: Linear
        norm1: LayerNorm
        norm2: LayerNorm
        dropout1: (no params)
        dropout2: (no params)
        norm_out: MyGroupNorm
        gamma_1: LayerScale
        gamma_2: LayerScale

    Structure (cross-attention variant, adds):
        cross_attn: MultiheadAttention
        norm3: LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.cross_attention = cross_attention

        if cross_attention:
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout)
        else:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if cross_attention:
            self.norm3 = nn.LayerNorm(d_model)

        self.norm_out = MyGroupNorm(d_model)
        self.gamma_1 = LayerScale(d_model, init=1e-4)
        self.gamma_2 = LayerScale(d_model, init=1e-4)

    def __call__(
        self,
        x: mx.array,
        cross: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input [B, T, D]
            cross: Cross-attention source [B, S, D] (only for cross-attn layers)

        Returns:
            Output [B, T, D]
        """
        if self.cross_attention:
            # Cross-attention
            x_norm = self.norm1(x)
            cross_norm = self.norm2(cross)
            attn_out = self.cross_attn(x_norm, cross_norm, cross_norm)
            x = x + self.gamma_1(attn_out)

            # FFN
            x_norm = self.norm3(x)
            ffn_out = self.linear2(nn.gelu(self.linear1(x_norm)))
            x = x + self.gamma_2(ffn_out)
        else:
            # Self-attention
            x_norm = self.norm1(x)
            attn_out = self.self_attn(x_norm, x_norm, x_norm)
            x = x + self.gamma_1(attn_out)

            # FFN
            x_norm = self.norm2(x)
            ffn_out = self.linear2(nn.gelu(self.linear1(x_norm)))
            x = x + self.gamma_2(ffn_out)

        # Final norm
        x = self.norm_out(x)

        return x


class CrossTransformerEncoder(nn.Module):
    """Cross-domain transformer matching PyTorch's CrossTransformerEncoder.

    Structure:
        norm_in: LayerNorm (freq input)
        norm_in_t: LayerNorm (time input)
        layers: list of MyTransformerEncoderLayer (freq branch)
        layers_t: list of MyTransformerEncoderLayer (time branch)

    Pattern: alternating self-attention (0,2,4) and cross-attention (1,3)

    Input format:
        freq: [B, C, F, T] (4D frequency features)
        time: [B, C, T] (3D time features)
    """

    def __init__(
        self,
        dim: int,
        depth: int = 5,
        heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_period = max_period

        # Position embedding weight (scalar, default 1.0)
        # This is not a learned parameter in pretrained model
        self.weight_pos_embed = 1.0

        if dim_feedforward is None:
            dim_feedforward = 4 * dim

        # Input layer norms
        self.norm_in = nn.LayerNorm(dim)
        self.norm_in_t = nn.LayerNorm(dim)

        # Build layers - alternating self and cross attention
        self.layers = []
        self.layers_t = []

        for i in range(depth):
            cross = (i % 2 == 1)  # odd layers are cross-attention
            self.layers.append(
                MyTransformerEncoderLayer(
                    dim, heads, dim_feedforward, dropout, cross_attention=cross
                )
            )
            self.layers_t.append(
                MyTransformerEncoderLayer(
                    dim, heads, dim_feedforward, dropout, cross_attention=cross
                )
            )

    def __call__(
        self,
        freq: mx.array,
        time: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Apply cross-domain transformer.

        Args:
            freq: Frequency features [B, C, F, T] (4D)
            time: Time features [B, C, T] (3D)

        Returns:
            Tuple of (freq_out, time_out):
                freq_out: [B, C, F, T]
                time_out: [B, C, T]
        """
        B, C, Fr, T1 = freq.shape
        _, _, T2 = time.shape

        # Create 2D positional embedding for freq branch
        pos_emb_2d = create_2d_sin_embedding(C, Fr, T1, self.max_period)

        # Rearrange: [B, C, Fr, T1] -> [B, (T1*Fr), C]
        # PyTorch does "b c fr t1 -> b (t1 fr) c" (time-major flattening)
        pos_emb_2d = pos_emb_2d.transpose(0, 3, 2, 1)  # [1, T1, Fr, C]
        pos_emb_2d = pos_emb_2d.reshape(1, T1 * Fr, C)  # [1, T1*Fr, C]

        freq = freq.transpose(0, 3, 2, 1)  # [B, T1, Fr, C]
        freq = freq.reshape(B, T1 * Fr, C)  # [B, T1*Fr, C]

        # Apply norm and add position embedding
        freq = self.norm_in(freq)
        freq = freq + self.weight_pos_embed * pos_emb_2d

        # Create 1D positional embedding for time branch
        # Returns [T2, 1, C] (TBC format), rearrange to [1, T2, C]
        pos_emb_1d = create_sin_embedding(T2, C, 0, self.max_period)  # [T2, 1, C]
        pos_emb_1d = pos_emb_1d.transpose(1, 0, 2)  # [1, T2, C]

        # Rearrange: [B, C, T2] -> [B, T2, C]
        time = time.transpose(0, 2, 1)

        # Apply norm and add position embedding
        time = self.norm_in_t(time)
        time = time + self.weight_pos_embed * pos_emb_1d

        # Apply layers
        for i, (freq_layer, time_layer) in enumerate(
            zip(self.layers, self.layers_t)
        ):
            if i % 2 == 0:
                # Self-attention
                freq = freq_layer(freq)
                time = time_layer(time)
            else:
                # Cross-attention (freq attends to time, time attends to freq)
                old_freq = freq
                freq = freq_layer(freq, time)
                time = time_layer(time, old_freq)

        # Rearrange back: [B, (T1*Fr), C] -> [B, C, Fr, T1]
        freq = freq.reshape(B, T1, Fr, C)
        freq = freq.transpose(0, 3, 2, 1)  # [B, C, Fr, T1]

        # Rearrange back: [B, T2, C] -> [B, C, T2]
        time = time.transpose(0, 2, 1)

        return freq, time
