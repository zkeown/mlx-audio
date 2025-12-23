"""Attentive statistics pooling for ECAPA-TDNN."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling for speaker embeddings.

    Pools variable-length sequences to fixed-dimensional embeddings
    using attention-weighted mean and standard deviation.

    Parameters
    ----------
    channels : int
        Number of input channels.
    attention_channels : int, default=128
        Hidden dimension for attention network.
    global_context : bool, default=True
        Whether to include global context (mean, std) in attention.
    """

    def __init__(
        self,
        channels: int,
        attention_channels: int = 128,
        global_context: bool = True,
    ):
        super().__init__()
        self.global_context = global_context

        # Input dimension for attention
        # If global_context, concatenate: [x, global_mean, global_std]
        in_dim = channels * 3 if global_context else channels

        # Attention network
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, attention_channels, 1),
            nn.ReLU(),
            nn.Conv1d(attention_channels, channels, 1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Pool sequence to fixed embedding.

        Parameters
        ----------
        x : mx.array
            Input of shape (batch, time, channels) - MLX channels-last.

        Returns
        -------
        mx.array
            Pooled output of shape (batch, channels * 2).
            Contains attention-weighted mean and std.
        """
        batch, time, channels = x.shape

        if self.global_context:
            # Compute global statistics over time (axis=1)
            global_mean = mx.mean(x, axis=1, keepdims=True)
            global_std = mx.std(x, axis=1, keepdims=True)

            # Expand to time dimension
            global_mean = mx.broadcast_to(global_mean, x.shape)
            global_std = mx.broadcast_to(global_std, x.shape)

            # Concatenate for attention input along channels (axis=2)
            attn_input = mx.concatenate([x, global_mean, global_std], axis=2)
        else:
            attn_input = x

        # Compute attention weights
        attn_weights = self.attention(attn_input)  # (batch, time, channels)
        attn_weights = mx.softmax(attn_weights, axis=1)  # Softmax over time

        # Weighted mean over time
        weighted_mean = mx.sum(x * attn_weights, axis=1)

        # Weighted std
        weighted_var = mx.sum(
            attn_weights * (x - weighted_mean[:, None, :]) ** 2,
            axis=1
        )
        weighted_std = mx.sqrt(weighted_var + 1e-8)

        # Concatenate mean and std
        return mx.concatenate([weighted_mean, weighted_std], axis=-1)


class TemporalAveragePooling(nn.Module):
    """Simple temporal average pooling.

    Averages over the time dimension to produce fixed-length output.
    """

    def __call__(self, x: mx.array) -> mx.array:
        """Pool by averaging over time.

        Parameters
        ----------
        x : mx.array
            Input of shape (batch, time, channels) - MLX channels-last.

        Returns
        -------
        mx.array
            Output of shape (batch, channels).
        """
        return mx.mean(x, axis=1)


class SelfAttentivePooling(nn.Module):
    """Self-attentive pooling layer.

    Uses self-attention to weight frames before pooling.

    Parameters
    ----------
    channels : int
        Number of channels.
    attention_dim : int, default=128
        Attention hidden dimension.
    """

    def __init__(self, channels: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply self-attentive pooling.

        Parameters
        ----------
        x : mx.array
            Input of shape (batch, time, channels) - MLX channels-last.

        Returns
        -------
        mx.array
            Pooled output of shape (batch, channels).
        """
        # Input is already (batch, time, channels)

        # Compute attention scores
        scores = self.attention(x)  # (batch, time, 1)
        weights = mx.softmax(scores, axis=1)

        # Weighted sum
        output = mx.sum(x * weights, axis=1)  # (batch, channels)

        return output
