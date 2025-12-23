"""Feature-wise Linear Modulation (FiLM) for Banquet model."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning.

    Applies query conditioning to band embeddings using learnable
    scale (gamma) and shift (beta) parameters.

    Architecture:
        x = GroupNorm(x)
        if multiplicative: x = gamma(w) * x
        if additive: x = x + beta(w)

    Where w is the conditioning embedding (from PaSST).
    """

    def __init__(
        self,
        cond_embedding_dim: int,
        channels: int,
        additive: bool = True,
        multiplicative: bool = True,
        depth: int = 2,
        activation: str = "ELU",
        channels_per_group: int = 16,
    ) -> None:
        super().__init__()

        self.cond_embedding_dim = cond_embedding_dim
        self.channels = channels
        self.additive = additive
        self.multiplicative = multiplicative
        self.depth = depth

        # GroupNorm for input normalization
        num_groups = max(1, channels // channels_per_group)
        self.gn = nn.GroupNorm(num_groups, channels)

        # Get activation function
        if activation == "ELU":
            self.activation_fn = nn.ELU
        elif activation == "ReLU":
            self.activation_fn = nn.ReLU
        elif activation == "GELU":
            self.activation_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build gamma (multiplicative) network
        if multiplicative:
            if depth == 1:
                self.gamma = nn.Linear(cond_embedding_dim, channels)
            else:
                layers = [nn.Linear(cond_embedding_dim, channels)]
                for _ in range(depth - 1):
                    layers.append(self.activation_fn())
                    layers.append(nn.Linear(channels, channels))
                self.gamma = nn.Sequential(*layers)
        else:
            self.gamma = None

        # Build beta (additive) network
        if additive:
            if depth == 1:
                self.beta = nn.Linear(cond_embedding_dim, channels)
            else:
                layers = [nn.Linear(cond_embedding_dim, channels)]
                for _ in range(depth - 1):
                    layers.append(self.activation_fn())
                    layers.append(nn.Linear(channels, channels))
                self.beta = nn.Sequential(*layers)
        else:
            self.beta = None

    def __call__(self, x: mx.array, w: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input features [batch, channels, n_bands, n_time] (NCHW format)
            w: Conditioning embedding [batch, cond_embedding_dim]

        Returns:
            Modulated features with same shape as input
        """
        # MLX GroupNorm expects channels-last format (NHWC)
        # Convert NCHW -> NHWC for GroupNorm, then back
        if x.ndim == 4:
            # [B, C, H, W] -> [B, H, W, C]
            x = mx.transpose(x, (0, 2, 3, 1))
            x = self.gn(x)
            # [B, H, W, C] -> [B, C, H, W]
            x = mx.transpose(x, (0, 3, 1, 2))
        elif x.ndim == 3:
            # [B, C, L] -> [B, L, C]
            x = mx.transpose(x, (0, 2, 1))
            x = self.gn(x)
            # [B, L, C] -> [B, C, L]
            x = mx.transpose(x, (0, 2, 1))
        else:
            x = self.gn(x)

        # Multiplicative modulation (gamma)
        if self.multiplicative and self.gamma is not None:
            gamma = self.gamma(w)
            # Broadcast gamma to match input dimensions
            if x.ndim == 4:
                gamma = gamma[:, :, None, None]  # [batch, channels, 1, 1]
            elif x.ndim == 3:
                gamma = gamma[:, :, None]  # [batch, channels, 1]
            # 2D case: no expansion needed
            x = gamma * x

        # Additive modulation (beta)
        if self.additive and self.beta is not None:
            beta = self.beta(w)
            # Broadcast beta to match input dimensions
            if x.ndim == 4:
                beta = beta[:, :, None, None]
            elif x.ndim == 3:
                beta = beta[:, :, None]
            x = x + beta

        return x
