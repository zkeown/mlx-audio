"""Dilated convolution residual block for HTDemucs.

This module exactly matches the PyTorch demucs implementation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LayerScale(nn.Module):
    """Learnable per-channel scaling (from ConvNeXt/Demucs).

    Weight key: .scale (not .weight)
    """

    def __init__(self, channels: int, init: float = 1e-4):
        super().__init__()
        self.scale = mx.ones((channels,)) * init

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, T, C] (NLC format), scale per channel on last dim
        return x * self.scale


class _Placeholder(nn.Module):
    """Placeholder for non-parametric ops to maintain index alignment."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


def _make_dconv_layer(
    channels: int,
    compress: int = 8,
    dilation: int = 1,
    init: float = 1e-4,
) -> list:
    """Create a DConv layer as a list matching PyTorch Sequential structure.

    PyTorch structure (Sequential with numeric indices):
        0: Conv1d (compress channels, dilated)
        1: GroupNorm (1 group = LayerNorm style)
        2: GELU (placeholder)
        3: Conv1d (expand to 2*channels for GLU)
        4: GroupNorm
        5: GLU (placeholder)
        6: LayerScale

    Returns list with 7 elements. Indices 2 and 5 are placeholders.
    """
    hidden = channels // compress
    # Padding = dilation * (kernel // 2) for same-size output
    padding = dilation * (3 // 2)  # = dilation * 1 = dilation

    return [
        nn.Conv1d(channels, hidden, kernel_size=3, padding=padding, dilation=dilation),  # 0
        nn.GroupNorm(1, hidden),  # 1
        _Placeholder(),  # 2: GELU placeholder
        nn.Conv1d(hidden, channels * 2, kernel_size=1),  # 3
        nn.GroupNorm(1, channels * 2),  # 4
        _Placeholder(),  # 5: GLU placeholder
        LayerScale(channels, init),  # 6
    ]


def _apply_dconv_layer(layer_list: list, x: mx.array) -> mx.array:
    """Apply a DConv layer (as a list) to input.

    Input x is in NLC format (MLX Conv1d/GroupNorm format): [B, T, C]
    """
    residual = x

    # 0: Dilated conv (NLC format)
    x = layer_list[0](x)
    # 1: GroupNorm (NLC format - MLX uses channels last)
    x = layer_list[1](x)
    # 2: GELU (inline)
    x = nn.gelu(x)
    # 3: Expand conv
    x = layer_list[3](x)
    # 4: GroupNorm
    x = layer_list[4](x)
    # 5: GLU (split + sigmoid gate) - split on last dim for NLC
    a, b = mx.split(x, 2, axis=-1)
    x = a * mx.sigmoid(b)
    # 6: LayerScale - operates on NLC format
    x = layer_list[6](x)

    return residual + x


class DConv(nn.Module):
    """Multi-depth dilated convolution with residual connections.

    Matches PyTorch demucs.hdemucs.DConv exactly.

    Structure:
        layers: list of lists (each inner list is a Sequential-like structure)
            layers[i][j] corresponds to PyTorch layers[i][j]
    """

    def __init__(
        self,
        channels: int,
        depth: int = 2,
        compress: int = 8,
        init: float = 1e-4,
    ):
        super().__init__()
        self.channels = channels
        self.depth = depth

        # Build layers with exponentially increasing dilation
        # Each layer is a list (matching PyTorch Sequential structure)
        self.layers = []
        for d in range(depth):
            dilation = 2 ** d
            layer = _make_dconv_layer(
                channels=channels,
                compress=compress,
                dilation=dilation,
                init=init,
            )
            self.layers.append(layer)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = _apply_dconv_layer(layer, x)
        return x
