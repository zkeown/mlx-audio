"""SE-Res2Net block for ECAPA-TDNN."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    Channel attention mechanism that adaptively recalibrates
    channel-wise feature responses.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    reduction : int, default=8
        Reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply SE attention.

        Parameters
        ----------
        x : mx.array
            Input of shape (batch, time, channels) - MLX channels-last.

        Returns
        -------
        mx.array
            Scaled output of same shape.
        """
        # Global average pooling over time (axis=1)
        s = mx.mean(x, axis=1)  # (batch, channels)

        # Excitation
        s = nn.relu(self.fc1(s))
        s = mx.sigmoid(self.fc2(s))

        # Scale: expand to (batch, 1, channels) for broadcasting
        return x * s[:, None, :]


class Res2NetBlock(nn.Module):
    """Res2Net multi-scale convolution block.

    Splits channels into groups and processes them with
    hierarchical residual connections.

    Parameters
    ----------
    channels : int
        Number of channels.
    kernel_size : int
        Convolution kernel size.
    dilation : int
        Dilation rate.
    scale : int, default=8
        Number of scale groups.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
    ):
        super().__init__()
        self.scale = scale
        self.width = channels // scale

        # Convolutions for each scale (except first which is identity)
        self.convs = [
            nn.Conv1d(
                self.width,
                self.width,
                kernel_size,
                padding=(kernel_size - 1) * dilation // 2,
                dilation=dilation,
            )
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Apply multi-scale convolution.

        Parameters
        ----------
        x : mx.array
            Input of shape (batch, time, channels) - MLX channels-last.

        Returns
        -------
        mx.array
            Output of same shape.
        """
        # Split into groups along channels axis (axis=2)
        splits = []
        for i in range(self.scale):
            start = i * self.width
            end = (i + 1) * self.width
            splits.append(x[:, :, start:end])

        # Process with hierarchical residual
        outputs = [splits[0]]  # First group is identity

        for i in range(1, self.scale):
            if i == 1:
                y = self.convs[i - 1](splits[i])
            else:
                y = self.convs[i - 1](splits[i] + outputs[-1])
            outputs.append(nn.relu(y))

        return mx.concatenate(outputs, axis=2)


class SERes2NetBlock(nn.Module):
    """Squeeze-Excitation Res2Net block for ECAPA-TDNN.

    Combines SE attention with Res2Net multi-scale processing.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, default=3
        Convolution kernel size.
    dilation : int, default=1
        Dilation rate.
    scale : int, default=8
        Number of Res2Net scale groups.
    se_channels : int, default=128
        SE bottleneck channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        se_channels: int = 128,
    ):
        super().__init__()

        # Input projection (1x1 conv)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm(out_channels)

        # Res2Net convolution
        self.res2net = Res2NetBlock(out_channels, kernel_size, dilation, scale)
        self.bn2 = nn.BatchNorm(out_channels)

        # Output projection
        self.conv3 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm(out_channels)

        # SE block
        self.se = SEBlock(out_channels, reduction=out_channels // se_channels)

        # Skip connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Parameters
        ----------
        x : mx.array
            Input of shape (batch, time, in_channels) - MLX channels-last.

        Returns
        -------
        mx.array
            Output of shape (batch, time, out_channels).
        """
        # Shortcut
        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x

        # Main path
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.res2net(out))
        out = nn.relu(self.bn3(self.conv3(out)))

        # SE attention
        out = self.se(out)

        # Residual connection
        return nn.relu(out + residual)
