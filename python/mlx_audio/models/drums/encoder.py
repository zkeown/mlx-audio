"""CNN encoder for spectrogram feature extraction.

Extracts local spectral features from mel-spectrograms, capturing:
- Attack transients (critical for drum onset detection)
- Frequency signatures (distinguishing cymbal types)
- Harmonic structure

Ported from PyTorch implementation.
"""

import mlx.core as mx
import mlx.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        groups: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()

        # Handle kernel_size tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        # Compute padding for "same" behavior
        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif isinstance(padding, int):
            padding = (padding, padding)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,
        )
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        if self.use_bn:
            # BatchNorm expects (N, H, W, C) in MLX, same as our Conv2d output
            x = self.bn(x)
        return nn.silu(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: (B, H, W, C) in MLX
        b, h, w, c = x.shape

        # Global average pooling over spatial dimensions
        scale = mx.mean(x, axis=(1, 2))  # (B, C)

        # FC layers
        scale = nn.silu(self.fc1(scale))  # (B, reduced)
        scale = mx.sigmoid(self.fc2(scale))  # (B, C)

        # Reshape for broadcasting
        scale = scale.reshape(b, 1, 1, c)

        return x * scale


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck block (MBConv).

    Standard building block for EfficientNet-style architectures.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] = 1,
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride)

        self.use_residual = stride == (1, 1) and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.drop_rate = drop_rate

        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = ConvBlock(in_channels, hidden_dim, kernel_size=1)

        # Depthwise convolution - MLX doesn't have groups, use regular conv
        # For simplicity, we use a regular conv here (not depthwise)
        self.dw_conv = ConvBlock(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
        )

        # Squeeze-and-excitation
        if se_ratio > 0:
            self.se = SqueezeExcitation(hidden_dim, int(1 / se_ratio))

        # Projection phase (no activation)
        self.proj_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        # Expansion
        if self.expand_ratio != 1:
            x = self.expand_conv(x)

        # Depthwise conv
        x = self.dw_conv(x)

        # SE
        if self.se_ratio > 0:
            x = self.se(x)

        # Projection
        x = self.proj_conv(x)
        x = self.proj_bn(x)

        # Residual connection
        if self.use_residual:
            if self.drop_rate > 0:
                # Stochastic depth not implemented for simplicity
                pass
            x = x + residual

        return x


class SpectrogramEncoder(nn.Module):
    """CNN encoder for mel-spectrogram feature extraction.

    Input: (batch, time, freq, 1) mel-spectrogram (MLX NHWC format)
    Output: (batch, time', embed_dim) feature sequence

    The time dimension is preserved (with possible striding) to allow
    frame-level predictions. Frequency dimension is collapsed.
    """

    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 512,
        base_channels: int = 32,
        drop_rate: float = 0.1,
    ):
        """Initialize encoder.

        Args:
            n_mels: Number of mel frequency bins (input height)
            embed_dim: Output embedding dimension
            base_channels: Base number of channels (scaled up in later stages)
            drop_rate: Dropout rate for MBConv blocks
        """
        super().__init__()

        self.n_mels = n_mels
        self.embed_dim = embed_dim

        # Stage 0: Initial convolution
        # Input: (B, T, 128, 1) -> (B, T, 128, 32)
        self.stem = ConvBlock(1, base_channels, kernel_size=3, stride=1)

        # Stage 1: (B, T, 128, 32) -> (B, T, 64, 64)
        self.stage1 = [
            MBConvBlock(base_channels, base_channels * 2, stride=(1, 2), drop_rate=drop_rate),
            MBConvBlock(base_channels * 2, base_channels * 2, drop_rate=drop_rate),
        ]

        # Stage 2: (B, T, 64, 64) -> (B, T, 32, 128)
        self.stage2 = [
            MBConvBlock(base_channels * 2, base_channels * 4, stride=(1, 2), drop_rate=drop_rate),
            MBConvBlock(base_channels * 4, base_channels * 4, drop_rate=drop_rate),
            MBConvBlock(base_channels * 4, base_channels * 4, drop_rate=drop_rate),
        ]

        # Stage 3: (B, T, 32, 128) -> (B, T, 16, 256)
        self.stage3 = [
            MBConvBlock(base_channels * 4, base_channels * 8, stride=(1, 2), drop_rate=drop_rate),
            MBConvBlock(base_channels * 8, base_channels * 8, drop_rate=drop_rate),
            MBConvBlock(base_channels * 8, base_channels * 8, drop_rate=drop_rate),
        ]

        # Stage 4: (B, T, 16, 256) -> (B, T, 8, 512)
        self.stage4 = [
            MBConvBlock(base_channels * 8, base_channels * 16, stride=(1, 2), drop_rate=drop_rate),
            MBConvBlock(base_channels * 16, base_channels * 16, drop_rate=drop_rate),
        ]

        # Project to embed_dim if necessary
        final_channels = base_channels * 16
        if final_channels != embed_dim:
            self.proj = nn.Linear(final_channels, embed_dim)
        else:
            self.proj = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input spectrogram (batch, 1, time, freq) in PyTorch format
               Will be transposed to MLX format internally

        Returns:
            Feature sequence (batch, time, embed_dim)
        """
        # Convert from PyTorch (B, C, T, F) to MLX (B, T, F, C)
        x = mx.transpose(x, (0, 2, 3, 1))

        # CNN stages
        x = self.stem(x)

        for block in self.stage1:
            x = block(x)
        for block in self.stage2:
            x = block(x)
        for block in self.stage3:
            x = block(x)
        for block in self.stage4:
            x = block(x)

        # x shape: (B, T, F', C) where F' = n_mels / 16

        # Global average pooling over frequency dimension
        x = mx.mean(x, axis=2)  # (B, T, C)

        # Project to embed_dim
        if self.proj is not None:
            x = self.proj(x)

        return x


class LightweightEncoder(nn.Module):
    """Lighter CNN encoder for faster training/inference.

    Uses fewer channels and simpler blocks.
    """

    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 256,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        # Simple 4-stage encoder
        self.conv1 = ConvBlock(1, 32, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.conv2 = ConvBlock(32, 64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=(1, 2))
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=(1, 2))

        self.dropout = nn.Dropout(drop_rate)
        self.proj = nn.Linear(256, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Convert from PyTorch (B, C, T, F) to MLX (B, T, F, C)
        x = mx.transpose(x, (0, 2, 3, 1))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)

        # Pool over frequency dimension
        x = mx.mean(x, axis=2)  # (B, T, 256)

        x = self.proj(x)

        return x
