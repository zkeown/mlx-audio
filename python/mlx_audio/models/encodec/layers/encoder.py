"""EnCodec convolutional encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.encodec.config import EnCodecConfig


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation.

    Applies: Conv1d -> Norm -> Activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        causal: bool = True,
        norm_type: str = "weight_norm",
        activation: str = "elu",
    ):
        """Initialize conv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            dilation: Convolution dilation
            groups: Number of groups for grouped convolution
            causal: Whether to use causal padding
            norm_type: Type of normalization ("weight_norm" or "none")
            activation: Activation function ("elu" or "none")
        """
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # Calculate padding for causal or same convolution
        effective_kernel = (kernel_size - 1) * dilation + 1
        if causal:
            self.padding_left = effective_kernel - 1
            self.padding_right = 0
        else:
            self.padding_left = (effective_kernel - 1) // 2
            self.padding_right = effective_kernel - 1 - self.padding_left

        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            dilation=dilation,
        )

        # Activation
        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        """Apply conv block.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C', T']
        """
        # MLX Conv1d expects [B, T, C] format, we have [B, C, T]
        x = x.transpose(0, 2, 1)  # [B, C, T] -> [B, T, C]

        # Apply padding (on time dimension, which is now axis 1)
        if self.padding_left > 0 or self.padding_right > 0:
            x = mx.pad(x, [(0, 0), (self.padding_left, self.padding_right), (0, 0)])

        # Apply convolution
        x = self.conv(x)

        # Back to [B, C, T] format
        x = x.transpose(0, 2, 1)  # [B, T, C] -> [B, C, T]

        # Apply activation
        if self.activation == "elu":
            x = nn.elu(x)

        return x


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions.

    Architecture: Conv(dilated) -> Conv(1x1) + skip connection
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = True,
        norm_type: str = "weight_norm",
    ):
        """Initialize residual unit.

        Args:
            channels: Number of channels
            kernel_size: Kernel size for dilated conv
            dilation: Dilation factor
            causal: Whether to use causal padding
            norm_type: Type of normalization
        """
        super().__init__()

        # Dilated convolution
        self.conv1 = ConvBlock(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            causal=causal,
            norm_type=norm_type,
            activation="elu",
        )

        # 1x1 convolution (no activation before residual)
        self.conv2 = ConvBlock(
            channels,
            channels,
            1,
            causal=causal,
            norm_type=norm_type,
            activation="none",
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply residual unit.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C, T]
        """
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class EncoderBlock(nn.Module):
    """Encoder block with residual units and downsampling.

    Architecture: [ResidualUnits] -> ELU -> DownsampleConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int = 7,
        residual_kernel_size: int = 3,
        num_residual_layers: int = 1,
        dilation_base: int = 2,
        causal: bool = True,
        norm_type: str = "weight_norm",
    ):
        """Initialize encoder block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Downsampling factor
            kernel_size: Kernel size for downsampling conv
            residual_kernel_size: Kernel size for residual convs
            num_residual_layers: Number of residual units
            dilation_base: Base for exponential dilation
            causal: Whether to use causal padding
            norm_type: Type of normalization
        """
        super().__init__()

        # Residual units with increasing dilation
        self.residuals = [
            ResidualUnit(
                in_channels,
                kernel_size=residual_kernel_size,
                dilation=dilation_base ** i,
                causal=causal,
                norm_type=norm_type,
            )
            for i in range(num_residual_layers)
        ]

        # Downsampling strided convolution
        self.downsample = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=stride * 2,  # EnCodec uses 2x stride for kernel
            stride=stride,
            causal=causal,
            norm_type=norm_type,
            activation="none",
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply encoder block.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C', T//stride]
        """
        # Apply residual units
        for residual in self.residuals:
            x = residual(x)

        # ELU before downsampling
        x = nn.elu(x)

        # Downsample
        x = self.downsample(x)

        return x


class EnCodecEncoder(nn.Module):
    """EnCodec convolutional encoder.

    Encodes audio waveform to latent embeddings with progressive downsampling.

    Architecture:
        Conv1d(initial) -> [EncoderBlock(stride) for stride in ratios] ->
        LSTM -> Conv1d(final)
    """

    def __init__(self, config: "EnCodecConfig"):
        """Initialize encoder.

        Args:
            config: EnCodec configuration
        """
        super().__init__()
        self.config = config

        # Initial convolution
        self.initial_conv = ConvBlock(
            config.channels,
            config.num_filters,
            config.kernel_size,
            causal=config.causal,
            norm_type=config.norm_type,
            activation="none",
        )

        # Encoder blocks with progressive downsampling
        self.blocks = []
        in_channels = config.num_filters
        for i, ratio in enumerate(config.ratios):
            out_channels = in_channels * 2
            self.blocks.append(
                EncoderBlock(
                    in_channels,
                    out_channels,
                    stride=ratio,
                    kernel_size=config.kernel_size,
                    residual_kernel_size=config.residual_kernel_size,
                    num_residual_layers=config.num_residual_layers,
                    dilation_base=config.dilation_base,
                    causal=config.causal,
                    norm_type=config.norm_type,
                )
            )
            in_channels = out_channels

        # LSTM layers (stack manually since MLX LSTM doesn't support num_layers)
        self.lstm_layers_list = []
        if config.lstm_layers > 0:
            for i in range(config.lstm_layers):
                self.lstm_layers_list.append(
                    nn.LSTM(in_channels, in_channels)
                )

        # Final convolution to codebook dimension
        self.final_conv = ConvBlock(
            in_channels,
            config.codebook_dim,
            config.last_kernel_size,
            causal=config.causal,
            norm_type=config.norm_type,
            activation="none",
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Encode audio waveform.

        Args:
            x: Audio waveform [B, C, T] or [B, T] (mono)

        Returns:
            Latent embeddings [B, T', D] where T' = T / hop_length
        """
        # Handle mono input
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=1)

        # Initial convolution
        x = self.initial_conv(x)
        x = nn.elu(x)

        # Encoder blocks
        for block in self.blocks:
            x = block(x)

        # LSTM (if enabled)
        if self.lstm_layers_list:
            # LSTM expects [B, T, C], we have [B, C, T]
            x = x.transpose(0, 2, 1)
            for lstm in self.lstm_layers_list:
                x, _ = lstm(x)
            # Back to [B, C, T]
            x = x.transpose(0, 2, 1)

        # Final convolution
        x = nn.elu(x)
        x = self.final_conv(x)

        # Return [B, T, D] format
        return x.transpose(0, 2, 1)
