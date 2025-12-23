"""EnCodec convolutional decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.encodec.config import EnCodecConfig


class ConvTransposeBlock(nn.Module):
    """Transposed convolutional block for upsampling.

    Applies: ConvTranspose1d -> Norm -> Activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = True,
        norm_type: str = "weight_norm",
        activation: str = "elu",
    ):
        """Initialize transposed conv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Upsampling factor
            causal: Whether to use causal padding/trimming
            norm_type: Type of normalization
            activation: Activation function ("elu" or "none")
        """
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.stride = stride

        # Calculate padding for transposed convolution
        # For transposed conv, output_size = (input_size - 1) * stride + kernel_size
        # We need to trim to get the right output size
        self.padding = kernel_size - stride

        # Transposed convolution
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
        )

        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        """Apply transposed conv block.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C', T * stride]
        """
        # MLX ConvTranspose1d expects [B, T, C] format, we have [B, C, T]
        x = x.transpose(0, 2, 1)  # [B, C, T] -> [B, T, C]

        # Apply transposed convolution
        x = self.conv_transpose(x)

        # Back to [B, C, T] format
        x = x.transpose(0, 2, 1)  # [B, T, C] -> [B, C, T]

        # Trim excess samples for proper output length
        if self.padding > 0:
            if self.causal:
                # Trim from the right for causal
                x = x[:, :, :-self.padding]
            else:
                # Trim equally from both sides
                trim_left = self.padding // 2
                trim_right = self.padding - trim_left
                if trim_right > 0:
                    x = x[:, :, trim_left:-trim_right]
                else:
                    x = x[:, :, trim_left:]

        # Apply activation
        if self.activation == "elu":
            x = nn.elu(x)

        return x


class ConvBlock(nn.Module):
    """Regular convolutional block (same as in encoder)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        causal: bool = True,
        norm_type: str = "weight_norm",
        activation: str = "elu",
    ):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        effective_kernel = (kernel_size - 1) * dilation + 1
        if causal:
            self.padding_left = effective_kernel - 1
            self.padding_right = 0
        else:
            self.padding_left = (effective_kernel - 1) // 2
            self.padding_right = effective_kernel - 1 - self.padding_left

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        # MLX Conv1d expects [B, T, C] format, we have [B, C, T]
        x = x.transpose(0, 2, 1)  # [B, C, T] -> [B, T, C]

        if self.padding_left > 0 or self.padding_right > 0:
            x = mx.pad(x, [(0, 0), (self.padding_left, self.padding_right), (0, 0)])

        x = self.conv(x)

        # Back to [B, C, T] format
        x = x.transpose(0, 2, 1)  # [B, T, C] -> [B, C, T]

        if self.activation == "elu":
            x = nn.elu(x)

        return x


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions (same as encoder)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = True,
        norm_type: str = "weight_norm",
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            causal=causal,
            norm_type=norm_type,
            activation="elu",
        )

        self.conv2 = ConvBlock(
            channels,
            channels,
            1,
            causal=causal,
            norm_type=norm_type,
            activation="none",
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and residual units.

    Architecture: UpsampleConv -> ELU -> [ResidualUnits]
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
        """Initialize decoder block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Upsampling factor
            kernel_size: Kernel size for upsampling conv
            residual_kernel_size: Kernel size for residual convs
            num_residual_layers: Number of residual units
            dilation_base: Base for exponential dilation
            causal: Whether to use causal padding
            norm_type: Type of normalization
        """
        super().__init__()

        # Upsampling transposed convolution
        self.upsample = ConvTransposeBlock(
            in_channels,
            out_channels,
            kernel_size=stride * 2,
            stride=stride,
            causal=causal,
            norm_type=norm_type,
            activation="none",
        )

        # Residual units with increasing dilation
        self.residuals = [
            ResidualUnit(
                out_channels,
                kernel_size=residual_kernel_size,
                dilation=dilation_base ** i,
                causal=causal,
                norm_type=norm_type,
            )
            for i in range(num_residual_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Apply decoder block.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C', T * stride]
        """
        # Upsample
        x = self.upsample(x)

        # ELU after upsampling
        x = nn.elu(x)

        # Apply residual units
        for residual in self.residuals:
            x = residual(x)

        return x


class EnCodecDecoder(nn.Module):
    """EnCodec convolutional decoder.

    Decodes latent embeddings back to audio waveform with progressive upsampling.

    Architecture:
        Conv1d(initial) -> LSTM -> [DecoderBlock(stride) for stride in reversed(ratios)]
        -> ELU -> Conv1d(final)
    """

    def __init__(self, config: "EnCodecConfig"):
        """Initialize decoder.

        Args:
            config: EnCodec configuration
        """
        super().__init__()
        self.config = config

        # Calculate channels at each stage (reverse of encoder)
        # Encoder: filters -> filters*2 -> filters*4 -> filters*8 -> ...
        # Decoder: ... -> filters*4 -> filters*2 -> filters
        mult = 2 ** len(config.ratios)
        hidden_channels = config.num_filters * mult

        # Initial convolution from codebook dimension
        self.initial_conv = ConvBlock(
            config.codebook_dim,
            hidden_channels,
            config.kernel_size,
            causal=config.causal,
            norm_type=config.norm_type,
            activation="none",
        )

        # LSTM layers (stack manually since MLX LSTM doesn't support num_layers)
        self.lstm_layers_list = []
        if config.lstm_layers > 0:
            for i in range(config.lstm_layers):
                self.lstm_layers_list.append(
                    nn.LSTM(hidden_channels, hidden_channels)
                )

        # Decoder blocks with progressive upsampling (reversed ratios)
        self.blocks = []
        in_channels = hidden_channels
        for i, ratio in enumerate(reversed(config.ratios)):
            out_channels = in_channels // 2
            self.blocks.append(
                DecoderBlock(
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

        # Final convolution to audio channels
        self.final_conv = ConvBlock(
            config.num_filters,
            config.channels,
            config.last_kernel_size,
            causal=config.causal,
            norm_type=config.norm_type,
            activation="none",
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Decode latent embeddings to audio.

        Args:
            x: Latent embeddings [B, T, D]

        Returns:
            Audio waveform [B, C, T'] where T' = T * hop_length
        """
        # Convert from [B, T, D] to [B, D, T]
        x = x.transpose(0, 2, 1)

        # Initial convolution
        x = self.initial_conv(x)
        x = nn.elu(x)

        # LSTM (if enabled)
        if self.lstm_layers_list:
            # LSTM expects [B, T, C], we have [B, C, T]
            x = x.transpose(0, 2, 1)
            for lstm in self.lstm_layers_list:
                x, _ = lstm(x)
            # Back to [B, C, T]
            x = x.transpose(0, 2, 1)

        # Decoder blocks
        for block in self.blocks:
            x = block(x)

        # Final convolution
        x = nn.elu(x)
        x = self.final_conv(x)

        return x
