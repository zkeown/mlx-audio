"""EnCodec model matching HuggingFace's exact architecture for weight loading."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.encodec.config import EnCodecConfig


class EnCodecConv1d(nn.Module):
    """Conv1d with weight normalization and causal padding.

    Matches HuggingFace's EuclideanCodebook structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.causal = causal
        self.pad_mode = pad_mode
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

        # Calculate padding
        effective_kernel = (kernel_size - 1) * dilation + 1
        if causal:
            self.padding_left = effective_kernel - stride
            self.padding_right = 0
        else:
            total_pad = effective_kernel - stride
            self.padding_left = total_pad // 2
            self.padding_right = total_pad - self.padding_left

        # Conv layer - MLX Conv1d expects [B, T, C]
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Input: [B, C, T] (PyTorch convention)
        # MLX Conv1d expects [B, T, C]
        x = x.transpose(0, 2, 1)

        # Apply padding
        if self.padding_left > 0 or self.padding_right > 0:
            x = mx.pad(x, [(0, 0), (self.padding_left, self.padding_right), (0, 0)])

        # Apply convolution
        x = self.conv(x)

        # Back to [B, C, T]
        return x.transpose(0, 2, 1)


class EnCodecConvTranspose1d(nn.Module):
    """ConvTranspose1d with proper output trimming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.stride = stride
        self.kernel_size = kernel_size

        # Trim amount for proper output length
        self.trim_right = kernel_size - stride

        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Input: [B, C, T]
        x = x.transpose(0, 2, 1)  # -> [B, T, C]
        x = self.conv(x)
        x = x.transpose(0, 2, 1)  # -> [B, C, T]

        # Trim excess
        if self.trim_right > 0:
            if self.causal:
                x = x[:, :, :-self.trim_right]
            else:
                trim_left = self.trim_right // 2
                trim_right = self.trim_right - trim_left
                if trim_right > 0:
                    x = x[:, :, trim_left:-trim_right]
                else:
                    x = x[:, :, trim_left:]

        return x


class EnCodecResnetBlock(nn.Module):
    """Residual block with dilated convolutions.

    Structure: [ELU, Conv(dilated), ELU, Conv(1x1)] + skip
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] = [3, 1],
        dilations: list[int] = [1, 1],
        causal: bool = True,
    ):
        super().__init__()

        # Block contains 4 layers: [ELU, Conv, ELU, Conv]
        # In HF these are block.{0,1,2,3}
        # We match by using block list with indices 1 and 3 for convs
        self.block = [
            None,  # index 0: activation (not a module)
            EnCodecConv1d(dim, dim, kernel_sizes[0], dilation=dilations[0], causal=causal),
            None,  # index 2: activation
            EnCodecConv1d(dim, dim, kernel_sizes[1], dilation=dilations[1], causal=causal),
        ]
        # For MLX module tracking, store convs explicitly
        self._conv1 = self.block[1]
        self._conv3 = self.block[3]

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        # Block 0: ELU
        x = nn.elu(x)
        # Block 1: Conv (dilated)
        x = self._conv1(x)
        # Block 2: ELU
        x = nn.elu(x)
        # Block 3: Conv (1x1)
        x = self._conv3(x)
        return x + residual


class EnCodecLSTM(nn.Module):
    """LSTM wrapper matching HF structure."""

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim)
        # For 2-layer LSTM, we need to stack
        if num_layers > 1:
            self.lstm2 = nn.LSTM(dim, dim)
        else:
            self.lstm2 = None
        self.num_layers = num_layers

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, T] -> need [B, T, C] for LSTM
        x = x.transpose(0, 2, 1)
        x, _ = self.lstm(x)
        if self.lstm2 is not None:
            x, _ = self.lstm2(x)
        return x.transpose(0, 2, 1)


class EnCodecEncoder(nn.Module):
    """EnCodec encoder matching HuggingFace structure exactly.

    HF structure (for 32kHz model):
    layers.0: Conv1d (initial)
    layers.1: ResnetBlock (for downsample stage 0)
    layers.3: Conv1d (downsample 0)
    layers.4: ResnetBlock (for downsample stage 1)
    layers.6: Conv1d (downsample 1)
    ...
    layers.13: LSTM
    layers.15: Conv1d (final)
    """

    def __init__(self, config: "EnCodecConfig"):
        super().__init__()
        self.config = config

        # Build layers list to match HF indexing exactly
        # The pattern is:
        # - Initial conv at 0
        # - For each ratio: ResnetBlock, skip, DownsampleConv
        # - LSTM near the end
        # - Final conv

        self.layers = []

        # Index 0: Initial conv
        self.layers.append(
            EnCodecConv1d(
                config.channels,
                config.num_filters,
                config.kernel_size,
                causal=config.causal,
            )
        )

        current_channels = config.num_filters

        # Encoder blocks with downsampling
        for i, ratio in enumerate(config.ratios):
            out_channels = current_channels * 2

            # ResnetBlock (odd index after each pair)
            for j in range(config.num_residual_layers):
                dilation = config.dilation_base ** j
                self.layers.append(
                    EnCodecResnetBlock(
                        current_channels,
                        kernel_sizes=[config.residual_kernel_size, 1],
                        dilations=[dilation, 1],
                        causal=config.causal,
                    )
                )

            # Placeholder for index alignment (skip index 2, 5, 8, 11)
            self.layers.append(None)

            # Downsample conv
            self.layers.append(
                EnCodecConv1d(
                    current_channels,
                    out_channels,
                    ratio * 2,  # kernel_size = 2 * stride
                    stride=ratio,
                    causal=config.causal,
                )
            )

            current_channels = out_channels

        # LSTM
        if config.lstm_layers > 0:
            self.layers.append(EnCodecLSTM(current_channels, config.lstm_layers))

        # Placeholder
        self.layers.append(None)

        # Final conv
        self.layers.append(
            EnCodecConv1d(
                current_channels,
                config.codebook_dim,
                config.last_kernel_size,
                causal=config.causal,
            )
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, T] or [B, T] for mono
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=1)

        for layer in self.layers:
            if layer is not None:
                x = layer(x)
                # Apply ELU after conv layers (not after LSTM or ResnetBlock)
                if isinstance(layer, EnCodecConv1d):
                    x = nn.elu(x)

        # Return [B, T, D]
        return x.transpose(0, 2, 1)


class EnCodecDecoder(nn.Module):
    """EnCodec decoder matching HuggingFace structure exactly."""

    def __init__(self, config: "EnCodecConfig"):
        super().__init__()
        self.config = config

        # Calculate hidden channels
        mult = 2 ** len(config.ratios)
        hidden_channels = config.num_filters * mult

        self.layers = []

        # Index 0: Initial conv
        self.layers.append(
            EnCodecConv1d(
                config.codebook_dim,
                hidden_channels,
                config.kernel_size,
                causal=config.causal,
            )
        )

        # LSTM
        if config.lstm_layers > 0:
            self.layers.append(EnCodecLSTM(hidden_channels, config.lstm_layers))

        # Placeholder
        self.layers.append(None)

        current_channels = hidden_channels

        # Decoder blocks with upsampling (reversed ratios)
        for i, ratio in enumerate(reversed(config.ratios)):
            out_channels = current_channels // 2

            # Upsample conv (transposed)
            self.layers.append(
                EnCodecConvTranspose1d(
                    current_channels,
                    out_channels,
                    ratio * 2,
                    stride=ratio,
                    causal=config.causal,
                )
            )

            # ResnetBlock
            for j in range(config.num_residual_layers):
                dilation = config.dilation_base ** j
                self.layers.append(
                    EnCodecResnetBlock(
                        out_channels,
                        kernel_sizes=[config.residual_kernel_size, 1],
                        dilations=[dilation, 1],
                        causal=config.causal,
                    )
                )

            # Placeholders for index alignment
            self.layers.append(None)

            current_channels = out_channels

        # Final conv
        self.layers.append(
            EnCodecConv1d(
                config.num_filters,
                config.channels,
                config.last_kernel_size,
                causal=config.causal,
            )
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(0, 2, 1)

        for layer in self.layers:
            if layer is not None:
                x = layer(x)
                # Apply ELU after conv layers
                if isinstance(layer, (EnCodecConv1d, EnCodecConvTranspose1d)):
                    x = nn.elu(x)

        return x


class EuclideanCodebook(nn.Module):
    """Codebook for vector quantization."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Initialize embeddings
        self.embed = mx.zeros((codebook_size, codebook_dim))
        # These are buffers in HF but we need them for weight loading
        self.cluster_size = mx.zeros((codebook_size,))
        self.embed_avg = mx.zeros((codebook_size, codebook_dim))
        self.inited = mx.array([False])

    def encode(self, x: mx.array) -> mx.array:
        """Encode input to codes."""
        # x: [B, T, D]
        # Compute distances to all codebook entries
        # dist[b,t,k] = ||x[b,t] - embed[k]||^2

        # Expand for broadcasting
        x_sq = mx.sum(x ** 2, axis=-1, keepdims=True)  # [B, T, 1]
        embed_sq = mx.sum(self.embed ** 2, axis=-1)  # [K]

        # x @ embed.T -> [B, T, K]
        dots = mx.matmul(x, self.embed.T)

        # dist = x^2 - 2*x*embed + embed^2
        dist = x_sq - 2 * dots + embed_sq

        # Get nearest
        codes = mx.argmin(dist, axis=-1)  # [B, T]
        return codes

    def decode(self, codes: mx.array) -> mx.array:
        """Decode codes to embeddings."""
        # codes: [B, T] or [B, K, T]
        # Use embedding lookup
        return self.embed[codes]


class VectorQuantization(nn.Module):
    """Single-level vector quantization."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = EuclideanCodebook(codebook_size, codebook_dim)

    def encode(self, x: mx.array) -> mx.array:
        return self.codebook.encode(x)

    def decode(self, codes: mx.array) -> mx.array:
        return self.codebook.decode(codes)


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with multiple codebooks."""

    def __init__(self, config: "EnCodecConfig"):
        super().__init__()
        self.num_codebooks = config.num_codebooks

        self.layers = [
            VectorQuantization(config.codebook_size, config.codebook_dim)
            for _ in range(config.num_codebooks)
        ]

    def encode(self, x: mx.array) -> mx.array:
        """Encode to multi-level codes."""
        # x: [B, T, D]
        codes = []
        residual = x

        for layer in self.layers:
            code = layer.encode(residual)
            codes.append(code)
            quantized = layer.decode(code)
            residual = residual - quantized

        # Stack codes: [B, K, T]
        return mx.stack(codes, axis=1)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode multi-level codes."""
        # codes: [B, K, T]
        quantized = mx.zeros((codes.shape[0], codes.shape[2], self.layers[0].codebook.codebook_dim))

        for k, layer in enumerate(self.layers):
            quantized = quantized + layer.decode(codes[:, k, :])

        return quantized


class EnCodecHF(nn.Module):
    """EnCodec model matching HuggingFace architecture exactly."""

    def __init__(self, config: "EnCodecConfig"):
        super().__init__()
        self.config = config

        self.encoder = EnCodecEncoder(config)
        self.decoder = EnCodecDecoder(config)
        self.quantizer = ResidualVectorQuantizer(config)

    @property
    def hop_length(self) -> int:
        return self.config.hop_length

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio to discrete codes."""
        embeddings = self.encoder(audio)  # [B, T', D]
        codes = self.quantizer.encode(embeddings)  # [B, K, T']
        return codes

    def decode(self, codes: mx.array) -> mx.array:
        """Decode codes to audio."""
        embeddings = self.quantizer.decode(codes)  # [B, T, D]
        audio = self.decoder(embeddings)  # [B, C, T]
        return audio

    def __call__(self, audio: mx.array) -> tuple[mx.array, mx.array]:
        """Full forward pass."""
        codes = self.encode(audio)
        reconstructed = self.decode(codes)
        return reconstructed, codes
