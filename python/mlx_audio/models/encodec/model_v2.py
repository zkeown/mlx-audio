"""EnCodec model matching HuggingFace architecture exactly for weight loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.encodec.config import EnCodecConfig


def _reflect_pad_1d(x: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """Apply reflect padding along the time dimension.

    Args:
        x: Input tensor [B, T, C]
        pad_left: Amount of padding on the left
        pad_right: Amount of padding on the right

    Returns:
        Padded tensor
    """
    # x shape: [B, T, C]
    length = x.shape[1]

    # Handle edge case where padding is larger than input
    # HF handles this by adding extra zero padding first
    max_pad = max(pad_left, pad_right)
    if length <= max_pad:
        # Add extra zeros on the right before reflecting
        extra_zeros = max_pad - length + 1
        x = mx.pad(x, [(0, 0), (0, extra_zeros), (0, 0)])
        length = x.shape[1]

    # Build reflected indices
    # For reflect padding, we don't include the edge value twice
    # e.g., for [1,2,3,4,5] with pad_left=2: [3,2,1,2,3,4,5,...]

    if pad_left > 0:
        # Reflect left: take indices [pad_left, pad_left-1, ..., 1]
        left_indices = list(range(pad_left, 0, -1))
        left_part = x[:, left_indices, :]
    else:
        left_part = None

    if pad_right > 0:
        # Reflect right: take indices [-2, -3, ..., -(pad_right+1)]
        right_indices = list(range(length - 2, length - 2 - pad_right, -1))
        right_part = x[:, right_indices, :]
    else:
        right_part = None

    # Concatenate
    parts = []
    if left_part is not None:
        parts.append(left_part)
    parts.append(x)
    if right_part is not None:
        parts.append(right_part)

    return mx.concatenate(parts, axis=1)


class EncodecConv1d(nn.Module):
    """Conv1d matching HuggingFace EncodecConv1d exactly.

    Weight key pattern: layers.{idx}.conv.weight, layers.{idx}.conv.bias
    Or for blocks: layers.{idx}.block.{bidx}.conv.weight

    Key differences from naive implementation:
    1. Computes extra_padding dynamically based on input length
    2. For non-causal: padding_right = padding_total // 2, padding_left = rest
    3. Extra padding is added to right side
    4. Uses reflect padding mode (not zero padding)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.causal = causal
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.pad_mode = pad_mode

        # Effective kernel size with dilation
        self.effective_kernel = (kernel_size - 1) * dilation + 1
        # Total padding needed (before extra padding adjustment)
        self.padding_total = max(0, self.effective_kernel - stride)

        # Conv layer - MLX uses [B, T, C] format
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def _get_extra_padding(self, length: int) -> int:
        """Calculate extra padding needed for proper frame alignment.

        This matches HuggingFace's _get_extra_padding_for_conv1d exactly.
        """
        import math
        # Number of output frames
        n_frames = (length - self.effective_kernel + self.padding_total) / self.stride + 1
        n_frames = math.ceil(n_frames) - 1
        # Ideal input length for this many frames
        ideal_length = n_frames * self.stride + self.effective_kernel - self.padding_total
        return max(0, ideal_length - length)

    def __call__(self, x: mx.array) -> mx.array:
        # Input: [B, C, T] (channel-first like PyTorch)
        length = x.shape[-1]

        # Calculate extra padding for frame alignment
        extra_padding = self._get_extra_padding(length)

        if self.causal:
            # Causal: all padding on left, extra on right
            padding_left = self.padding_total
            padding_right = extra_padding
        else:
            # Non-causal: split padding, extra goes to right
            # HF: padding_right = padding_total // 2
            # HF: padding_left = padding_total - padding_right
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
            padding_right = padding_right + extra_padding

        # MLX Conv1d expects [B, T, C]
        x = x.transpose(0, 2, 1)

        # Apply padding
        if padding_left > 0 or padding_right > 0:
            if self.pad_mode == "reflect":
                x = _reflect_pad_1d(x, padding_left, padding_right)
            else:
                # Zero padding
                x = mx.pad(x, [(0, 0), (padding_left, padding_right), (0, 0)])

        x = self.conv(x)

        # Back to [B, C, T]
        return x.transpose(0, 2, 1)


class EncodecConvTranspose1d(nn.Module):
    """ConvTranspose1d matching HuggingFace EncodecConvTranspose1d.

    HF forward logic:
    1. Apply transposed conv
    2. Calculate padding_total = kernel_size - stride
    3. For causal: padding_right = ceil(padding_total * trim_right_ratio)
       For non-causal: padding_right = padding_total // 2
    4. padding_left = padding_total - padding_right
    5. Slice: hidden_states[..., padding_left:end-padding_right]
    """

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
        self.padding_total = kernel_size - stride

        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 1)  # [B, C, T] -> [B, T, C]
        x = self.conv(x)
        x = x.transpose(0, 2, 1)  # [B, T, C] -> [B, C, T]

        # Trim to correct length (matching HF exactly)
        if self.padding_total > 0:
            if self.causal:
                # For causal: trim all from right
                # HF uses ceil(padding_total * 1.0) = padding_total
                padding_right = self.padding_total
                padding_left = 0
            else:
                # For non-causal: split asymmetrically
                # HF: padding_right = padding_total // 2
                # HF: padding_left = padding_total - padding_right
                padding_right = self.padding_total // 2
                padding_left = self.padding_total - padding_right

            # Slice: x[..., padding_left:-padding_right] or x[..., padding_left:]
            if padding_right > 0:
                x = x[:, :, padding_left:-padding_right]
            else:
                x = x[:, :, padding_left:]

        return x


class EncodecResnetBlock(nn.Module):
    """Residual block matching HuggingFace EncodecResnetBlock.

    Structure: block = [ELU, Conv, ELU, Conv]
    block.0: ELU (not a module param)
    block.1: EncodecConv1d (dilated) - bottleneck (dim -> dim/2)
    block.2: ELU (not a module param)
    block.3: EncodecConv1d (1x1) - expand back (dim/2 -> dim)

    HF uses compress=2 by default which creates a bottleneck structure.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tuple[int, int] = (3, 1),
        dilations: tuple[int, int] = (1, 1),
        causal: bool = True,
        compress: int = 2,
    ):
        super().__init__()

        # HF uses a bottleneck: dim -> dim/compress -> dim
        hidden = dim // compress

        # Match HF structure exactly with a "block" attribute
        # block.1 and block.3 are the conv layers
        self.block = [
            None,  # ELU (index 0)
            EncodecConv1d(dim, hidden, kernel_sizes[0], dilation=dilations[0], causal=causal),
            None,  # ELU (index 2)
            EncodecConv1d(hidden, dim, kernel_sizes[1], dilation=dilations[1], causal=causal),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        # block[0]: ELU
        x = nn.elu(x)
        # block[1]: Conv (dilated, bottleneck)
        x = self.block[1](x)
        # block[2]: ELU
        x = nn.elu(x)
        # block[3]: Conv (1x1, expand)
        x = self.block[3](x)
        return x + residual


class EncodecLSTM(nn.Module):
    """LSTM matching HuggingFace EncodecLSTM.

    HF has 2-layer LSTM. Weight keys are:
    - lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0
    - lstm.weight_ih_l1, lstm.weight_hh_l1, lstm.bias_ih_l1, lstm.bias_hh_l1

    We use a list of LSTMs to match weights:
    - lstm.0.Wx, lstm.0.Wh, lstm.0.bias
    - lstm.1.Wx, lstm.1.Wh, lstm.1.bias

    IMPORTANT: HF EncodecLSTM has a residual connection:
    output = lstm(x) + x
    """

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers

        # Create a list of LSTM layers to match HF weight structure
        # This creates lstm.0, lstm.1, etc.
        self.lstm = [nn.LSTM(dim, dim) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, T] -> [B, T, C] for LSTM
        x = x.transpose(0, 2, 1)

        # Save input for residual connection
        residual = x

        # Run through each LSTM layer
        for layer in self.lstm:
            x, _ = layer(x)

        # HF adds residual connection: output = lstm_out + input
        x = x + residual

        return x.transpose(0, 2, 1)


class EncodecEuclideanCodebook(nn.Module):
    """Codebook matching HuggingFace EncodecEuclideanCodebook.

    Weight keys: quantizer.layers.{idx}.codebook.embed, cluster_size, embed_avg, inited
    """

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Main embedding - this is what we need for inference
        self.embed = mx.zeros((codebook_size, codebook_dim))

        # These are buffers used during training, needed for weight loading
        self.cluster_size = mx.zeros((codebook_size,))
        self.embed_avg = mx.zeros((codebook_size, codebook_dim))
        self.inited = mx.array([True])

    def encode(self, x: mx.array) -> mx.array:
        """Find nearest codebook entries."""
        # x: [B, T, D]
        # Compute L2 distance to all codebook entries

        # ||x - e||^2 = ||x||^2 - 2<x,e> + ||e||^2
        x_sq = mx.sum(x ** 2, axis=-1, keepdims=True)  # [B, T, 1]
        embed_sq = mx.sum(self.embed ** 2, axis=-1)  # [K]
        dots = mx.matmul(x, self.embed.T)  # [B, T, K]

        dist = x_sq - 2 * dots + embed_sq  # [B, T, K]
        return mx.argmin(dist, axis=-1)  # [B, T]

    def decode(self, codes: mx.array) -> mx.array:
        """Look up embeddings."""
        return self.embed[codes]


class EncodecVectorQuantization(nn.Module):
    """Single VQ layer matching HuggingFace."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = EncodecEuclideanCodebook(codebook_size, codebook_dim)

    def encode(self, x: mx.array) -> mx.array:
        return self.codebook.encode(x)

    def decode(self, codes: mx.array) -> mx.array:
        return self.codebook.decode(codes)


class EncodecResidualVectorQuantizer(nn.Module):
    """RVQ matching HuggingFace structure.

    Weight keys: quantizer.layers.{idx}.codebook.{attr}
    """

    def __init__(self, num_codebooks: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks

        self.layers = [
            EncodecVectorQuantization(codebook_size, codebook_dim)
            for _ in range(num_codebooks)
        ]

    def encode(self, x: mx.array) -> mx.array:
        """Encode to multi-level codes."""
        codes = []
        residual = x

        for layer in self.layers:
            code = layer.encode(residual)
            codes.append(code)
            quantized = layer.decode(code)
            residual = residual - quantized

        return mx.stack(codes, axis=1)  # [B, K, T]

    def decode(self, codes: mx.array) -> mx.array:
        """Decode multi-level codes."""
        quantized = None

        for k, layer in enumerate(self.layers):
            q = layer.decode(codes[:, k, :])
            quantized = q if quantized is None else quantized + q

        return quantized

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize input embeddings.

        Args:
            x: Input embeddings [B, T, D]

        Returns:
            Tuple of (quantized embeddings, codes)
        """
        codes = self.encode(x)
        quantized = self.decode(codes)
        return quantized, codes


class EncodecEncoder(nn.Module):
    """Encoder matching HuggingFace exactly.

    Layer structure for 32kHz:
    0: EncodecConv1d (initial)
    1: EncodecResnetBlock
    2: ELU
    3: EncodecConv1d (downsample)
    4: EncodecResnetBlock
    5: ELU
    6: EncodecConv1d (downsample)
    ... pattern repeats ...
    13: EncodecLSTM
    14: ELU
    15: EncodecConv1d (final)
    """

    def __init__(self, config: EnCodecConfig):
        super().__init__()
        self.config = config

        self.layers = []

        # Layer 0: Initial conv
        self.layers.append(
            EncodecConv1d(
                config.channels,
                config.num_filters,
                config.kernel_size,
                causal=config.causal,
            )
        )

        current_channels = config.num_filters

        # For each ratio: ResnetBlock, ELU, DownsampleConv
        # Note: ratios are stored as upsampling order [8,5,4,4], but encoder
        # uses them in reversed order [4,4,5,8] for downsampling
        for _i, ratio in enumerate(reversed(config.ratios)):
            out_channels = current_channels * 2

            # ResnetBlock
            for j in range(config.num_residual_layers):
                dilation = config.dilation_base ** j
                self.layers.append(
                    EncodecResnetBlock(
                        current_channels,
                        kernel_sizes=(config.residual_kernel_size, 1),
                        dilations=(dilation, 1),
                        causal=config.causal,
                    )
                )

            # ELU (placeholder - we'll handle in forward)
            self.layers.append(None)  # ELU marker

            # Downsample conv
            self.layers.append(
                EncodecConv1d(
                    current_channels,
                    out_channels,
                    ratio * 2,  # kernel = 2 * stride
                    stride=ratio,
                    causal=config.causal,
                )
            )

            current_channels = out_channels

        # LSTM
        self.layers.append(EncodecLSTM(current_channels, config.lstm_layers))

        # ELU
        self.layers.append(None)

        # Final conv
        self.layers.append(
            EncodecConv1d(
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

        for _i, layer in enumerate(self.layers):
            if layer is None:
                # ELU activation
                x = nn.elu(x)
            elif isinstance(layer, EncodecResnetBlock):
                # ResnetBlock (has internal activations)
                x = layer(x)
            elif isinstance(layer, EncodecLSTM):
                # LSTM (no activation after)
                x = layer(x)
            else:
                # Conv layer - apply ELU after
                x = layer(x)
                # ELU is applied via the None markers in the layer list

        # Return [B, T, D] for quantizer
        return x.transpose(0, 2, 1)


class EncodecDecoder(nn.Module):
    """Decoder matching HuggingFace exactly.

    Layer structure for 32kHz:
    0: EncodecConv1d (initial)
    1: EncodecLSTM
    2: ELU
    3: EncodecConvTranspose1d (upsample)
    4: EncodecResnetBlock
    5: ELU
    6: EncodecConvTranspose1d (upsample)
    ... pattern repeats ...
    15: EncodecConv1d (final)
    """

    def __init__(self, config: EnCodecConfig):
        super().__init__()
        self.config = config

        mult = 2 ** len(config.ratios)
        hidden_channels = config.num_filters * mult

        self.layers = []

        # Layer 0: Initial conv
        self.layers.append(
            EncodecConv1d(
                config.codebook_dim,
                hidden_channels,
                config.kernel_size,
                causal=config.causal,
            )
        )

        # LSTM
        self.layers.append(EncodecLSTM(hidden_channels, config.lstm_layers))

        # ELU
        self.layers.append(None)

        current_channels = hidden_channels

        # For each ratio: UpsampleConv, ResnetBlock, ELU
        # Decoder uses ratios in forward order [8,5,4,4] for upsampling
        for _i, ratio in enumerate(config.ratios):
            out_channels = current_channels // 2

            # Upsample conv
            self.layers.append(
                EncodecConvTranspose1d(
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
                    EncodecResnetBlock(
                        out_channels,
                        kernel_sizes=(config.residual_kernel_size, 1),
                        dilations=(dilation, 1),
                        causal=config.causal,
                    )
                )

            # ELU
            self.layers.append(None)

            current_channels = out_channels

        # Final conv
        self.layers.append(
            EncodecConv1d(
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
            if layer is None:
                x = nn.elu(x)
            elif isinstance(layer, (EncodecResnetBlock, EncodecLSTM)):
                x = layer(x)
            else:
                x = layer(x)

        return x


class EnCodecV2(nn.Module):
    """EnCodec model matching HuggingFace architecture for weight loading."""

    def __init__(self, config: EnCodecConfig):
        super().__init__()
        self.config = config

        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)
        self.quantizer = EncodecResidualVectorQuantizer(
            config.num_codebooks,
            config.codebook_size,
            config.codebook_dim,
        )

    @property
    def hop_length(self) -> int:
        return self.config.hop_length

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

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
        """Full forward pass: encode then decode."""
        codes = self.encode(audio)
        reconstructed = self.decode(codes)
        return reconstructed, codes

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> EnCodecV2:
        """Load model from pretrained weights."""
        from mlx_audio.models.encodec.config import EnCodecConfig

        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = EnCodecConfig.from_dict(config_dict)
        else:
            config = EnCodecConfig()

        # Create model
        model = cls(config, **kwargs)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            model.load_weights(str(weights_path))

        return model
