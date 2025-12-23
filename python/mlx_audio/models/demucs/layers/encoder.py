"""Encoder layers for HTDemucs.

Matches PyTorch demucs.hdemucs.HEncLayer exactly.

OPTIMIZATION: Uses MLX-native NHWC/NLC format internally to avoid transposes.
Format conversion happens at model boundaries in model.py.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.demucs.layers.dconv import DConv


class HEncLayer(nn.Module):
    """Hybrid encoder layer for HTDemucs.

    Structure matches PyTorch exactly:
        conv: Conv2d (freq) or Conv1d (time) - main downsampling
        norm1: Identity (not used, kept for weight compatibility)
        rewrite: Conv2d (freq) or Conv1d (time) - GLU rewrite
        norm2: Identity (not used, kept for weight compatibility)
        dconv: DConv - dilated residual block

    INTERNAL FORMAT (optimized for MLX):
        - freq=True: [B, F, T, C] (NHWC format)
        - freq=False: [B, T, C] (NLC format)

    NOTE: model.py converts from PyTorch format (NCHW/NCL) at entry
    and converts back at exit. Layers operate in MLX-native format.
    """

    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int = 8,
        stride: int = 4,
        freq: bool = True,
        dconv_depth: int = 2,
        dconv_compress: int = 8,
        dconv_init: float = 1e-4,
    ):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.kernel_size = kernel_size
        self.stride = stride
        self.freq = freq

        # Padding for "same" output (roughly)
        pad = (kernel_size - stride) // 2

        if freq:
            # Frequency branch uses Conv2d
            self.conv = nn.Conv2d(
                chin, chout,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(pad, 0),
            )
            # Rewrite conv: 1x1 that doubles channels for GLU
            self.rewrite = nn.Conv2d(chout, chout * 2, kernel_size=(1, 1))
        else:
            # Time branch uses Conv1d
            self.conv = nn.Conv1d(
                chin, chout,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
            )
            # Rewrite conv: 1x1 that doubles channels for GLU
            self.rewrite = nn.Conv1d(chout, chout * 2, kernel_size=1)

        # norm1 and norm2 are Identity in PyTorch (no params)
        # We don't need them since Identity has no parameters

        # DConv residual block
        self.dconv = DConv(
            channels=chout,
            depth=dconv_depth,
            compress=dconv_compress,
            init=dconv_init,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        PyTorch order:
            1. conv(x)
            2. gelu(norm1(y)) - norm1 is identity
            3. dconv(y) with reshape for freq
            4. norm2(rewrite(y)) - norm2 is identity
            5. glu(z)

        Args:
            x: Input tensor (NHWC/NLC format for MLX efficiency)
                - freq=True: [B, F, T, C] (NHWC format)
                - freq=False: [B, T, C] (NLC format)

        Returns:
            Downsampled output with same format as input
        """
        if self.freq:
            # x is [B, F, T, C] - already NHWC, no transpose needed!

            # 1. Conv2d (MLX uses NHWC natively)
            x = self.conv(x)

            # 2. GELU (norm1 is identity)
            x = nn.gelu(x)

            # 3. DConv: collapse freq into batch
            # [B, F, T, C] -> [B*F, T, C] - already NLC for DConv!
            B, Fr, T, C = x.shape
            x = x.reshape(B * Fr, T, C)
            x = self.dconv(x)
            x = x.reshape(B, Fr, T, C)

            # 4. Rewrite (NHWC, no transpose)
            x = self.rewrite(x)

            # 5. GLU: split along channel dim (axis=-1 in NHWC)
            a, b = mx.split(x, 2, axis=-1)
            x = a * mx.sigmoid(b)
        else:
            # x is [B, T, C] - already NLC

            # Pad input to be divisible by stride
            le = x.shape[1]  # Time is axis 1 in NLC
            if le % self.stride != 0:
                pad_amount = self.stride - (le % self.stride)
                x = mx.pad(x, [(0, 0), (0, pad_amount), (0, 0)])

            # 1. Conv1d (MLX uses NLC natively)
            x = self.conv(x)

            # 2. GELU
            x = nn.gelu(x)

            # 3. DConv (already NLC)
            x = self.dconv(x)

            # 4. Rewrite (NLC, no transpose)
            x = self.rewrite(x)

            # 5. GLU: split along channel dim (axis=-1 in NLC)
            a, b = mx.split(x, 2, axis=-1)
            x = a * mx.sigmoid(b)

        return x
