"""Encoder layers for HTDemucs.

Matches PyTorch demucs.hdemucs.HEncLayer exactly.
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

    For frequency branch (freq=True):
        - Uses Conv2d with kernel (K, 1), stride (S, 1)
        - Input: [B, C, F, T] (batch, channels, freq bins, time frames)

    For time branch (freq=False):
        - Uses Conv1d with kernel K, stride S
        - Input: [B, C, T] (batch, channels, time samples)
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
            x: Input tensor
                - freq=True: [B, C, F, T] (NCHW format)
                - freq=False: [B, C, T] (NCL format)

        Returns:
            Downsampled output with same format as input
        """
        if self.freq:
            # 1. Conv: NCHW -> NHWC for MLX Conv2d
            x = x.transpose(0, 2, 3, 1)  # [B, F, T, C]
            x = self.conv(x)
            x = x.transpose(0, 3, 1, 2)  # back to [B, C, F, T]

            # 2. GELU (norm1 is identity)
            x = nn.gelu(x)

            # 3. DConv: collapse freq into batch, keep C channels
            B, C, Fr, T = x.shape
            x = x.transpose(0, 2, 1, 3)  # [B, Fr, C, T]
            x = x.reshape(B * Fr, C, T)  # [B*Fr, C, T]
            x = x.transpose(0, 2, 1)  # NCL -> NLC for MLX
            x = self.dconv(x)
            x = x.transpose(0, 2, 1)  # NLC -> NCL
            x = x.reshape(B, Fr, C, T)
            x = x.transpose(0, 2, 1, 3)  # [B, C, Fr, T]

            # 4. Rewrite: NCHW -> NHWC -> NCHW
            x = x.transpose(0, 2, 3, 1)
            x = self.rewrite(x)
            x = x.transpose(0, 3, 1, 2)

            # 5. GLU: split along channel dim (axis=1 in NCHW)
            a, b = mx.split(x, 2, axis=1)
            x = a * mx.sigmoid(b)
        else:
            # PyTorch pads input to be divisible by stride
            le = x.shape[-1]
            if le % self.stride != 0:
                pad_amount = self.stride - (le % self.stride)
                x = mx.pad(x, [(0, 0), (0, 0), (0, pad_amount)])

            # 1. Conv: NCL -> NLC for MLX
            x = x.transpose(0, 2, 1)  # [B, T, C]
            x = self.conv(x)
            x = x.transpose(0, 2, 1)  # [B, C, T]

            # 2. GELU
            x = nn.gelu(x)

            # 3. DConv
            x = x.transpose(0, 2, 1)
            x = self.dconv(x)
            x = x.transpose(0, 2, 1)

            # 4. Rewrite
            x = x.transpose(0, 2, 1)
            x = self.rewrite(x)
            x = x.transpose(0, 2, 1)

            # 5. GLU
            a, b = mx.split(x, 2, axis=1)
            x = a * mx.sigmoid(b)

        return x
