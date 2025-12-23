"""Decoder layers for HTDemucs.

Matches PyTorch demucs.hdemucs.HDecLayer exactly.

OPTIMIZATION: Uses MLX-native NHWC/NLC format internally to avoid transposes.
Format conversion happens at model boundaries in model.py.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.demucs.layers.dconv import DConv


class HDecLayer(nn.Module):
    """Hybrid decoder layer for HTDemucs.

    Structure matches PyTorch exactly:
        conv_tr: ConvTranspose2d (freq) or ConvTranspose1d (time) - upsampling
        norm2: Identity (not used)
        rewrite: Conv2d (freq) or Conv1d (time) - GLU rewrite (kernel 3)
        norm1: Identity (not used)
        dconv: DConv - dilated residual block

    INTERNAL FORMAT (optimized for MLX):
        - freq=True: [B, F, T, C] (NHWC format)
        - freq=False: [B, T, C] (NLC format)

    NOTE: model.py converts from PyTorch format (NCHW/NCL) at entry
    and converts back at exit. Layers operate in MLX-native format.

    IMPORTANT: The decoder uses length-based trimming to ensure output matches
    the expected encoder input length. PyTorch stores lengths before each
    encoder pass and uses them to trim decoder output.
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
        last: bool = False,
    ):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.kernel_size = kernel_size
        self.stride = stride
        self.freq = freq
        self.last = last

        # PyTorch uses pad = kernel_size // 4 for output trimming
        # This is different from the transposed conv padding!
        self.pad = kernel_size // 4

        # Padding for transposed conv (no padding, we handle it via trimming)
        # PyTorch ConvTranspose uses no padding, then trims output
        if freq:
            # Frequency branch uses ConvTranspose2d
            self.conv_tr = nn.ConvTranspose2d(
                chin, chout,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(0, 0),
            )
            # Rewrite conv: 3x3 that doubles channels for GLU
            self.rewrite = nn.Conv2d(
                chin, chin * 2, kernel_size=(3, 3), padding=1
            )
        else:
            # Time branch uses ConvTranspose1d
            self.conv_tr = nn.ConvTranspose1d(
                chin, chout,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,  # No padding, we trim after
            )
            # Rewrite conv: kernel 3 that doubles channels for GLU
            self.rewrite = nn.Conv1d(chin, chin * 2, kernel_size=3, padding=1)

        # norm1 and norm2 are Identity in PyTorch (no params)

        # DConv residual block (operates on chin, before upsampling)
        self.dconv = DConv(
            channels=chin,
            depth=dconv_depth,
            compress=dconv_compress,
            init=dconv_init,
        )

    def __call__(
        self, x: mx.array, skip: mx.array | None, length: int
    ) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input tensor from previous decoder layer (NHWC/NLC format)
                - freq=True: [B, F, T, C] (NHWC format)
                - freq=False: [B, T, C] (NLC format)
            skip: Skip connection from corresponding encoder layer
            length: Expected output time dimension length (from encoder input)

        Returns:
            Tuple of (output, pre_output):
                - output: Upsampled and trimmed output
                - pre_output: Output before transposed conv (for branch merge)
        """
        # PyTorch order:
        # 1. x = x + skip
        # 2. y = glu(norm1(rewrite(x))) - norm1 is identity
        # 3. dconv(y) with reshape for freq
        # 4. z = norm2(conv_tr(y)) - norm2 is identity
        # 5. Trim z to expected length
        # 6. gelu(z) if not last

        if self.freq:
            # x is [B, F, T, C] - NHWC format

            # 1. Add skip connection
            if skip is not None:
                # Handle length mismatch by trimming to shorter length
                # In NHWC: F is axis 1
                if x.shape[1] != skip.shape[1]:
                    min_f = min(x.shape[1], skip.shape[1])
                    x = x[:, :min_f, :, :]
                    skip = skip[:, :min_f, :, :]
                x = x + skip

            B, Fr, T, C = x.shape

            # 2. Rewrite -> GLU (no transpose needed, already NHWC)
            x = self.rewrite(x)
            a, b = mx.split(x, 2, axis=-1)
            y = a * mx.sigmoid(b)

            # 3. DConv: collapse freq into batch
            # [B, F, T, C] -> [B*F, T, C] - already NLC!
            _, Fr2, T2, C2 = y.shape
            y = y.reshape(B * Fr2, T2, C2)
            y = self.dconv(y)
            y = y.reshape(B, Fr2, T2, C2)

            # Save pre-output for branch merge
            pre = y

            # 4. ConvTranspose2d (no transpose, already NHWC)
            z = self.conv_tr(y)

            # 5. Trim: z[:, pad:-pad, :, :] for freq (F is axis 1 in NHWC)
            if self.pad > 0:
                z = z[:, self.pad:-self.pad, :, :]

            # 6. GELU if not last
            if not self.last:
                z = nn.gelu(z)
        else:
            # x is [B, T, C] - NLC format

            # 1. Add skip connection
            if skip is not None:
                # Handle length mismatch (T is axis 1 in NLC)
                if x.shape[1] != skip.shape[1]:
                    min_t = min(x.shape[1], skip.shape[1])
                    x = x[:, :min_t, :]
                    skip = skip[:, :min_t, :]
                x = x + skip

            # 2. Rewrite -> GLU (no transpose, already NLC)
            x = self.rewrite(x)
            a, b = mx.split(x, 2, axis=-1)
            y = a * mx.sigmoid(b)

            # 3. DConv (already NLC)
            y = self.dconv(y)

            # Save pre-output for branch merge
            pre = y

            # 4. ConvTranspose1d (no transpose, already NLC)
            z = self.conv_tr(y)

            # 5. Trim: z[:, pad:pad+length, :] for time (T is axis 1 in NLC)
            z = z[:, self.pad:self.pad + length, :]

            # 6. GELU if not last
            if not self.last:
                z = nn.gelu(z)

        return z, pre
