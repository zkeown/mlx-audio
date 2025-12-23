"""Decoder layers for HTDemucs.

Matches PyTorch demucs.hdemucs.HDecLayer exactly.
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

    For frequency branch (freq=True):
        - Uses ConvTranspose2d with kernel (K, 1), stride (S, 1)
        - Input: [B, C, F, T]

    For time branch (freq=False):
        - Uses ConvTranspose1d with kernel K, stride S
        - Input: [B, C, T]

    IMPORTANT: The decoder uses length-based trimming to ensure output matches
    the expected encoder input length. PyTorch stores lengths before each encoder
    pass and uses them to trim decoder output.
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
                padding=(0, 0),  # No padding, we trim after
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
            x: Input tensor from previous decoder layer
                - freq=True: [B, C, F, T] (NCHW format)
                - freq=False: [B, C, T] (NCL format)
            skip: Skip connection from corresponding encoder layer
            length: Expected output time dimension length (from encoder input)

        Returns:
            Tuple of (output, pre_output):
                - output: Upsampled and trimmed output
                - pre_output: Output before transposed conv (used for branch merge)
        """
        # PyTorch order:
        # 1. x = x + skip
        # 2. y = glu(norm1(rewrite(x))) - norm1 is identity
        # 3. dconv(y) with reshape for freq
        # 4. z = norm2(conv_tr(y)) - norm2 is identity
        # 5. Trim z to expected length
        # 6. gelu(z) if not last

        # 1. Add skip connection
        if skip is not None:
            # Handle length mismatch by trimming to shorter length
            if self.freq:
                if x.shape[2] != skip.shape[2]:
                    min_f = min(x.shape[2], skip.shape[2])
                    x = x[:, :, :min_f, :]
                    skip = skip[:, :, :min_f, :]
            else:
                if x.shape[2] != skip.shape[2]:
                    min_t = min(x.shape[2], skip.shape[2])
                    x = x[:, :, :min_t]
                    skip = skip[:, :, :min_t]
            x = x + skip

        if self.freq:
            B, C, Fr, T = x.shape

            # 2. Rewrite -> GLU: NCHW -> NHWC -> NCHW
            x = x.transpose(0, 2, 3, 1)
            x = self.rewrite(x)
            x = x.transpose(0, 3, 1, 2)
            a, b = mx.split(x, 2, axis=1)
            y = a * mx.sigmoid(b)

            # 3. DConv: collapse freq into batch
            B, C, Fr, T = y.shape
            y = y.transpose(0, 2, 1, 3)  # [B, Fr, C, T]
            y = y.reshape(B * Fr, C, T)  # [B*Fr, C, T]
            y = y.transpose(0, 2, 1)  # NCL -> NLC
            y = self.dconv(y)
            y = y.transpose(0, 2, 1)  # NLC -> NCL
            y = y.reshape(B, Fr, C, T)
            y = y.transpose(0, 2, 1, 3)  # [B, C, Fr, T]

            # Save pre-output for branch merge
            pre = y

            # 4. ConvTranspose2d: NCHW -> NHWC -> NCHW
            y = y.transpose(0, 2, 3, 1)
            z = self.conv_tr(y)
            z = z.transpose(0, 3, 1, 2)

            # 5. Trim: z[..., pad:-pad, :] for freq
            # PyTorch uses self.pad on both sides of freq dimension
            if self.pad > 0:
                z = z[:, :, self.pad:-self.pad, :]

            # 6. GELU if not last
            if not self.last:
                z = nn.gelu(z)
        else:
            # 2. Rewrite -> GLU
            x = x.transpose(0, 2, 1)  # NCL -> NLC
            x = self.rewrite(x)
            x = x.transpose(0, 2, 1)  # NLC -> NCL
            a, b = mx.split(x, 2, axis=1)
            y = a * mx.sigmoid(b)

            # 3. DConv
            y = y.transpose(0, 2, 1)
            y = self.dconv(y)
            y = y.transpose(0, 2, 1)

            # Save pre-output for branch merge
            pre = y

            # 4. ConvTranspose1d
            y = y.transpose(0, 2, 1)
            z = self.conv_tr(y)
            z = z.transpose(0, 2, 1)

            # 5. Trim: z[..., pad:pad+length] for time
            # PyTorch trims to exact expected length
            z = z[:, :, self.pad:self.pad + length]

            # 6. GELU if not last
            if not self.last:
                z = nn.gelu(z)

        return z, pre
