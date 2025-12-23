"""Patch embedding for HTSAT/Swin Transformer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.clap.config import CLAPAudioConfig


class AFFBlock(nn.Module):
    """Attentional Feature Fusion block.

    Fuses global and local features using learned attention weights.

    Args:
        channels: Number of input/output channels
        reduction: Reduction ratio for attention (default 4)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = channels // reduction

        # Local attention branch: Conv1x1 -> BN -> ReLU -> Conv1x1 -> BN
        self.local_att = [
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.BatchNorm(reduced),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.BatchNorm(channels),
        ]

        # Global attention branch: AvgPool -> Conv1x1 -> BN -> ReLU -> Conv1x1 -> BN
        # Note: We'll apply global avg pool in forward
        self.global_att = [
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.BatchNorm(reduced),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.BatchNorm(channels),
        ]

    def __call__(self, hidden_states: mx.array, residual: mx.array) -> mx.array:
        """Fuse hidden_states with residual using attention.

        Args:
            hidden_states: Global features [B, H, W, C]
            residual: Local features [B, H, W, C]

        Returns:
            Fused features [B, H, W, C]
        """
        attention_input = hidden_states + residual

        # Local attention
        local_out = attention_input
        for layer in self.local_att:
            local_out = layer(local_out)

        # Global attention (with global average pooling)
        global_out = mx.mean(attention_input, axis=(1, 2), keepdims=True)
        for layer in self.global_att:
            global_out = layer(global_out)

        # Combine with sigmoid
        fused_weight = mx.sigmoid(local_out + global_out)

        # Weighted fusion
        output = 2 * hidden_states * fused_weight + 2 * residual * (1 - fused_weight)
        return output


class PatchEmbed(nn.Module):
    """Patch embedding layer for audio spectrograms with optional fusion.

    Converts mel spectrogram [B, 1, F, T] to patch tokens [B, H*W, C].
    Uses Conv2d with kernel_size=patch_size and stride=patch_stride.

    When enable_fusion=True, handles multi-channel input [B, 4, F, T] where:
    - Channel 0: Global mel spectrogram
    - Channels 1-3: Local mel patches for longer audio

    Args:
        patch_size: Size of each patch (height and width)
        patch_stride: Stride for patch extraction
        in_chans: Number of input channels (1 for mel spectrogram)
        embed_dim: Output embedding dimension
        flatten: Whether to flatten spatial dimensions
        enable_fusion: Whether to enable fusion for variable-length audio
        img_size: Expected input size (F, T) for validation
    """

    def __init__(
        self,
        patch_size: int = 4,
        patch_stride: int | tuple[int, int] = 4,
        in_chans: int = 1,
        embed_dim: int = 96,
        flatten: bool = True,
        enable_fusion: bool = False,
        img_size: tuple[int, int] = (64, 1024),
    ):
        super().__init__()
        self.patch_size = patch_size
        if isinstance(patch_stride, int):
            self.patch_stride = (patch_stride, patch_stride)
        else:
            self.patch_stride = patch_stride
        self.flatten = flatten
        self.enable_fusion = enable_fusion
        self.img_size = img_size

        # Main projection for global features
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=self.patch_stride,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Fusion components
        if enable_fusion:
            # Conv for local mel patches (3 channels -> embed_dim)
            # Kernel: [patch_size, patch_size * 3] = [4, 12]
            # Stride: [patch_stride, patch_stride * 3] = [4, 12]
            # This matches HuggingFace's mel_conv2d configuration
            self.mel_conv2d = nn.Conv2d(
                1,
                embed_dim,
                kernel_size=(patch_size, patch_size * 3),
                stride=(self.patch_stride[0], self.patch_stride[1] * 3),
            )
            # AFF block for fusing global and local
            self.fusion_model = AFFBlock(embed_dim)

    def __call__(
        self,
        x: mx.array,
        is_longer_idx: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W] (mel spectrogram)
               If enable_fusion, expects [B, 4, H, W] with first channel as global
            is_longer_idx: Indices of samples that need fusion (have local features)

        Returns:
            Patch embeddings [B, N, embed_dim] if flatten else [B, H', W', embed_dim]
        """
        if self.enable_fusion:
            # Extract global features (first channel)
            global_x = x[:, 0:1, :, :]  # [B, 1, H, W]

            # Conv2d expects [B, H, W, C] in MLX
            global_x = mx.transpose(global_x, (0, 2, 3, 1))  # [B, H, W, 1]
            global_out = self.proj(global_x)  # [B, H', W', embed_dim]
            output_width = global_out.shape[2]

            # Handle fusion for longer samples
            if is_longer_idx is not None and len(is_longer_idx) > 0:
                # Extract local features (channels 1-3) for longer samples
                local_x = x[is_longer_idx, 1:, :, :]  # [N, 3, H, W]
                B_local, num_channels, H, W = local_x.shape

                # Reshape to process each channel independently
                local_x = mx.reshape(local_x, (B_local * num_channels, 1, H, W))
                local_x = mx.transpose(local_x, (0, 2, 3, 1))  # [N*3, H, W, 1]

                # Apply mel conv
                local_out = self.mel_conv2d(local_x)  # [N*3, H', W', embed_dim]
                _, local_H, local_W, embed_dim = local_out.shape

                # Reshape back: [N, 3, H', W', embed_dim] -> [N, H', 3, W', embed_dim]
                local_out = mx.reshape(local_out, (B_local, num_channels, local_H, local_W, embed_dim))
                local_out = mx.transpose(local_out, (0, 2, 1, 3, 4))  # [N, H', 3, W', embed_dim]

                # Flatten the 3 channels into width dimension
                local_out = mx.reshape(local_out, (B_local, local_H, -1, embed_dim))  # [N, H', 3*W', embed_dim]

                local_width = local_out.shape[2]

                # Match widths between local and global features
                if local_width < output_width:
                    # Pad local to match global
                    pad_width = output_width - local_width
                    local_out = mx.pad(local_out, [(0, 0), (0, 0), (0, pad_width), (0, 0)])
                elif local_width > output_width:
                    # Crop local to match global
                    local_out = local_out[:, :, :output_width, :]

                # Apply fusion
                global_out_subset = global_out[is_longer_idx]
                fused = self.fusion_model(global_out_subset, local_out)

                # Update global_out with fused results
                # Check if all samples need fusion
                B = global_out.shape[0]
                if len(is_longer_idx) == B:
                    # All samples get fused - just use fused directly
                    global_out = fused
                else:
                    # Partial fusion - need to scatter fused values back
                    # Use a loop since MLX doesn't have scatter/index_put
                    import numpy as np
                    global_out_np = np.array(global_out)
                    fused_np = np.array(fused)
                    idx_np = np.array(is_longer_idx)
                    global_out_np[idx_np] = fused_np
                    global_out = mx.array(global_out_np)

            x = global_out
        else:
            # Standard path without fusion
            # Conv2d expects [B, H, W, C] in MLX
            x = mx.transpose(x, (0, 2, 3, 1))  # [B, H, W, C]
            x = self.proj(x)  # [B, H', W', embed_dim]

        if self.flatten:
            B, H, W, C = x.shape
            x = mx.reshape(x, (B, H * W, C))  # [B, N, C]

        x = self.norm(x)
        return x
