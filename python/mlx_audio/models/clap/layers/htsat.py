"""HTSAT (Hierarchical Token-Semantic Audio Transformer) audio encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.clap.layers.patch_embed import PatchEmbed
from mlx_audio.models.clap.layers.swin_block import BasicLayer

if TYPE_CHECKING:
    from mlx_audio.models.clap.config import CLAPAudioConfig


def _cubic_interp_weights(t: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute cubic interpolation weights for keys kernel (a=-0.5).

    PyTorch's bicubic uses the Keys kernel with a=-0.5:
        w0 = a*t^3 - 2*a*t^2 + a*t
        w1 = (a+2)*t^3 - (a+3)*t^2 + 1
        w2 = -(a+2)*t^3 + (2a+3)*t^2 - a*t
        w3 = -a*t^3 + a*t^2
    where t is the fractional distance from pixel 1.
    """
    a = -0.5
    t2 = t * t
    t3 = t2 * t

    w0 = a * t3 - 2 * a * t2 + a * t
    w1 = (a + 2) * t3 - (a + 3) * t2 + 1
    w2 = -(a + 2) * t3 + (2 * a + 3) * t2 - a * t
    w3 = -a * t3 + a * t2

    return w0, w1, w2, w3


def interpolate_2d(
    x: mx.array,
    target_shape: tuple[int, int],
    mode: str = "bicubic",
    align_corners: bool = True,
) -> mx.array:
    """Interpolate 2D tensor to target shape.

    Args:
        x: Input tensor [B, C, H, W]
        target_shape: Target (H, W)
        mode: Interpolation mode ("bicubic" or "bilinear")
        align_corners: If True, corner pixels are aligned (matching PyTorch)

    Returns:
        Interpolated tensor [B, C, target_H, target_W]
    """
    B, C, H, W = x.shape
    target_H, target_W = target_shape

    if target_H == H and target_W == W:
        return x

    # Create coordinate mapping with align_corners=True
    # This maps output corners exactly to input corners
    if align_corners:
        y_coords = mx.linspace(0, H - 1, target_H) if target_H > 1 else mx.array([0.0])
        x_coords = mx.linspace(0, W - 1, target_W) if target_W > 1 else mx.array([0.0])
    else:
        # Scale factor approach (not used when align_corners=True)
        scale_h = H / target_H
        scale_w = W / target_W
        y_coords = (mx.arange(target_H) + 0.5) * scale_h - 0.5
        x_coords = (mx.arange(target_W) + 0.5) * scale_w - 0.5

    if mode == "bilinear":
        # Bilinear interpolation
        y0 = mx.floor(y_coords).astype(mx.int32)
        y1 = mx.minimum(y0 + 1, H - 1)
        x0 = mx.floor(x_coords).astype(mx.int32)
        x1 = mx.minimum(x0 + 1, W - 1)

        wy = y_coords - y0.astype(mx.float32)
        wx = x_coords - x0.astype(mx.float32)

        wy = wy.reshape(1, 1, target_H, 1)
        wx = wx.reshape(1, 1, 1, target_W)

        val_00 = x[:, :, y0, :][:, :, :, x0]
        val_01 = x[:, :, y0, :][:, :, :, x1]
        val_10 = x[:, :, y1, :][:, :, :, x0]
        val_11 = x[:, :, y1, :][:, :, :, x1]

        result = (
            val_00 * (1 - wy) * (1 - wx)
            + val_01 * (1 - wy) * wx
            + val_10 * wy * (1 - wx)
            + val_11 * wy * wx
        )
    else:
        # Bicubic interpolation using Keys kernel (vectorized)
        # For each target pixel, we sample 4x4 source pixels
        y_floor = mx.floor(y_coords).astype(mx.int32)
        x_floor = mx.floor(x_coords).astype(mx.int32)

        ty = y_coords - y_floor.astype(mx.float32)
        tx = x_coords - x_floor.astype(mx.float32)

        # Vectorized index generation: offsets broadcast with floor values
        offsets = mx.array([-1, 0, 1, 2])  # [4]
        # y_floor: [target_H], offsets: [4] -> y_indices: [4, target_H]
        y_indices = mx.clip(y_floor[None, :] + offsets[:, None], 0, H - 1)
        # x_floor: [target_W], offsets: [4] -> x_indices: [4, target_W]
        x_indices = mx.clip(x_floor[None, :] + offsets[:, None], 0, W - 1)

        # Compute cubic weights and stack into tensors
        wy = _cubic_interp_weights(ty)  # tuple of 4 arrays, each [target_H]
        wx = _cubic_interp_weights(tx)  # tuple of 4 arrays, each [target_W]
        wy_stacked = mx.stack(wy, axis=0)  # [4, target_H]
        wx_stacked = mx.stack(wx, axis=0)  # [4, target_W]

        # Compute outer product of weights: [4, 4, target_H, target_W]
        # wy: [4, target_H, 1] * wx: [4, 1, target_W] via broadcasting
        weight_grid = wy_stacked[:, None, :, None] * wx_stacked[None, :, None, :]

        # Gather all 16 sample positions using advanced indexing
        # x shape: [B, C, H, W]
        # We need samples at all combinations of y_indices[i] and x_indices[j]
        # Result shape: [B, C, 4, 4, target_H, target_W]
        #
        # For each (i, j) pair, we gather x[:, :, y_indices[i], x_indices[j]]
        # Using reshape + take for efficient gathering:
        #   - First gather along H with y_indices: [B, C, 4, target_H, W]
        #   - Then gather along W with x_indices: [B, C, 4, target_H, 4, target_W]
        #   - Transpose to [B, C, 4, 4, target_H, target_W]

        # Step 1: Gather along H dimension for all 4 y offsets
        # x[:, :, y_indices, :] where y_indices is [4, target_H]
        # Use take to gather: flatten y_indices and reshape result
        y_flat = y_indices.flatten()  # [4 * target_H]
        gathered_y = mx.take(x, y_flat, axis=2)  # [B, C, 4*target_H, W]
        gathered_y = gathered_y.reshape(B, C, 4, target_H, W)  # [B, C, 4, target_H, W]

        # Step 2: Gather along W dimension for all 4 x offsets
        # For each of the 4 y-positions, gather 4 x-positions
        x_flat = x_indices.flatten()  # [4 * target_W]
        # gathered_y: [B, C, 4, target_H, W] -> take along axis=4
        gathered_xy = mx.take(gathered_y, x_flat, axis=4)  # [B, C, 4, target_H, 4*target_W]
        gathered_xy = gathered_xy.reshape(B, C, 4, target_H, 4, target_W)  # [B, C, 4, target_H, 4, target_W]

        # Transpose to [B, C, 4, 4, target_H, target_W] for weight multiplication
        samples = gathered_xy.transpose(0, 1, 2, 4, 3, 5)  # [B, C, 4, 4, target_H, target_W]

        # Apply weights and sum over the 4x4 kernel dimensions
        # weight_grid: [4, 4, target_H, target_W] broadcasts with samples
        result = mx.sum(samples * weight_grid[None, None, :, :, :, :], axis=(2, 3))

    return result


class HTSAT(nn.Module):
    """Hierarchical Token-Semantic Audio Transformer.

    A Swin Transformer-based audio encoder that processes mel spectrograms
    into fixed-size embeddings.

    Architecture:
        Input: Log-mel spectrogram [B, C, F, T] (C=1 or 4 for fusion)
        → BatchNorm (on frequency axis)
        → reshape_mel2img → [B, C, 256, 256]
        → PatchEmbed (with optional fusion) → [B, N, embed_dim]
        → 4 stages of Swin Transformer blocks
        → LayerNorm → Reshape → Grouped pooling
        → [B, hidden_size]

    Args:
        config: Audio encoder configuration
    """

    def __init__(self, config: CLAPAudioConfig):
        super().__init__()
        self.config = config
        self.enable_fusion = config.enable_fusion
        self.freq_ratio = 4  # Used in reshape and pooling
        self.spec_size = config.spec_size  # 256

        # Batch normalization on frequency dimension (64 mels)
        self.batch_norm = nn.BatchNorm(config.n_mels)

        # Patch embedding with optional fusion
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            in_chans=1,
            embed_dim=config.embed_dim,
            flatten=False,  # Keep spatial dims for Swin
            enable_fusion=config.enable_fusion,
        )

        # Stochastic depth decay rule
        num_layers = sum(config.depths)
        dpr = [
            x.item()
            for x in mx.linspace(0, config.drop_path_rate, num_layers)
        ]

        # Build stages
        self.layers = []
        dim = config.embed_dim
        dpr_idx = 0

        for i, (depth, num_heads) in enumerate(zip(config.depths, config.num_heads, strict=False)):
            layer = BasicLayer(
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                window_size=config.window_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[dpr_idx : dpr_idx + depth],
                downsample=(i < len(config.depths) - 1),  # No downsample on last stage
            )
            self.layers.append(layer)
            dpr_idx += depth

            # Update dim for next stage (doubled by patch merging)
            if i < len(config.depths) - 1:
                dim *= 2

        self.norm = nn.LayerNorm(dim)

        # Store final dim for reference (should equal hidden_size for default config)
        self._final_dim = dim

    def reshape_mel2img(self, x: mx.array) -> mx.array:
        """Reshape mel spectrogram to image format for Swin Transformer.

        This transforms the input from [B, C, T, F] to [B, C, 256, 256] by:
        1. Interpolating time dimension to spec_size * freq_ratio (1024)
        2. Reshaping to reorganize frequency bins

        Args:
            x: Input tensor [B, C, T, F] where T=time, F=freq (64 mels)

        Returns:
            Reshaped tensor [B, C, 256, 256]
        """
        _, _, time_length, freq_length = x.shape

        spec_width = int(self.spec_size * self.freq_ratio)  # 256 * 4 = 1024
        spec_height = self.spec_size // self.freq_ratio  # 256 // 4 = 64

        # Interpolate time dimension if needed
        if time_length < spec_width:
            x = interpolate_2d(x, (spec_width, freq_length))

        # Interpolate freq dimension if needed
        if freq_length < spec_height:
            x = interpolate_2d(x, (x.shape[2], spec_height))

        batch, channels, time, freq = x.shape

        # Reshape: [B, C, T, F] -> [B, C*freq_ratio, T//freq_ratio, F]
        x = mx.reshape(x, (batch, channels * self.freq_ratio, time // self.freq_ratio, freq))

        # Permute: [B, C*freq_ratio, T//freq_ratio, F] -> [B, C*freq_ratio, F, T//freq_ratio]
        x = mx.transpose(x, (0, 1, 3, 2))

        # Reshape: [B, C*freq_ratio, F, T//freq_ratio] -> [B, C, F*freq_ratio, T//freq_ratio]
        x = mx.reshape(x, (batch, channels, freq * self.freq_ratio, time // self.freq_ratio))

        return x

    def forward_features(
        self,
        x: mx.array,
        is_longer: mx.array | None = None,
    ) -> mx.array:
        """Extract features before final projection.

        Args:
            x: Input mel spectrogram [B, C, F, T] where C=1 or 4 for fusion
            is_longer: Boolean tensor [B] indicating which samples need fusion

        Returns:
            Features [B, hidden_size] before projection
        """
        B = x.shape[0]

        # MLX input: [B, C, F, T] where C=1 or 4, F=64, T=time
        # HuggingFace input: [B, C, T, F] - note T and F are swapped
        # First, convert to HF format for consistent processing
        x = mx.transpose(x, (0, 1, 3, 2))  # [B, C, F, T] -> [B, C, T, F]

        # Apply batch norm on frequency axis (F=64 mels)
        # HuggingFace: transpose(1,3) -> BN2d(64) -> transpose(1,3)
        # This normalizes over the F dimension
        # For MLX, we need F in the last position since BatchNorm normalizes last dim
        # [B, C, T, F] -> permute so F is last -> normalize -> permute back
        # Simplest: [B, C, T, F] already has F last, so just apply!
        x = self.batch_norm(x)  # Normalizes over F=64 (last dim)

        # Reshape mel spectrogram to image format
        # Input: [B, C, T, F] -> Output: [B, C, H=256, W=256]
        x = self.reshape_mel2img(x)

        # Output is [B, C, 256, 256] - ready for patch embedding

        # Determine which samples need fusion
        is_longer_idx = None
        if self.enable_fusion and is_longer is not None:
            # MLX doesn't have argwhere/nonzero, so we convert to numpy briefly
            import numpy as np
            is_longer_np = np.array(is_longer.flatten())
            idx_np = np.nonzero(is_longer_np)[0]
            if len(idx_np) > 0:
                is_longer_idx = mx.array(idx_np)

        # Patch embedding (keeps spatial dims)
        x = self.patch_embed(x, is_longer_idx=is_longer_idx)  # [B, H, W, C]
        B, H, W, C = x.shape
        x = mx.reshape(x, (B, H * W, C))  # [B, L, C]

        # Pass through Swin stages
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        # Final norm
        x = self.norm(x)

        # Reshape and pool using HuggingFace's grouped pooling
        # x is [B, L, C] where L = H * W
        n_channels = x.shape[-1]

        # Reshape to [B, H, W, C] then to [B, C, H, W]
        x = mx.reshape(x, (B, H, W, n_channels))
        x = mx.transpose(x, (0, 3, 1, 2))  # [B, C, H, W]

        # Grouped pooling like HuggingFace:
        # c_freq_bin = n_frequencies // freq_ratio
        # Reshape to [B, C, n_freq // c_freq_bin, c_freq_bin, n_temp]
        # Then permute and reshape to [B, C, c_freq_bin, -1]
        # Then flatten from dim 2 -> [B, C, c_freq_bin * ...]
        # Finally adaptive avg pool to [B, C, 1] -> [B, C]
        n_frequencies = x.shape[2]  # H
        n_temp = x.shape[3]  # W
        c_freq_bin = n_frequencies // self.freq_ratio

        x = mx.reshape(x, (B, n_channels, n_frequencies // c_freq_bin, c_freq_bin, n_temp))
        # Permute: [B, C, n_freq//c_freq_bin, c_freq_bin, n_temp] -> [B, C, c_freq_bin, n_freq//c_freq_bin, n_temp]
        x = mx.transpose(x, (0, 1, 3, 2, 4))
        # Reshape: [B, C, c_freq_bin, (n_freq//c_freq_bin)*n_temp]
        x = mx.reshape(x, (B, n_channels, c_freq_bin, -1))

        # Flatten from dim 2: [B, C, c_freq_bin, ...] -> [B, C, flat]
        x = mx.reshape(x, (B, n_channels, -1))

        # Adaptive average pooling: mean over last dimension -> [B, C]
        x = mx.mean(x, axis=-1)

        return x

    def __call__(
        self,
        x: mx.array,
        is_longer: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input mel spectrogram [B, C, F, T] where C=1 or 4 for fusion
            is_longer: Boolean tensor [B] indicating which samples need fusion

        Returns:
            Audio embeddings [B, hidden_size]
        """
        x = self.forward_features(x, is_longer=is_longer)
        # Note: No fc layer - output goes directly to audio_projection
        return x


class AudioFusion(nn.Module):
    """Fusion module for variable-length audio.

    Handles audio longer than the model's fixed input size by:
    1. Splitting audio into overlapping chunks
    2. Processing each chunk independently
    3. Aggregating chunk embeddings

    Args:
        config: Audio encoder configuration
    """

    def __init__(self, config: CLAPAudioConfig):
        super().__init__()
        self.config = config

        # Fusion type determines aggregation method
        if config.fusion_type == "aff_2d":
            # Attention-based feature fusion
            self.fusion_weight = mx.ones((1,))
        else:
            self.fusion_weight = None

    def __call__(
        self,
        mel: mx.array,
        encoder: HTSAT,
        chunk_size: int = 1024,
        overlap: float = 0.5,
    ) -> mx.array:
        """Process variable-length audio with fusion.

        Args:
            mel: Mel spectrogram [B, 1, F, T]
            encoder: HTSAT encoder
            chunk_size: Size of each chunk (time frames)
            overlap: Overlap ratio between chunks

        Returns:
            Fused embeddings [B, hidden_size]
        """
        B, _, F, T = mel.shape

        if chunk_size >= T:
            # No fusion needed
            return encoder(mel)

        # Calculate chunk parameters
        hop = int(chunk_size * (1 - overlap))
        num_chunks = (T - chunk_size) // hop + 1

        # Process chunks
        chunk_embeddings = []
        for i in range(num_chunks):
            start = i * hop
            end = start + chunk_size
            chunk = mel[:, :, :, start:end]
            emb = encoder(chunk)
            chunk_embeddings.append(emb)

        # Stack and aggregate
        chunk_embeddings = mx.stack(chunk_embeddings, axis=1)  # [B, num_chunks, hidden]

        # Simple mean fusion (can be replaced with attention)
        fused = mx.mean(chunk_embeddings, axis=1)

        return fused
