"""Tests for CLAP model layers."""

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestPatchEmbed:
    """Tests for PatchEmbed layer."""

    def test_output_shape_flatten(self):
        """Test output shape with flatten=True."""
        from mlx_audio.models.clap.layers.patch_embed import PatchEmbed

        embed = PatchEmbed(
            patch_size=4,
            patch_stride=4,
            embed_dim=96,
            flatten=True,
        )

        # Input: [B, C, H, W]
        x = mx.random.normal((2, 1, 64, 256))
        out = embed(x)

        # H' = 64 / 4 = 16, W' = 256 / 4 = 64
        # N = 16 * 64 = 1024
        assert out.shape == (2, 1024, 96)

    def test_output_shape_no_flatten(self):
        """Test output shape with flatten=False."""
        from mlx_audio.models.clap.layers.patch_embed import PatchEmbed

        embed = PatchEmbed(
            patch_size=4,
            patch_stride=4,
            embed_dim=96,
            flatten=False,
        )

        x = mx.random.normal((2, 1, 64, 256))
        out = embed(x)

        # Output: [B, H', W', C]
        assert out.shape == (2, 16, 64, 96)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestWindowAttention:
    """Tests for WindowAttention layer."""

    def test_output_shape(self):
        """Test attention output shape."""
        from mlx_audio.models.clap.layers.swin_block import WindowAttention

        attn = WindowAttention(
            dim=96,
            window_size=(8, 8),
            num_heads=4,
        )

        # Input: [num_windows*B, window_size^2, dim]
        x = mx.random.normal((8, 64, 96))  # 8 windows, 8*8=64 tokens
        out = attn(x)

        assert out.shape == x.shape

    def test_attention_weights(self):
        """Test that attention produces valid outputs."""
        from mlx_audio.models.clap.layers.swin_block import WindowAttention

        attn = WindowAttention(
            dim=96,
            window_size=(4, 4),
            num_heads=4,
        )

        x = mx.random.normal((4, 16, 96))
        out = attn(x)

        # Output should be finite
        assert mx.all(mx.isfinite(out))


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestSwinTransformerBlock:
    """Tests for SwinTransformerBlock layer."""

    def test_output_shape(self):
        """Test block output shape."""
        from mlx_audio.models.clap.layers.swin_block import SwinTransformerBlock

        block = SwinTransformerBlock(
            dim=96,
            num_heads=4,
            window_size=8,
            shift_size=0,
        )

        # Input: [B, L, C] where L = H * W
        x = mx.random.normal((2, 256, 96))  # 16x16 spatial
        out = block(x, H=16, W=16)

        assert out.shape == x.shape

    def test_shifted_window(self):
        """Test shifted window attention."""
        from mlx_audio.models.clap.layers.swin_block import SwinTransformerBlock

        block = SwinTransformerBlock(
            dim=96,
            num_heads=4,
            window_size=8,
            shift_size=4,  # Half window shift
        )

        x = mx.random.normal((2, 256, 96))
        out = block(x, H=16, W=16)

        assert out.shape == x.shape
        assert mx.all(mx.isfinite(out))


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestPatchMerging:
    """Tests for PatchMerging layer."""

    def test_output_shape(self):
        """Test patch merging output shape."""
        from mlx_audio.models.clap.layers.swin_block import PatchMerging

        merge = PatchMerging(dim=96)

        # Input: [B, L, C] where L = H * W
        x = mx.random.normal((2, 256, 96))  # 16x16 spatial
        out, new_H, new_W = merge(x, H=16, W=16)

        # Output: [B, L/4, 2*C]
        assert out.shape == (2, 64, 192)
        assert new_H == 8
        assert new_W == 8


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestBasicLayer:
    """Tests for BasicLayer (stage of Swin blocks)."""

    def test_output_shape_with_downsample(self):
        """Test basic layer with downsampling."""
        from mlx_audio.models.clap.layers.swin_block import BasicLayer

        layer = BasicLayer(
            dim=96,
            depth=2,
            num_heads=4,
            window_size=8,
            downsample=True,
        )

        x = mx.random.normal((2, 256, 96))  # 16x16 spatial
        out, new_H, new_W = layer(x, H=16, W=16)

        # After 2 blocks + downsample: H/2, W/2, 2*C
        assert out.shape == (2, 64, 192)
        assert new_H == 8
        assert new_W == 8

    def test_output_shape_no_downsample(self):
        """Test basic layer without downsampling."""
        from mlx_audio.models.clap.layers.swin_block import BasicLayer

        layer = BasicLayer(
            dim=96,
            depth=2,
            num_heads=4,
            window_size=8,
            downsample=False,
        )

        x = mx.random.normal((2, 256, 96))
        out, new_H, new_W = layer(x, H=16, W=16)

        # No downsample: same shape
        assert out.shape == x.shape
        assert new_H == 16
        assert new_W == 16
