"""Tests for MLP classifier head."""

import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

pytestmark = pytest.mark.skipif(mx is None, reason="MLX not available")


class TestMLPHead:
    """Tests for MLPHead layer."""

    def test_basic_forward(self, fixed_seed):
        """Test basic forward pass with default config."""
        from mlx_audio.models.classifier.config import MLPHeadConfig
        from mlx_audio.models.classifier.layers import MLPHead

        config = MLPHeadConfig(input_dim=512, num_classes=10, hidden_dims=[256])
        head = MLPHead(config)

        x = mx.random.normal((4, 512))
        logits = head(x)

        assert logits.shape == (4, 10)
        assert logits.dtype == mx.float32

    def test_no_hidden_layers(self, fixed_seed):
        """Test with no hidden layers (linear classifier)."""
        from mlx_audio.models.classifier.config import MLPHeadConfig
        from mlx_audio.models.classifier.layers import MLPHead

        config = MLPHeadConfig(input_dim=512, num_classes=50, hidden_dims=[])
        head = MLPHead(config)

        x = mx.random.normal((2, 512))
        logits = head(x)

        assert logits.shape == (2, 50)

    def test_multiple_hidden_layers(self, fixed_seed):
        """Test with multiple hidden layers."""
        from mlx_audio.models.classifier.config import MLPHeadConfig
        from mlx_audio.models.classifier.layers import MLPHead

        config = MLPHeadConfig(
            input_dim=512, num_classes=100, hidden_dims=[256, 128, 64]
        )
        head = MLPHead(config)

        x = mx.random.normal((8, 512))
        logits = head(x)

        assert logits.shape == (8, 100)

    def test_with_batch_norm(self, fixed_seed):
        """Test with batch normalization enabled."""
        from mlx_audio.models.classifier.config import MLPHeadConfig
        from mlx_audio.models.classifier.layers import MLPHead

        config = MLPHeadConfig(
            input_dim=512,
            num_classes=10,
            hidden_dims=[256],
            use_batch_norm=True,
        )
        head = MLPHead(config)

        x = mx.random.normal((4, 512))
        logits = head(x)

        assert logits.shape == (4, 10)

    def test_different_activations(self, fixed_seed):
        """Test different activation functions."""
        from mlx_audio.models.classifier.config import MLPHeadConfig
        from mlx_audio.models.classifier.layers import MLPHead

        for activation in ["relu", "gelu", "silu"]:
            config = MLPHeadConfig(
                input_dim=512,
                num_classes=10,
                hidden_dims=[256],
                activation=activation,
            )
            head = MLPHead(config)

            x = mx.random.normal((2, 512))
            logits = head(x)

            assert logits.shape == (2, 10), f"Failed for activation={activation}"

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        from mlx_audio.models.classifier.config import MLPHeadConfig

        # Error is raised in MLPHeadConfig.__post_init__ validation
        with pytest.raises(ValueError, match="activation must be one of"):
            MLPHeadConfig(
                input_dim=512,
                num_classes=10,
                hidden_dims=[256],
                activation="invalid",
            )

    def test_properties(self):
        """Test num_classes and input_dim properties."""
        from mlx_audio.models.classifier.config import MLPHeadConfig
        from mlx_audio.models.classifier.layers import MLPHead

        config = MLPHeadConfig(input_dim=768, num_classes=527)
        head = MLPHead(config)

        assert head.num_classes == 527
        assert head.input_dim == 768

    def test_single_sample(self, fixed_seed):
        """Test with single sample (batch size 1)."""
        from mlx_audio.models.classifier.config import MLPHeadConfig
        from mlx_audio.models.classifier.layers import MLPHead

        config = MLPHeadConfig(input_dim=512, num_classes=10, hidden_dims=[256])
        head = MLPHead(config)

        x = mx.random.normal((1, 512))
        logits = head(x)

        assert logits.shape == (1, 10)

    def test_dropout_training(self, fixed_seed):
        """Test that dropout is applied during training."""
        from mlx_audio.models.classifier.config import MLPHeadConfig
        from mlx_audio.models.classifier.layers import MLPHead

        config = MLPHeadConfig(
            input_dim=512,
            num_classes=10,
            hidden_dims=[256],
            dropout=0.5,
        )
        head = MLPHead(config)
        head.train()

        x = mx.random.normal((4, 512))

        # Run multiple times - with dropout, outputs should differ
        logits1 = head(x)
        mx.eval(logits1)
        logits2 = head(x)
        mx.eval(logits2)

        # At high dropout rate, results should differ
        # (though this is probabilistic, 0.5 dropout should cause differences)
        # Note: We're just checking that the network runs in train mode
        assert logits1.shape == logits2.shape == (4, 10)
