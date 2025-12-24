"""Tests for DeepFilterNet model architecture."""

import numpy as np
import pytest

import mlx.core as mx

from mlx_audio.models.enhance import DeepFilterNet, DeepFilterNetConfig


class TestDeepFilterNet:
    """Tests for DeepFilterNet model."""

    @pytest.fixture
    def config(self):
        """Small config for faster tests."""
        return DeepFilterNetConfig(
            sample_rate=16000,
            fft_size=640,
            frame_size=320,
            hop_size=160,
            erb_bands=32,
            df_bins=64,
            hidden_size=64,
            num_groups=1,
            enc_layers=1,
        )

    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        return DeepFilterNet(config)

    def test_model_creation(self, model):
        """Test model can be created."""
        assert isinstance(model, DeepFilterNet)
        assert model.config is not None

    def test_model_has_encoder(self, model):
        """Test model has encoder components."""
        assert hasattr(model, "erb_encoder")

    def test_model_has_decoder(self, model):
        """Test model has decoder components."""
        assert hasattr(model, "df_decoder")

    def test_forward_pass_shape(self, model, config):
        """Test forward pass produces correct output shape."""
        # Create input: (batch, samples)
        batch_size = 2
        duration_samples = config.sample_rate  # 1 second
        audio = mx.array(
            np.random.randn(batch_size, duration_samples).astype(np.float32)
        )

        # Forward pass
        output = model(audio)

        # Output should match input shape
        assert output.shape == audio.shape

    def test_forward_single_sample(self, model, config):
        """Test forward pass with single sample (no batch dim)."""
        duration_samples = config.sample_rate // 2  # 0.5 seconds
        audio = mx.array(np.random.randn(duration_samples).astype(np.float32))

        output = model(audio)

        assert output.shape == audio.shape

    def test_different_input_lengths(self, model, config):
        """Test model handles various input lengths."""
        for duration_sec in [0.1, 0.5, 1.0]:
            samples = int(config.sample_rate * duration_sec)
            audio = mx.array(np.random.randn(samples).astype(np.float32))

            output = model(audio)
            assert output.shape == audio.shape

    def test_output_dtype(self, model, config):
        """Test output is float32."""
        audio = mx.array(
            np.random.randn(config.sample_rate).astype(np.float32)
        )

        output = model(audio)
        assert output.dtype == mx.float32


class TestDeepFilterNetLayers:
    """Tests for individual DeepFilterNet layers."""

    def test_erb_filterbank(self):
        """Test ERB filterbank function."""
        from mlx_audio.models.enhance.layers.erb import erb_filterbank

        fb = erb_filterbank(
            n_fft=640,
            sample_rate=16000,
            n_bands=32,
        )

        # Should produce a filterbank matrix
        assert fb.shape == (32, 321)  # (n_bands, n_fft//2+1)

    def test_grouped_gru(self):
        """Test GroupedGRU layer."""
        from mlx_audio.models.enhance.layers.grouped import GroupedGRU

        gru = GroupedGRU(
            input_size=64,
            hidden_size=128,
            num_groups=1,
        )

        # Test forward pass
        x = mx.array(np.random.randn(2, 10, 64).astype(np.float32))

        output, h_n = gru(x)

        assert output.shape == (2, 10, 128)

    def test_grouped_linear(self):
        """Test GroupedLinear layer."""
        from mlx_audio.models.enhance.layers.grouped import GroupedLinear

        linear = GroupedLinear(
            input_size=64,
            output_size=128,
            num_groups=1,
        )

        x = mx.array(np.random.randn(2, 10, 64).astype(np.float32))
        output = linear(x)

        assert output.shape == (2, 10, 128)
